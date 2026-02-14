"""
utils.py - PrunableDeiT model and training functions

Two-phase approach:
  Phase 1: Standard DeiT training (100% params, cross-entropy)
  Phase 2: Importance-based pruning (max 20%, KL + CE loss)
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
import timm
from tqdm import tqdm


# =============================================================================
# Model
# =============================================================================

class PrunableDeiT(nn.Module):
    """
    Standard DeiT with importance-based neuron pruning on MLP layers.

    Phase 1: Train normally with cross-entropy (all neurons active).
    Phase 2: Score neurons by activation × gradient, prune bottom 5%
             every 20 batches, up to 20% total compression.
    """

    def __init__(self, num_classes=100, pretrained=False):
        super().__init__()

        self.base_model = timm.create_model(
            "deit_tiny_patch16_224",
            pretrained=pretrained,
            num_classes=num_classes
        )

        self.num_blocks = len(self.base_model.blocks)

        # Pruning config
        self.prune_step_ratio = 0.05      # Remove at most 5% at once
        self.cooling_period = 20          # Wait 20 batches between prune steps
        self.min_neurons = 64             # Minimum neurons per block (safety floor)

        # Per-block tracking
        self.initial_widths = {}
        self._layer_activations = {}
        self._batches_since_last_prune = 0
        self._importance_scores = {}
        self._importance_count = {}

        for i, block in enumerate(self.base_model.blocks):
            hidden_dim = block.mlp.fc1.out_features
            self.initial_widths[i] = hidden_dim
            self._register_hook(i, block)

    def _register_hook(self, block_idx, block):
        """Hook to capture MLP hidden activations during training."""
        def hook_fn(module, input, output):
            if self.training:
                with torch.no_grad():
                    h = module.fc1(input[0])
                    h = module.act(h)
                    self._layer_activations[block_idx] = h.detach().abs().mean(dim=(0, 1))
        block.mlp.register_forward_hook(hook_fn)

    # -------------------------------------------------------------------------
    # Pruning logic
    # -------------------------------------------------------------------------

    def check_and_prune(self):
        """
        Importance-based pruning (called after each optimizer step in Phase 2):
          1. Every batch: accumulate importance = activation × gradient
          2. Every `cooling_period` batches: prune bottom 5% per block
          3. Respect 20% max total cap per block

        Returns list of (block_idx, num_pruned, old_width).
        """
        if not self.training:
            return []

        self._batches_since_last_prune += 1
        self._accumulate_importance()

        # Cooling period
        if self._batches_since_last_prune < self.cooling_period:
            return []

        all_pruned = []

        for block_idx in range(self.num_blocks):
            if block_idx not in self._importance_scores:
                continue

            mlp = self.base_model.blocks[block_idx].mlp
            hidden_dim = mlp.fc1.out_features
            initial_dim = self.initial_widths[block_idx]
            device = next(mlp.parameters()).device

            # Safety floor - don't prune below minimum
            if hidden_dim <= self.min_neurons:
                continue

            # Get importance scores
            scores = self._importance_scores[block_idx].to(device)
            if scores.numel() != hidden_dim:
                continue

            # Prune bottom 5% of initial width (but don't go below min)
            num_to_prune = max(1, int(initial_dim * self.prune_step_ratio))
            num_to_prune = min(num_to_prune, hidden_dim - self.min_neurons)

            # Sort by importance, remove least important
            _, sorted_indices = scores.sort()
            to_remove = sorted_indices[:num_to_prune]

            keep_mask = torch.ones(hidden_dim, dtype=torch.bool, device=device)
            keep_mask[to_remove] = False
            keep_indices = torch.where(keep_mask)[0]

            all_pruned.append((block_idx, num_to_prune, hidden_dim))
            self._prune_mlp(block_idx, keep_indices)

        if all_pruned:
            self._batches_since_last_prune = 0
            self._importance_scores = {}

        return all_pruned

    def _accumulate_importance(self):
        """Accumulate importance = activation_magnitude × gradient_magnitude."""
        for block_idx in range(self.num_blocks):
            if block_idx not in self._layer_activations:
                continue

            mlp = self.base_model.blocks[block_idx].mlp
            activations = self._layer_activations[block_idx]
            hidden_dim = mlp.fc1.out_features

            importance = activations.clone()
            if mlp.fc1.weight.grad is not None:
                grad_norm = mlp.fc1.weight.grad.abs().sum(dim=1)
                importance = importance * grad_norm

            importance = importance.detach().cpu()

            if block_idx not in self._importance_scores or \
               len(self._importance_scores[block_idx]) != hidden_dim:
                self._importance_scores[block_idx] = importance
                self._importance_count[block_idx] = 1
            else:
                self._importance_scores[block_idx] += importance
                self._importance_count[block_idx] += 1

    def _prune_mlp(self, block_idx, keep_indices):
        """Remove neurons from fc1 (output dim) and fc2 (input dim)."""
        mlp = self.base_model.blocks[block_idx].mlp
        device = next(mlp.parameters()).device
        keep_indices = keep_indices.to(device)
        new_width = len(keep_indices)

        old_fc1, old_fc2 = mlp.fc1, mlp.fc2

        new_fc1 = nn.Linear(old_fc1.in_features, new_width, bias=True).to(device)
        new_fc2 = nn.Linear(new_width, old_fc2.out_features, bias=True).to(device)

        with torch.no_grad():
            new_fc1.weight.data = old_fc1.weight.data[keep_indices]
            new_fc1.bias.data = old_fc1.bias.data[keep_indices]
            new_fc2.weight.data = old_fc2.weight.data[:, keep_indices]
            new_fc2.bias.data = old_fc2.bias.data

        mlp.fc1 = new_fc1
        mlp.fc2 = new_fc2

    # -------------------------------------------------------------------------
    # Stats
    # -------------------------------------------------------------------------

    def get_width_stats(self):
        """Returns list of (block_idx, initial_width, current_width)."""
        return [
            (i, self.initial_widths[i], self.base_model.blocks[i].mlp.fc1.out_features)
            for i in range(self.num_blocks)
        ]

    def get_total_neurons(self):
        return sum(
            self.base_model.blocks[i].mlp.fc1.out_features
            for i in range(self.num_blocks)
        )

    def get_initial_total(self):
        return sum(self.initial_widths.values())

    def forward(self, x):
        return self.base_model(x)


# =============================================================================
# Training functions
# =============================================================================

def train_phase1(model, loader, optimizer, device):
    """Phase 1: Standard cross-entropy training (100% params)."""
    model.train()
    total_loss, correct, total = 0, 0, 0

    for imgs, labels in tqdm(loader, desc="Phase1-Train", leave=False):
        imgs, labels = imgs.to(device), labels.to(device)
        optimizer.zero_grad()
        outputs = model(imgs)
        loss = F.cross_entropy(outputs, labels)
        loss.backward()
        optimizer.step()

        total_loss += loss.item() * imgs.size(0)
        correct += (outputs.argmax(1) == labels).sum().item()
        total += labels.size(0)

    return total_loss / total, correct / total


def train_phase2(model, loader, optimizer, device, reference_model,
                 temperature=4.0, alpha=0.5):
    """
    Phase 2: Pruning with KL + CE loss.

    Loss = alpha * CE(outputs, labels) + (1-alpha) * KL(outputs, ref_outputs)

    Args:
        model: PrunableDeiT being pruned
        reference_model: frozen Phase 1 model (plain timm DeiT, no hooks)
        temperature: KL softmax temperature
        alpha: balance between CE and KL (0.5 = equal)

    Returns:
        (total_loss, ce_loss, kl_loss, accuracy, neurons_pruned)
    """
    model.train()
    reference_model.eval()
    total_loss, total_ce, total_kl, correct, total = 0, 0, 0, 0, 0
    total_pruned = 0

    for imgs, labels in tqdm(loader, desc="Phase2-Prune", leave=False):
        imgs, labels = imgs.to(device), labels.to(device)
        optimizer.zero_grad()

        outputs = model(imgs)

        with torch.no_grad():
            ref_outputs = reference_model(imgs)

        ce_loss = F.cross_entropy(outputs, labels)
        kl_loss = F.kl_div(
            F.log_softmax(outputs / temperature, dim=-1),
            F.softmax(ref_outputs / temperature, dim=-1),
            reduction='batchmean'
        ) * (temperature ** 2)

        loss = alpha * ce_loss + (1 - alpha) * kl_loss

        loss.backward()
        optimizer.step()

        # Prune after backward
        pruned = model.check_and_prune()
        if pruned:
            for _, num_pruned, _ in pruned:
                total_pruned += num_pruned
            # Recreate optimizer to track new parameters
            optimizer = torch.optim.AdamW(
                model.parameters(),
                lr=optimizer.param_groups[0]['lr'],
                weight_decay=0.05
            )

        total_loss += loss.item() * imgs.size(0)
        total_ce += ce_loss.item() * imgs.size(0)
        total_kl += kl_loss.item() * imgs.size(0)
        correct += (outputs.argmax(1) == labels).sum().item()
        total += labels.size(0)

    return total_loss / total, total_ce / total, total_kl / total, correct / total, total_pruned


@torch.no_grad()
def validate(model, loader, device):
    """Validation with cross-entropy loss."""
    model.eval()
    total_loss, correct, total = 0, 0, 0

    for imgs, labels in tqdm(loader, desc="Val", leave=False):
        imgs, labels = imgs.to(device), labels.to(device)
        outputs = model(imgs)
        loss = F.cross_entropy(outputs, labels)
        total_loss += loss.item() * imgs.size(0)
        correct += (outputs.argmax(1) == labels).sum().item()
        total += labels.size(0)

    return total_loss / total, correct / total
