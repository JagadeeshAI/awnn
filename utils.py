import torch
import torch.nn as nn
import torch.nn.functional as F
import timm
import math
from tqdm import tqdm

class AWNNTransformerMLP(nn.Module):
    """
    AWNN MLP layers for transformer following the paper exactly
    Replaces the standard MLP in transformer blocks
    """

    def __init__(self, embed_dim, mlp_ratio=4.0, num_layers=2, k=0.8, max_neurons=512):
        super(AWNNTransformerMLP, self).__init__()
        self.embed_dim = embed_dim
        self.hidden_dim = int(embed_dim * mlp_ratio)
        self.num_layers = num_layers
        self.k = k
        self.max_neurons = max_neurons

        # Variational parameters ν_ℓ (nu in paper)
        self.nu = nn.ParameterList([
            nn.Parameter(torch.tensor(-2.5)) for _ in range(num_layers)
        ])

        # Dynamic layers - will grow/shrink during training
        self.dynamic_layers = nn.ModuleList()
        self.current_widths = [int(self.hidden_dim * 0.5)] * num_layers  # Start with 50% of target size

        # Priors from paper
        self.mu_lambda = -1.5
        self.sigma_lambda = 2.0
        self.sigma_theta = 20.0

        # Initialize layers
        self._initialize_layers()

    def _initialize_layers(self):
        """Initialize with Kaiming+ from paper"""
        self.dynamic_layers = nn.ModuleList()

        # First layer: embed_dim -> current_width[0]
        layer = nn.Linear(self.embed_dim, self.current_widths[0], bias=True)
        self._kaiming_plus_init(layer, self.embed_dim, self.current_widths[0])
        self.dynamic_layers.append(layer)

        # Hidden layers
        for i in range(self.num_layers - 1):
            prev_width = self.current_widths[i]
            curr_width = self.current_widths[i + 1]
            layer = nn.Linear(prev_width, curr_width, bias=True)
            self._kaiming_plus_init(layer, prev_width, curr_width)
            self.dynamic_layers.append(layer)

        # Output layer back to embed_dim
        self.output_layer = nn.Linear(self.current_widths[-1], self.embed_dim, bias=True)
        self._kaiming_plus_init(self.output_layer, self.current_widths[-1], self.embed_dim)

    def _kaiming_plus_init(self, layer, prev_width, curr_width):
        """Kaiming+ initialization from Theorem 3.1"""
        # Calculate ∑f_ℓ²(j) for importance function
        f_squared_sum = 0.0
        lam = 0.1  # Reasonable lambda for initialization

        for j in range(1, curr_width + 1):
            f_j = self.discretized_exponential(torch.tensor(j, dtype=torch.float32), lam)
            f_squared_sum += f_j.item() ** 2

        if f_squared_sum > 0:
            std = math.sqrt(2.0 / (prev_width * f_squared_sum))
        else:
            std = math.sqrt(2.0 / prev_width)

        with torch.no_grad():
            layer.weight.normal_(0, std)
            layer.bias.zero_()

    def discretized_exponential(self, x, lam):
        """Equation 6 from paper"""
        return (1 - torch.exp(-lam * (x + 1))) - (1 - torch.exp(-lam * x))

    def compute_truncated_width(self, nu_l):
        """Compute D_l as quantile function"""
        lam = torch.exp(nu_l) + 1e-8

        cumulative = 0.0
        for d in range(1, self.max_neurons + 1):
            prob = self.discretized_exponential(torch.tensor(d, dtype=torch.float32), lam)
            cumulative += prob.item()
            if cumulative >= self.k:
                return min(d, self.max_neurons)
        return self.max_neurons

    def compute_importance_weights(self, D_l, nu_l):
        """Compute f_l(j; nu_l) for soft ordering"""
        lam = torch.exp(nu_l) + 1e-8
        weights = []
        for j in range(1, D_l + 1):
            weight = self.discretized_exponential(torch.tensor(j, dtype=torch.float32), lam)
            weights.append(weight)
        weights_tensor = torch.stack(weights)
        weights_tensor = weights_tensor / weights_tensor.sum()
        return weights_tensor * D_l

    def update_width(self):
        """Algorithm 1: update_width function"""
        new_widths = []

        # Compute new widths
        for l in range(self.num_layers):
            D_l = self.compute_truncated_width(self.nu[l])
            new_widths.append(D_l)

        # Create new layers with updated widths
        new_dynamic_layers = nn.ModuleList()

        for l in range(self.num_layers):
            # Input dimension for this layer
            if l == 0:
                input_dim = self.embed_dim
            else:
                input_dim = new_widths[l-1]

            new_width = new_widths[l]
            old_width = self.current_widths[l]

            # Create new layer
            new_layer = nn.Linear(input_dim, new_width, bias=True)

            # Copy weights if possible
            if l < len(self.dynamic_layers):
                old_layer = self.dynamic_layers[l]
                with torch.no_grad():
                    min_out = min(old_width, new_width)
                    min_in = min(old_layer.in_features, input_dim)
                    if min_out > 0 and min_in > 0:
                        new_layer.weight[:min_out, :min_in] = old_layer.weight[:min_out, :min_in]
                        new_layer.bias[:min_out] = old_layer.bias[:min_out]

                    # Initialize new neurons
                    if new_width > min_out or input_dim > min_in:
                        self._kaiming_plus_init_partial(new_layer, min_out, new_width, input_dim)
            else:
                self._kaiming_plus_init(new_layer, input_dim, new_width)

            new_dynamic_layers.append(new_layer)

        # Update output layer
        new_output = nn.Linear(new_widths[-1], self.embed_dim, bias=True)
        if hasattr(self, 'output_layer'):
            with torch.no_grad():
                min_in = min(self.output_layer.in_features, new_widths[-1])
                if min_in > 0:
                    new_output.weight[:, :min_in] = self.output_layer.weight[:, :min_in]
                    new_output.bias[:] = self.output_layer.bias[:]
        else:
            self._kaiming_plus_init(new_output, new_widths[-1], self.embed_dim)

        # Replace layers - ensure they're on the same device
        device = next(self.parameters()).device

        # Move each layer to the correct device before replacing
        for layer in new_dynamic_layers:
            layer.to(device)
        new_output.to(device)

        self.dynamic_layers = new_dynamic_layers
        self.output_layer = new_output
        self.current_widths = new_widths

        return new_widths

    def _kaiming_plus_init_partial(self, layer, old_width, new_width, input_dim):
        """Initialize only new neurons"""
        if new_width > old_width:
            f_squared_sum = 0.0
            lam = 0.1
            for j in range(old_width + 1, new_width + 1):
                f_j = self.discretized_exponential(torch.tensor(j, dtype=torch.float32), lam)
                f_squared_sum += f_j.item() ** 2

            if f_squared_sum > 0:
                std = math.sqrt(2.0 / (input_dim * f_squared_sum))
            else:
                std = math.sqrt(2.0 / input_dim)

            layer.weight[old_width:new_width].normal_(0, std)
            layer.bias[old_width:new_width].zero_()

    def forward(self, x):
        """Forward with soft ordering and bounded activations"""
        h = x

        for l, layer in enumerate(self.dynamic_layers):
            D_l = self.current_widths[l]

            # Forward pass
            h_raw = layer(h)

            # Apply importance weighting (Equation 10)
            importance_weights = self.compute_importance_weights(D_l, self.nu[l])
            if importance_weights.device != h_raw.device:
                importance_weights = importance_weights.to(h_raw.device)

            # Expand weights for batch and sequence dimensions
            # x shape: [batch_size, seq_len, embed_dim]
            # weights shape: [D_l] -> [1, 1, D_l]
            h_weighted = h_raw * importance_weights.unsqueeze(0).unsqueeze(0)

            # Bounded activation (ReLU6 from paper)
            h = F.relu6(h_weighted)

        # Output layer
        output = self.output_layer(h)

        # For transformer integration, only return the tensor
        return output

    def compute_kl_terms(self):
        """Separate method to compute KL terms for ELBO"""
        total_kl_lambda = 0.0
        total_kl_theta = 0.0

        for l in range(self.num_layers):
            # KL divergences for ELBO
            kl_lambda = self._kl_divergence_lambda(self.nu[l])
            total_kl_lambda += kl_lambda

            layer = self.dynamic_layers[l]
            D_l = self.current_widths[l]
            kl_theta = self._kl_divergence_theta(layer, D_l)
            total_kl_theta += kl_theta

        return total_kl_lambda, total_kl_theta

    def _kl_divergence_lambda(self, nu_l):
        """KL divergence for width parameter"""
        mu_q, sigma_q = nu_l, 1.0
        mu_p, sigma_p = self.mu_lambda, self.sigma_lambda

        kl = math.log(sigma_p / sigma_q) + (sigma_q**2 + (mu_q - mu_p)**2) / (2 * sigma_p**2) - 0.5
        return kl

    def _kl_divergence_theta(self, layer, D_l):
        """KL divergence for weights"""
        weights = layer.weight
        bias = layer.bias
        kl = (torch.sum(weights**2) + torch.sum(bias**2)) / (2 * self.sigma_theta**2)
        return kl


class AWNNDeiTTiny(nn.Module):
    """
    DeiT Tiny with AWNN MLPs replacing standard transformer MLPs
    Following paper exactly - no LoRA, pretrained=False, full training
    """

    def __init__(self, num_classes=100, pretrained=False):
        super(AWNNDeiTTiny, self).__init__()

        # Create base DeiT model WITHOUT pretraining (as requested)
        self.base_model = timm.create_model(
            "deit_tiny_patch16_224",
            pretrained=pretrained,  # Set to False as requested
            num_classes=num_classes
        )

        # Get model dimensions
        self.embed_dim = self.base_model.embed_dim  # 192 for deit_tiny

        # Replace MLP layers in transformer blocks with AWNN MLPs
        self.awnn_mlps = nn.ModuleList()
        for i, block in enumerate(self.base_model.blocks):
            # Create AWNN MLP to replace standard MLP
            awnn_mlp = AWNNTransformerMLP(
                embed_dim=self.embed_dim,
                mlp_ratio=4.0,  # Standard transformer ratio
                num_layers=2,   # Two-layer MLP as in transformers
                k=0.8,          # Quantile parameter
                max_neurons=1024  # Maximum neurons for CIFAR-100
            )
            self.awnn_mlps.append(awnn_mlp)

            # Replace the MLP in the transformer block
            block.mlp = awnn_mlp

        self.num_transformer_blocks = len(self.base_model.blocks)

    def update_all_widths(self):
        """Update widths for all AWNN MLPs (Algorithm 1)"""
        total_widths = []
        for awnn_mlp in self.awnn_mlps:
            widths = awnn_mlp.update_width()
            total_widths.append(widths)
        return total_widths

    def get_current_widths(self):
        """Get current widths of all AWNN MLPs"""
        all_widths = []
        for awnn_mlp in self.awnn_mlps:
            all_widths.append(awnn_mlp.current_widths)
        return all_widths

    def compute_total_elbo_terms(self):
        """Compute total ELBO KL terms from all AWNN MLPs"""
        total_kl_lambda = 0.0
        total_kl_theta = 0.0

        for awnn_mlp in self.awnn_mlps:
            kl_lambda, kl_theta = awnn_mlp.compute_kl_terms()
            total_kl_lambda += kl_lambda
            total_kl_theta += kl_theta

        return total_kl_lambda, total_kl_theta

    def forward(self, x):
        """Forward pass through AWNN-enhanced DeiT"""
        # Standard forward pass - the AWNN MLPs are already integrated
        logits = self.base_model(x)

        # Compute ELBO terms from all AWNN MLPs
        kl_lambda, kl_theta = self.compute_total_elbo_terms()

        return logits, kl_lambda, kl_theta


def compute_awnn_elbo_loss(model, x, y, N, M):
    """Compute ELBO loss following Equation 9 from paper"""
    logits, kl_lambda, kl_theta = model(x)

    # Predictive loss
    pred_loss = F.cross_entropy(logits, y)

    # Scale by N/M for mini-batch (from paper)
    scaled_pred_loss = pred_loss * (N / M)

    # ELBO = predictive_loss + KL_lambda + KL_theta
    elbo = scaled_pred_loss + kl_lambda + kl_theta

    return elbo, pred_loss, kl_lambda, kl_theta


def train_one_epoch_awnn(model, loader, optimizer, criterion_fn, N, M, device):
    """Training with AWNN ELBO optimization"""
    model.train()
    total_elbo, total_pred, total_kl_l, total_kl_t = 0, 0, 0, 0
    correct, total = 0, 0

    for imgs, labels in tqdm(loader, desc="Train", leave=False):
        imgs, labels = imgs.to(device), labels.to(device)

        optimizer.zero_grad()

        # Compute AWNN ELBO loss
        elbo, pred_loss, kl_lambda, kl_theta = criterion_fn(model, imgs, labels, N, M)

        elbo.backward()
        optimizer.step()

        # Statistics
        total_elbo += elbo.item() * imgs.size(0)
        total_pred += pred_loss.item() * imgs.size(0)
        total_kl_l += kl_lambda.item() * imgs.size(0)
        total_kl_t += kl_theta.item() * imgs.size(0)

        # Accuracy
        with torch.no_grad():
            outputs, _, _ = model(imgs)
            correct += (outputs.argmax(1) == labels).sum().item()
            total += labels.size(0)

    return (total_elbo / total, total_pred / total,
            total_kl_l / total, total_kl_t / total, correct / total)


@torch.no_grad()
def validate_awnn(model, loader, device):
    """Validation for AWNN model"""
    model.eval()
    total_loss, correct, total = 0, 0, 0

    for imgs, labels in tqdm(loader, desc="Val", leave=False):
        imgs, labels = imgs.to(device), labels.to(device)

        outputs, _, _ = model(imgs)
        loss = F.cross_entropy(outputs, labels)

        total_loss += loss.item() * imgs.size(0)
        correct += (outputs.argmax(1) == labels).sum().item()
        total += labels.size(0)

    return total_loss / total, correct / total


def train_one_epoch_standard(model, loader, optimizer, device):
    """Standard training without AWNN"""
    model.train()
    total_loss, correct, total = 0, 0, 0

    for imgs, labels in tqdm(loader, desc="Train", leave=False):
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


@torch.no_grad()
def validate_standard(model, loader, device):
    """Standard validation without AWNN"""
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
