"""
awnn_cifar100.py - Main training script

Two-phase DeiT training with dynamic neuron pruning:
  Phase 1: Full training (20 epochs, cross-entropy, 100% params)
  Phase 2: Pruning (10 epochs, KL+CE loss, max 20% compression)
"""

import os
import copy
import json
import torch
import argparse
from data import get_dataloaders
from utils import PrunableDeiT, train_phase1, train_phase2, validate


def parse_args():
    parser = argparse.ArgumentParser(description='DeiT CIFAR-100 with Pruning')
    parser.add_argument('--pretrained', type=str, default='no', choices=['yes', 'no'])
    parser.add_argument('--compress', type=str, default='yes', choices=['yes', 'no'])
    return parser.parse_args()


def main():
    args = parse_args()
    use_pretrained = args.pretrained == 'yes'
    enable_compression = args.compress == 'yes'

    print("=" * 70)
    print("   DeiT CIFAR-100 Training with Dynamic Neuron Pruning")
    print("=" * 70)

    # Config
    DATA_DIR = "/media/jag/volD2/cifer100/cifer"
    PHASE1_EPOCHS = 20
    PHASE2_EPOCHS = 10
    BATCH_SIZE = 64
    LR_PHASE1 = 3e-4
    LR_PHASE2 = 1e-4
    DEVICE = "cuda" if torch.cuda.is_available() else "cpu"

    print(f"\nConfiguration:")
    print(f"  Pretrained:       {'YES' if use_pretrained else 'NO'}")
    print(f"  Phase 1 (Train):  {PHASE1_EPOCHS} epochs, LR={LR_PHASE1}")
    print(f"  Phase 2 (Prune):  {PHASE2_EPOCHS} epochs, LR={LR_PHASE2}")
    print(f"  Max Pruning:      Unlimited (floor: 64 neurons)")
    print(f"  Prune Step:       5% at a time")
    print(f"  Cooling Period:   20 batches")
    print(f"  Phase 2 Loss:     0.5×CE + 0.5×KL (self-distillation)")
    print(f"  Device:           {DEVICE}")

    # Data
    train_loader, val_loader, num_classes = get_dataloaders(
        DATA_DIR, BATCH_SIZE, class_range=(0, 9), data_ratio=1.0
    )
    print(f"  Training samples: {len(train_loader.dataset)}")
    print(f"  Classes:          {num_classes}")

    # Model
    model = PrunableDeiT(num_classes=num_classes, pretrained=use_pretrained).to(DEVICE)
    total_params = sum(p.numel() for p in model.parameters())
    print(f"  Parameters:       {total_params:,}")

    os.makedirs('logs', exist_ok=True)
    tag = 'pretrained' if use_pretrained else 'no_pretrained'
    phase1_path = f'logs/phase1_{tag}.pth'

    # =========================================================================
    # PHASE 1
    # =========================================================================
    if os.path.exists(phase1_path):
        print(f"\n{'='*70}")
        print(f"  PHASE 1: Loading from checkpoint ({phase1_path})")
        print(f"{'='*70}")

        checkpoint = torch.load(phase1_path, map_location=DEVICE, weights_only=False)
        model.load_state_dict(checkpoint['model_state_dict'])
        phase1_train_acc = checkpoint['train_acc']
        phase1_val_acc = checkpoint['val_acc']
        phase1_train_loss = checkpoint['train_loss']
        phase1_val_loss = checkpoint['val_loss']
        phase1_val_final = phase1_val_acc[-1]

        print(f"  ✅ Loaded! Phase 1 Val Accuracy: {phase1_val_final:.2f}%")
    else:
        print(f"\n{'='*70}")
        print(f"  PHASE 1: Full Training ({PHASE1_EPOCHS} epochs)")
        print(f"{'='*70}")

        optimizer = torch.optim.AdamW(model.parameters(), lr=LR_PHASE1, weight_decay=0.05)
        phase1_train_acc, phase1_val_acc = [], []
        phase1_train_loss, phase1_val_loss = [], []

        for epoch in range(1, PHASE1_EPOCHS + 1):
            train_loss, train_acc = train_phase1(model, train_loader, optimizer, DEVICE)
            val_loss, val_acc = validate(model, val_loader, DEVICE)

            phase1_train_acc.append(train_acc * 100)
            phase1_val_acc.append(val_acc * 100)
            phase1_train_loss.append(train_loss)
            phase1_val_loss.append(val_loss)

            print(f"  Epoch {epoch:2d}/{PHASE1_EPOCHS} | "
                  f"Train Loss: {train_loss:.4f} | Train Acc: {train_acc:6.2%} | "
                  f"Val Loss: {val_loss:.4f} | Val Acc: {val_acc:6.2%}")

        phase1_val_final = phase1_val_acc[-1]
        print(f"\n  ✅ Phase 1 Complete! Val Accuracy: {phase1_val_final:.2f}%")

        torch.save({
            'model_state_dict': model.state_dict(),
            'train_acc': phase1_train_acc,
            'val_acc': phase1_val_acc,
            'train_loss': phase1_train_loss,
            'val_loss': phase1_val_loss,
        }, phase1_path)
        print(f"  💾 Phase 1 saved to {phase1_path}")

    # =========================================================================
    # Reference model for KL distillation (frozen copy, no hooks)
    # =========================================================================
    reference_model = copy.deepcopy(model.base_model)
    for module in reference_model.modules():
        module._forward_hooks.clear()
        module._forward_pre_hooks.clear()
        module._backward_hooks.clear()
    reference_model.eval()
    for p in reference_model.parameters():
        p.requires_grad = False

    print(f"\n  📋 Reference model ready for KL distillation")

    # =========================================================================
    # PHASE 2
    # =========================================================================
    print(f"\n{'='*70}")
    print(f"  PHASE 2: Pruning ({PHASE2_EPOCHS} epochs, Unlimited Pruning)")
    print(f"{'='*70}")

    optimizer = torch.optim.AdamW(model.parameters(), lr=LR_PHASE2, weight_decay=0.05)
    phase2_train_acc, phase2_val_acc = [], []
    phase2_train_loss, phase2_val_loss = [], []
    phase2_compression, phase2_pruned = [], []

    initial_params = sum(p.numel() for p in model.parameters())
    if enable_compression:
        initial_mb = initial_params * 4 / (1024 * 1024)
        print(f"  Initial Params:   {initial_params:,} ({initial_mb:.2f} MB)")
        for epoch in range(1, PHASE2_EPOCHS + 1):
            train_loss, ce_loss, kl_loss, train_acc, neurons_pruned = train_phase2(
                model, train_loader, optimizer, DEVICE, reference_model
            )
            val_loss, val_acc = validate(model, val_loader, DEVICE)

            total_neurons = model.get_total_neurons()
            initial_neurons = model.get_initial_total()
            compression = (1 - total_neurons / initial_neurons) * 100

            current_params = sum(p.numel() for p in model.parameters())
            param_reduction = (1 - current_params / initial_params) * 100

            # Size in MB (assuming float32 = 4 bytes)
            size_mb = current_params * 4 / (1024 * 1024)
            initial_mb = initial_params * 4 / (1024 * 1024)

            phase2_train_acc.append(train_acc * 100)
            phase2_val_acc.append(val_acc * 100)
            phase2_train_loss.append(train_loss)
            phase2_val_loss.append(val_loss)
            phase2_compression.append(compression)
            phase2_pruned.append(neurons_pruned)

            print(f"  Epoch {epoch:2d}/{PHASE2_EPOCHS} | "
                  f"Loss: {train_loss:.4f} (CE: {ce_loss:.4f} + KL: {kl_loss:.4f}) | "
                  f"Train Acc: {train_acc:6.2%} | Val Acc: {val_acc:6.2%} | "
                  f"Pruned: {neurons_pruned} | Compression: {compression:.1f}% | "
                  f"Params: {current_params:,} (-{param_reduction:.1f}%) | Size: {size_mb:.2f} MB")

            if neurons_pruned > 0 or epoch == PHASE2_EPOCHS:
                stats = model.get_width_stats()
                print(f"    Block widths: ", end="")
                for i, init, curr in stats:
                    if curr < init:
                        print(f"B{i}:{init}→{curr}", end=" ")
                print()
    else:
        print("  ⏩ Skipping Phase 2 (Compression Disabled)")

    # =========================================================================
    # Final Summary
    # =========================================================================
    total_neurons = model.get_total_neurons()
    initial_neurons = model.get_initial_total()
    compression = (1 - total_neurons / initial_neurons) * 100

    if enable_compression and phase2_val_acc:
        final_acc = phase2_val_acc[-1]
    else:
        final_acc = phase1_val_final

    print(f"\n{'='*70}")
    print(f"  FINAL RESULTS")
    print(f"{'='*70}")
    print(f"  Phase 1 Val Accuracy:  {phase1_val_final:.2f}%")
    print(f"  Final Val Accuracy:    {final_acc:.2f}%")
    print(f"  Accuracy Change:       {final_acc - phase1_val_final:.2f}%")
    print(f"  Neurons:               {total_neurons}/{initial_neurons}")
    print(f"  Compression:           {compression:.1f}%")
    print(f"  Total Pruned:          {sum(phase2_pruned)}")
    print(f"{'='*70}")

    # Save results for plotting
    tag = 'pretrained' if use_pretrained else 'no_pretrained'
    tag += '_pruned' if enable_compression else '_baseline'

    results = {
        'pretrained': 'yes' if use_pretrained else 'no',
        'cumpress': 'yes' if enable_compression else 'no',
        'phase1': {
            'epochs': list(range(1, PHASE1_EPOCHS + 1)),
            'train_acc': phase1_train_acc,
            'val_acc': phase1_val_acc,
            'train_loss': phase1_train_loss,
            'val_loss': phase1_val_loss
        },
        'phase2': {
            'epochs': list(range(PHASE1_EPOCHS + 1, PHASE1_EPOCHS + PHASE2_EPOCHS + 1)) if enable_compression else [],
            'train_acc': phase2_train_acc,
            'val_acc': phase2_val_acc,
            'train_loss': phase2_train_loss,
            'val_loss': phase2_val_loss,
            'compression': phase2_compression,
            'neurons_pruned': phase2_pruned
        },
        'final': {
            'phase1_val_acc': phase1_val_final,
            'phase2_val_acc': final_acc,
            'accuracy_change': final_acc - phase1_val_final,
            'compression': compression,
            'total_pruned': sum(phase2_pruned)
        }
    }

    with open(f'logs/results_{tag}.json', 'w') as f:
        json.dump(results, f, indent=2)
    print(f"\n  Results saved to logs/results_{tag}.json")


if __name__ == "__main__":
    main()
