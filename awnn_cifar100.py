import torch
import torch.nn as nn
import torch.optim as optim
import timm
from data import get_dataloaders
from utils import (
    AWNNTransformerMLP,
    AWNNDeiTTiny,
    compute_awnn_elbo_loss,
    train_one_epoch_awnn,
    validate_awnn,
    train_one_epoch_standard,
    validate_standard
)

def main():
    """
    CIFAR-100 training with optional AWNN:
    - use_awnn=True: AWNN MLPs with Algorithm 1, ELBO optimization
    - use_awnn=False: Standard DeiT tiny training
    - No LoRA (full training)
    - pretrained=False
    """

    # ============ CONFIGURATION ============
    use_awnn = True  # Set to False for standard training
    # =======================================

    if use_awnn:
        print("🚀 AWNN CIFAR-100 Training - Complete Paper Implementation")
    else:
        print("🚀 Standard DeiT CIFAR-100 Training")
    print("=" * 70)

    # Configuration
    DATA_DIR = "/media/jag/volD2/cifer100/cifer"
    EPOCHS = 20
    BATCH_SIZE = 64
    LR = 1e-4 if use_awnn else 3e-4  # Lower LR for AWNN stability
    DEVICE = "cuda" if torch.cuda.is_available() else "cpu"

    print("Configuration:")
    print(f"- AWNN Mode: {use_awnn}")
    print(f"- No LoRA (full training): ✓")
    print(f"- Pretrained: False ✓")
    if use_awnn:
        print(f"- AWNN MLPs in transformer: ✓")
        print(f"- Algorithm 1 width updates: ✓")
        print(f"- ELBO optimization: ✓")
    else:
        print(f"- Standard transformer MLPs: ✓")
        print(f"- Standard cross-entropy loss: ✓")
    print(f"- Device: {DEVICE}")

    # Data loaders using existing data.py
    train_loader, val_loader, num_classes = get_dataloaders(
        DATA_DIR, BATCH_SIZE, class_range=(0, 9), data_ratio=1.0
    )

    N = len(train_loader.dataset)  # Total training samples
    M = BATCH_SIZE                  # Mini-batch size

    print(f"- Training samples: {N}")
    print(f"- Batch size: {M}")
    print(f"- Classes: {num_classes}")

    # Create model based on use_awnn flag
    if use_awnn:
        model = AWNNDeiTTiny(num_classes=num_classes, pretrained=False).to(DEVICE)
    else:
        model = timm.create_model(
            "deit_tiny_patch16_224",
            pretrained=False,
            num_classes=num_classes
        ).to(DEVICE)

    # Count parameters
    total_params = sum(p.numel() for p in model.parameters())
    trainable_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
    print(f"- Total parameters: {total_params:,}")
    print(f"- Trainable parameters: {trainable_params:,} (100% - no LoRA)")

    # Optimizer
    optimizer = torch.optim.AdamW(model.parameters(), lr=LR, weight_decay=0.05)

    if use_awnn:
        print("\nTraining with AWNN...")
    else:
        print("\nTraining with Standard DeiT...")
    print("=" * 70)

    for epoch in range(1, EPOCHS + 1):
        if use_awnn:
            # Algorithm 1: update_width every few epochs
            if epoch > 1 and epoch % 3 == 0:
                print(f"Updating widths at epoch {epoch}...")
                model.update_all_widths()

            # AWNN Training
            elbo, pred_loss, kl_l, kl_t, train_acc = train_one_epoch_awnn(
                model, train_loader, optimizer, compute_awnn_elbo_loss, N, M, DEVICE
            )

            # AWNN Validation
            val_loss, val_acc = validate_awnn(model, val_loader, DEVICE)

            # Get current widths
            all_widths = model.get_current_widths()
            total_neurons = sum(sum(widths) for widths in all_widths)
            max_possible = len(all_widths) * len(all_widths[0]) * model.awnn_mlps[0].max_neurons
            compression = 100 * (1 - total_neurons / max_possible)

            print(f"\nEpoch {epoch:2d}/{EPOCHS}")
            print(f"  ELBO Loss: {elbo:8.4f} (pred: {pred_loss:.4f}, KL_λ: {kl_l:.4f}, KL_θ: {kl_t:.4f})")
            print(f"  Train Acc: {train_acc:6.2%}")
            print(f"  Val Acc:   {val_acc:6.2%}")
            print(f"  Total AWNN neurons: {total_neurons} (compression: {compression:.1f}%)")

            # Show some layer widths
            if len(all_widths) > 0:
                print(f"  Example widths: Block 0: {all_widths[0]}, Block {len(all_widths)//2}: {all_widths[len(all_widths)//2]}")

        else:
            # Standard Training
            train_loss, train_acc = train_one_epoch_standard(
                model, train_loader, optimizer, DEVICE
            )

            # Standard Validation
            val_loss, val_acc = validate_standard(model, val_loader, DEVICE)

            print(f"\nEpoch {epoch:2d}/{EPOCHS}")
            print(f"  Train Loss: {train_loss:.4f}")
            print(f"  Train Acc:  {train_acc:6.2%}")
            print(f"  Val Loss:   {val_loss:.4f}")
            print(f"  Val Acc:    {val_acc:6.2%}")

        # Save checkpoint
        checkpoint_data = {
            'epoch': epoch,
            'model_state_dict': model.state_dict(),
            'optimizer_state_dict': optimizer.state_dict(),
            'val_acc': val_acc,
            'use_awnn': use_awnn
        }

        if use_awnn:
            checkpoint_data['awnn_widths'] = all_widths

        filename = f"{'awnn' if use_awnn else 'standard'}_cifar100_epoch_{epoch}.pth"
        torch.save(checkpoint_data, filename)

    if use_awnn:
        print(f"\n🎉 AWNN CIFAR-100 training completed!")
        print(f"Final validation accuracy: {val_acc:.2%}")
        print(f"Final compression: {compression:.1f}%")
    else:
        print(f"\n🎉 Standard DeiT CIFAR-100 training completed!")
        print(f"Final validation accuracy: {val_acc:.2%}")

if __name__ == "__main__":
    main()
