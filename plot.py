#!/usr/bin/env python3
"""
Plot DeiT CIFAR-100 Pruning Results
Shows Phase 1 (training) → Phase 2 (pruning) progression
"""

import json
import matplotlib.pyplot as plt
import numpy as np
from pathlib import Path

def load_results(path):
    """Load JSON results file"""
    if not Path(path).exists():
        return None
    with open(path) as f:
        return json.load(f)

def create_plots(results_list):
    """Create comprehensive plots"""

    fig, axes = plt.subplots(2, 3, figsize=(18, 10))
    fig.suptitle('DeiT CIFAR-100: Phase 1 (Training) → Phase 2 (Pruning)',
                 fontsize=16, fontweight='bold')

    colors = {'no': ('#1f77b4', '#ff7f0e'), 'yes': ('#2ca02c', '#d62728')}

    for res in results_list:
        if res is None:
            continue

        tag = res['pretrained']
        label = 'Pretrained' if tag == 'yes' else 'No Pretrain'
        c1, c2 = colors[tag]

        p1 = res['phase1']
        p2 = res['phase2']
        all_epochs = p1['epochs'] + p2['epochs']
        phase2_start = p1['epochs'][-1] + 0.5  # For vertical line

        # Plot 1: Validation Accuracy
        ax = axes[0, 0]
        ax.plot(p1['epochs'], p1['val_acc'], '-o', color=c1, linewidth=2, markersize=4,
                label=f'{label} Phase1')
        if p2['epochs']:
            ax.plot(p2['epochs'], p2['val_acc'], '-s', color=c2, linewidth=2, markersize=4,
                    label=f'{label} Phase2')

        # Plot 2: Training Accuracy
        ax = axes[0, 1]
        ax.plot(p1['epochs'], p1['train_acc'], '-o', color=c1, linewidth=2, markersize=4,
                label=f'{label} Phase1')
        ax.plot(p2['epochs'], p2['train_acc'], '-s', color=c2, linewidth=2, markersize=4,
                label=f'{label} Phase2')

        # Plot 3: Loss (Train Loss for P1, KL Loss for P2)
        ax = axes[0, 2]
        ax.plot(p1['epochs'], p1['train_loss'], '-o', color=c1, linewidth=2, markersize=4,
                label=f'{label} CE Loss')
        ax.plot(p2['epochs'], p2['train_loss'], '-s', color=c2, linewidth=2, markersize=4,
                label=f'{label} KL Loss')

        # Plot 4: Validation Loss
        ax = axes[1, 0]
        ax.plot(p1['epochs'], p1['val_loss'], '-o', color=c1, linewidth=2, markersize=4,
                label=f'{label} Phase1')
        ax.plot(p2['epochs'], p2['val_loss'], '-s', color=c2, linewidth=2, markersize=4,
                label=f'{label} Phase2')

        # Plot 5: Neurons Pruned per Epoch
        ax = axes[1, 1]
        offset = 0.15 if tag == 'yes' else -0.15
        ax.bar(np.array(p2['epochs']) + offset, p2['neurons_pruned'], width=0.3,
               color=c2, alpha=0.7, label=f'{label}')

        # Plot 6: Compression %
        ax = axes[1, 2]
        ax.plot(p2['epochs'], p2['compression'], '-*', color=c2, linewidth=2, markersize=8,
                label=f'{label}')

    # Add phase separator & labels
    for ax in axes.flat:
        ax.legend(fontsize=8)
        ax.grid(True, alpha=0.3)

    # Add vertical line at phase boundary
    for ax in [axes[0, 0], axes[0, 1], axes[0, 2], axes[1, 0]]:
        for res in results_list:
            if res:
                boundary = res['phase1']['epochs'][-1] + 0.5
                ax.axvline(x=boundary, color='gray', linestyle='--', alpha=0.5)
                break

    axes[0, 0].set_title('Validation Accuracy', fontweight='bold')
    axes[0, 0].set_ylabel('Accuracy (%)')
    axes[0, 1].set_title('Training Accuracy', fontweight='bold')
    axes[0, 1].set_ylabel('Accuracy (%)')
    axes[0, 2].set_title('Loss (CE → KL)', fontweight='bold')
    axes[0, 2].set_ylabel('Loss')
    axes[1, 0].set_title('Validation Loss', fontweight='bold')
    axes[1, 0].set_ylabel('Loss')
    axes[1, 1].set_title('Neurons Pruned per Epoch', fontweight='bold')
    axes[1, 1].set_ylabel('Neurons Pruned')
    axes[1, 2].set_title('Compression (Max 20%)', fontweight='bold')
    axes[1, 2].set_ylabel('Compression (%)')
    axes[1, 2].axhline(y=20, color='red', linestyle='--', alpha=0.5, label='20% cap')
    axes[1, 2].legend(fontsize=8)

    for ax in axes[1]:
        ax.set_xlabel('Epoch')
    for ax in axes[0]:
        ax.set_xlabel('Epoch')

    plt.tight_layout()
    return fig

def main():
    print("Loading results...")

    results = []
    for tag in ['no_pretrained', 'pretrained']:
        path = f'logs/results_{tag}.json'
        print(f"  - {path}...", end=" ")
        data = load_results(path)
        if data:
            print("✓")
            results.append(data)
        else:
            print("✗ not found")
            results.append(None)

    if not any(results):
        print("\n❌ No results found! Run: ./run.sh")
        return

    fig = create_plots(results)
    fig.savefig("results.png", dpi=300, bbox_inches='tight')
    print("\n✓ Saved: results.png")

    # Print summary
    print("\n" + "="*60)
    print("SUMMARY")
    print("="*60)
    for res in results:
        if res is None:
            continue
        f = res['final']
        tag = 'Pretrained' if res['pretrained'] == 'yes' else 'No Pretrain'
        print(f"\n  {tag}:")
        print(f"    Phase 1 Val Acc:  {f['phase1_val_acc']:.2f}%")
        print(f"    Phase 2 Val Acc:  {f['phase2_val_acc']:.2f}%")
        print(f"    Accuracy Drop:    {f['accuracy_drop']:.2f}%")
        print(f"    Compression:      {f['compression']:.1f}%")
        print(f"    Total Pruned:     {f['total_pruned']}")

    print(f"\n{'='*60}")
    print("✅ Done!")

if __name__ == "__main__":
    main()
