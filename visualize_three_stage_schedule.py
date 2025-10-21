#!/usr/bin/env python3
"""
Visualize the Three Stage Scheduler learning rate schedules
"""

import numpy as np
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
from matplotlib.gridspec import GridSpec

def get_lr(epoch, stage_start, stage_end, base_lr, warmup_epochs=3, min_lr_factor=0.0):
    """
    Calculate learning rate with warmup + cosine annealing
    Matches ThreeStageScheduler.get_lr() logic
    """
    epoch_in_stage = epoch - stage_start + 1
    stage_length = stage_end - stage_start + 1

    # Warmup phase
    if epoch_in_stage <= warmup_epochs:
        warmup_factor = epoch_in_stage / warmup_epochs
        return base_lr * (0.01 + 0.99 * warmup_factor)

    # Cosine annealing after warmup
    epochs_after_warmup = epoch_in_stage - warmup_epochs
    total_annealing_epochs = stage_length - warmup_epochs

    if total_annealing_epochs > 0:
        cosine_factor = 0.5 * (1 + np.cos(np.pi * epochs_after_warmup / total_annealing_epochs))
        lr = min_lr_factor * base_lr + (base_lr - min_lr_factor * base_lr) * cosine_factor
    else:
        lr = base_lr

    return lr


def get_component_schedule(component_name, epoch, stage1_end=30, stage2_end=50, total_epochs=90):
    """
    Get learning rate for a component at a given epoch
    Matches ThreeStageScheduler._get_stage_config() logic
    """
    # Determine stage
    if epoch <= stage1_end:
        stage = 1
        stage_start, stage_end = 1, stage1_end
    elif epoch <= stage2_end:
        stage = 2
        stage_start, stage_end = stage1_end + 1, stage2_end
    else:
        stage = 3
        stage_start, stage_end = stage2_end + 1, total_epochs

    # Component configurations (base LRs)
    configs = {
        'encoder': {
            1: {'active': True, 'lr': 1e-4},
            2: {'active': False, 'lr': 0.0},
            3: {'active': True, 'lr': 1e-4 * 0.05},  # 5e-6
        },
        'action_encoder': {
            1: {'active': True, 'lr': 5e-4},
            2: {'active': False, 'lr': 0.0},
            3: {'active': True, 'lr': 5e-4},
        },
        'predictor': {
            1: {'active': True, 'lr': 3e-4},
            2: {'active': True, 'lr': 3e-4},
            3: {'active': True, 'lr': 3e-4},
        },
        'decoder': {
            1: {'active': True, 'lr': 3e-4},
            2: {'active': True, 'lr': 3e-4},
            3: {'active': True, 'lr': 3e-4},
        },
        'state_quantizer': {
            1: {'active': False, 'lr': 0.0},
            2: {'active': True, 'lr': 1e-4},
            3: {'active': True, 'lr': 1e-4},
        },
        'action_quantizer': {
            1: {'active': False, 'lr': 0.0},
            2: {'active': True, 'lr': 1e-4},
            3: {'active': True, 'lr': 1e-4},
        },
    }

    config = configs[component_name][stage]

    if not config['active']:
        return 0.0

    return get_lr(epoch, stage_start, stage_end, config['lr'])


def main():
    # Training configuration
    stage1_end = 30
    stage2_end = 50
    total_epochs = 90

    # Generate learning rate schedules
    epochs = np.arange(1, total_epochs + 1)

    components = {
        'encoder': {'color': '#E74C3C', 'label': 'Encoder'},
        'action_encoder': {'color': '#9B59B6', 'label': 'Action Encoder'},
        'predictor': {'color': '#3498DB', 'label': 'Predictor'},
        'decoder': {'color': '#1ABC9C', 'label': 'Decoder'},
        'state_quantizer': {'color': '#F39C12', 'label': 'State Quantizer'},
        'action_quantizer': {'color': '#E67E22', 'label': 'Action Quantizer'},
    }

    schedules = {}
    for comp_name in components.keys():
        schedules[comp_name] = [get_component_schedule(comp_name, e, stage1_end, stage2_end, total_epochs)
                                for e in epochs]

    # Create figure with subplots
    fig = plt.figure(figsize=(20, 12))
    gs = GridSpec(3, 2, figure=fig, hspace=0.3, wspace=0.25)

    # Main plot: All components together
    ax_main = fig.add_subplot(gs[0, :])

    for comp_name, comp_info in components.items():
        ax_main.plot(epochs, schedules[comp_name],
                    label=comp_info['label'],
                    color=comp_info['color'],
                    linewidth=2.5,
                    alpha=0.9)

    # Stage backgrounds
    ax_main.axvspan(1, stage1_end, alpha=0.1, color='blue', label='Stage 1: Continuous Repr.')
    ax_main.axvspan(stage1_end + 1, stage2_end, alpha=0.1, color='green', label='Stage 2: Quantizer Learning')
    ax_main.axvspan(stage2_end + 1, total_epochs, alpha=0.1, color='orange', label='Stage 3: Joint Fine-Tuning')

    # Stage boundaries
    ax_main.axvline(stage1_end + 0.5, color='black', linestyle='--', linewidth=1.5, alpha=0.5)
    ax_main.axvline(stage2_end + 0.5, color='black', linestyle='--', linewidth=1.5, alpha=0.5)

    # Stage labels
    ax_main.text(stage1_end / 2, 5.2e-4, '🔵 STAGE 1',
                ha='center', va='center', fontsize=14, fontweight='bold',
                bbox=dict(boxstyle='round,pad=0.5', facecolor='lightblue', alpha=0.8))
    ax_main.text((stage1_end + stage2_end) / 2, 5.2e-4, '🟢 STAGE 2',
                ha='center', va='center', fontsize=14, fontweight='bold',
                bbox=dict(boxstyle='round,pad=0.5', facecolor='lightgreen', alpha=0.8))
    ax_main.text((stage2_end + total_epochs) / 2, 5.2e-4, '🟡 STAGE 3',
                ha='center', va='center', fontsize=14, fontweight='bold',
                bbox=dict(boxstyle='round,pad=0.5', facecolor='lightyellow', alpha=0.8))

    ax_main.set_xlabel('Epoch', fontsize=14, fontweight='bold')
    ax_main.set_ylabel('Learning Rate', fontsize=14, fontweight='bold')
    ax_main.set_title('Three-Stage Training Pipeline: All Components', fontsize=16, fontweight='bold', pad=20)
    ax_main.legend(loc='upper right', fontsize=11, framealpha=0.95)
    ax_main.grid(True, alpha=0.3, linestyle=':')
    ax_main.set_xlim(0, total_epochs + 1)
    ax_main.set_ylim(-0.02e-4, 5.5e-4)

    # Individual component plots (2x2 grid for key components)

    # Plot 1: Encoder (most important)
    ax1 = fig.add_subplot(gs[1, 0])
    ax1.plot(epochs, schedules['encoder'], color=components['encoder']['color'], linewidth=2.5)
    ax1.axvspan(1, stage1_end, alpha=0.1, color='blue')
    ax1.axvspan(stage1_end + 1, stage2_end, alpha=0.1, color='green')
    ax1.axvspan(stage2_end + 1, total_epochs, alpha=0.1, color='orange')
    ax1.axvline(stage1_end + 0.5, color='black', linestyle='--', linewidth=1, alpha=0.5)
    ax1.axvline(stage2_end + 0.5, color='black', linestyle='--', linewidth=1, alpha=0.5)
    ax1.set_xlabel('Epoch', fontsize=11)
    ax1.set_ylabel('Learning Rate', fontsize=11)
    ax1.set_title('Encoder (Critical: Frozen in Stage 2)', fontsize=12, fontweight='bold')
    ax1.grid(True, alpha=0.3, linestyle=':')
    ax1.text(15, 8e-5, 'Active\n1e-4', ha='center', fontsize=9, bbox=dict(boxstyle='round', facecolor='white', alpha=0.8))
    ax1.text(40, 5e-6, '❄️ FROZEN', ha='center', fontsize=10, fontweight='bold', color='blue')
    ax1.text(70, 4e-6, 'Very Low\n5e-6', ha='center', fontsize=9, bbox=dict(boxstyle='round', facecolor='white', alpha=0.8))

    # Plot 2: Quantizers
    ax2 = fig.add_subplot(gs[1, 1])
    ax2.plot(epochs, schedules['state_quantizer'], color=components['state_quantizer']['color'],
            linewidth=2.5, label='State Quantizer')
    ax2.plot(epochs, schedules['action_quantizer'], color=components['action_quantizer']['color'],
            linewidth=2.5, label='Action Quantizer', linestyle='--')
    ax2.axvspan(1, stage1_end, alpha=0.1, color='blue')
    ax2.axvspan(stage1_end + 1, stage2_end, alpha=0.1, color='green')
    ax2.axvspan(stage2_end + 1, total_epochs, alpha=0.1, color='orange')
    ax2.axvline(stage1_end + 0.5, color='black', linestyle='--', linewidth=1, alpha=0.5)
    ax2.axvline(stage2_end + 0.5, color='black', linestyle='--', linewidth=1, alpha=0.5)
    ax2.set_xlabel('Epoch', fontsize=11)
    ax2.set_ylabel('Learning Rate', fontsize=11)
    ax2.set_title('Quantizers (Frozen → Active)', fontsize=12, fontweight='bold')
    ax2.legend(fontsize=9)
    ax2.grid(True, alpha=0.3, linestyle=':')
    ax2.text(15, 5e-5, '❄️ FROZEN', ha='center', fontsize=10, fontweight='bold', color='blue')
    ax2.text(40, 8e-5, 'Learning\n1e-4', ha='center', fontsize=9, bbox=dict(boxstyle='round', facecolor='white', alpha=0.8))
    ax2.text(70, 8e-5, 'Active\n1e-4', ha='center', fontsize=9, bbox=dict(boxstyle='round', facecolor='white', alpha=0.8))

    # Plot 3: Predictor (always active)
    ax3 = fig.add_subplot(gs[2, 0])
    ax3.plot(epochs, schedules['predictor'], color=components['predictor']['color'], linewidth=2.5)
    ax3.axvspan(1, stage1_end, alpha=0.1, color='blue')
    ax3.axvspan(stage1_end + 1, stage2_end, alpha=0.1, color='green')
    ax3.axvspan(stage2_end + 1, total_epochs, alpha=0.1, color='orange')
    ax3.axvline(stage1_end + 0.5, color='black', linestyle='--', linewidth=1, alpha=0.5)
    ax3.axvline(stage2_end + 0.5, color='black', linestyle='--', linewidth=1, alpha=0.5)
    ax3.set_xlabel('Epoch', fontsize=11)
    ax3.set_ylabel('Learning Rate', fontsize=11)
    ax3.set_title('Predictor (Always Active)', fontsize=12, fontweight='bold')
    ax3.grid(True, alpha=0.3, linestyle=':')
    ax3.text(15, 2.5e-4, 'Active\n3e-4', ha='center', fontsize=9, bbox=dict(boxstyle='round', facecolor='white', alpha=0.8))
    ax3.text(40, 2.5e-4, 'Active\n3e-4', ha='center', fontsize=9, bbox=dict(boxstyle='round', facecolor='white', alpha=0.8))
    ax3.text(70, 2.5e-4, 'Active\n3e-4', ha='center', fontsize=9, bbox=dict(boxstyle='round', facecolor='white', alpha=0.8))

    # Plot 4: Action Encoder (frozen in stage 2, active otherwise)
    ax4 = fig.add_subplot(gs[2, 1])
    ax4.plot(epochs, schedules['action_encoder'], color=components['action_encoder']['color'], linewidth=2.5)
    ax4.axvspan(1, stage1_end, alpha=0.1, color='blue')
    ax4.axvspan(stage1_end + 1, stage2_end, alpha=0.1, color='green')
    ax4.axvspan(stage2_end + 1, total_epochs, alpha=0.1, color='orange')
    ax4.axvline(stage1_end + 0.5, color='black', linestyle='--', linewidth=1, alpha=0.5)
    ax4.axvline(stage2_end + 0.5, color='black', linestyle='--', linewidth=1, alpha=0.5)
    ax4.set_xlabel('Epoch', fontsize=11)
    ax4.set_ylabel('Learning Rate', fontsize=11)
    ax4.set_title('Action Encoder (Frozen in Stage 2)', fontsize=12, fontweight='bold')
    ax4.grid(True, alpha=0.3, linestyle=':')
    ax4.text(15, 4e-4, 'Active\n5e-4', ha='center', fontsize=9, bbox=dict(boxstyle='round', facecolor='white', alpha=0.8))
    ax4.text(40, 2.5e-4, '❄️ FROZEN', ha='center', fontsize=10, fontweight='bold', color='blue')
    ax4.text(70, 4e-4, 'Active\n5e-4', ha='center', fontsize=9, bbox=dict(boxstyle='round', facecolor='white', alpha=0.8))

    # Overall title
    fig.suptitle('Single-Stage 3-Phase Training: Learning Rate Schedules\n' +
                'Warmup (3 epochs) + Cosine Annealing per Stage',
                fontsize=18, fontweight='bold', y=0.995)

    # Save figure
    output_path = 'three_stage_scheduler_visualization.png'
    plt.savefig(output_path, dpi=300, bbox_inches='tight', facecolor='white')
    print(f"✅ Saved visualization to: {output_path}")

    # Create a second figure: Stage-by-stage comparison
    fig2, axes = plt.subplots(1, 3, figsize=(20, 5))

    # Stage 1
    stage1_epochs = epochs[epochs <= stage1_end]
    ax = axes[0]
    for comp_name, comp_info in components.items():
        stage1_lrs = [schedules[comp_name][e-1] for e in stage1_epochs]
        if max(stage1_lrs) > 0:
            ax.plot(stage1_epochs, stage1_lrs, label=comp_info['label'],
                   color=comp_info['color'], linewidth=2.5)
    ax.set_xlabel('Epoch', fontsize=12)
    ax.set_ylabel('Learning Rate', fontsize=12)
    ax.set_title('🔵 STAGE 1: Continuous Representations\nEncoder/Predictor/Decoder Active',
                fontsize=13, fontweight='bold')
    ax.legend(fontsize=10)
    ax.grid(True, alpha=0.3, linestyle=':')
    ax.set_facecolor('#E8F4FF')

    # Stage 2
    stage2_epochs = epochs[(epochs > stage1_end) & (epochs <= stage2_end)]
    ax = axes[1]
    for comp_name, comp_info in components.items():
        stage2_lrs = [schedules[comp_name][e-1] for e in stage2_epochs]
        if max(stage2_lrs) > 0:
            ax.plot(stage2_epochs, stage2_lrs, label=comp_info['label'],
                   color=comp_info['color'], linewidth=2.5)
    ax.set_xlabel('Epoch', fontsize=12)
    ax.set_ylabel('Learning Rate', fontsize=12)
    ax.set_title('🟢 STAGE 2: Quantizer Learning\nEncoder FROZEN, Quantizers Active',
                fontsize=13, fontweight='bold')
    ax.legend(fontsize=10)
    ax.grid(True, alpha=0.3, linestyle=':')
    ax.set_facecolor('#E8FFE8')

    # Stage 3
    stage3_epochs = epochs[epochs > stage2_end]
    ax = axes[2]
    for comp_name, comp_info in components.items():
        stage3_lrs = [schedules[comp_name][e-1] for e in stage3_epochs]
        if max(stage3_lrs) > 0:
            ax.plot(stage3_epochs, stage3_lrs, label=comp_info['label'],
                   color=comp_info['color'], linewidth=2.5)
    ax.set_xlabel('Epoch', fontsize=12)
    ax.set_ylabel('Learning Rate', fontsize=12)
    ax.set_title('🟡 STAGE 3: Joint Fine-Tuning\nAll Active (Encoder Very Low LR)',
                fontsize=13, fontweight='bold')
    ax.legend(fontsize=10)
    ax.grid(True, alpha=0.3, linestyle=':')
    ax.set_facecolor('#FFF8E8')

    fig2.suptitle('Single-Stage 3-Phase Training: Stage-by-Stage Breakdown',
                 fontsize=16, fontweight='bold', y=1.02)
    plt.tight_layout()

    output_path2 = 'three_stage_scheduler_stages_breakdown.png'
    plt.savefig(output_path2, dpi=300, bbox_inches='tight', facecolor='white')
    print(f"✅ Saved stage breakdown to: {output_path2}")

    print("\n" + "="*70)
    print("KEY OBSERVATIONS FROM THE SCHEDULES:")
    print("="*70)
    print(f"📊 Stage 1 (epochs 1-{stage1_end}):")
    print(f"   • Encoder: {schedules['encoder'][0]:.2e} → {schedules['encoder'][stage1_end-1]:.2e}")
    print(f"   • Quantizers: FROZEN (LR = 0)")
    print(f"\n📊 Stage 2 (epochs {stage1_end+1}-{stage2_end}):")
    print(f"   • Encoder: FROZEN (LR = 0)")
    print(f"   • State Quantizer: {schedules['state_quantizer'][stage1_end]:.2e} → {schedules['state_quantizer'][stage2_end-1]:.2e}")
    print(f"\n📊 Stage 3 (epochs {stage2_end+1}-{total_epochs}):")
    print(f"   • Encoder: {schedules['encoder'][stage2_end]:.2e} → {schedules['encoder'][total_epochs-1]:.2e} (VERY LOW)")
    print(f"   • Action Encoder: {schedules['action_encoder'][stage2_end]:.2e} → {schedules['action_encoder'][total_epochs-1]:.2e}")
    print(f"   • All components active for joint fine-tuning")
    print("="*70)

    plt.show()


if __name__ == "__main__":
    main()
