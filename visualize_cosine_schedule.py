#!/usr/bin/env python3
"""
Single-Stage Learning Rate Schedule with Cosine Annealing + Warmup
Mimics 3-stage training with smooth cosine curves for better optimization

DESIGN PHILOSOPHY:
==================
Uses cosine annealing within each stage for smoother optimization compared to
constant learning rates. Cosine annealing has been shown to improve convergence
and generalization in deep learning.

Stage 1 (Epochs 1-30): Learn Continuous Representations
  - Warmup (1-3): Linear ramp to peak LR
  - Cosine anneal (4-30): Smooth decay to minimum LR
  - Components: encoder, action_encoder, predictor, decoder
  - Frozen: quantizers

Stage 2 (Epochs 31-50): Learn Discrete Codebooks
  - Warmup (31-33): Linear ramp to peak LR for quantizers
  - Cosine anneal (34-50): Smooth decay
  - Frozen: encoder, action_encoder
  - Active: quantizers (HIGH LR), predictor, decoder

Stage 3 (Epochs 51-70): Joint Fine-Tuning
  - Warmup (51-53): Linear ramp to peak LR (for previously frozen components)
  - Cosine anneal (54-70): Smooth decay to very low LR
  - All components active with reduced peak LRs
"""

import numpy as np
import matplotlib
matplotlib.use('Agg')  # Non-interactive backend
import matplotlib.pyplot as plt

# =============================================================================
# CONFIGURATION
# =============================================================================

# Stage boundaries
STAGE1_START = 1
STAGE1_END = 30
STAGE2_START = 31
STAGE2_END = 50
STAGE3_START = 51
STAGE3_END = 70
TOTAL_EPOCHS = 70

# Warmup configuration (epochs at start of each stage)
WARMUP_EPOCHS = 3
WARMUP_START_FACTOR = 0.1  # Start at 10% of peak LR

# Peak learning rates for each stage (max LR after warmup)
STAGE1_PEAK_LR = {
    'encoder': 5e-5,
    'action_encoder': 3e-4,
    'predictor': 3e-4,
    'decoder': 2e-4,
    'state_quantizer': 0,      # Frozen
    'action_quantizer': 0,     # Frozen
}

STAGE2_PEAK_LR = {
    'encoder': 0,              # Frozen
    'action_encoder': 0,       # Frozen
    'predictor': 3e-4,
    'decoder': 2e-4,
    'state_quantizer': 1e-3,   # HIGH peak for fast learning
    'action_quantizer': 1e-3,
}

STAGE3_PEAK_LR = {
    'encoder': 1e-5,           # Lower peak for gentle fine-tuning
    'action_encoder': 5e-5,
    'predictor': 1e-4,
    'decoder': 5e-5,
    'state_quantizer': 3e-4,
    'action_quantizer': 3e-4,
}

# Minimum learning rates (cosine annealing floor) - relative to peak
# This prevents LR from going to exactly 0 in the middle of training
MIN_LR_FACTOR = {
    'stage1': 0.1,   # Anneal down to 10% of peak
    'stage2': 0.1,   # Anneal down to 10% of peak
    'stage3': 0.01,  # Anneal down to 1% of peak (very low for final fine-tuning)
}

# =============================================================================
# COSINE ANNEALING FUNCTIONS
# =============================================================================

def cosine_annealing(current_epoch, start_epoch, end_epoch, peak_lr, min_lr):
    """
    Cosine annealing schedule: smooth decay from peak_lr to min_lr.

    Formula: lr = min_lr + 0.5 * (peak_lr - min_lr) * (1 + cos(π * progress))
    where progress = (current_epoch - start_epoch) / (end_epoch - start_epoch)
    """
    if current_epoch < start_epoch:
        return peak_lr
    if current_epoch > end_epoch:
        return min_lr

    progress = (current_epoch - start_epoch) / (end_epoch - start_epoch)
    lr = min_lr + 0.5 * (peak_lr - min_lr) * (1 + np.cos(np.pi * progress))
    return lr


def linear_warmup(current_epoch, warmup_start_epoch, warmup_epochs, target_lr, start_factor=0.1):
    """
    Linear warmup: ramp from start_factor * target_lr to target_lr.
    """
    if current_epoch < warmup_start_epoch:
        return 0

    epochs_since_start = current_epoch - warmup_start_epoch + 1

    if epochs_since_start > warmup_epochs:
        return target_lr

    progress = epochs_since_start / warmup_epochs
    lr = target_lr * (start_factor + (1 - start_factor) * progress)
    return lr


def get_lr_at_epoch(component, epoch):
    """
    Get learning rate for a component at a specific epoch.
    Combines warmup + cosine annealing for each stage.
    """

    # =========================================================================
    # STAGE 1: Continuous representation learning
    # =========================================================================
    if epoch <= STAGE1_END:
        peak_lr = STAGE1_PEAK_LR[component]

        if peak_lr == 0:
            return 0

        min_lr = peak_lr * MIN_LR_FACTOR['stage1']

        # Warmup phase (epochs 1-3)
        if epoch <= WARMUP_EPOCHS:
            return linear_warmup(epoch, STAGE1_START, WARMUP_EPOCHS, peak_lr, WARMUP_START_FACTOR)

        # Cosine annealing phase (epochs 4-30)
        return cosine_annealing(
            epoch,
            WARMUP_EPOCHS + 1,  # Start annealing after warmup
            STAGE1_END,
            peak_lr,
            min_lr
        )

    # =========================================================================
    # STAGE 2: Quantizer learning
    # =========================================================================
    elif epoch <= STAGE2_END:
        peak_lr = STAGE2_PEAK_LR[component]

        if peak_lr == 0:
            return 0

        min_lr = peak_lr * MIN_LR_FACTOR['stage2']

        # Warmup phase for components that were frozen in stage 1
        warmup_end = STAGE2_START + WARMUP_EPOCHS - 1
        if epoch <= warmup_end and STAGE1_PEAK_LR[component] == 0:
            return linear_warmup(epoch, STAGE2_START, WARMUP_EPOCHS, peak_lr, WARMUP_START_FACTOR)

        # Cosine annealing phase
        anneal_start = warmup_end + 1 if STAGE1_PEAK_LR[component] == 0 else STAGE2_START
        return cosine_annealing(
            epoch,
            anneal_start,
            STAGE2_END,
            peak_lr,
            min_lr
        )

    # =========================================================================
    # STAGE 3: Joint fine-tuning
    # =========================================================================
    else:
        peak_lr = STAGE3_PEAK_LR[component]

        if peak_lr == 0:
            return 0

        min_lr = peak_lr * MIN_LR_FACTOR['stage3']

        # Warmup phase for components that were frozen in stage 2
        warmup_end = STAGE3_START + WARMUP_EPOCHS - 1
        if epoch <= warmup_end and STAGE2_PEAK_LR[component] == 0:
            return linear_warmup(epoch, STAGE3_START, WARMUP_EPOCHS, peak_lr, WARMUP_START_FACTOR)

        # Cosine annealing phase
        anneal_start = warmup_end + 1 if STAGE2_PEAK_LR[component] == 0 else STAGE3_START
        return cosine_annealing(
            epoch,
            anneal_start,
            STAGE3_END,
            peak_lr,
            min_lr
        )


# =============================================================================
# GENERATE SCHEDULES
# =============================================================================

components = ['encoder', 'action_encoder', 'predictor', 'decoder', 'state_quantizer', 'action_quantizer']
epochs = np.arange(1, TOTAL_EPOCHS + 1)

schedules = {}
for component in components:
    schedules[component] = [get_lr_at_epoch(component, e) for e in epochs]

# =============================================================================
# VISUALIZATION
# =============================================================================

# Color scheme
colors = {
    'encoder': '#1976D2',           # Blue
    'action_encoder': '#388E3C',    # Green
    'predictor': '#F57C00',         # Orange
    'decoder': '#C2185B',           # Pink
    'state_quantizer': '#7B1FA2',   # Purple
    'action_quantizer': '#0097A7',  # Cyan
}

stage_colors = {
    1: '#e8f4f8',  # Light blue
    2: '#fff4e6',  # Light orange
    3: '#f0f8e8',  # Light green
}

# Create figure with subplots
fig, axes = plt.subplots(2, 3, figsize=(20, 11))
fig.suptitle('Single-Stage Learning Rate Schedule with Cosine Annealing + Warmup',
             fontsize=18, fontweight='bold', y=0.98)

axes = axes.flatten()

for idx, component in enumerate(components):
    ax = axes[idx]

    # Add stage background colors
    ax.axvspan(0, STAGE1_END, color=stage_colors[1], alpha=0.3, label='Stage 1')
    ax.axvspan(STAGE1_END, STAGE2_END, color=stage_colors[2], alpha=0.3, label='Stage 2')
    ax.axvspan(STAGE2_END, TOTAL_EPOCHS, color=stage_colors[3], alpha=0.3, label='Stage 3')

    # Plot learning rate with smooth line
    ax.plot(epochs, schedules[component], linewidth=3, color=colors[component],
            label=component.replace('_', ' ').title(), alpha=0.9)

    # Styling
    ax.set_xlabel('Epoch', fontsize=12, fontweight='bold')
    ax.set_ylabel('Learning Rate', fontsize=12, fontweight='bold')
    ax.set_title(component.replace('_', ' ').title(), fontsize=14, fontweight='bold')
    ax.grid(True, alpha=0.3, linestyle='--', linewidth=0.5)
    ax.set_xlim(0, TOTAL_EPOCHS)

    # Add stage boundary lines
    ax.axvline(STAGE1_END, color='black', linestyle='--', linewidth=2, alpha=0.7)
    ax.axvline(STAGE2_END, color='black', linestyle='--', linewidth=2, alpha=0.7)

    # Mark warmup regions with subtle shading
    ax.axvspan(0, WARMUP_EPOCHS, color='yellow', alpha=0.1)
    if STAGE1_PEAK_LR[component] == 0 and STAGE2_PEAK_LR[component] > 0:
        ax.axvspan(STAGE2_START, STAGE2_START + WARMUP_EPOCHS - 1, color='yellow', alpha=0.1)
    if STAGE2_PEAK_LR[component] == 0 and STAGE3_PEAK_LR[component] > 0:
        ax.axvspan(STAGE3_START, STAGE3_START + WARMUP_EPOCHS - 1, color='yellow', alpha=0.1)

    # Format y-axis
    ax.ticklabel_format(axis='y', style='scientific', scilimits=(0, 0))

    # Add legend only to first subplot
    if idx == 0:
        ax.legend(loc='upper right', fontsize=9)

# Add stage descriptions
fig.text(0.175, 0.02,
         'Stage 1: Continuous Representations\n'
         '(Epochs 1-30)\n'
         'Warmup (1-3) → Cosine Anneal (4-30)\n'
         'Train: Encoders, Predictor, Decoder',
         ha='center', fontsize=10,
         bbox=dict(boxstyle='round', facecolor=stage_colors[1], alpha=0.6, pad=0.5))

fig.text(0.5, 0.02,
         'Stage 2: Quantizer Learning\n'
         '(Epochs 31-50)\n'
         'Warmup (31-33) → Cosine Anneal (34-50)\n'
         'Train: Quantizers (HIGH), Predictor, Decoder',
         ha='center', fontsize=10,
         bbox=dict(boxstyle='round', facecolor=stage_colors[2], alpha=0.6, pad=0.5))

fig.text(0.825, 0.02,
         'Stage 3: Joint Fine-Tuning\n'
         '(Epochs 51-70)\n'
         'Warmup (51-53) → Cosine Anneal (54-70)\n'
         'Train: ALL components (REDUCED)',
         ha='center', fontsize=10,
         bbox=dict(boxstyle='round', facecolor=stage_colors[3], alpha=0.6, pad=0.5))

plt.tight_layout(rect=[0, 0.08, 1, 0.96])

# Save figure
output_path = 'cosine_annealing_lr_schedule.png'
plt.savefig(output_path, dpi=300, bbox_inches='tight')
print(f"\n✅ Saved cosine annealing LR schedule to: {output_path}")

# =============================================================================
# SUMMARY TABLE
# =============================================================================

print("\n" + "="*110)
print("COSINE ANNEALING + WARMUP LEARNING RATE SCHEDULE")
print("="*110)
print(f"{'Component':<20} {'Stage 1 Peak':<18} {'Stage 1 Min':<18} {'Stage 2 Peak':<18} {'Stage 3 Peak':<18}")
print("-"*110)

for component in components:
    s1_peak = STAGE1_PEAK_LR[component]
    s1_min = s1_peak * MIN_LR_FACTOR['stage1'] if s1_peak > 0 else 0
    s2_peak = STAGE2_PEAK_LR[component]
    s3_peak = STAGE3_PEAK_LR[component]

    s1_peak_str = f"{s1_peak:.2e}" if s1_peak > 0 else "0 (frozen)"
    s1_min_str = f"{s1_min:.2e}" if s1_min > 0 else "-"
    s2_peak_str = f"{s2_peak:.2e}" if s2_peak > 0 else "0 (frozen)"
    s3_peak_str = f"{s3_peak:.2e}" if s3_peak > 0 else "0 (frozen)"

    print(f"{component.replace('_', ' ').title():<20} {s1_peak_str:<18} {s1_min_str:<18} {s2_peak_str:<18} {s3_peak_str:<18}")

print("="*110)

# =============================================================================
# DESIGN PRINCIPLES
# =============================================================================

print("\n" + "="*110)
print("COSINE ANNEALING SCHEDULE DESIGN PRINCIPLES")
print("="*110)
print("""
1. WARMUP PHASE (First 3 epochs of each stage):
   - Linear ramp from 10% to 100% of peak LR
   - Applied when component becomes active (unfreezes)
   - Prevents training instability from sudden LR changes

2. COSINE ANNEALING PHASE:
   - Smooth decay following: lr = min_lr + 0.5 * (peak_lr - min_lr) * (1 + cos(π * progress))
   - Stage 1: Anneal to 10% of peak (keeps some learning throughout)
   - Stage 2: Anneal to 10% of peak (quantizers continue adapting)
   - Stage 3: Anneal to 1% of peak (very gentle fine-tuning at end)

3. STAGE 1 (Epochs 1-30) - Continuous Representations:
   ├─ Warmup (1-3): Ramp to peak LR
   └─ Cosine (4-30): Decay to 10% of peak
   - Components: encoder, action_encoder, predictor, decoder
   - Quantizers: FROZEN (exist but LR=0)

4. STAGE 2 (Epochs 31-50) - Quantizer Learning:
   ├─ Warmup (31-33): Quantizers ramp to HIGH peak (1e-3)
   └─ Cosine (34-50): Decay to 10% of peak
   - Encoders: FROZEN (preserve learned representations)
   - Quantizers: HIGH peak LR for fast learning
   - Predictor/Decoder: Continue with cosine annealing

5. STAGE 3 (Epochs 51-70) - Joint Fine-Tuning:
   ├─ Warmup (51-53): Encoders ramp to LOW peak (gentle restart)
   └─ Cosine (54-70): Decay to 1% of peak (very low final LR)
   - ALL components active
   - Lower peaks than previous stages (fine-tuning mode)
   - Final LRs very low (1% of peak) for stable convergence

6. WHY COSINE ANNEALING?
   ✅ Smoother optimization than step decay
   ✅ Better exploration early, exploitation late (per stage)
   ✅ Proven to improve generalization in deep learning
   ✅ Natural "cyclic" behavior across stages aids optimization
   ✅ Avoids sharp LR drops that can harm training
""")
print("="*110)

# =============================================================================
# IMPLEMENTATION NOTES
# =============================================================================

print("\n" + "="*110)
print("IMPLEMENTATION NOTES")
print("="*110)
print("""
PyTorch Implementation:

```python
import math

class CosineAnnealingThreeStageScheduler:
    def __init__(self, optimizer, component_name, total_epochs=70):
        self.optimizer = optimizer
        self.component = component_name
        self.total_epochs = total_epochs
        self.current_epoch = 0

    def get_lr(self):
        return get_lr_at_epoch(self.component, self.current_epoch)

    def step(self):
        self.current_epoch += 1
        new_lr = self.get_lr()
        for param_group in self.optimizer.param_groups:
            param_group['lr'] = new_lr
        return new_lr
```

Alternatively, use PyTorch's built-in schedulers with careful configuration:
- torch.optim.lr_scheduler.CosineAnnealingLR for each stage
- torch.optim.lr_scheduler.LinearLR for warmup
- torch.optim.lr_scheduler.SequentialLR to chain them

ADVANTAGES OVER STEP-BASED SCHEDULE:
✅ Smoother gradient updates (less shock from LR changes)
✅ Better final convergence (gradual decay to low LR)
✅ Proven theoretical and empirical benefits
✅ More "natural" optimization trajectory
✅ Easier to tune (just adjust peak and min LRs)
""")
print("="*110)

print(f"\n✅ Visualization complete! Check '{output_path}' for the graphs.\n")

# Save the schedule data to a file for later use
np.savez('lr_schedules.npz',
         epochs=epochs,
         encoder=schedules['encoder'],
         action_encoder=schedules['action_encoder'],
         predictor=schedules['predictor'],
         decoder=schedules['decoder'],
         state_quantizer=schedules['state_quantizer'],
         action_quantizer=schedules['action_quantizer'])
print("✅ Saved schedule data to 'lr_schedules.npz' for implementation reference\n")
