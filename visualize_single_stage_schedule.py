#!/usr/bin/env python3
"""
Single-Stage Learning Rate Schedule Design
Mimics 3-stage training without checkpoint passing or k-means initialization

DESIGN PHILOSOPHY:
==================
Instead of 3 separate training runs, we use sophisticated LR scheduling to achieve
the same effect in a single continuous run:

Stage 1 (Epochs 1-30): Learn Continuous Representations
  - Train: encoder, action_encoder, predictor, decoder
  - Frozen: quantizers (LR=0) - they exist but don't interfere
  - Goal: Learn good feature representations without quantization

Stage 2 (Epochs 31-50): Learn Discrete Codebooks
  - Train: quantizers (HIGH LR), predictor, decoder
  - Frozen: encoder, action_encoder (LR=0)
  - Goal: Quantizers learn to discretize the frozen encoder outputs

Stage 3 (Epochs 51-70): Joint Fine-Tuning
  - Train: ALL components (REDUCED LRs)
  - Goal: Fine-tune everything together for optimal performance
"""

import numpy as np
import matplotlib
matplotlib.use('Agg')  # Non-interactive backend
import matplotlib.pyplot as plt
from matplotlib.patches import Rectangle
import sys

# =============================================================================
# CONFIGURATION
# =============================================================================

# Stage boundaries
STAGE1_END = 30    # End of continuous representation learning
STAGE2_END = 50    # End of quantizer learning
STAGE3_END = 70    # End of joint fine-tuning
TOTAL_EPOCHS = 70

# Warmup configuration
WARMUP_EPOCHS = 3
WARMUP_START_FACTOR = 0.01

# Learning rate values (tuned based on your 3-stage config)
STAGE1_LR = {
    'encoder': 5e-5,
    'action_encoder': 3e-4,
    'predictor': 3e-4,
    'decoder': 2e-4,
    'state_quantizer': 0,      # Frozen
    'action_quantizer': 0,     # Frozen
}

STAGE2_LR = {
    'encoder': 0,              # Frozen
    'action_encoder': 0,       # Frozen
    'predictor': 3e-4,         # Continue training
    'decoder': 2e-4,           # Continue training
    'state_quantizer': 1e-3,   # HIGH LR to learn codebook fast
    'action_quantizer': 1e-3,  # HIGH LR to learn codebook fast
}

STAGE3_LR = {
    'encoder': 1e-5,           # 5x lower than stage 1
    'action_encoder': 5e-5,    # 6x lower than stage 1
    'predictor': 1e-4,         # 3x lower
    'decoder': 5e-5,           # 4x lower
    'state_quantizer': 3e-4,   # 3x lower than stage 2
    'action_quantizer': 3e-4,  # 3x lower than stage 2
}

# Transition configuration
STAGE3_RAMP_EPOCHS = 3  # Epochs to ramp from 0 to target LR in stage 3

# =============================================================================
# SCHEDULE FUNCTIONS
# =============================================================================

def warmup_schedule(epoch, base_lr, warmup_epochs=WARMUP_EPOCHS, warmup_start_factor=WARMUP_START_FACTOR):
    """Linear warmup from warmup_start_factor * base_lr to base_lr."""
    if epoch <= warmup_epochs:
        return base_lr * (warmup_start_factor + (1 - warmup_start_factor) * (epoch / warmup_epochs))
    return base_lr


def get_lr_at_epoch(component, epoch):
    """Get learning rate for a component at a specific epoch."""

    # Stage 1: Continuous representation learning
    if epoch <= STAGE1_END:
        base_lr = STAGE1_LR[component]
        if base_lr == 0:
            return 0
        return warmup_schedule(epoch, base_lr)

    # Stage 2: Quantizer learning (encoders frozen)
    elif epoch <= STAGE2_END:
        return STAGE2_LR[component]

    # Stage 3: Joint fine-tuning
    else:
        target_lr = STAGE3_LR[component]

        # For components that were frozen in stage 2, ramp up gradually
        if STAGE2_LR[component] == 0 and target_lr > 0:
            epochs_since_stage3 = epoch - STAGE2_END
            if epochs_since_stage3 <= STAGE3_RAMP_EPOCHS:
                return target_lr * (epochs_since_stage3 / STAGE3_RAMP_EPOCHS)

        return target_lr


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
fig.suptitle('Single-Stage Learning Rate Schedule (Mimics 3-Stage Training)',
             fontsize=18, fontweight='bold', y=0.98)

axes = axes.flatten()

for idx, component in enumerate(components):
    ax = axes[idx]

    # Add stage background colors
    ax.axvspan(0, STAGE1_END, color=stage_colors[1], alpha=0.3, label='Stage 1')
    ax.axvspan(STAGE1_END, STAGE2_END, color=stage_colors[2], alpha=0.3, label='Stage 2')
    ax.axvspan(STAGE2_END, TOTAL_EPOCHS, color=stage_colors[3], alpha=0.3, label='Stage 3')

    # Plot learning rate
    ax.plot(epochs, schedules[component], linewidth=3, color=colors[component],
            marker='o', markersize=2.5, label=component.replace('_', ' ').title())

    # Styling
    ax.set_xlabel('Epoch', fontsize=12, fontweight='bold')
    ax.set_ylabel('Learning Rate', fontsize=12, fontweight='bold')
    ax.set_title(component.replace('_', ' ').title(), fontsize=14, fontweight='bold')
    ax.grid(True, alpha=0.3, linestyle='--', linewidth=0.5)
    ax.set_xlim(0, TOTAL_EPOCHS)

    # Add stage boundary lines
    ax.axvline(STAGE1_END, color='black', linestyle='--', linewidth=2, alpha=0.7)
    ax.axvline(STAGE2_END, color='black', linestyle='--', linewidth=2, alpha=0.7)

    # Format y-axis
    ax.ticklabel_format(axis='y', style='scientific', scilimits=(0, 0))

    # Add legend only to first subplot
    if idx == 0:
        ax.legend(loc='upper right', fontsize=9)

# Add stage descriptions at the bottom
fig.text(0.175, 0.02,
         'Stage 1: Continuous Representations\n'
         '(Epochs 1-30)\n'
         'Train: Encoder, Action Encoder, Predictor, Decoder\n'
         'Frozen: Quantizers (LR=0)',
         ha='center', fontsize=10,
         bbox=dict(boxstyle='round', facecolor=stage_colors[1], alpha=0.6, pad=0.5))

fig.text(0.5, 0.02,
         'Stage 2: Quantizer Learning\n'
         '(Epochs 31-50)\n'
         'Train: Quantizers (HIGH LR), Predictor, Decoder\n'
         'Frozen: Encoder, Action Encoder (LR=0)',
         ha='center', fontsize=10,
         bbox=dict(boxstyle='round', facecolor=stage_colors[2], alpha=0.6, pad=0.5))

fig.text(0.825, 0.02,
         'Stage 3: Joint Fine-Tuning\n'
         '(Epochs 51-70)\n'
         'Train: ALL components (REDUCED LRs)\n'
         'Gradual unfreeze for encoders',
         ha='center', fontsize=10,
         bbox=dict(boxstyle='round', facecolor=stage_colors[3], alpha=0.6, pad=0.5))

plt.tight_layout(rect=[0, 0.08, 1, 0.96])

# Save figure
output_path = 'single_stage_lr_schedule.png'
plt.savefig(output_path, dpi=300, bbox_inches='tight')
print(f"\n✅ Saved learning rate schedule visualization to: {output_path}")

# =============================================================================
# SUMMARY TABLE
# =============================================================================

print("\n" + "="*100)
print("SINGLE-STAGE LEARNING RATE SCHEDULE SUMMARY")
print("="*100)
print(f"{'Component':<20} {'Stage 1 (1-30)':<22} {'Stage 2 (31-50)':<22} {'Stage 3 (51-70)':<22}")
print("-"*100)
for component in components:
    s1 = f"{STAGE1_LR[component]:.0e}" if STAGE1_LR[component] > 0 else "0 (frozen)"
    s2 = f"{STAGE2_LR[component]:.0e}" if STAGE2_LR[component] > 0 else "0 (frozen)"
    s3 = f"{STAGE3_LR[component]:.0e}" if STAGE3_LR[component] > 0 else "0 (frozen)"

    # Add warmup note for stage 1
    if STAGE1_LR[component] > 0:
        s1 += " (warmup)"

    # Add ramp note for stage 3 if unfrozen after stage 2
    if STAGE2_LR[component] == 0 and STAGE3_LR[component] > 0:
        s3 += " (ramp)"

    print(f"{component.replace('_', ' ').title():<20} {s1:<22} {s2:<22} {s3:<22}")
print("="*100)

# =============================================================================
# KEY DESIGN PRINCIPLES
# =============================================================================

print("\n" + "="*100)
print("KEY DESIGN PRINCIPLES")
print("="*100)
print("""
1. WARMUP (Epochs 1-3):
   - All active components start with 1% LR and linearly ramp to full LR
   - Prevents instability at training start

2. STAGE 1 - Continuous Representation Learning (Epochs 1-30):
   - Encoder, action_encoder, predictor, decoder all train normally
   - Quantizers exist in the model but are FROZEN (LR=0)
   - This prevents quantization from interfering with representation learning
   - VCReg loss (weight=25.0) ensures diverse features

3. STAGE 2 - Quantizer Learning (Epochs 31-50):
   - FREEZE encoders (LR=0) to preserve learned representations
   - Quantizers train with HIGH LR (1e-3) to quickly learn codebook from scratch
   - NO k-means initialization needed - high LR allows fast learning
   - Predictor/decoder continue training to adapt to quantized features

4. STAGE 3 - Joint Fine-Tuning (Epochs 51-70):
   - Unfreeze encoders with GRADUAL ramp-up over 3 epochs (0 → target LR)
   - All components train with REDUCED learning rates (3-10x lower)
   - Gentle fine-tuning prevents catastrophic forgetting
   - Achieves optimal joint performance

5. WHY THIS WORKS WITHOUT K-MEANS:
   - Stage 1 learns good continuous representations
   - Stage 2's high quantizer LR (1e-3) can learn codebook from scratch
   - Random initialization + high LR ≈ k-means for codebook learning
   - Avoids complexity of data collection and clustering
""")
print("="*100)

# =============================================================================
# IMPLEMENTATION GUIDE
# =============================================================================

print("\n" + "="*100)
print("IMPLEMENTATION GUIDE - HOW TO USE THIS SCHEDULE")
print("="*100)
print("""
To implement this schedule, you need a custom LR scheduler that changes LR
based on epoch number. Here's the pseudocode:

```python
class ThreeStageScheduler:
    def __init__(self, optimizer, component_name):
        self.optimizer = optimizer
        self.component = component_name
        self.epoch = 0

    def step(self):
        self.epoch += 1
        new_lr = get_lr_at_epoch(self.component, self.epoch)
        for param_group in self.optimizer.param_groups:
            param_group['lr'] = new_lr
```

Then in your training loop:
```python
# Create separate schedulers for each component
encoder_scheduler = ThreeStageScheduler(encoder_optimizer, 'encoder')
action_encoder_scheduler = ThreeStageScheduler(action_encoder_optimizer, 'action_encoder')
# ... etc for all components

# After each epoch
for scheduler in [encoder_scheduler, action_encoder_scheduler, ...]:
    scheduler.step()
```

ADVANTAGES OVER 3-STAGE TRAINING:
✅ No checkpoint passing between stages
✅ No k-means initialization needed
✅ Single continuous training run
✅ Simpler to run and debug
✅ Can easily adjust stage boundaries
✅ Better gradient flow with smooth LR transitions
""")
print("="*100)

print(f"\n✅ Visualization complete! Check '{output_path}' for the graphs.\n")
