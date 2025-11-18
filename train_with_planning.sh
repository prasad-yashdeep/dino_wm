#!/bin/bash

################################################################################
# Training with Inline Planning Script
#
# This script runs training with inline MPC/CEM planning after each epoch.
# Planning is integrated directly into the training process.
#
# Usage:
#   ./train_with_planning.sh                    # Start new training
#   RESUME_FROM=/absolute/path/to/checkpoint ./train_with_planning.sh  # Resume from checkpoint
################################################################################

set -e

# ============================================================================
# Activate Conda Environment
# ============================================================================

eval "$(conda shell.bash hook)"
conda activate /scratch/yp2693/world_models/penv

echo "✅ Activated conda environment: /scratch/yp2693/world_models/penv"
echo ""

# ============================================================================
# Configuration
# ============================================================================

# Training configuration
TOTAL_EPOCHS=3
N_ROLLOUT=200
OUTPUTS_DIR="small_outputs"
WANDB_PROJECT="dino_wm_small"

# Resume from existing checkpoint (set via environment variable)
RESUME_FROM=${RESUME_FROM:-""}

# ============================================================================
# Display Configuration
# ============================================================================

echo "════════════════════════════════════════════════════════════"
echo "  Training with Inline MPC/CEM Planning"
echo "════════════════════════════════════════════════════════════"
echo ""
echo "Training Configuration:"
echo "  • Total epochs: $TOTAL_EPOCHS"
echo "  • Dataset: $N_ROLLOUT rollouts"
echo "  • Outputs directory: $OUTPUTS_DIR"
echo "  • W&B project: $WANDB_PROJECT"
echo ""

if [ -n "$RESUME_FROM" ]; then
    echo "Resume Configuration:"
    echo "  • Resume from: $RESUME_FROM"
    echo ""
fi

echo "Planning Configuration (Inline):"
echo "  • Planner: MPC with CEM sub-planner"
echo "  • N evaluations: 50"
echo "  • Goal horizon: 5"
echo "  • Goal source: random_state"
echo "  • Seed: 99"
echo ""
echo "  MPC Parameters:"
echo "    - Max iterations: 5"
echo "    - Actions taken per replan: 5"
echo ""
echo "  CEM Sub-planner Parameters:"
echo "    - Horizon: 5"
echo "    - Samples: 100"
echo "    - Top-k: 10"
echo "    - Optimization steps: 10"
echo ""
echo "════════════════════════════════════════════════════════════"
echo ""

# ============================================================================
# Training with Inline Planning
# ============================================================================

echo "🚀 Starting training with inline planning..."
echo ""

# Build the training command
TRAIN_CMD="python train.py \
    --config-name train.yaml \
    env=wall \
    env.dataset.n_rollout=$N_ROLLOUT \
    outputs_dir=$OUTPUTS_DIR \
    wandb_project=$WANDB_PROJECT \
    img_size=224 \
    frameskip=5 \
    concat_dim=1 \
    normalize_action=True \
    action_emb_dim=10 \
    num_action_repeat=1 \
    num_hist=1 \
    num_pred=1 \
    has_predictor=True \
    has_decoder=True \
    training.seed=0 \
    training.epochs=$TOTAL_EPOCHS \
    training.batch_size=32 \
    training.save_every_x_epoch=1 \
    training.reconstruct_every_x_batch=500 \
    training.num_reconstruct_samples=6 \
    training.encoder_lr=1e-4 \
    training.decoder_lr=3e-4 \
    training.predictor_lr=5e-4 \
    training.action_encoder_lr=5e-4 \
    training.action_quantizer_lr=1e-4 \
    training.state_quantizer_lr=1e-4 \
    training.max_grad_norm=1.0 \
    training.scheduler.type=cosine_with_warmup \
    training.scheduler.warmup_epochs=5 \
    training.scheduler.warmup_start_lr_factor=0.01 \
    training.scheduler.min_lr_factor=0.0 \
    model.train_encoder=True \
    model.train_predictor=True \
    model.train_decoder=True \
    model.vcreg_loss_weight=10.0 \
    model.quantization_loss_weight=1.0 \
    quantize=True \
    state_vocabulary_size=128 \
    action_vocabulary_size=128 \
    encoder=conv2d \
    action_encoder=proprio \
    decoder=vqvae \
    predictor=conv3d \
    state_quantizer=ema_vector_quantizer \
    action_quantizer=ema_vector_quantizer \
    debug=False \
    disable_wandb=False"

# Only add resume_from if RESUME_FROM is not empty
if [ -n "$RESUME_FROM" ]; then
    TRAIN_CMD="$TRAIN_CMD resume_from=\"${RESUME_FROM}\""
fi

# Execute the training command
eval $TRAIN_CMD

echo ""
echo "════════════════════════════════════════════════════════════"
echo "✅ Training with inline planning completed!"
echo "════════════════════════════════════════════════════════════"
echo ""
