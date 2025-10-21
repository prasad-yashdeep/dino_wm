#!/bin/bash

################################################################################
# RESUME STAGE 3 FULL TRAINING
################################################################################
# Resumes stage3_full training from epoch 72 → 90 (18 remaining epochs)
################################################################################

set -e  # Exit on error

# Configuration
PYTHON="/scratch/yp2693/world_models/penv/bin/python"
PROJECT_DIR="/scratch/yp2693/world_models/quantised_dinowm_bhumi"

# Checkpoint directory
CHECKPOINT_DIR="/scratch/yp2693/world_models/quantised_dinowm_bhumi/outputs/2025-10-20/18-41-16"

# Log directory
LOG_DIR="$PROJECT_DIR/logs"
mkdir -p $LOG_DIR

# Training hyperparameters (matching train_3stage_full.sh)
STATE_CODEBOOK_SIZE=128  # Full run uses 128 state codes
ACTION_CODEBOOK_SIZE=16

# # Stage 3 Learning rates
# STAGE3_ENCODER_LR=5e-6
# STAGE3_PREDICTOR_LR=3e-4
# STAGE3_DECODER_LR=3e-4
# STAGE3_ACTION_ENCODER_LR=5e-6
# STAGE3_ACTION_QUANTIZER_LR=1e-4
# STAGE3_STATE_QUANTIZER_LR=1e-4


# Stage 3 Learning rates (FIXED: Match original train_3stage_full.sh)
STAGE3_ENCODER_LR=5e-6  # CRITICAL: Was 1e-5 (2x too high), causing encoder drift and increasing z_collapse_loss
STAGE3_PREDICTOR_LR=3e-4
STAGE3_DECODER_LR=3e-4
STAGE3_ACTION_ENCODER_LR=5e-6  # CRITICAL: Was 1e-5 (2x too high), causing encoder drift
STAGE3_ACTION_QUANTIZER_LR=1e-4
STAGE3_STATE_QUANTIZER_LR=1e-4


# Remaining epochs (stopped at 72, target 90)
REMAINING_EPOCHS=18  # FIXED: Was 40, should be 18 (90 - 72 = 18)

echo ""
echo "========================================================================"
echo "▶️  RESUMING: STAGE 3 FULL TRAINING"
echo "========================================================================"
echo "Checkpoint: $CHECKPOINT_DIR"
echo "Remaining epochs: $REMAINING_EPOCHS"
echo "Log file: $LOG_DIR/stage3_full_resume.log"
echo "========================================================================"
echo ""

# Verify checkpoint exists
if [ ! -f "$CHECKPOINT_DIR/checkpoints/model_latest.pth" ]; then
    echo "❌ ERROR: Checkpoint not found: $CHECKPOINT_DIR/checkpoints/model_latest.pth"
    exit 1
fi

cd $PROJECT_DIR

$PYTHON train.py \
    --config-name train.yaml \
    env=wall \
    frameskip=5 \
    num_hist=1 \
    training.epochs=$REMAINING_EPOCHS \
    training.encoder_lr=$STAGE3_ENCODER_LR \
    training.predictor_lr=$STAGE3_PREDICTOR_LR \
    training.decoder_lr=$STAGE3_DECODER_LR \
    training.action_encoder_lr=$STAGE3_ACTION_ENCODER_LR \
    training.action_quantizer_lr=$STAGE3_ACTION_QUANTIZER_LR \
    training.state_quantizer_lr=$STAGE3_STATE_QUANTIZER_LR \
    training.scheduler.type=cosine_with_warmup \
    training.scheduler.warmup_epochs=2 \
    training.scheduler.warmup_start_lr_factor=0.1 \
    training.max_grad_norm=3.0 \
    quantize=True \
    state_vocabulary_size=$STATE_CODEBOOK_SIZE \
    action_vocabulary_size=$ACTION_CODEBOOK_SIZE \
    model.train_encoder=True \
    model.train_predictor=True \
    model.train_decoder=True \
    model.vcreg_loss_weight=10.0 \
    model.quantization_loss_weight=4.0 \
    encoder=conv2d \
    predictor=conv3d \
    decoder=vqvae \
    disable_wandb=false \
    force_restart=false \
    resume_from=$CHECKPOINT_DIR \
    2>&1 | tee $LOG_DIR/stage3_full_resume.log

echo ""
echo "✅ STAGE 3 FULL TRAINING COMPLETED!"
echo ""
