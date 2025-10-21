#!/bin/bash

################################################################################
# 3-STAGE TRAINING PIPELINE - full TEST
################################################################################
# This script runs the complete 3-stage training pipeline with reduced epochs
# and smaller dataset for full testing and validation.
################################################################################

set -e  # Exit on error

# Configuration #edited by B
PYTHON="/scratch/yp2693/world_models/penv/bin/python"
PROJECT_DIR="/scratch/yp2693/world_models/quantised_dinowm_bhumi"  #edited by B
DATASET_DIR="/scratch/yp2693/world_models/datasets/"

# Training hyperparameters (full TEST -> MEDIUM TEST) #edited by B
DATASET_ROLLOUTS=1920 # Increased from 100 for larger codebooks #edited by B
STATE_CODEBOOK_SIZE=128  # Increased from 16 for more capacity #edited by B
ACTION_CODEBOOK_SIZE=16  # Increased from 4 for more capacity #edited by B

# Stage-specific epochs (MEDIUM TEST - adjusted for 2x data and 4x codebook) #edited by B
STAGE1_EPOCHS=25  # Increased from 15 for better convergence #edited by B
STAGE2_EPOCHS=25  # Increased from 15 for larger codebook #edited by B
STAGE3_EPOCHS=40 # Increased from 15 for larger codebook #edited by B

# Learning rates - Stage 1 (reduced by 2x to prevent collapse) #edited by B
STAGE1_ENCODER_LR=5e-4  # Reduced from 1e-4 #edited by B
STAGE1_PREDICTOR_LR=1e-5  # Reduced from 5e-4 #edited by B
STAGE1_DECODER_LR=3e-4  # Reduced from 3e-4 #edited by B
STAGE1_ACTION_ENCODER_LR=5e-4  # Reduced from 5e-4 #edited by B

# Learning rates - Stage 2
STAGE2_PREDICTOR_LR=5e-4
STAGE2_DECODER_LR=3e-4
STAGE2_ACTION_ENCODER_LR=3e-4
STAGE2_ACTION_QUANTIZER_LR=3e-4
STAGE2_STATE_QUANTIZER_LR=3e-4

# Learning rates - Stage 3
STAGE3_ENCODER_LR=5e-6
STAGE3_PREDICTOR_LR=3e-4
STAGE3_DECODER_LR=3e-4
STAGE3_ACTION_ENCODER_LR=5e-6
STAGE3_ACTION_QUANTIZER_LR=1e-4
STAGE3_STATE_QUANTIZER_LR=1e-4

# Export environment variables
export DATASET_DIR=$DATASET_DIR

echo "========================================================================"
echo "3-STAGE TRAINING PIPELINE - MEDIUM TEST"
echo "========================================================================"
echo "⚡ Running medium test with 2x data and full codebook sizes"
echo ""
echo "Configuration:"
echo "  Dataset rollouts: $DATASET_ROLLOUTS (2x data)"
echo "  State codebook size: $STATE_CODEBOOK_SIZE (4x larger)"
echo "  Action codebook size: $ACTION_CODEBOOK_SIZE (4x larger)"
echo "  Stage 1 epochs: $STAGE1_EPOCHS"
echo "  Stage 2 epochs: $STAGE2_EPOCHS"
echo "  Stage 3 epochs: $STAGE3_EPOCHS"
echo ""
echo "========================================================================"
echo ""

cd $PROJECT_DIR

################################################################################
# STAGE 1: Train Continuous Representations
################################################################################

echo "▶️  STAGE 1: Training Continuous Representations ($STAGE1_EPOCHS epochs)"
echo "========================================================================"

$PYTHON train.py \
    --config-name train.yaml \
    env=wall \
    frameskip=5 \
    num_hist=1 \
    training.epochs=$STAGE1_EPOCHS \
    training.encoder_lr=$STAGE1_ENCODER_LR \
    training.predictor_lr=$STAGE1_PREDICTOR_LR \
    training.decoder_lr=$STAGE1_DECODER_LR \
    training.action_encoder_lr=$STAGE1_ACTION_ENCODER_LR \
    training.scheduler.type=cosine_with_warmup \
    training.scheduler.warmup_epochs=3 \
    training.scheduler.warmup_start_lr_factor=0.1 \
    training.max_grad_norm=4.0 \
    quantize=False \
    model.train_encoder=True \
    model.train_predictor=True \
    model.train_decoder=True \
    model.vcreg_loss_weight=10.0 \
    encoder=conv2d \
    predictor=conv3d \
    decoder=vqvae \
    disable_wandb=false\
    force_restart=true \
    2>&1 | tee stage1_full.log

# Extract checkpoint directory from log
STAGE1_DIR=$(grep "Model saved dir:" stage1_full.log 2>/dev/null | tail -1 | sed 's/.*Model saved dir: //')

if [ -z "$STAGE1_DIR" ]; then
    echo "❌ Error: Could not find Stage 1 checkpoint directory"
    echo "Check stage1_full.log for errors"
    exit 1
fi

if [ ! -d "$STAGE1_DIR/checkpoints" ]; then
    echo "❌ Error: Checkpoint directory not found: $STAGE1_DIR/checkpoints"
    exit 1
fi

echo "✅ Stage 1 complete: $STAGE1_DIR"
echo ""
sleep 1

################################################################################
# STAGE 2: Add Quantization
################################################################################

echo "▶️  STAGE 2: Add Quantization ($STAGE2_EPOCHS epochs)"
echo "========================================================================"
echo "Loading Stage 1 checkpoint from: $STAGE1_DIR"
echo ""

$PYTHON train.py \
    --config-name train.yaml \
    env=wall \
    frameskip=5 \
    num_hist=1 \
    training.epochs=$STAGE2_EPOCHS \
    training.encoder_lr=0.0 \
    training.predictor_lr=$STAGE2_PREDICTOR_LR \
    training.decoder_lr=$STAGE2_DECODER_LR \
    training.action_encoder_lr=0.0 \
    training.action_quantizer_lr=$STAGE2_ACTION_QUANTIZER_LR \
    training.state_quantizer_lr=$STAGE2_STATE_QUANTIZER_LR \
    training.max_grad_norm=3.0 \
    quantize=True \
    state_vocabulary_size=$STATE_CODEBOOK_SIZE \
    action_vocabulary_size=$ACTION_CODEBOOK_SIZE \
    model.train_encoder=False \
    model.train_predictor=True \
    model.train_decoder=True \
    model.vcreg_loss_weight=10.0 \
    model.quantization_loss_weight=3.0 \
    encoder=conv2d \
    predictor=conv3d \
    decoder=vqvae \
    disable_wandb=false\
    force_restart=false \
    resume_from=$STAGE1_DIR \
    2>&1 | tee stage2_full.log  #edited by B

# Extract checkpoint directory from log
STAGE2_DIR=$(grep "Model saved dir:" stage2_full.log 2>/dev/null | tail -1 | sed 's/.*Model saved dir: //')

if [ -z "$STAGE2_DIR" ]; then
    echo "❌ Error: Could not find Stage 2 checkpoint directory"
    echo "Check stage2_full.log for errors"
    exit 1
fi

if [ ! -d "$STAGE2_DIR/checkpoints" ]; then
    echo "❌ Error: Checkpoint directory not found: $STAGE2_DIR/checkpoints"
    exit 1
fi

echo "✅ Stage 2 complete: $STAGE2_DIR"
echo ""
sleep 1

################################################################################
# STAGE 3: Joint Fine-tuning
################################################################################

echo "▶️  STAGE 3: Joint Fine-tuning ($STAGE3_EPOCHS epochs)"
echo "========================================================================"
echo "Loading Stage 2 checkpoint from: $STAGE2_DIR"
echo ""

$PYTHON train.py \
    --config-name train.yaml \
    env=wall \
    frameskip=5 \
    num_hist=1 \
    training.epochs=$STAGE3_EPOCHS \
    training.encoder_lr=$STAGE3_ENCODER_LR \
    training.predictor_lr=$STAGE3_PREDICTOR_LR \
    training.decoder_lr=$STAGE3_DECODER_LR \
    training.action_encoder_lr=$STAGE3_ACTION_ENCODER_LR \
    training.action_quantizer_lr=$STAGE3_ACTION_QUANTIZER_LR \
    training.state_quantizer_lr=$STAGE3_STATE_QUANTIZER_LR \
    training.scheduler.type=cosine_with_warmup \
    training.scheduler.warmup_epochs=3 \
    training.scheduler.warmup_start_lr_factor=0.1 \
    training.max_grad_norm=3.0 \
    quantize=True \
    state_vocabulary_size=$STATE_CODEBOOK_SIZE \
    action_vocabulary_size=$ACTION_CODEBOOK_SIZE \
    model.train_encoder=True \
    model.train_predictor=True \
    model.train_decoder=True \
    model.vcreg_loss_weight=10.0 \
    model.quantization_loss_weight=2.0 \
    encoder=conv2d \
    predictor=conv3d \
    decoder=vqvae \
    disable_wandb=false \
    force_restart=false \
    resume_from=$STAGE2_DIR \
    2>&1 | tee stage3_full.log  #edited by B

# Extract checkpoint directory from log
STAGE3_DIR=$(grep "Model saved dir:" stage3_full.log 2>/dev/null | tail -1 | sed 's/.*Model saved dir: //')

if [ -z "$STAGE3_DIR" ]; then
    echo "❌ Error: Could not find Stage 3 checkpoint directory"
    echo "Check stage3_full.log for errors"
    exit 1
fi

if [ ! -d "$STAGE3_DIR/checkpoints" ]; then
    echo "❌ Error: Checkpoint directory not found: $STAGE3_DIR/checkpoints"
    exit 1
fi

echo "✅ Stage 3 complete: $STAGE3_DIR"
echo ""

################################################################################
# Summary
################################################################################

echo "========================================================================"
echo "🎉 full TEST COMPLETE!"
echo "========================================================================"
echo ""
echo "Training Summary:"
echo "  Stage 1 (Continuous):    $STAGE1_DIR"
echo "  Stage 2 (Quantization):  $STAGE2_DIR"
echo "  Stage 3 (Fine-tuning):   $STAGE3_DIR"
echo ""
echo "Logs saved:"
echo "  Stage 1: $(pwd)/stage1_full.log"
echo "  Stage 2: $(pwd)/stage2_full.log"
echo "  Stage 3: $(pwd)/stage3_full.log"
echo ""
echo "To visualize:"
echo "  export DATASET_DIR=/scratch/yp2693/world_models/datasets/"
echo "  $PYTHON visualize.py --config-name visualize.yaml model_name=$STAGE3_DIR"
echo ""
echo "Codebook utilization (from final epoch):"
grep "State Codebook:" stage3_full.log | tail -1 || echo "  (check stage3_full.log)"
grep "Action Codebook:" stage3_full.log | tail -1 || echo "  (check stage3_full.log)"
echo ""
echo "Final collapse loss:"
grep "z_collapse_loss:" stage3_full.log | tail -1 || echo "  (check stage3_full.log)"
echo ""
echo "========================================================================"
