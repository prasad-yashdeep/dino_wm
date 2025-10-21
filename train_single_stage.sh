#!/bin/bash
#
# Single-Stage 3-Phase Training Pipeline
# This script trains the VQ-WM model in a single continuous run with 3 phases:
#   - Phase 1 (epochs 1-30): Continuous representations (encoder/predictor/decoder train, quantizers frozen)
#   - Phase 2 (epochs 31-50): Quantizer learning (encoder frozen, quantizers/predictor/decoder train)
#   - Phase 3 (epochs 51-90): Joint fine-tuning (all components train, encoder with very low LR)
#
# Benefits over 3-stage approach:
#   - No need to manage multiple checkpoints
#   - No manual k-means initialization (uses built-in quantizer initialization)
#   - Seamless transitions between phases via custom LR schedulers
#   - Single wandb run for complete training trajectory
#

set -e  # Exit on error

# Configuration
PYTHON="/scratch/yp2693/world_models/penv/bin/python"
export DATASET_DIR=/scratch/yp2693/world_models/datasets/

# Dataset settings
DATASET_ROLLOUTS=1920  # Full dataset (use 100 for quick testing)

# Codebook sizes (match your full-scale setup)
STATE_CODEBOOK_SIZE=128
ACTION_CODEBOOK_SIZE=16

# Total epochs (90 total: 30 stage1 + 20 stage2 + 40 stage3)
TOTAL_EPOCHS=100
STAGE1_END=40   # End of continuous representation learning
STAGE2_END=60   # End of quantizer learning (30 + 20)

# Learning rates (match the 3-stage schedule)
# Encoder learning rates (separate for Stage 1 and Stage 3)
STAGE1_ENCODER_LR=3e-4  # Stage 1: Learn diverse continuous representations
STAGE3_ENCODER_LR=5e-6  # Stage 3: Fine-tune with very low LR to prevent drift

# Other component LRs
PREDICTOR_LR=3e-4
DECODER_LR=3e-4
ACTION_ENCODER_LR=5e-4
STATE_QUANTIZER_LR=1e-4
ACTION_QUANTIZER_LR=1e-4

# Scheduler settings
WARMUP_EPOCHS=3
MIN_LR_FACTOR=0.3

# Other training settings
BATCH_SIZE=32
VCREG_LOSS_WEIGHT=10.0
QUANTIZATION_LOSS_WEIGHT=3.0

echo "=========================================="
echo "Single-Stage 3-Phase Training"
echo "=========================================="
echo "Dataset: Wall environment, $DATASET_ROLLOUTS rollouts"
echo "Codebook sizes: State=$STATE_CODEBOOK_SIZE, Action=$ACTION_CODEBOOK_SIZE"
echo "Total epochs: $TOTAL_EPOCHS"
echo "  Stage 1 (epochs 1-$STAGE1_END): Continuous representations"
echo "  Stage 2 (epochs $((STAGE1_END+1))-$STAGE2_END): Quantizer learning"
echo "  Stage 3 (epochs $((STAGE2_END+1))-$TOTAL_EPOCHS): Joint fine-tuning"
echo "=========================================="

# Run training
$PYTHON train.py \
    --config-name train.yaml \
    env=wall \
    frameskip=5 \
    num_hist=1 \
    training.epochs=$TOTAL_EPOCHS \
    training.batch_size=$BATCH_SIZE \
    training.encoder_lr=$STAGE1_ENCODER_LR \
    training.predictor_lr=$PREDICTOR_LR \
    training.decoder_lr=$DECODER_LR \
    training.action_encoder_lr=$ACTION_ENCODER_LR \
    training.state_quantizer_lr=$STATE_QUANTIZER_LR \
    training.action_quantizer_lr=$ACTION_QUANTIZER_LR \
    training.scheduler.type=three_stage \
    training.scheduler.stage1_end=$STAGE1_END \
    training.scheduler.stage2_end=$STAGE2_END \
    training.scheduler.warmup_epochs=$WARMUP_EPOCHS \
    training.scheduler.min_lr_factor=$MIN_LR_FACTOR \
    training.scheduler.stage3_encoder_lr=$STAGE3_ENCODER_LR \
    quantize=True \
    state_vocabulary_size=$STATE_CODEBOOK_SIZE \
    action_vocabulary_size=$ACTION_CODEBOOK_SIZE \
    model.vcreg_loss_weight=$VCREG_LOSS_WEIGHT \
    model.quantization_loss_weight=$QUANTIZATION_LOSS_WEIGHT \
    encoder=conv2d \
    predictor=conv3d \
    decoder=vqvae \
    disable_wandb=false \
    force_restart=true

echo ""
echo "=========================================="
echo "Training complete!"
echo "=========================================="
echo ""
echo "The model was trained in a single continuous run with:"
echo "  - Phase 1: Built continuous representations"
echo "  - Phase 2: Learned quantizer codebooks (encoder frozen)"
echo "  - Phase 3: Fine-tuned all components together"
echo ""
echo "Checkpoints saved in: outputs/[timestamp]/checkpoints/"
echo "To visualize results, use visualize.py with the checkpoint path."
echo ""
