#!/bin/bash
#
# Quick Single-Stage 3-Phase Training (for testing)
# Reduced dataset and epochs for fast validation of the single-stage approach
#

set -e  # Exit on error

# Configuration
PYTHON="/scratch/yp2693/world_models/penv/bin/python"
export DATASET_DIR=/scratch/yp2693/world_models/datasets/

# Dataset settings (REDUCED for quick testing)
DATASET_ROLLOUTS=100  # Minimal dataset for speed

# Codebook sizes (REDUCED for quick testing)
STATE_CODEBOOK_SIZE=128
ACTION_CODEBOOK_SIZE=16

# Total epochs (REDUCED: 18 total = 8 stage1 + 5 stage2 + 5 stage3)
TOTAL_EPOCHS=4
STAGE1_END=2    # End of continuous representation learning
STAGE2_END=3   # End of quantizer learning

# Learning rates (same proportions as full training)
# Encoder learning rates (separate for Stage 1 and Stage 3)
STAGE1_ENCODER_LR=1e-4  # Stage 1: Learn diverse continuous representations
STAGE3_ENCODER_LR=5e-6  # Stage 3: Fine-tune with very low LR to prevent drift

PREDICTOR_LR=3e-4
DECODER_LR=3e-4
ACTION_ENCODER_LR=5e-4
STATE_QUANTIZER_LR=1e-4
ACTION_QUANTIZER_LR=1e-4

# Scheduler settings
WARMUP_EPOCHS=1 # Reduced warmup
MIN_LR_FACTOR=0.0

# Other training settings
BATCH_SIZE=16  # Smaller batch for speed
VCREG_LOSS_WEIGHT=10.0
QUANTIZATION_LOSS_WEIGHT=3.0

echo "=========================================="
echo "QUICK Single-Stage 3-Phase Training"
echo "=========================================="
echo "Dataset: Wall environment, $DATASET_ROLLOUTS rollouts (MINIMAL)"
echo "Codebook sizes: State=$STATE_CODEBOOK_SIZE, Action=$ACTION_CODEBOOK_SIZE (SMALL)"
echo "Total epochs: $TOTAL_EPOCHS (REDUCED for quick testing)"
echo "  Stage 1 (epochs 1-$STAGE1_END): Continuous representations"
echo "  Stage 2 (epochs $((STAGE1_END+1))-$STAGE2_END): Quantizer learning"
echo "  Stage 3 (epochs $((STAGE2_END+1))-$TOTAL_EPOCHS): Joint fine-tuning"
echo "=========================================="
echo "This is a QUICK test run (~10-15 minutes)"
echo "For full training, use train_single_stage.sh"
echo "=========================================="

# Run training
$PYTHON train.py \
    --config-name train.yaml \
    env=wall \
    env.dataset.n_rollout=$DATASET_ROLLOUTS \
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
    disable_wandb=true \
    force_restart=true

echo ""
echo "=========================================="
echo "Quick test complete!"
echo "=========================================="
echo ""
echo "Verify that:"
echo "  1. All 3 stages executed (check logs for stage transition messages)"
echo "  2. Learning rates changed appropriately (encoder LR drops to ~5e-6 in stage 3)"
echo "  3. Components froze/unfroze correctly:"
echo "     - Stage 1: Encoder/Predictor/Decoder trained, Quantizers frozen"
echo "     - Stage 2: Quantizers/Predictor/Decoder trained, Encoder frozen"
echo "     - Stage 3: All components trained"
echo "  4. Codebook utilization increased in stages 2-3"
echo ""
echo "If tests pass, run full training with train_single_stage.sh"
echo ""
