#!/bin/bash
#
# Enhanced Single-Stage 3-Phase Training Pipeline
# Supports both fresh training and resuming from checkpoints
#
# This script trains the VQ-WM model in a single continuous run with 3 phases:
#   - Phase 1 (epochs 1-30): Continuous representations (encoder/predictor/decoder train, quantizers frozen)
#   - Phase 2 (epochs 31-50): Quantizer learning (encoder frozen, quantizers/predictor/decoder train)
#   - Phase 3 (epochs 51-90): Joint fine-tuning (all components train, encoder with very low LR)
#

set -e  # Exit on error

# Default configuration
export DATASET_DIR=/scratch/yp2693/world_models/datasets/

# Parse command line arguments
MODE="full"  # full, quick, test
FORCE_RESTART=true
RESUME_FROM=""
DISABLE_WANDB=false
CUSTOM_EPOCHS=""

print_usage() {
    cat << EOF
Usage: $0 [OPTIONS]

Single-stage 3-phase training for VQ-WM with support for resume and fresh starts.

OPTIONS:
    --mode MODE              Training mode: full, quick, or test (default: full)
                            - full: 90 epochs, 1920 rollouts, codebook 128
                            - quick: 30 epochs, 400 rollouts, codebook 64
                            - test: 10 epochs, 100 rollouts, codebook 16

    --resume PATH           Resume from checkpoint (disables force_restart)
    --fresh                 Force fresh start (default, removes old checkpoints)
    --continue              Continue from existing checkpoint if available

    --epochs N              Override total epochs (default: mode-dependent)

    --wandb                 Enable wandb logging (default: disabled)
    --no-wandb              Disable wandb logging

    -h, --help              Show this help message

EXAMPLES:
    # Fresh full training
    $0 --mode full --fresh

    # Quick test run
    $0 --mode test

    # Resume from checkpoint
    $0 --resume outputs/2025-10-21/12-34-56

    # Continue if checkpoint exists, else start fresh
    $0 --mode full --continue

    # Custom epoch count
    $0 --mode quick --epochs 50

EOF
    exit 0
}

# Parse arguments
while [[ $# -gt 0 ]]; do
    case $1 in
        --mode)
            MODE="$2"
            shift 2
            ;;
        --resume)
            RESUME_FROM="$2"
            FORCE_RESTART=false
            shift 2
            ;;
        --fresh)
            FORCE_RESTART=true
            RESUME_FROM=""
            shift
            ;;
        --continue)
            FORCE_RESTART=false
            shift
            ;;
        --epochs)
            CUSTOM_EPOCHS="$2"
            shift 2
            ;;
        --wandb)
            DISABLE_WANDB=false
            shift
            ;;
        --no-wandb)
            DISABLE_WANDB=true
            shift
            ;;
        -h|--help)
            print_usage
            ;;
        *)
            echo "Unknown option: $1"
            print_usage
            ;;
    esac
done

# Set configuration based on mode
case $MODE in
    full)
        DATASET_ROLLOUTS=1920
        STATE_CODEBOOK_SIZE=128
        ACTION_CODEBOOK_SIZE=128
        TOTAL_EPOCHS=${CUSTOM_EPOCHS:-90}
        STAGE1_END=30
        STAGE2_END=50
        BATCH_SIZE=32
        ;;
    quick)
        DATASET_ROLLOUTS=400
        STATE_CODEBOOK_SIZE=64
        ACTION_CODEBOOK_SIZE=64
        TOTAL_EPOCHS=${CUSTOM_EPOCHS:-30}
        STAGE1_END=10
        STAGE2_END=20
        BATCH_SIZE=32
        ;;
    test)
        DATASET_ROLLOUTS=100
        STATE_CODEBOOK_SIZE=16
        ACTION_CODEBOOK_SIZE=4
        TOTAL_EPOCHS=${CUSTOM_EPOCHS:-10}
        STAGE1_END=3
        STAGE2_END=6
        BATCH_SIZE=32
        DISABLE_WANDB=true  # Always disable wandb for tests
        ;;
    *)
        echo "Error: Invalid mode '$MODE'. Must be: full, quick, or test"
        exit 1
        ;;
esac

# Learning rates (same across all modes)
ENCODER_LR=1e-4
PREDICTOR_LR=3e-4
DECODER_LR=3e-4
ACTION_ENCODER_LR=5e-4
STATE_QUANTIZER_LR=1e-4
ACTION_QUANTIZER_LR=1e-4
WARMUP_EPOCHS=3
MIN_LR_FACTOR=0.0
VCREG_LOSS_WEIGHT=10.0
QUANTIZATION_LOSS_WEIGHT=1.0

# Print configuration
echo "=========================================================================="
echo "🚀 Single-Stage 3-Phase Training - Enhanced"
echo "=========================================================================="
echo ""
echo "Mode: $MODE"
echo "  Dataset: $DATASET_ROLLOUTS rollouts"
echo "  Codebook sizes: State=$STATE_CODEBOOK_SIZE, Action=$ACTION_CODEBOOK_SIZE"
echo "  Total epochs: $TOTAL_EPOCHS"
echo "    Phase 1 (epochs 1-$STAGE1_END): Continuous representations"
echo "    Phase 2 (epochs $((STAGE1_END+1))-$STAGE2_END): Quantizer learning"
echo "    Phase 3 (epochs $((STAGE2_END+1))-$TOTAL_EPOCHS): Joint fine-tuning"
echo ""
echo "Resume configuration:"
if [ "$FORCE_RESTART" = true ]; then
    echo "  Mode: Fresh start (force_restart=true)"
elif [ -n "$RESUME_FROM" ]; then
    echo "  Mode: Resume from $RESUME_FROM"
else
    echo "  Mode: Continue from existing checkpoint if available"
fi
echo ""
echo "Logging:"
echo "  Wandb: $([ "$DISABLE_WANDB" = true ] && echo "disabled" || echo "enabled")"
echo ""
echo "=========================================================================="
echo ""

# Activate environment
if [ -f /scratch/yp2693/world_models/penv/bin/activate ]; then
    source /scratch/yp2693/world_models/penv/bin/activate
    echo "✅ Virtual environment activated"
else
    echo "⚠️  Warning: Virtual environment not found, using system Python"
fi

# Build command
CMD="python train.py \
    --config-name train.yaml \
    env=wall \
    env.dataset.n_rollout=$DATASET_ROLLOUTS \
    frameskip=5 \
    num_hist=1 \
    training.epochs=$TOTAL_EPOCHS \
    training.batch_size=$BATCH_SIZE \
    training.encoder_lr=$ENCODER_LR \
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
    quantize=True \
    state_vocabulary_size=$STATE_CODEBOOK_SIZE \
    action_vocabulary_size=$ACTION_CODEBOOK_SIZE \
    model.vcreg_loss_weight=$VCREG_LOSS_WEIGHT \
    model.quantization_loss_weight=$QUANTIZATION_LOSS_WEIGHT \
    encoder=conv2d \
    predictor=conv3d \
    decoder=vqvae \
    disable_wandb=$DISABLE_WANDB \
    force_restart=$FORCE_RESTART"

# Add resume_from if specified
if [ -n "$RESUME_FROM" ]; then
    CMD="$CMD resume_from=$RESUME_FROM"
fi

echo "Starting training..."
echo ""

# Run training
eval $CMD

# Get exit code
EXIT_CODE=$?

echo ""
echo "=========================================================================="
if [ $EXIT_CODE -eq 0 ]; then
    echo "✅ Training complete!"
else
    echo "❌ Training failed with exit code $EXIT_CODE"
    exit $EXIT_CODE
fi
echo "=========================================================================="
echo ""

if [ $EXIT_CODE -eq 0 ]; then
    echo "Training completed successfully!"
    echo ""
    echo "The model was trained in a single continuous run with:"
    echo "  - Phase 1: Built continuous representations"
    echo "  - Phase 2: Learned quantizer codebooks (encoder frozen)"
    echo "  - Phase 3: Fine-tuned all components together"
    echo ""
    echo "Checkpoints saved in: outputs/[timestamp]/checkpoints/"
    echo ""
    echo "Next steps:"
    echo "  1. Check training logs for collapse warnings"
    echo "  2. Visualize results: python visualize.py --checkpoint outputs/[timestamp]"
    echo "  3. Resume if needed: $0 --resume outputs/[timestamp]"
    echo ""
fi
