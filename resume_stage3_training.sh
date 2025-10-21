#!/bin/bash

################################################################################
# RESUME STAGE 3 TRAINING SCRIPT - SEQUENTIAL EXECUTION
################################################################################
# This script resumes both stage3_quick and stage3_full training runs from
# their last saved checkpoints. Runs SEQUENTIALLY - stage3_full starts only
# after stage3_quick completes successfully.
#
# Training Status:
# - stage3_quick: Stopped at epoch 67/90 (23 epochs remaining)
# - stage3_full: Stopped at epoch 72/90 (18 epochs remaining)
################################################################################

set -e  # Exit on error

# Configuration
PYTHON="/scratch/yp2693/world_models/penv/bin/python"
PROJECT_DIR="/scratch/yp2693/world_models/quantised_dinowm_bhumi"

# Checkpoint directories
STAGE3_QUICK_DIR="/scratch/yp2693/world_models/quantised_dinowm_bhumi/outputs/2025-10-20/13-04-57"
STAGE3_FULL_DIR="/scratch/yp2693/world_models/quantised_dinowm_bhumi/outputs/2025-10-20/13-16-54"

# Log directory
LOG_DIR="$PROJECT_DIR/logs"
mkdir -p $LOG_DIR

# Training hyperparameters
# NOTE: quick uses 64 state codes, full uses 128 state codes
STATE_CODEBOOK_SIZE_QUICK=64
STATE_CODEBOOK_SIZE_FULL=128
ACTION_CODEBOOK_SIZE=16

# Stage 3 Learning rates
STAGE3_ENCODER_LR=1e-5
STAGE3_PREDICTOR_LR=3e-4
STAGE3_DECODER_LR=3e-4
STAGE3_ACTION_ENCODER_LR=1e-5
STAGE3_ACTION_QUANTIZER_LR=1e-4
STAGE3_STATE_QUANTIZER_LR=1e-4

# Remaining epochs
# Total target: 90 epochs (stage1: 25, stage2: 25, stage3: 40)
STAGE3_QUICK_REMAINING=23  # Stopped at epoch 67, needs 67+23=90
STAGE3_FULL_REMAINING=18   # Stopped at epoch 72, needs 72+18=90

################################################################################
# Function to resume training
################################################################################
resume_training() {
    local name=$1
    local checkpoint_dir=$2
    local epochs=$3
    local log_file=$4
    local state_codebook_size=$5

    echo ""
    echo "========================================================================"
    echo "▶️  RESUMING: $name"
    echo "========================================================================"
    echo "Checkpoint: $checkpoint_dir"
    echo "Remaining epochs: $epochs"
    echo "State codebook size: $state_codebook_size"
    echo "Action codebook size: $ACTION_CODEBOOK_SIZE"
    echo "Log file: $log_file"
    echo "========================================================================"
    echo ""

    cd $PROJECT_DIR

    $PYTHON train.py \
        --config-name train.yaml \
        env=wall \
        frameskip=5 \
        num_hist=1 \
        training.epochs=$epochs \
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
        state_vocabulary_size=$state_codebook_size \
        action_vocabulary_size=$ACTION_CODEBOOK_SIZE \
        model.train_encoder=True \
        model.train_predictor=True \
        model.train_decoder=True \
        model.vcreg_loss_weight=10.0 \
        model.quantization_loss_weight=3.0 \
        encoder=conv2d \
        predictor=conv3d \
        decoder=vqvae \
        disable_wandb=false \
        force_restart=false \
        resume_from=$checkpoint_dir \
        2>&1 | tee $log_file

    local exit_code=$?

    if [ $exit_code -eq 0 ]; then
        echo ""
        echo "✅ Completed: $name"
        echo ""
        return 0
    else
        echo ""
        echo "❌ FAILED: $name (exit code: $exit_code)"
        echo "Check log file: $log_file"
        echo ""
        return $exit_code
    fi
}

################################################################################
# Main execution
################################################################################

echo ""
echo "################################################################################"
echo "# RESUMING STAGE 3 TRAINING - SEQUENTIAL EXECUTION"
echo "################################################################################"
echo ""
echo "This script will resume Stage 3 training from the last saved checkpoints:"
echo ""
echo "1. stage3_quick (RUNS FIRST):"
echo "   - Checkpoint: $STAGE3_QUICK_DIR"
echo "   - Last epoch: 67"
echo "   - Remaining: $STAGE3_QUICK_REMAINING epochs (→ epoch 90)"
echo "   - State codebook: $STATE_CODEBOOK_SIZE_QUICK codes"
echo "   - Log: $LOG_DIR/stage3_quick_resume.log"
echo ""
echo "2. stage3_full (RUNS AFTER stage3_quick completes):"
echo "   - Checkpoint: $STAGE3_FULL_DIR"
echo "   - Last epoch: 72"
echo "   - Remaining: $STAGE3_FULL_REMAINING epochs (→ epoch 90)"
echo "   - State codebook: $STATE_CODEBOOK_SIZE_FULL codes"
echo "   - Log: $LOG_DIR/stage3_full_resume.log"
echo ""
echo "⏱️  Estimated time: ~4.8 hours total (2.7h + 2.1h)"
echo ""
echo "Press Ctrl+C within 10 seconds to cancel..."
sleep 10

# Verify checkpoints exist
echo ""
echo "Verifying checkpoints..."
if [ ! -f "$STAGE3_QUICK_DIR/checkpoints/model_latest.pth" ]; then
    echo "❌ ERROR: Checkpoint not found: $STAGE3_QUICK_DIR/checkpoints/model_latest.pth"
    exit 1
fi
echo "✅ stage3_quick checkpoint found"

if [ ! -f "$STAGE3_FULL_DIR/checkpoints/model_latest.pth" ]; then
    echo "❌ ERROR: Checkpoint not found: $STAGE3_FULL_DIR/checkpoints/model_latest.pth"
    exit 1
fi
echo "✅ stage3_full checkpoint found"

# Record start time
START_TIME=$(date +%s)
echo ""
echo "🚀 Starting sequential training at $(date)"
echo ""

################################################################################
# STEP 1: Resume stage3_quick
################################################################################

echo ""
echo "################################################################################"
echo "# STEP 1/2: RESUMING STAGE3_QUICK"
echo "################################################################################"
echo ""

QUICK_START=$(date +%s)

resume_training \
    "STAGE3_QUICK" \
    "$STAGE3_QUICK_DIR" \
    "$STAGE3_QUICK_REMAINING" \
    "$LOG_DIR/stage3_quick_resume.log" \
    "$STATE_CODEBOOK_SIZE_QUICK"

QUICK_EXIT=$?
QUICK_END=$(date +%s)
QUICK_DURATION=$((QUICK_END - QUICK_START))

if [ $QUICK_EXIT -ne 0 ]; then
    echo ""
    echo "################################################################################"
    echo "# ❌ STAGE3_QUICK FAILED"
    echo "################################################################################"
    echo ""
    echo "stage3_quick failed with exit code: $QUICK_EXIT"
    echo "Check log: $LOG_DIR/stage3_quick_resume.log"
    echo ""
    echo "stage3_full will NOT be started."
    echo ""
    exit $QUICK_EXIT
fi

echo ""
echo "✅ STAGE3_QUICK completed successfully in $((QUICK_DURATION / 60)) minutes"
echo ""

################################################################################
# STEP 2: Resume stage3_full
################################################################################

echo ""
echo "################################################################################"
echo "# STEP 2/2: RESUMING STAGE3_FULL"
echo "################################################################################"
echo ""
echo "stage3_quick completed successfully. Starting stage3_full now..."
echo ""

FULL_START=$(date +%s)

resume_training \
    "STAGE3_FULL" \
    "$STAGE3_FULL_DIR" \
    "$STAGE3_FULL_REMAINING" \
    "$LOG_DIR/stage3_full_resume.log" \
    "$STATE_CODEBOOK_SIZE_FULL"

FULL_EXIT=$?
FULL_END=$(date +%s)
FULL_DURATION=$((FULL_END - FULL_START))

################################################################################
# Final summary
################################################################################

END_TIME=$(date +%s)
TOTAL_DURATION=$((END_TIME - START_TIME))

echo ""
echo "################################################################################"
echo "# TRAINING SUMMARY"
echo "################################################################################"
echo ""
echo "stage3_quick:"
echo "  - Status: ✅ COMPLETED"
echo "  - Duration: $((QUICK_DURATION / 60)) minutes"
echo "  - Log: $LOG_DIR/stage3_quick_resume.log"
echo ""

if [ $FULL_EXIT -eq 0 ]; then
    echo "stage3_full:"
    echo "  - Status: ✅ COMPLETED"
    echo "  - Duration: $((FULL_DURATION / 60)) minutes"
    echo "  - Log: $LOG_DIR/stage3_full_resume.log"
    echo ""
    echo "🎉 ALL TRAINING COMPLETED SUCCESSFULLY!"
    echo ""
    echo "Total time: $((TOTAL_DURATION / 60)) minutes ($((TOTAL_DURATION / 3600)) hours)"
else
    echo "stage3_full:"
    echo "  - Status: ❌ FAILED (exit code: $FULL_EXIT)"
    echo "  - Duration: $((FULL_DURATION / 60)) minutes"
    echo "  - Log: $LOG_DIR/stage3_full_resume.log"
    echo ""
    echo "⚠️  stage3_quick completed, but stage3_full FAILED"
    echo ""
    exit $FULL_EXIT
fi

echo ""
echo "Logs saved to:"
echo "  - $LOG_DIR/stage3_quick_resume.log"
echo "  - $LOG_DIR/stage3_full_resume.log"
echo ""
echo "Final checkpoints:"
echo "  - $STAGE3_QUICK_DIR/checkpoints/model_90.pth"
echo "  - $STAGE3_FULL_DIR/checkpoints/model_90.pth"
echo ""
echo "Completed at: $(date)"
echo ""
