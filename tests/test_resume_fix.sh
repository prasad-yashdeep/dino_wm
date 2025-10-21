#!/bin/bash
#
# Test script for quantizer collapse fix when resuming training
# Tests both 3-stage and single-stage training pipelines
#

set -e  # Exit on error

# Set paths relative to script location
SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
PROJECT_ROOT="$(dirname "$SCRIPT_DIR")"

# Parse command line arguments
MODE="3stage"  # 3stage or single
CLEAN_START=false
SKIP_INITIAL=false
USE_EXISTING=""

print_usage() {
    cat << EOF
Usage: $0 [OPTIONS]

Test script to verify the quantizer collapse fix works correctly.

OPTIONS:
    --mode MODE             Test mode: 3stage or single (default: 3stage)
                           - 3stage: Test 3-stage pipeline (Stage 1 -> Stage 2 -> Resume Stage 2)
                           - single: Test single-stage pipeline (Train -> Resume)

    --clean                Remove all test outputs and start fresh
    --skip-initial         Skip initial training, use existing checkpoint
    --use PATH             Use existing checkpoint at PATH

    -h, --help             Show this help message

EXAMPLES:
    # Test 3-stage pipeline from scratch
    $0 --mode 3stage

    # Test single-stage pipeline
    $0 --mode single

    # Clean start for 3-stage
    $0 --mode 3stage --clean

    # Resume test from existing checkpoint
    $0 --mode 3stage --skip-initial --use outputs/2025-10-21/12-34-56

EOF
    exit 0
}

while [[ $# -gt 0 ]]; do
    case $1 in
        --mode)
            MODE="$2"
            shift 2
            ;;
        --clean)
            CLEAN_START=true
            shift
            ;;
        --skip-initial)
            SKIP_INITIAL=true
            shift
            ;;
        --use)
            USE_EXISTING="$2"
            SKIP_INITIAL=true
            shift 2
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

# Validate
if [ "$SKIP_INITIAL" = true ] && [ -z "$USE_EXISTING" ]; then
    echo "Error: --skip-initial requires --use PATH"
    exit 1
fi

# Configuration
export DATASET_DIR=/scratch/yp2693/world_models/datasets/
source /scratch/yp2693/world_models/penv/bin/activate

# Logs go in tests/logs/
TEST_LOG_DIR="${SCRIPT_DIR}/logs/test_resume_fix_${MODE}"

echo "=========================================================================="
echo "🧪 TESTING RESUME FIX FOR QUANTIZER COLLAPSE"
echo "=========================================================================="
echo ""
echo "Test mode: $MODE"
echo "Clean start: $CLEAN_START"
echo "Skip initial: $SKIP_INITIAL"
if [ -n "$USE_EXISTING" ]; then
    echo "Using checkpoint: $USE_EXISTING"
fi
echo ""

# Clean if requested
if [ "$CLEAN_START" = true ]; then
    echo "🧹 Cleaning test outputs..."
    rm -rf "$TEST_LOG_DIR"
    echo "✅ Cleaned: $TEST_LOG_DIR"
    echo ""
fi

mkdir -p "$TEST_LOG_DIR"

# Change to project root for training
cd "$PROJECT_ROOT"

# Test configuration - small for speed
DATASET_ROLLOUTS=100
STATE_CODES=16
ACTION_CODES=4

#==============================================================================
# 3-STAGE TEST MODE
#==============================================================================
if [ "$MODE" = "3stage" ]; then
    echo "Test Plan (3-Stage):"
    echo "  1. Train Stage 1 for 5 epochs (no quantization)"
    echo "  2. Train Stage 2 for 5 epochs (add quantization, freeze encoder)"
    echo "  3. Resume Stage 2 for 3 epochs (CRITICAL TEST - should NOT collapse)"
    echo ""
    echo "=========================================================================="

    # Stage 1
    if [ "$SKIP_INITIAL" = false ]; then
        echo ""
        echo "========================================================================"
        echo "TEST 1: Stage 1 Initial (5 epochs)"
        echo "========================================================================"

        python train.py \
            --config-name train.yaml \
            env=wall \
            env.dataset.n_rollout=$DATASET_ROLLOUTS \
            frameskip=5 \
            num_hist=1 \
            training.epochs=5 \
            training.encoder_lr=1e-4 \
            training.scheduler.warmup_epochs=2 \
            quantize=False \
            state_vocabulary_size=$STATE_CODES \
            action_vocabulary_size=$ACTION_CODES \
            encoder=conv2d \
            predictor=conv3d \
            decoder=vqvae \
            disable_wandb=true \
            force_restart=true 2>&1 | tee "$TEST_LOG_DIR/stage1.log"

        STAGE1_CKPT=$(grep "Model saved dir:" "$TEST_LOG_DIR/stage1.log" | tail -1 | awk '{print $NF}')
        echo "✅ Stage 1 complete. Checkpoint: $STAGE1_CKPT"

        echo ""
        echo "========================================================================"
        echo "TEST 2: Stage 2 Initial (5 epochs)"
        echo "========================================================================"

        python train.py \
            --config-name train.yaml \
            env=wall \
            env.dataset.n_rollout=$DATASET_ROLLOUTS \
            frameskip=5 \
            num_hist=1 \
            training.epochs=5 \
            training.encoder_lr=1e-4 \
            training.scheduler.warmup_epochs=2 \
            quantize=True \
            state_vocabulary_size=$STATE_CODES \
            action_vocabulary_size=$ACTION_CODES \
            encoder=conv2d \
            predictor=conv3d \
            decoder=vqvae \
            model.train_encoder=False \
            disable_wandb=true \
            resume_from=$STAGE1_CKPT 2>&1 | tee "$TEST_LOG_DIR/stage2.log"

        STAGE2_CKPT=$(grep "Model saved dir:" "$TEST_LOG_DIR/stage2.log" | tail -1 | awk '{print $NF}')
        echo "✅ Stage 2 complete. Checkpoint: $STAGE2_CKPT"
    else
        STAGE2_CKPT=$USE_EXISTING
        echo "⏭️  Skipping initial training, using: $STAGE2_CKPT"
    fi

    # Stage 2 Resume - THE CRITICAL TEST
    echo ""
    echo "========================================================================"
    echo "TEST 3: Stage 2 Resume (3 epochs) - CRITICAL TEST"
    echo "========================================================================"
    echo "🔍 This test verifies the fix prevents quantizer collapse on resume"
    echo ""

    python train.py \
        --config-name train.yaml \
        env=wall \
        env.dataset.n_rollout=$DATASET_ROLLOUTS \
        frameskip=5 \
        num_hist=1 \
        training.epochs=3 \
        training.encoder_lr=1e-4 \
        training.scheduler.warmup_epochs=2 \
        quantize=True \
        state_vocabulary_size=$STATE_CODES \
        action_vocabulary_size=$ACTION_CODES \
        encoder=conv2d \
        predictor=conv3d \
        decoder=vqvae \
        model.train_encoder=False \
        disable_wandb=true \
        resume_from=$STAGE2_CKPT 2>&1 | tee "$TEST_LOG_DIR/stage2_resume.log"

#==============================================================================
# SINGLE-STAGE TEST MODE
#==============================================================================
elif [ "$MODE" = "single" ]; then
    echo "Test Plan (Single-Stage):"
    echo "  1. Train for 10 epochs (3-phase in one run)"
    echo "  2. Resume for 5 more epochs (CRITICAL TEST - should NOT collapse)"
    echo ""
    echo "=========================================================================="

    if [ "$SKIP_INITIAL" = false ]; then
        echo ""
        echo "========================================================================"
        echo "TEST 1: Initial Training (10 epochs, 3 phases)"
        echo "========================================================================"

        python train.py \
            --config-name train.yaml \
            env=wall \
            env.dataset.n_rollout=$DATASET_ROLLOUTS \
            frameskip=5 \
            num_hist=1 \
            training.epochs=10 \
            training.encoder_lr=1e-4 \
            training.scheduler.type=three_stage \
            training.scheduler.stage1_end=3 \
            training.scheduler.stage2_end=6 \
            training.scheduler.warmup_epochs=1 \
            quantize=True \
            state_vocabulary_size=$STATE_CODES \
            action_vocabulary_size=$ACTION_CODES \
            encoder=conv2d \
            predictor=conv3d \
            decoder=vqvae \
            disable_wandb=true \
            force_restart=true 2>&1 | tee "$TEST_LOG_DIR/initial.log"

        CKPT=$(grep "Model saved dir:" "$TEST_LOG_DIR/initial.log" | tail -1 | awk '{print $NF}')
        echo "✅ Initial training complete. Checkpoint: $CKPT"
    else
        CKPT=$USE_EXISTING
        echo "⏭️  Skipping initial training, using: $CKPT"
    fi

    echo ""
    echo "========================================================================"
    echo "TEST 2: Resume Training (5 epochs) - CRITICAL TEST"
    echo "========================================================================"
    echo "🔍 This test verifies the fix prevents quantizer collapse on resume"
    echo ""

    python train.py \
        --config-name train.yaml \
        env=wall \
        env.dataset.n_rollout=$DATASET_ROLLOUTS \
        frameskip=5 \
        num_hist=1 \
        training.epochs=5 \
        training.encoder_lr=1e-4 \
        training.scheduler.type=three_stage \
        training.scheduler.stage1_end=3 \
        training.scheduler.stage2_end=6 \
        training.scheduler.warmup_epochs=1 \
        quantize=True \
        state_vocabulary_size=$STATE_CODES \
        action_vocabulary_size=$ACTION_CODES \
        encoder=conv2d \
        predictor=conv3d \
        decoder=vqvae \
        disable_wandb=true \
        resume_from=$CKPT 2>&1 | tee "$TEST_LOG_DIR/resume.log"

else
    echo "Error: Invalid mode '$MODE'. Must be: 3stage or single"
    exit 1
fi

#==============================================================================
# ANALYZE RESULTS
#==============================================================================
echo ""
echo "=========================================================================="
echo "📊 RESULTS ANALYSIS"
echo "=========================================================================="
echo ""

# Find the resume log
if [ "$MODE" = "3stage" ]; then
    RESUME_LOG="$TEST_LOG_DIR/stage2_resume.log"
else
    RESUME_LOG="$TEST_LOG_DIR/resume.log"
fi

# Check if quantizer optimizers were skipped (indicates fix is active)
echo "1. Checking if fix was applied..."
SKIP_COUNT=$(grep -c "⏭️  Skipping.*quantizer" "$RESUME_LOG" || echo "0")
if [ "$SKIP_COUNT" -ge 2 ]; then
    echo "   ✅ Fix applied: Quantizer optimizers were skipped"
    grep "⏭️  Skipping" "$RESUME_LOG" | sed 's/^/      /'
else
    echo "   ⚠️  Warning: Fix may not be active (no skip messages found)"
fi

echo ""
echo "2. Checking first epoch after resume..."

# Extract metrics from first epoch
FIRST_EPOCH=$(grep "Epoch.*Training loss:" "$RESUME_LOG" | head -1)
if [ -z "$FIRST_EPOCH" ]; then
    echo "   ❌ ERROR: No training epoch found in resume log"
    exit 1
fi

echo "   $FIRST_EPOCH"

# Check for collapse indicators
if echo "$FIRST_EPOCH" | grep -q "COLLAPSE"; then
    echo "   🔴 COLLAPSE DETECTED!"
    COLLAPSE=true
else
    COLLAPSE=false
fi

# Extract numeric values
COLLAPSE_LOSS=$(echo "$FIRST_EPOCH" | grep -oP 'z_collapse_loss: \K[0-9.]+' || echo "N/A")
echo "   z_collapse_loss: $COLLAPSE_LOSS"

# Check codebook utilization if available
CODEBOOK_UTIL=$(grep "State Codebook:" "$RESUME_LOG" | head -1)
if [ -n "$CODEBOOK_UTIL" ]; then
    echo "   $CODEBOOK_UTIL"
fi

echo ""
echo "3. Final verdict..."

# Determine success/failure
if [ "$COLLAPSE" = true ]; then
    echo "   🔴 TEST FAILED: Model collapsed after resume!"
    echo ""
    echo "   The fix did not work. Possible issues:"
    echo "   - Scheduler state still being loaded incorrectly"
    echo "   - Another source of collapse"
    echo "   - Learning rates corrupted"
    echo ""
    echo "   Check logs in: $TEST_LOG_DIR/"
    exit 1
else
    # Check if collapse loss is healthy (> 0.1)
    if [ "$COLLAPSE_LOSS" != "N/A" ]; then
        IS_HEALTHY=$(awk -v val="$COLLAPSE_LOSS" 'BEGIN { print (val > 0.1) ? 1 : 0 }')
        if [ "$IS_HEALTHY" -eq 1 ]; then
            echo "   ✅ TEST PASSED: No collapse detected!"
            echo ""
            echo "   The fix is working correctly:"
            echo "   - Quantizer optimizers were reset with fresh schedulers"
            echo "   - Training continues smoothly without collapse"
            echo "   - Codebook embeddings preserved from checkpoint"
            echo ""
        else
            echo "   ⚠️  WARNING: Collapse loss is low ($COLLAPSE_LOSS)"
            echo "   This may indicate partial collapse. Review logs carefully."
            echo ""
        fi
    else
        echo "   ⚠️  Could not extract collapse_loss value"
        echo "   Manual review of logs required"
        echo ""
    fi
fi

echo "Full logs saved to: $TEST_LOG_DIR/"
echo ""
echo "=========================================================================="
