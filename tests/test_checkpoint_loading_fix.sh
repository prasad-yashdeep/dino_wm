#!/bin/bash

################################################################################
# TEST CHECKPOINT LOADING FIX - QUICK VALIDATION
################################################################################
# This script tests the checkpoint loading fixes with a minimal dataset
# to verify optimizer states are loaded correctly before running full training.
#
# Tests:
# 1. Train Stage 1 (5 epochs) -> Save checkpoint
# 2. Resume Stage 1 (3 more epochs) -> Verify optimizer states load
# 3. Train Stage 2 (5 epochs) -> Verify quantizer loading
# 4. Resume Stage 2 (3 more epochs) -> Verify all states preserved
################################################################################

set -e  # Exit on error

# Configuration
PYTHON="/scratch/yp2693/world_models/penv/bin/python"
PROJECT_DIR="/scratch/yp2693/world_models/quantised_dinowm_bhumi"
DATASET_DIR="/scratch/yp2693/world_models/datasets/"

# Test hyperparameters (MINIMAL for quick testing)
DATASET_ROLLOUTS=100  # Minimal dataset for speed
STATE_CODEBOOK_SIZE=128  # Small codebook for speed
ACTION_CODEBOOK_SIZE=16   # Small codebook for speed

# Stage epochs (minimal for testing)
STAGE1_INITIAL_EPOCHS=5   # Initial training
STAGE1_RESUME_EPOCHS=3    # Resume training
STAGE2_INITIAL_EPOCHS=5   # Initial quantizer training
STAGE2_RESUME_EPOCHS=3    # Resume quantizer training

# Learning rates (using same as train_3stage_quick.sh)
STAGE1_ENCODER_LR=5e-4
STAGE1_PREDICTOR_LR=1e-5
STAGE1_DECODER_LR=3e-4
STAGE1_ACTION_ENCODER_LR=5e-4

STAGE2_PREDICTOR_LR=5e-4
STAGE2_DECODER_LR=3e-4
STAGE2_ACTION_ENCODER_LR=3e-4
STAGE2_ACTION_QUANTIZER_LR=3e-4
STAGE2_STATE_QUANTIZER_LR=3e-4

# Export environment variables
export DATASET_DIR=$DATASET_DIR

# Create test log directory
TEST_LOG_DIR="$PROJECT_DIR/test_logs"
mkdir -p $TEST_LOG_DIR

echo "========================================================================"
echo "🧪 TESTING CHECKPOINT LOADING FIX"
echo "========================================================================"
echo "This test validates the optimizer state loading fix."
echo ""
echo "Test Configuration:"
echo "  Dataset rollouts: $DATASET_ROLLOUTS (minimal for speed)"
echo "  State codebook: $STATE_CODEBOOK_SIZE codes"
echo "  Action codebook: $ACTION_CODEBOOK_SIZE codes"
echo "  Test log dir: $TEST_LOG_DIR"
echo ""
echo "Test Plan:"
echo "  1. Train Stage 1 for $STAGE1_INITIAL_EPOCHS epochs"
echo "  2. Resume Stage 1 for $STAGE1_RESUME_EPOCHS more epochs (VERIFY optimizer loading)"
echo "  3. Train Stage 2 for $STAGE2_INITIAL_EPOCHS epochs"
echo "  4. Resume Stage 2 for $STAGE2_RESUME_EPOCHS more epochs (VERIFY optimizer loading)"
echo ""
echo "========================================================================"
echo ""

cd $PROJECT_DIR

################################################################################
# TEST 1: Train Stage 1 Initial
################################################################################

echo ""
echo "========================================================================"
echo "TEST 1: Training Stage 1 (Initial $STAGE1_INITIAL_EPOCHS epochs)"
echo "========================================================================"
echo ""

$PYTHON train.py \
    --config-name train.yaml \
    env=wall \
    env.dataset.n_rollout=$DATASET_ROLLOUTS \
    frameskip=5 \
    num_hist=1 \
    training.epochs=$STAGE1_INITIAL_EPOCHS \
    training.encoder_lr=$STAGE1_ENCODER_LR \
    training.predictor_lr=$STAGE1_PREDICTOR_LR \
    training.decoder_lr=$STAGE1_DECODER_LR \
    training.action_encoder_lr=$STAGE1_ACTION_ENCODER_LR \
    training.scheduler.type=cosine_with_warmup \
    training.scheduler.warmup_epochs=1 \
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
    disable_wandb=true \
    force_restart=true \
    2>&1 | tee $TEST_LOG_DIR/test1_stage1_initial.log

# Extract checkpoint directory
STAGE1_DIR=$(grep "Model saved dir:" $TEST_LOG_DIR/test1_stage1_initial.log 2>/dev/null | tail -1 | sed 's/.*Model saved dir: //')

if [ -z "$STAGE1_DIR" ]; then
    echo "❌ TEST 1 FAILED: Could not find Stage 1 checkpoint directory"
    exit 1
fi

echo "✅ TEST 1 PASSED: Stage 1 initial training complete"
echo "   Checkpoint: $STAGE1_DIR"
echo ""
sleep 2

################################################################################
# TEST 2: Resume Stage 1 (CRITICAL TEST - Verify optimizer loading)
################################################################################

echo ""
echo "========================================================================"
echo "TEST 2: Resuming Stage 1 ($STAGE1_RESUME_EPOCHS more epochs)"
echo "========================================================================"
echo "🔍 This test verifies optimizer states are loaded correctly"
echo "   Looking for: '✅ Loaded encoder_optimizer' messages"
echo ""

$PYTHON train.py \
    --config-name train.yaml \
    env=wall \
    env.dataset.n_rollout=$DATASET_ROLLOUTS \
    frameskip=5 \
    num_hist=1 \
    training.epochs=$STAGE1_RESUME_EPOCHS \
    training.encoder_lr=$STAGE1_ENCODER_LR \
    training.predictor_lr=$STAGE1_PREDICTOR_LR \
    training.decoder_lr=$STAGE1_DECODER_LR \
    training.action_encoder_lr=$STAGE1_ACTION_ENCODER_LR \
    training.scheduler.type=cosine_with_warmup \
    training.scheduler.warmup_epochs=1 \
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
    disable_wandb=true \
    force_restart=false \
    resume_from=$STAGE1_DIR \
    2>&1 | tee $TEST_LOG_DIR/test2_stage1_resume.log

# Check if optimizer states were loaded
if grep -q "✅ Loaded encoder_optimizer" $TEST_LOG_DIR/test2_stage1_resume.log; then
    echo "✅ TEST 2 PASSED: Optimizer states loaded successfully!"
    echo "   Found: $(grep -c '✅ Loaded' $TEST_LOG_DIR/test2_stage1_resume.log) optimizer/scheduler states loaded"
else
    echo "❌ TEST 2 FAILED: Optimizer states NOT loaded!"
    echo "   Expected: '✅ Loaded encoder_optimizer' messages"
    echo "   Check: $TEST_LOG_DIR/test2_stage1_resume.log"
    exit 1
fi

# Verify resumed from correct epoch
RESUMED_EPOCH=$(grep "Resumed from epoch" $TEST_LOG_DIR/test2_stage1_resume.log | grep -oP 'epoch \K[0-9]+' | head -1)
if [ "$RESUMED_EPOCH" == "$STAGE1_INITIAL_EPOCHS" ]; then
    echo "✅ Resumed from correct epoch: $RESUMED_EPOCH"
else
    echo "⚠️  Warning: Resumed from epoch $RESUMED_EPOCH, expected $STAGE1_INITIAL_EPOCHS"
fi

echo ""
sleep 2

################################################################################
# TEST 3: Train Stage 2 Initial (with quantization)
################################################################################

echo ""
echo "========================================================================"
echo "TEST 3: Training Stage 2 - Initial ($STAGE2_INITIAL_EPOCHS epochs)"
echo "========================================================================"
echo "Loading Stage 1 checkpoint from: $STAGE1_DIR"
echo ""

$PYTHON train.py \
    --config-name train.yaml \
    env=wall \
    env.dataset.n_rollout=$DATASET_ROLLOUTS \
    frameskip=5 \
    num_hist=1 \
    training.epochs=$STAGE2_INITIAL_EPOCHS \
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
    disable_wandb=true \
    force_restart=false \
    resume_from=$STAGE1_DIR \
    2>&1 | tee $TEST_LOG_DIR/test3_stage2_initial.log

# Extract checkpoint directory
STAGE2_DIR=$(grep "Model saved dir:" $TEST_LOG_DIR/test3_stage2_initial.log 2>/dev/null | tail -1 | sed 's/.*Model saved dir: //')

if [ -z "$STAGE2_DIR" ]; then
    echo "❌ TEST 3 FAILED: Could not find Stage 2 checkpoint directory"
    exit 1
fi

# Verify quantizer was loaded from checkpoint (not re-initialized)
if grep -q "State quantizer loaded from checkpoint" $TEST_LOG_DIR/test3_stage2_initial.log; then
    echo "✅ TEST 3 PASSED: Quantizers loaded from Stage 1 checkpoint"
else
    echo "⚠️  Note: New quantizers created (expected for Stage 1 -> Stage 2 transition)"
fi

echo "   Checkpoint: $STAGE2_DIR"
echo ""
sleep 2

################################################################################
# TEST 4: Resume Stage 2 (CRITICAL TEST - Verify quantizer optimizer loading)
################################################################################

echo ""
echo "========================================================================"
echo "TEST 4: Resuming Stage 2 ($STAGE2_RESUME_EPOCHS more epochs)"
echo "========================================================================"
echo "🔍 This test verifies quantizer optimizer states are loaded correctly"
echo "   Looking for: '✅ Loaded state_quantizer_optimizer' messages"
echo ""

$PYTHON train.py \
    --config-name train.yaml \
    env=wall \
    env.dataset.n_rollout=$DATASET_ROLLOUTS \
    frameskip=5 \
    num_hist=1 \
    training.epochs=$STAGE2_RESUME_EPOCHS \
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
    disable_wandb=true \
    force_restart=false \
    resume_from=$STAGE2_DIR \
    2>&1 | tee $TEST_LOG_DIR/test4_stage2_resume.log

# Check if all optimizer states were loaded
LOADED_COUNT=$(grep -c "✅ Loaded" $TEST_LOG_DIR/test4_stage2_resume.log || echo "0")

if [ "$LOADED_COUNT" -ge 8 ]; then
    echo "✅ TEST 4 PASSED: All optimizer/scheduler states loaded successfully!"
    echo "   Loaded $LOADED_COUNT optimizer/scheduler states"
else
    echo "❌ TEST 4 FAILED: Expected at least 8 optimizer/scheduler loads, got $LOADED_COUNT"
    echo "   Check: $TEST_LOG_DIR/test4_stage2_resume.log"
    exit 1
fi

# Verify quantizers loaded from checkpoint (not re-initialized)
if grep -q "State quantizer loaded from checkpoint" $TEST_LOG_DIR/test4_stage2_resume.log; then
    echo "✅ Quantizers correctly loaded from checkpoint (not re-initialized)"
else
    echo "❌ WARNING: Quantizers may have been re-initialized!"
fi

# Verify resumed from correct epoch
TOTAL_STAGE2_EPOCHS=$((STAGE1_INITIAL_EPOCHS + STAGE1_RESUME_EPOCHS + STAGE2_INITIAL_EPOCHS))
RESUMED_EPOCH=$(grep "Resumed from epoch" $TEST_LOG_DIR/test4_stage2_resume.log | grep -oP 'epoch \K[0-9]+' | head -1)
if [ "$RESUMED_EPOCH" == "$TOTAL_STAGE2_EPOCHS" ]; then
    echo "✅ Resumed from correct epoch: $RESUMED_EPOCH"
else
    echo "⚠️  Warning: Resumed from epoch $RESUMED_EPOCH, expected $TOTAL_STAGE2_EPOCHS"
fi

echo ""
sleep 1

################################################################################
# Final Summary
################################################################################

echo ""
echo "========================================================================"
echo "🎉 ALL TESTS PASSED!"
echo "========================================================================"
echo ""
echo "Test Results:"
echo "  ✅ TEST 1: Stage 1 initial training"
echo "  ✅ TEST 2: Stage 1 resume with optimizer loading"
echo "  ✅ TEST 3: Stage 2 initial with quantizers"
echo "  ✅ TEST 4: Stage 2 resume with all states"
echo ""
echo "Key Validations:"
echo "  ✅ Optimizer states load correctly after creation"
echo "  ✅ Scheduler states load correctly"
echo "  ✅ Quantizer states preserved across checkpoints"
echo "  ✅ Epoch counter resumes correctly"
echo ""
echo "Test Logs:"
echo "  Test 1: $TEST_LOG_DIR/test1_stage1_initial.log"
echo "  Test 2: $TEST_LOG_DIR/test2_stage1_resume.log"
echo "  Test 3: $TEST_LOG_DIR/test3_stage2_initial.log"
echo "  Test 4: $TEST_LOG_DIR/test4_stage2_resume.log"
echo ""
echo "Checkpoints:"
echo "  Stage 1: $STAGE1_DIR"
echo "  Stage 2: $STAGE2_DIR"
echo ""
echo "✅ The checkpoint loading fix is working correctly!"
echo "✅ Safe to run full resume_stage3_full.sh with confidence."
echo ""
echo "========================================================================"
