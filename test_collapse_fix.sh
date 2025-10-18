#!/bin/bash

# Script to test representation collapse fixes
# This runs a quick 5-epoch test on a small subset of data

echo "================================================"
echo "🔬 Testing Representation Collapse Fixes"
echo "================================================"
echo ""

# Set environment
export DATASET_DIR=/scratch/yp2693/world_models/datasets/

# Activate conda environment
source /scratch/yp2693/world_models/penv/bin/activate

echo "✅ Environment configured"
echo ""

# Run training with small dataset and debug mode
echo "🚀 Starting training with:"
echo "  - 50 rollouts (small dataset for fast testing)"
echo "  - 5 epochs"
echo "  - Force restart (ignore old checkpoints)"
echo "  - Encoder LR: 1e-4 (100x higher than before)"
echo "  - VCReg cov_coeff: 1.0 (25x higher than before)"
echo "  - Gradient clipping: 1.0"
echo ""

/scratch/yp2693/world_models/penv/bin/python train.py \
    --config-name train.yaml \
    env=wall \
    env.dataset.n_rollout=50 \
    frameskip=5 \
    num_hist=1 \
    training.epochs=5 \
    training.encoder_lr=1e-4 \
    quantize=True \
    model.vcreg_loss_weight=10.0 \
    encoder=conv2d \
    predictor=conv3d \
    decoder=vqvae \
    disable_wandb=true \
    force_restart=true

echo ""
echo "================================================"
echo "✅ Test complete!"
echo "================================================"
echo ""
echo "Check the output above for:"
echo "  1. '🔍 COLLAPSE LOSS DEBUG' messages showing embedding statistics"
echo "  2. z_collapse_loss should be > 0.1 (ideally 0.5-2.0)"
echo "  3. '⚠️ WARNING' messages if embeddings are still collapsed"
echo ""
