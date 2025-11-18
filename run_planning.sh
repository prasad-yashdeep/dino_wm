#!/bin/bash
# Planning script for DINO-WM
# Based on configuration from the experiment image

# Load modules and activate conda environment
source /scratch/yp2693/world_models/load_modules.sh

# Change to the project directory
cd /scratch/yp2693/world_models/bhumi_exp_2

# Set PyTorch CUDA memory optimization to reduce fragmentation
export PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True

# Run planning with specified parameters
# Outputs will be saved to plan_outputs/ directory as configured in plan_wall.yaml
# Use the penv Python explicitly to ensure correct environment
/scratch/yp2693/world_models/penv/bin/python -u plan.py \
    --config-name plan_wall \
    model_name=2025-11-04/10-09-40 \
    n_evals=50 \
    planner.n_taken_actions=5 \
    planner.sub_planner.horizon=5 \
    planner.sub_planner.num_samples=100 \
    planner.sub_planner.topk=10 \
    planner.sub_planner.opt_steps=10 \
    seed=99 \
    ckpt_base_path=./

