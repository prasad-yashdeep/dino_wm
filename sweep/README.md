# Hyperparameter Sweep Management

This directory contains all tools and configurations for running hyperparameter sweeps with W&B.

## Directory Contents

```
sweep/
├── README.md                      (This file)
├── sweep_manager.sh              (Main sweep orchestration script)
├── sweep.yaml                    (W&B sweep configuration)
├── train_sweep_wrapper.py        (Converts W&B args to Hydra format)
├── train_with_sweep.py           (Training wrapper for sweep)
├── hyperparameter_sweep.sh       (Alternative sweep launcher)
├── run_sweep.sh                  (Older sweep executor)
├── SWEEP_SETUP_GUIDE.md          (Detailed setup and usage guide)
└── SWEEP_OUTPUT_STRUCTURE.txt    (Output directory structure reference)
```

## Quick Start

### Recommended: From Parent Directory (`bhumi_exp_2/`)
```bash
# Activate environment
source /scratch/yp2693/world_models/penv/bin/activate

# Run the sweep manager (convenient launcher at root)
./run_sweep.sh
```

### Alternative: From This Directory (`sweep/`)
```bash
# Activate environment first
source /scratch/yp2693/world_models/penv/bin/activate

# Run sweep manager directly
./sweep_manager.sh
```

**⚠️ Don't use `run_sweep_legacy.sh`** - that's the old version. Use `sweep_manager.sh` or the launcher at root.

## What Each File Does

### sweep_manager.sh
The main orchestration script that:
- Allows reviewing/editing hyperparameters
- Creates a new W&B sweep
- Prompts for execution mode (single vs. parallel agents)
- Automatically runs the sweep agents
- Saves logs and configurations

**Usage:**
```bash
./sweep_manager.sh
```

### sweep.yaml
W&B hyperparameter sweep configuration that defines:
- Which hyperparameters to sweep
- How to sample them (grid, log-uniform, etc.)
- Fixed parameters for all runs

**Edit to change sweep space:**
```bash
nano sweep.yaml
```

### train_sweep_wrapper.py
Converts W&B command-line arguments to Hydra format and redirects training outputs to `sweep_outputs/runs/`.

**Called by:** W&B agent automatically
**Do not run directly.**

### hyperparameter_sweep.sh
Alternative sweep launcher (older version). Similar to sweep_manager.sh but with less interactive features.

## Output Organization

All sweep outputs are saved to: `../sweep_outputs/`

Structure:
```
sweep_outputs/
├── logs/                ← Agent execution logs
├── runs/                ← Training outputs (models, checkpoints)
├── configs/             ← Sweep configuration backups
└── README.md           ← Output directory guide
```

See `../sweep_outputs/README.md` for detailed information.

## Hyperparameter Sweep Configuration

### Current Sweep Space

**Loss Weights (Grid - 3×3 combinations):**
- `model.quantization_loss_weight`: [0.5, 1.0, 2.0]
- `model.vcreg_loss_weight`: [1, 5, 10]

**Learning Rates (Log-uniform random):**
- `training.state_quantizer_lr`: 5e-6 to 5e-3
- `training.predictor_lr`: 5e-6 to 5e-3
- `training.encoder_lr`: 5e-6 to 5e-3

**Fixed Parameters:**
- Epochs: 30
- Batch Size: 32
- Dataset Size: 400 rollouts

**Total Configurations:** 27 runs (3×3×3)

### Modify Sweep Configuration

1. Edit `sweep.yaml`:
   ```bash
   nano sweep.yaml
   ```

2. Run sweep manager with edit option:
   ```bash
   ./sweep_manager.sh
   # Select 'y' when prompted to edit hyperparameters
   ```

For detailed configuration options, see W&B documentation:
https://docs.wandb.ai/guides/sweeps/overview

## Running Sweeps

### Method 1: Interactive Manager (Recommended)
```bash
cd /scratch/yp2693/world_models/bhumi_exp_2
source /scratch/yp2693/world_models/penv/bin/activate
./run_sweep.sh
```

Then:
1. Choose whether to edit hyperparameters (n)
2. Select execution mode (1 for single, 2 for parallel)
3. Confirm to start (y)

### Method 2: Direct Script
```bash
cd sweep/
./sweep_manager.sh
```

### Method 3: Manual W&B Command
```bash
# Create sweep
python -m wandb sweep \
    --entity yashdeep18121-new-york-university \
    --project dino_wm_sweeps \
    sweep.yaml

# Start agent(s)
python -m wandb agent yashdeep18121-new-york-university/dino_wm_sweeps/SWEEP_ID
```

## Execution Modes

### Single Agent (Sequential)
- Runs one configuration at a time
- Time: ~9-12 hours for all 27 configurations
- Resources: Minimal (1 GPU)
- Best for: Limited resources

### Parallel Agents
- Runs multiple configurations simultaneously
- Time: ~2-3 hours with 4 agents
- Resources: High (multiple GPUs)
- Best for: Faster results, multiple GPUs available

## Monitoring

### View Agent Logs (Real-time)
```bash
tail -f ../sweep_outputs/logs/sweep_agent_1.log
```

### W&B Dashboard
Visit: https://wandb.ai/yashdeep18121-new-york-university/dino_wm_sweeps

Shows:
- All running and completed runs
- Real-time loss/metric curves
- Best performing configurations
- Hyperparameter comparison plots

### Check Running Processes
```bash
ps aux | grep "wandb agent"
ps aux | grep train.py
```

## After Sweep Completes

### Find Best Model
1. Check W&B dashboard - sorts by best metric
2. Note the hyperparameters
3. Find checkpoint:
   ```bash
   find ../sweep_outputs/runs -name "model_latest.pth"
   ```

### Access Configuration
```bash
# View configuration of best run
cat ../sweep_outputs/runs/YYYY-MM-DD/HH-MM-SS/.hydra/config.yaml
```

### Create Final Training Run
```bash
cd /scratch/yp2693/world_models/bhumi_exp_2
python train.py \
    model.quantization_loss_weight=0.5 \
    model.vcreg_loss_weight=5 \
    training.state_quantizer_lr=0.001 \
    training.predictor_lr=0.002 \
    training.encoder_lr=0.0015 \
    training.epochs=100
```

## Storage & Time Estimates

### Disk Space
- Per run: ~500MB - 2GB
- All 27 runs: ~10-50GB
- Agent logs: ~50MB

### Training Time
- Per run: ~20-25 minutes
- Sequential (1 agent): ~9-12 hours
- Parallel (4 agents): ~2-3 hours

## Troubleshooting

### "ModuleNotFoundError: No module named 'hydra'"
Activate the Python environment:
```bash
source /scratch/yp2693/world_models/penv/bin/activate
```

### Agent not starting
Check W&B authentication:
```bash
wandb login  # Use your W&B API key
./sweep_manager.sh
```

### Training errors in sweep
View full logs:
```bash
tail -100 ../sweep_outputs/logs/sweep_agent_1.log
tail -100 ../sweep_outputs/runs/YYYY-MM-DD/HH-MM-SS/outputs.log
```

### Low disk space
Check usage:
```bash
du -sh ../sweep_outputs/
```

Archive old runs:
```bash
tar -czf old_sweep_$(date +%Y-%m-%d).tar.gz ../sweep_outputs/runs/
rm -rf ../sweep_outputs/runs/
```

## Documentation Files

- **SWEEP_SETUP_GUIDE.md** - Detailed setup and workflow guide
- **SWEEP_OUTPUT_STRUCTURE.txt** - Visual output directory structure
- **../sweep_outputs/README.md** - Output directory reference

## Important Notes

1. **Environment Required**: Always activate the Python environment before running sweeps
2. **W&B Account**: Must have W&B account with proper entity/project setup
3. **Disk Space**: Ensure ~30-50GB available for full sweep
4. **GPU Memory**: Each training run needs sufficient GPU memory (typically 12GB+)
5. **Outputs Organized**: All outputs automatically go to `../sweep_outputs/`

## Quick Reference

```bash
# From bhumi_exp_2/ directory:
./run_sweep.sh                                    # Run sweep manager
tail -f sweep_outputs/logs/sweep_agent_1.log     # Monitor progress
ls -ltra sweep_outputs/runs/*/*/                 # View all runs
du -sh sweep_outputs/                            # Check disk usage
find sweep_outputs/runs -name "*.pth" | head -5  # Find checkpoints
```

---

**Location:** `/scratch/yp2693/world_models/bhumi_exp_2/sweep/`
**Status:** Ready to use
**Last Updated:** 2025-11-17
