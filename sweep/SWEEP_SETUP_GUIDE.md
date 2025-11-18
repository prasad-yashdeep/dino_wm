# Hyperparameter Sweep Setup Guide

## Quick Start

```bash
cd /scratch/yp2693/world_models/bhumi_exp_2
source /scratch/yp2693/world_models/penv/bin/activate
./sweep_manager.sh
```

Then follow the interactive prompts:
1. Edit hyperparameters? → `n` (or `y` to modify)
2. Single or multiple agents? → `1` (sequential) or `2` (parallel)
3. Start sweep? → `y`

## Output Organization

All sweep outputs are organized in **`sweep_outputs/`**:

```
sweep_outputs/
├── logs/              # W&B agent execution logs
├── runs/              # Training outputs (models, checkpoints, logs)
├── configs/           # Sweep configuration snapshots
└── README.md          # Detailed documentation
```

### logs/ - Agent Execution Logs
- **sweep_agent_1.log** - Agent 1 output
- **sweep_agent_2.log** - Agent 2 output (if parallel)
- **sweep_agent_N.log** - Agent N output (if parallel)

Each contains W&B agent logs showing:
- Sweep run assignments
- Command execution details
- Training status

**View in real-time:**
```bash
tail -f sweep_outputs/logs/sweep_agent_1.log
```

### runs/ - Training Outputs
Organized by timestamp: `runs/YYYY-MM-DD/HH-MM-SS/`

Each run contains:
- **checkpoints/** - Model checkpoints
  - `model_latest.pth` - Most recent checkpoint
  - `model_1.pth`, `model_2.pth` - Epoch-specific checkpoints
- **hydra/** - Configuration snapshots
  - `config.yaml` - Full training configuration
  - `.hydra/` - Hydra internal configs
- **outputs.log** - Complete training logs
- Other outputs: tensorboard logs, etc.

**View all runs:**
```bash
ls -ltra sweep_outputs/runs/*/*/
```

**View a specific run's config:**
```bash
cat sweep_outputs/runs/2025-11-17/19-30-45/.hydra/config.yaml
```

### configs/ - Sweep Configuration Backups
- **sweep_{SWEEP_ID}_{TIMESTAMP}.yaml** - Backup of sweep configuration
- Useful for reproducing specific sweeps

**View configs:**
```bash
ls -la sweep_outputs/configs/
cat sweep_outputs/configs/sweep_abc123*.yaml
```

## Hyperparameter Sweep Configuration

File: `sweep.yaml`

### Current Sweep Space

**Loss Weights (3×3 grid):**
- Quantization Loss: 0.5, 1.0, 2.0
- VCReg Loss: 1, 5, 10
- **Total combinations: 9**

**Learning Rates (log-uniform random):**
- State Quantizer LR: 5e-6 to 5e-3
- Predictor LR: 5e-6 to 5e-3
- Encoder LR: 5e-6 to 5e-3
- **3 random samples per loss weight combination**

**Fixed Parameters:**
- Epochs: 30
- Batch Size: 32
- Dataset Size: 400 rollouts

**Total Configurations: 3×3×3 = 27 runs**

### Modifying Hyperparameters

To change what's being swept:
```bash
# Edit the sweep configuration
nano sweep.yaml

# Or run the manager with edit option
./sweep_manager.sh
# Select 'y' when asked to edit sweep.yaml
```

## Execution Modes

### Single Agent (Sequential)
- Runs one configuration at a time
- **Time**: ~9-12 hours for all 27 configurations
- **Resources**: Minimal (1 GPU)
- **Best for**: Limited resource availability

```bash
./sweep_manager.sh
# Select: 1 (Single Agent)
```

### Multiple Agents (Parallel)
- Runs multiple configurations simultaneously
- **Time**: ~2-3 hours with 4 agents for 27 configurations
- **Resources**: High (multiple GPUs)
- **Best for**: Faster results, multiple GPUs available

```bash
./sweep_manager.sh
# Select: 2 (Multiple Agents)
# Then enter number of agents (e.g., 4)
```

## Monitoring

### Real-time Agent Logs
```bash
tail -f sweep_outputs/logs/sweep_agent_1.log
```

### W&B Dashboard
Visit: https://wandb.ai/yashdeep18121-new-york-university/dino_wm_sweeps

Shows:
- All running and completed runs
- Hyperparameter values for each run
- Training metrics and loss curves
- Best performing configurations

### Check Running Processes
```bash
ps aux | grep "wandb agent"
ps aux | grep train.py
```

### Stop Running Agents
```bash
# Kill single agent
killall wandb

# Or kill specific agent
kill -9 <PID>
```

## After Sweep Completes

### Review Results
1. Go to W&B dashboard
2. Sort by best metric (e.g., validation accuracy)
3. Note the best hyperparameters

### Analyze Outputs
```bash
# Find best performing run
ls -ltra sweep_outputs/runs/*/*/

# Copy best model checkpoint
cp sweep_outputs/runs/YYYY-MM-DD/HH-MM-SS/checkpoints/model_latest.pth \
   ./best_model.pth

# View best configuration
cat sweep_outputs/runs/YYYY-MM-DD/HH-MM-SS/.hydra/config.yaml
```

### Create Final Training Run
Once best hyperparameters are identified, create a full training run:
```bash
python train.py \
    model.quantization_loss_weight=0.5 \
    model.vcreg_loss_weight=5 \
    training.state_quantizer_lr=0.001 \
    training.predictor_lr=0.002 \
    training.encoder_lr=0.0015 \
    training.epochs=100
```

## Cleanup

### Remove Old Sweep Outputs
```bash
# Keep configs, remove training outputs
rm -rf sweep_outputs/runs/

# Remove logs
rm -rf sweep_outputs/logs/*.log

# Complete cleanup (keep structure)
rm -rf sweep_outputs/*
mkdir -p sweep_outputs/{logs,runs,configs}
```

### Remove W&B Cache (if needed)
```bash
rm -rf wandb/
```

## Troubleshooting

### "ModuleNotFoundError: No module named 'hydra'"
**Solution:** Activate Python environment before running sweep_manager.sh
```bash
source /scratch/yp2693/world_models/penv/bin/activate
```

### Agent fails with "Training error"
**Solution:** Check full logs
```bash
tail -100 sweep_outputs/logs/sweep_agent_1.log
# Or view training output
tail -100 sweep_outputs/runs/YYYY-MM-DD/HH-MM-SS/outputs.log
```

### Agent not starting
**Solution:** Verify W&B authentication
```bash
wandb login  # Use your W&B API key
./sweep_manager.sh
```

### Running out of disk space
**Solution:** Monitor sweep_outputs/runs size
```bash
du -sh sweep_outputs/runs/
# Each run ~500MB - 2GB
# 27 runs × 1GB = ~27GB space needed
```

## Storage Requirements

- **Agent Logs**: ~5-10MB per agent
- **Per Training Run**: ~500MB - 2GB (depending on checkpoint frequency)
- **Total for 27 runs**: ~10-50GB

Estimated time per run: 20-25 minutes

## Support

For more details on sweep configuration, see:
- `sweep_outputs/README.md` - Detailed directory structure
- `sweep.yaml` - W&B sweep configuration
- W&B Documentation: https://docs.wandb.ai/guides/sweeps

---

**Generated by:** Hyperparameter Sweep Setup
**Location:** `/scratch/yp2693/world_models/bhumi_exp_2/`
