# Planning Debug & Verification Guide

## What Was Fixed

### 1. **Lock Scope Issue** (train.py)
- **Problem**: `lock` variable was defined inside `if self.accelerator.is_main_process:` block but used outside
- **Solution**: Moved lock declaration to outside the if block, initialized to None for non-main processes

### 2. **Conda Environment Activation**
- Added `source /scratch/yp2693/world_models/penv/bin/activate` to test scripts
- Ensures all dependencies are available before running training

### 3. **Logging & Debugging** (train.py)
Added comprehensive logging to track planning:
- Confirms planning status at startup
- Logs when planning jobs are submitted
- Includes checkpoint path and model information

## How to Verify Planning is Working

### Quick Test (6 epochs)
```bash
bash test_planning_quick.sh
```

This trains for only 6 epochs and triggers planning at:
- Epoch 3: First checkpoint, first planning job submitted
- Epoch 6: Second checkpoint, second planning job submitted

### Expected Log Messages

**At Training Start:**
```
✅ Planning ENABLED - Config: conf/plan_wall.yaml
   Planning will run every 3 epochs
   Planners: ['gd', 'cem']
   Goal source: ['dset', 'random_state'], Goal H: [5]
```

**At Epoch 3:**
```
🎯 Launching planning jobs for epoch 3...
   Checkpoint: /path/to/checkpoints/model_3.pth
   Model name: 2025-11-04/HH-MM-SS_wall_f5_h1_p1, epoch: 3
Submitted evaluation job for checkpoint: path/to/submitit-evals/epoch_3/..., job id: 12345
```

**At Epoch 6:**
```
🎯 Launching planning jobs for epoch 6...
   Checkpoint: /path/to/checkpoints/model_6.pth
   Model name: 2025-11-04/HH-MM-SS_wall_f5_h1_p1, epoch: 6
Submitted evaluation job for checkpoint: path/to/submitit-evals/epoch_6/..., job id: 12346
```

## Full Training Run

```bash
bash test_new_quantizers.sh
```

This trains for 100 epochs with planning jobs submitted at:
- Epoch 3, 6, 9, 12, ..., 99

## Troubleshooting

### Planning logs not appearing?

1. **Check plan_cfg_path in config:**
   ```bash
   grep "plan_cfg_path:" conf/train.yaml
   # Should show: plan_cfg_path: conf/plan_wall.yaml
   ```

2. **Check test script override:**
   ```bash
   grep "plan_settings.plan_cfg_path" test_new_quantizers.sh
   # Should show: plan_settings.plan_cfg_path=conf/plan_wall.yaml
   ```

3. **Check conda environment:**
   ```bash
   source /scratch/yp2693/world_models/penv/bin/activate
   python -c "import submitit; print('✅ Submitit available')"
   ```

### Planning jobs submitted but not running?

1. Check Submitit logs:
   ```bash
   ls -la submitit-evals/epoch_*/
   ```

2. Check job status:
   ```bash
   squeue -u yp2693  # Check SLURM jobs
   ```

### Planning not defined in plan_wall.yaml?

Verify planner configs exist:
```bash
ls -la conf/planner/
# Should show: gd.yaml, cem.yaml, mpc_cem.yaml, mpc_gd.yaml
```

## Verification Checklist

- [ ] Conda environment activated: `source /scratch/yp2693/world_models/penv/bin/activate`
- [ ] `plan_cfg_path: conf/plan_wall.yaml` in conf/train.yaml (line 103)
- [ ] `plan_settings.plan_cfg_path=conf/plan_wall.yaml` in test script (line 95)
- [ ] Planner config files exist in `conf/planner/`
- [ ] Submitit installed in conda env
- [ ] SLURM available for job submission
- [ ] Checkpoints saved successfully (check `checkpoints/` directory)

## Log Location

Training logs are saved to:
```
small_outputs/YYYY-MM-DD/HH-MM-SS/.hydra/output.log
```

Check this file for planning-related messages:
```bash
tail -f small_outputs/*/*/logs  # Watch live
```

## Output Structure

Planning job outputs:
```
small_outputs/YYYY-MM-DD/HH-MM-SS/
└── submitit-evals/
    ├── epoch_3/
    │   ├── gd_goal_source=dset_goal_H=5_alpha=0.1/
    │   ├── gd_goal_source=dset_goal_H=5_alpha=1/
    │   ├── gd_goal_source=random_state_goal_H=5_alpha=0.1/
    │   ├── cem_goal_source=dset_goal_H=5_alpha=0.1/
    │   └── ... (multiple planner + param combinations)
    ├── epoch_6/
    │   └── ...
    └── epoch_9/
        └── ...
```
