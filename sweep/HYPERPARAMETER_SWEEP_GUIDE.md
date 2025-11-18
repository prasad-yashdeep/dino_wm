# W&B Hyperparameter Sweep Guide

## Overview

This guide explains how to run a Weights & Biases (W&B) hyperparameter sweep to optimize your 3-stage world model training. The sweep will test different combinations of loss weights and learning rates to find the best configuration.

---

## What is a Hyperparameter Sweep?

A hyperparameter sweep automatically trains your model with different hyperparameter combinations and logs results to W&B. You can then analyze which combination works best.

**Example workflow:**
```
Sweep Configuration (9 different combinations)
    ↓
W&B generates all combinations
    ↓
Run training for each combination
    ↓
Log metrics to W&B
    ↓
Analyze results and find best configuration
```

---

## Parameters Being Swept

### Loss Weights (Grid: Fixed)

#### 1. **Quantization Loss Weight** [0.5, 1.0, 2.0]

**What it does:** Controls the balance between reconstruction quality and learning discrete codes.

**Values:**
- **0.5** = Less emphasis on quantization
  - ✅ Pros: Better image reconstruction quality
  - ❌ Cons: Weaker discrete code learning, poorer codebook usage
  - **Best for:** When you need clear/sharp images

- **1.0** = Balanced (recommended starting point)
  - ✅ Pros: Good balance between reconstruction and quantization
  - ❌ Cons: May diverge if other LRs are wrong
  - **Best for:** General purpose training

- **2.0** = More emphasis on quantization
  - ✅ Pros: Forces learning of good discrete codes, prevents codebook collapse
  - ❌ Cons: Blurrier reconstructions
  - **Best for:** When codebook utilization is low

#### 2. **VCReg Loss Weight** [1, 5, 10]

**What it does:** Regularizes the codebook to prevent collapse (when few codes are used).

**Values:**
- **1** = Minimal diversity enforcement
  - ✅ Pros: Higher codebook utilization (more codes used)
  - ❌ Cons: May collapse (codes become redundant)
  - **Best for:** Large datasets with many distinct embeddings

- **5** = Moderate diversity (recommended for 100 rollouts)
  - ✅ Pros: Good balance between utilization and stability
  - ❌ Cons: Middle-of-the-road, may not be optimal
  - **Best for:** Medium-sized datasets

- **10** = Strong diversity enforcement
  - ✅ Pros: More stable, prevents collapse
  - ❌ Cons: Lower codebook utilization (fewer codes used)
  - **Best for:** Small datasets with high collapse risk

### Learning Rates (Log-Uniform Distribution)

These use **log-uniform distribution**, meaning equal spacing on logarithmic scale. The sweep will test values roughly:
- 5e-6, 1e-5, 2e-5, 5e-5, 1e-4, 2e-4, 5e-4, 1e-3, 2e-3, 5e-3

#### 3. **State Quantizer Learning Rate** [5e-6 to 5e-3]

**What it does:** Controls how fast the state (visual) codebook entries update.

**Guidelines:**
- **Too Low (5e-6):** Codebook barely changes, training very slow
- **Optimal (~1e-4):** Stable codebook updates, balanced learning
- **Too High (5e-3):** Codebook changes too fast, decoder can't track

**Expected range:** 1e-5 to 1e-4 (based on previous experiments)

#### 4. **Predictor Learning Rate** [5e-6 to 5e-3]

**What it does:** Controls how fast the next-frame predictor learns.

**Guidelines:**
- **Too Low (5e-6):** Model learns very slowly
- **Optimal (~3e-4):** Good convergence without divergence
- **Too High (5e-3):** May diverge or oscillate

**Expected range:** 1e-4 to 5e-4 (based on previous experiments)

#### 5. **Encoder Learning Rate** [5e-6 to 5e-3]

**What it does:** Controls how fast visual encoder learns features.

**Important notes:**
- **Stage 1:** Encoder is trained with this LR
- **Stage 2:** Encoder is **frozen** (LR=0), regardless of this value
- **Stage 3:** Encoder trains with reduced LR (stage3_encoder_lr)

**Guidelines:**
- **Too Low (5e-6):** Encoder barely learns, poor feature quality
- **Optimal (~1e-5 to 1e-4):** Good feature learning without overfitting
- **Too High (5e-3):** May overfit, especially with small dataset (100 rollouts)

**Expected range:** 5e-6 to 1e-4 (small values for limited data)

---

## Total Sweep Size

With current configuration:

```
Quantization Loss × VCReg Loss × Random LR combinations
= 3 × 3 × (random) × (random) × (random)
= 9 + variations in learning rates
```

**Each run takes:** ~1 hour (depending on dataset size and epochs)
**Total time estimate:** 9-12 hours for all combinations

---

## How to Run the Sweep

### Step 1: Initialize the Sweep

This creates the sweep on W&B and generates a sweep ID.

```bash
cd /scratch/yp2693/world_models/bhumi_exp3
bash run_sweep.sh --init
```

**Output:**
```
📊 Initializing new W&B sweep...

Sweep Parameters:
  • Quantization Loss Weight: 0.5, 1.0, 2.0
  • VCReg Loss Weight: 1, 5, 10
  • State Quantizer LR: 5e-6 to 5e-3
  • Predictor LR: 5e-6 to 5e-3
  • Encoder LR: 5e-6 to 5e-3

Creating sweep...
✅ Sweep created successfully!

Sweep ID: yashdeep18121-new-york-university/dino_wm_sweeps/abc123def456

To run sweep agents:
  bash run_sweep.sh yashdeep18121-new-york-university/dino_wm_sweeps/abc123def456

To run multiple agents in parallel:
  bash run_sweep.sh yashdeep18121-new-york-university/dino_wm_sweeps/abc123def456 4
```

### Step 2: Run Sweep Agents

After getting the sweep ID, run sweep agents to train models:

**Single agent (sequential):**
```bash
bash run_sweep.sh yashdeep18121-new-york-university/dino_wm_sweeps/abc123def456
```

**Multiple agents in parallel (faster):**
```bash
bash run_sweep.sh yashdeep18121-new-york-university/dino_wm_sweeps/abc123def456 4
```

This launches 4 agents in parallel, each training a different hyperparameter combination.

### Step 3: Monitor Progress

**Option 1: On W&B Dashboard**
Visit: https://wandb.ai/yashdeep18121-new-york-university/dino_wm_sweeps

You'll see:
- Real-time training curves for each run
- Hyperparameter values for each run
- Final metrics (validation loss, codebook utilization, etc.)

**Option 2: Logs**
```bash
# Check individual agent logs
tail -f sweep_agent_1.log
tail -f sweep_agent_2.log

# Check training logs
tail -f outputs/2025-11-04/16-42-01/training.log
```

---

## Analyzing Sweep Results

### 1. View Results on W&B

Go to your sweep page: https://wandb.ai/yashdeep18121-new-york-university/dino_wm_sweeps

You'll see:
- **Parallel coordinates plot:** Shows how parameters affect final loss
- **Table view:** Compare all runs side-by-side
- **Charts:** Loss curves, metrics over time

### 2. Identify Best Configuration

Look for runs with:
- ✅ **Lowest validation loss** - Best prediction accuracy
- ✅ **High codebook utilization** (>75%) - Good code diversity
- ✅ **Stable training curves** - No divergence or spikes
- ✅ **Reasonable training time** - Not too slow

### 3. Example Analysis

**Bad combination (diverging loss):**
```
Quantization Loss: 2.0
VCReg Loss: 1
State Quant LR: 5e-3    ← Too high!
Predictor LR: 1e-3
Encoder LR: 1e-4
Result: Loss explodes at epoch 8 ❌
```

**Good combination (stable loss):**
```
Quantization Loss: 1.0
VCReg Loss: 5
State Quant LR: 1e-4    ✓ Stable
Predictor LR: 3e-4      ✓ Stable
Encoder LR: 1e-5        ✓ Prevents overfitting
Result: Smooth decrease from 4.0 → 0.8 ✓
```

### 4. Export Best Configuration

Once you find the best run:

1. Click on the run in W&B
2. Note the hyperparameters
3. Create a new training script with these fixed values

Example:
```bash
# After identifying best config
bash train_3stage_quick.sh  # Update with best hyperparameters
```

---

## Common Issues & Solutions

### Issue 1: Sweep Creates Too Many Runs

**Problem:** You only wanted 9 runs but got 50+

**Cause:** Log-uniform LR distribution creates combinations, multiplied by grid parameters

**Solution:**
- Option A: Use `method: random` with `max_runs: 20` in sweep.yaml
- Option B: Reduce parameter ranges
- Option C: Use grid for LRs instead of log-uniform

### Issue 2: Agents Keep Crashing

**Problem:** W&B agents die unexpectedly

**Solutions:**
1. Check logs: `cat sweep_agent_1.log | tail -50`
2. Verify wandb login: `wandb login`
3. Check disk space: `df -h`
4. Restart agent: `bash run_sweep.sh <sweep_id>`

### Issue 3: W&B Connection Lost

**Problem:** "ConnectionError: Failed to upload metrics"

**Solutions:**
1. Check internet connection
2. Offline mode: `wandb offline`
3. Later sync: Metrics save locally and sync when back online

### Issue 4: Hyperparameters Not Changing

**Problem:** All runs use same hyperparameters

**Cause:** `disable_wandb: true` in config (ignores W&B parameters)

**Solution:** Set `disable_wandb: false` in sweep.yaml

---

## Advanced: Customizing the Sweep

### Modify sweep.yaml to:

1. **Add more parameters:**
```yaml
model.num_layers:
  values: [2, 3, 4, 5]
```

2. **Change distribution:**
```yaml
training.batch_size:
  distribution: categorical
  values: [16, 32, 64]
```

3. **Set fixed value:**
```yaml
training.epochs:
  value: 100  # Fixed (not swept)
```

4. **Limit total runs:**
```yaml
method: random
early_terminate:
  type: hyperband
  max_iter: 27
  eta: 3
  s: 5
```

---

## Expected Results

Based on your previous experiments:

**Best Configuration (predicted):**
```
Quantization Loss Weight: 1.0
VCReg Loss Weight: 5
State Quantizer LR: 1e-4
Predictor LR: 3e-4
Encoder LR: 1e-5

Expected metrics:
  - Final validation loss: ~0.6-0.8
  - State codebook utilization: ~98%
  - Action codebook utilization: ~45%
  - Training time: ~1 hour
```

---

## Next Steps After Sweep

1. **Identify best run** on W&B dashboard
2. **Document the configuration** in a new script
3. **Retrain with best parameters** for final model
4. **Test on validation/test set** for deployment

---

## References

- [W&B Sweep Documentation](https://docs.wandb.ai/guides/sweeps)
- [Example Colab (from your link)](https://colab.research.google.com/github/wandb/examples/blob/master/colabs/pytorch/Organizing_Hyperparameter_Sweeps_in_PyTorch_with_W%26B.ipynb)
- Your previous experiments (Stage 2 instability analysis)

---

## File Structure

```
bhumi_exp3/
├── sweep.yaml                    ← Sweep configuration (9 combinations)
├── run_sweep.sh                  ← Script to initialize & run sweeps
├── train_with_sweep.py          ← Python wrapper for W&B integration
├── HYPERPARAMETER_SWEEP_GUIDE.md ← This file
├── train.py                       ← Original training script
└── conf/
    ├── train.yaml               ← Base Hydra configuration
    ├── encoder/
    │   └── conv2d.yaml
    └── ...
```

---

**Good luck with your sweep! 🚀**

Questions? Check W&B docs or review the sweep.yaml comments.
