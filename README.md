# VQ-WM: Vector-Quantized World Model

A PyTorch implementation of a vector-quantized world model for the Wall environment, featuring 3-phase training pipeline with continuous and discrete representations.

## 🚀 Quick Start

```bash
# 0. Verify Tanh setup (1 minute - NEW!)
python verify_tanh_setup.py

# 1. Quick test with EMA quantizer (10 minutes - UPDATED!)
./test_new_quantizers.sh

# 2. Verify resume works (20 minutes)
./tests/test_resume_fix.sh --mode single

# 3. Full training (12-24 hours)
./train_single_stage_enhanced.sh --mode full --fresh --wandb

# 4. Resume if interrupted
./train_single_stage_enhanced.sh --resume outputs/[timestamp]
```

## 📚 Documentation

### 🌟 Start Here
- **[Training Scripts Guide](docs/TRAINING_SCRIPTS_GUIDE.md)** - Complete guide to all training scripts, usage examples, and workflows ⭐

### Training Modes
- **[Single-Stage Training](docs/SINGLE_STAGE_TRAINING_README.md)** - Recommended: 3 phases in one continuous run
- **[3-Stage Training](docs/TRAINING_GUIDE.md)** - Advanced: Separate Stage 1 → Stage 2 → Stage 3

### Technical Details
- **[Tanh Activation Update](docs/TANH_ACTIVATION_UPDATE.md)** - ⭐ NEW: Tanh for full codebook capacity & symmetric representations
- **[Improved Quantizers](docs/IMPROVED_QUANTIZERS.md)** - ⭐ NEW: EMA & Gumbel quantizers for better codebook utilization
- **[Codebook Initialization](docs/CODEBOOK_INITIALIZATION.md)** - Deep dive into matching encoder output distribution
- **[Quantizer Collapse Fix](docs/QUANTIZER_COLLAPSE_FIX.md)** - Deep dive into the resume bug and fix
- **[Quick Fix Summary](docs/QUICK_FIX_SUMMARY.md)** - One-page reference for the fix
- **[Implementation Summary](docs/IMPLEMENTATION_SUMMARY.md)** - Overview of all recent changes

### Additional Documentation
- **[Changes Applied](docs/CHANGES_APPLIED.md)** - Recent modifications to the codebase
- **[Project Structure](docs/PROJECT_STRUCTURE.md)** - Codebase organization

## 🎯 Training Scripts

### Recommended: Enhanced Single-Stage Training

**Script:** `train_single_stage_enhanced.sh`

Single-stage training with all 3 phases in one run plus comprehensive resume support.

```bash
# Full training (90 epochs, 1920 rollouts, codebook 128/128)
./train_single_stage_enhanced.sh --mode full --fresh

# Quick experiment (30 epochs, 400 rollouts, codebook 64/64)
./train_single_stage_enhanced.sh --mode quick

# Fast test (10 epochs, 100 rollouts, codebook 16/4)
./train_single_stage_enhanced.sh --mode test

# Resume from checkpoint
./train_single_stage_enhanced.sh --resume outputs/2025-10-21/12-34-56

# Continue if checkpoint exists, else fresh start
./train_single_stage_enhanced.sh --mode full --continue
```

**Features:**
- ✅ Three training phases in one continuous run
- ✅ Automatic 3-phase learning rate scheduling
- ✅ Built-in resume support (no collapse!)
- ✅ Multiple training modes
- ✅ Easy checkpoint management

### New: 3-Stage Training with Smooth Learning Rate Decay

**Script:** `train_single_stage_dynamic_lr.sh` ⭐ NEW

Implements proper 3-stage training with smooth cosine annealing within each stage (not just step decay).

```bash
./train_single_stage_dynamic_lr.sh
```

**Stage Breakdown:**
- **Stage 1 (1-30):** Encoder learns | Quantizers FROZEN → continuous representations
- **Stage 2 (31-60):** Encoder FROZEN | Quantizers learn → discrete codebook
- **Stage 3 (61-100):** All train (encoder at 5e-6 LR) → fine-tuning

**Features:**
- ✅ Component freezing via learning rate (LR=0 = frozen)
- ✅ Smooth cosine annealing within each stage
- ✅ With new improvements: Tanh encoder, EMA quantizer, KL divergence scaling

### Legacy: 3-Stage Separate Training

For users who need manual control between stages:

```bash
# Train all 3 stages separately
./train_3stage_full.sh

# Resume Stage 3 from checkpoint
./resume_stage3_full.sh
```

**See [Training Guide](docs/TRAINING_GUIDE.md) for details.**

## 🧪 Testing

### Test Resume Functionality

Verify the quantizer collapse fix is working:

```bash
# Test single-stage pipeline (recommended, 20 min)
./tests/test_resume_fix.sh --mode single

# Test 3-stage pipeline
./tests/test_resume_fix.sh --mode 3stage

# Clean start
./tests/test_resume_fix.sh --mode single --clean
```

**Expected output:**
```
✅ Fix applied: Quantizer optimizers were skipped
✅ TEST PASSED: No collapse detected!
```

See [Test Resume Fix Guide](docs/QUICK_FIX_SUMMARY.md) for more details.

## 📁 Project Structure

```
quantised_dinowm_bhumi/
├── README.md                           # This file
├── train.py                            # Main training script
│
├── Training Scripts (Main)
├── train_single_stage_enhanced.sh      # ⭐ Recommended (3-phase in 1 run)
├── train_single_stage_dynamic_lr.sh    # ⭐ NEW (all train together, dynamic LR)
├── train_single_stage.sh               # Basic single-stage
├── train_3stage_full.sh                # Legacy 3-stage (full)
├── train_3stage_quick.sh               # Legacy 3-stage (quick)
├── resume_stage3_full.sh               # Resume Stage 3
└── resume_stage3_quick.sh              # Resume Stage 3 (quick)
│
├── tests/                              # Testing scripts and logs
│   ├── test_resume_fix.sh              # Test quantizer resume fix
│   ├── test_checkpoint_loading_fix.sh  # Test checkpoint loading
│   └── logs/                           # Test logs go here
│
├── docs/                               # Documentation
│   ├── TRAINING_SCRIPTS_GUIDE.md       # ⭐ Complete usage guide
│   ├── SINGLE_STAGE_TRAINING_README.md # Single-stage details
│   ├── SINGLE_STAGE_DYNAMIC_LR.md      # ⭐ NEW: Dynamic LR guide
│   ├── TRAINING_GUIDE.md               # 3-stage details
│   ├── QUANTIZER_COLLAPSE_FIX.md       # Technical deep-dive
│   ├── QUICK_FIX_SUMMARY.md            # Quick reference
│   ├── IMPLEMENTATION_SUMMARY.md       # Recent changes overview
│   └── ...                             # Additional docs
│
├── models/                             # Model implementations
│   ├── encoder/                        # Visual encoders
│   ├── predictor/                      # Dynamics predictors
│   ├── decoder/                        # Visual decoders
│   ├── quantizer/                      # Vector quantizers
│   └── visual_world_model.py           # Main model
│
├── env/                                # Environment wrappers
├── conf/                               # Hydra configuration
└── outputs/                            # Training outputs
```

## 🎓 Training Pipeline

All training (single-stage or 3-stage) goes through these phases:

### Phase 1: Continuous Representations (30% of training)
- **Trains:** Encoder, Predictor, Decoder
- **Frozen:** Quantizers (don't exist yet)
- **Goal:** Learn good continuous representations

### Phase 2: Quantizer Learning (20% of training)
- **Trains:** Quantizers, Predictor, Decoder
- **Frozen:** Encoder
- **Goal:** Learn discrete codebook while preserving encoder

### Phase 3: Joint Fine-Tuning (50% of training)
- **Trains:** Everything (encoder at very low LR)
- **Goal:** Fine-tune all components together

**Difference:**
- **Single-stage:** All phases in one run with automatic transitions
- **3-stage:** Each phase in separate runs with manual checkpoint management

## ✅ Recent Improvements

### Quantizer Collapse Fix (2025-10-21)

Fixed a critical bug where quantizers would collapse immediately when resuming training.

**The problem:**
- Learning rate scheduler state mismatch
- Resuming with different epoch counts caused corrupt LRs
- Quantizers collapsed to single code

**The solution:**
- Skip loading quantizer optimizer/scheduler states on resume
- Quantizer weights still preserved
- Fresh schedulers prevent corruption

**Status:** ✅ Fixed and tested

See [Quantizer Collapse Fix](docs/QUANTIZER_COLLAPSE_FIX.md) for technical details.

## 🛠️ Configuration

Main configuration files:

- `conf/train.yaml` - Training hyperparameters
- `conf/env/wall.yaml` - Environment settings
- `conf/encoder/conv2d.yaml` - Encoder configuration
- `conf/predictor/conv3d.yaml` - Predictor configuration
- `conf/decoder/vqvae.yaml` - Decoder configuration
- `conf/state_quantizer/vector_quantizer.yaml` - State quantizer
- `conf/action_quantizer/vector_quantizer.yaml` - Action quantizer

## 🔧 Requirements

```bash
# Create environment
conda create -n vqwm python=3.9
conda activate vqwm

# Install dependencies
pip install -r requirements.txt

# Set dataset path
export DATASET_DIR=/path/to/datasets
```

## 📊 Monitoring Training

### Key Metrics to Watch

**Collapse Detection:**
```
z_collapse_loss: 0.945  |  ✅ Healthy range (>0.1)
State Codebook: 100.0% utilized  |  ✅ Good (>75%)
```

**Warning Signs:**
```
z_collapse_loss: 0.000  |  ⚠️ COLLAPSE DETECTED!
State Codebook: 6.2% utilized  |  🔴 SEVERE (<25%)
```

**Learning Rates:**
```
📊 Learning rates at epoch 11: encoder=1.00e-04 | state_q=1.00e-04
```

### Using Wandb

```bash
# Enable wandb logging
./train_single_stage_enhanced.sh --mode full --wandb
```

## 🐛 Troubleshooting

### Quantizers Collapse on Resume

**Solution:** Ensure you're using the latest code with the fix applied.

**Verify:**
```bash
./tests/test_resume_fix.sh --mode single
```

Look for `⏭️ Skipping` messages in logs.

### Out of Memory

**Solutions:**
- Use smaller mode: `--mode quick` or `--mode test`
- Reduce batch size in `conf/train.yaml`
- Use fewer rollouts

### Training Diverges

**Check:**
- Learning rates are reasonable (1e-6 to 1e-3)
- Gradient clipping is enabled
- VCReg loss weight not too high

See [Training Scripts Guide](docs/TRAINING_SCRIPTS_GUIDE.md) for more troubleshooting.

## 📖 Getting Help

1. **Check documentation:**
   - [Training Scripts Guide](docs/TRAINING_SCRIPTS_GUIDE.md) - Start here!
   - [Technical Docs](docs/) - All documentation

2. **Run tests:**
   ```bash
   ./tests/test_resume_fix.sh --mode single
   ```

3. **Check logs:**
   - Training logs: `outputs/[timestamp]/`
   - Test logs: `tests/logs/`

## 📝 Citation

If you use this code, please cite:

```bibtex
@misc{vqwm2025,
  title={Vector-Quantized World Model for Wall Environment},
  author={Your Name},
  year={2025}
}
```

## 📜 License

[Your License Here]

---

**Version:** 2.0
**Last Updated:** 2025-10-21
**Status:** ✅ Production Ready

For detailed usage instructions, see [Training Scripts Guide](docs/TRAINING_SCRIPTS_GUIDE.md).
