# Quantized DINO World Model Implementation Summary

## Overview
Successfully implemented the quantized DINO world model architecture as specified in the flowchart. The implementation includes:
- Student-teacher encoder architecture
- Vector quantization for both state and action embeddings
- VCReg regularization loss
- 3D convolutional predictor with causal masking
- Full training pipeline with multi-GPU support

## Architecture Flow

### 1. Data Split (Source/Target)
```
obs (B, num_hist + num_pred, 3, 224, 224) → obs_src (B, num_hist, ...) + obs_tgt (B, num_pred, ...)
act (B, num_hist + num_pred, action_dim) → act_src (B, num_hist, ...) + act_tgt (B, num_pred, ...)
```

### 2. Student Encoding Path
```
obs_src → Conv2D Encoder → (B, num_hist, H, W, 64) visual embeddings
act_src → Action Encoder → (B, num_hist, 10) action embeddings
  ↓
Concatenate (channel-wise) → z_src (B, num_hist, H, W, 74)
  ↓
Vector Quantization:
  - State: 64 channels → 128 codebook entries
  - Action: 10 channels → 128 codebook entries
  ↓
z_src_quantized (B, num_hist, H, W, 74)
```

### 3. Teacher Encoding Path (no_grad)
```
obs_tgt → Conv2D Encoder → (B, num_pred, H, W, 64)
act_tgt → Action Encoder → (B, num_pred, 10)
  ↓
Concatenate → z_tgt (B, num_pred, H, W, 74)
  ↓
Vector Quantization (state only) → z_tgt_quantized
```

### 4. Prediction and Loss
```
z_src_quantized → Conv3D Predictor → z_pred (B, num_hist, H, W, 74)
  ↓
Re-Quantize (state only) → z_pred_quantized
  ↓
Embedding Loss: MSE(z_pred_obs, z_tgt_obs)
```

### 5. Decoder Reconstruction
```
z_concat = [z_src, z_tgt] → VQ-VAE Decoder → visual_reconstructed
  ↓
Reconstruction Loss: MSE(visual_reconstructed, obs['visual'])
```

### 6. Total Loss
```
Total Loss = VCReg Loss + Quantization Loss + Embedding Loss + Reconstruction Loss
```

## Files Modified/Created

### New Files
1. **models/quantizer/vector_quantizer.py** - Vector quantizer implementation
2. **objectives/vcreg.py** - VCReg regularization objective
3. **models/action_encoder/proprio.py** - Action encoder implementation
4. **test_architecture.py** - Test script to verify implementation

### Modified Files
1. **models/visual_world_model.py**
   - Added quantization support with `state_quantizer` and `action_quantizer`
   - Implemented `separate_src_tgt()` method
   - Implemented `quantize_embeddings()` method
   - Updated `forward()` to follow student-teacher architecture
   - Updated `encode_obs()` to return (B, T, H, W, C) format
   - Updated `encode()` to support both concat_dim=0 and concat_dim=1
   - Added VCReg loss computation
   - Modified predictor flow with quantization
   - Updated decoder reconstruction logic

2. **train.py**
   - Added quantizer initialization in `init_models()`
   - Added action_encoder_optimizer to save keys
   - Added quantizer checkpointing support
   - Updated model instantiation to pass quantizers

3. **conf/train.yaml**
   - Added `state_quantizer` and `action_quantizer` to defaults
   - Added `vcreg_loss_weight: 1.0`
   - Added `quantization_loss_weight: 1.0`
   - Added `state_vocabulary_size: 128`
   - Added `action_vocabulary_size: 128`

## Key Features

### 1. Student-Teacher Architecture
- **Student**: Trainable encoder that learns representations
- **Teacher**: Fixed encoder (same weights, no gradients) for target embeddings
- Enables stable training by preventing representation collapse

### 2. Vector Quantization
- **State Quantizer**: 128 codebook entries for 64-dim visual features
- **Action Quantizer**: 128 codebook entries for 10-dim action features
- Commitment cost: 0.25
- Straight-through estimator for gradients

### 3. VCReg Regularization
- **Variance Loss**: Prevents representation collapse
- **Covariance Loss**: Decorrelates feature dimensions
- Applied only to state embeddings (not actions)
- Weight: 1.0 (configurable)

### 4. 3D Convolutional Predictor
- Causal convolutions in temporal dimension
- Processes spatial and temporal information jointly
- Input/Output: (B, T, H, W, C) format
- Residual connections for stable training

### 5. Multi-GPU Training Support
- Uses Accelerate library for distributed training
- Automatic gradient accumulation
- Synchronized batch normalization
- Checkpoint saving/loading

## Configuration

### For Wall Environment
```yaml
# Use these settings in conf/train.yaml
env: wall
encoder: conv2d  # or dino for DINO encoder
action_encoder: proprio
decoder: conv2d  # or vqvae
predictor: conv3d
state_quantizer: vector_quantizer
action_quantizer: vector_quantizer

# Model settings
model:
  train_encoder: False  # Set to True to train encoder
  train_predictor: True
  train_decoder: True
  vcreg_loss_weight: 1.0
  quantization_loss_weight: 1.0

# Quantization
state_vocabulary_size: 128
action_vocabulary_size: 128

# Action encoding
action_emb_dim: 10
num_action_repeat: 1
concat_dim: 1  # Concatenate along channel dimension
```

## Testing

Run the test script to verify the implementation:
```bash
conda activate /scratch/yp2693/world_models/penv
cd /scratch/yp2693/world_models/quantised_dinowm
python test_architecture.py
```

Expected output:
- ✓ Forward pass works correctly
- ✓ Loss components computed properly
- ✓ Rollout functionality works
- ✓ All shapes are correct

## Training

To start training:
```bash
python train.py env=wall
```

For multi-GPU training with SLURM:
```bash
python train.py -m env=wall hydra/launcher=submitit_slurm
```

## Loss Components Logged

1. **vcreg_std_loss** - Variance loss component
2. **vcreg_cov_loss** - Covariance loss component
3. **vcreg_loss** - Total VCReg loss
4. **quantization_loss** - Vector quantization loss
5. **z_loss** - Embedding prediction loss (used for training)
6. **z_visual_loss** - Visual embedding loss (monitoring only)
7. **z_collapse_loss** - Collapse detection (monitoring only)
8. **decoder_loss_pred** - Decoder loss on predictions (monitoring only)
9. **decoder_loss_reconstructed** - Decoder reconstruction loss (used for training)
10. **loss** - Total loss

## Spatial Dimension Handling

The implementation uses a 2D spatial grid representation:
- Conv2D encoder output: 224×224 → 14×14 grid (with depth=4, stride=2)
- DINO encoder output: 196 patches → 14×14 grid
- Embeddings: (B, T, H=14, W=14, C)
- Quantizers work on per-position basis
- Predictor processes full 5D tensor

## Notes

1. **Encoder Training**: Set `train_encoder: False` initially, can be enabled later for fine-tuning
2. **Quantization**: Only state embeddings are quantized in target (not actions)
3. **Decoder**: Trained separately with reconstruction loss
4. **VCReg**: Applied only to source embeddings, not target
5. **Accelerate**: All models are wrapped with accelerator.prepare()

## Next Steps

1. Train on wall environment with multi-GPU setup
2. Monitor loss components in wandb
3. Evaluate planning performance
4. Tune hyperparameters (vocabulary sizes, loss weights)
5. Experiment with different encoder architectures
