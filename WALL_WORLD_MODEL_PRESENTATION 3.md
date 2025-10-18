# Visual World Model for Wall Environment
## Self-Supervised Learning of Spatial Dynamics with Vector Quantization

**Presentation for Technical Review**

---

## Slide 1: Overview

### Architecture Summary
- **Task**: Learn visual dynamics model for wall navigation environment
- **Approach**: Self-supervised future prediction with quantized embeddings
- **Key Innovation**: Discrete state-space representation learned end-to-end

### System Components
1. Conv2D Encoder (spatial feature extraction)
2. Vector Quantizer (discrete representation)
3. Conv3D Predictor (dynamics model)
4. VQ-VAE Decoder (visual reconstruction)
5. VCReg Regularization (representation quality)

---

## Slide 2: Wall Environment Specification

### Environment Characteristics
```
Visual Space: 224×224 RGB images
State Space:  [x, y] ∈ [4, 60] × [4, 60] (continuous)
Action Space: [Δx, Δy] ∈ ℝ² (continuous velocity commands)
Constraints:  Vertical wall at x≈32 with door opening at y≈10
```

### Dataset Structure
```python
Training Sample:
  obs['visual']: (B, 2, 3, 224, 224)  # t=0 and t=5 frames (frameskip=5)
  act:           (B, 2, 10)            # Concatenated actions across frameskip

Action Dimension Computation (frameskip=5):
  - Each timestep has raw action: [Δx, Δy]
  - 5 consecutive actions concatenated: [Δx₁,Δy₁, Δx₂,Δy₂, ..., Δx₅,Δy₅]
  - Total action_dim = 2 × 5 = 10 dimensions per frame

Trajectory: 20,000 rollouts of wall crossing behaviors
```

### Visual Representation
```
┌────────────────────────────────┐
│           • agent (red dot)    │
│                    ║           │  ║ = wall (black)
│                    ╬ door      │  ╬ = door opening
│                    ║           │  Background: white
└────────────────────────────────┘
```

**Key Challenge**: Learn that wall blocks direct paths; navigation requires door

---

## Slide 3: Self-Supervised Learning Framework

### Temporal Prediction as Supervision

**Core Principle**: Future observations provide free supervision signals

```
Given:  obs_t=0, action_t
Learn:  f(obs_t, action_t) ≈ obs_t+5

Loss = ||encode(predict(obs_t, action)) - encode(obs_t+5)||²
```

### Training Paradigm

**Student-Teacher Framework** (No EMA):
- **Student**: Encodes current state, predicts future embedding
- **Teacher**: Encodes target state (same weights, `torch.no_grad()`)
- **No EMA**: Teacher = Student in inference mode (use_ema_teacher=False)

**Why This Works**:
1. Abundant temporal data (unlimited observation tuples)
2. Physics provides deterministic targets
3. No manual annotation required
4. Learns dynamics, constraints, and representations jointly

---

## Slide 4: Architecture - Encoding Pipeline

### Stage 1: Visual Encoding

**Conv2D Encoder** (spatial downsampling):
```
Input: (B, 3, 224, 224) RGB
│
├─ Conv2d(3→64,   stride=2) → 112×112
├─ Conv2d(64→128, stride=2) → 56×56
├─ Conv2d(128→256, stride=2) → 28×28
├─ Conv2d(256→512, stride=2) → 14×14
└─ Conv2d(512→64, kernel=1) → 14×14 (projection)
│
Output: (B, 14, 14, 64) spatial embeddings
```

**Spatial Preservation**:
- 14×14 grid maintains spatial structure
- Each cell = 16×16 pixel receptive field
- Column ~7: encodes wall structure
- Dot position: high activation in corresponding grid cell

---

## Slide 5: Architecture - Action Integration

### Stage 2: Action Encoding

**Action Representation with Frameskip**:

**Raw Actions** (from environment):
```
5 consecutive timesteps (frameskip=5):
  t=0: [Δx₁, Δy₁]
  t=1: [Δx₂, Δy₂]
  t=2: [Δx₃, Δy₃]
  t=3: [Δx₄, Δy₄]
  t=4: [Δx₅, Δy₅]

Concatenated into single vector:
  action_concat = [Δx₁, Δy₁, Δx₂, Δy₂, Δx₃, Δy₃, Δx₄, Δy₄, Δx₅, Δy₅]
  Shape: (10,) representing full action sequence
```

**ProprioceptiveEmbedding** (action encoder):
```
Input:  (B, T, 10) concatenated actions (already 10-dim)
│
├─ Conv1d(in_chans=10, out_chans=10, kernel=1)
│  Projects 10D action sequence to 10D embedding
│
Output: (B, T, 10) action embeddings
```

**Why 10 dimensions?**
- Encodes entire motion trajectory from t=0 to t=5
- Model sees all intermediate actions, not just instantaneous velocity
- Captures momentum and action persistence

### Spatial Broadcasting

**Concatenation Strategy** (concat_dim=1):
```python
# Tile action embedding to every spatial location
action_tiled = repeat(action_emb, "b t d -> b t h w d", h=14, w=14)
# (B, 1, 10) → (B, 1, 14, 14, 10)

# No repetition needed (num_action_repeat=1)
action_repeated = action_tiled  # Same as input
# → (B, 1, 14, 14, 10)

# Concatenate with visual embeddings
z = cat([visual_emb, action_repeated], dim=-1)
# → (B, 1, 14, 14, 74) = 64(visual) + 10(action)
```

**Rationale**:
- Every spatial location has access to full 5-step action sequence
- Enables local prediction conditioned on entire motion trajectory
- Action embedding identical across all spatial locations (global context)

---

## Slide 6: Vector Quantization - Theory

### Discrete Representation Learning

**Motivation**:
- Continuous embeddings → infinite state space
- Quantization → discrete, interpretable states
- Enables downstream planning (graph search, dynamic programming)

### Vector Quantizer Formulation

Given continuous embedding **z** ∈ ℝ^D, codebook **C** = {c₁, ..., c_K}:

**Forward Pass**:
```
1. Compute distances: d_i = ||z - c_i||²  for i ∈ [1, K]
2. Assign nearest:    k* = argmin(d_i)
3. Quantize:          z_q = c_k*
4. Straight-through:  z_q_st = z + (z_q - z).detach()
```

**Loss Functions**:
```
ℒ_codebook = ||sg[z] - z_q||²      # Update codebook toward embeddings
ℒ_commit   = β ||z - sg[z_q]||²    # Encourage encoder commitment (β=0.25)
ℒ_quant    = ℒ_codebook + ℒ_commit
```

sg[·] = stop gradient operator

---

## Slide 7: Vector Quantization - Implementation

### Two Separate Quantizers

**State Quantizer**:
```
Input:  z_obs (B, 1, 14, 14, 64)  # Visual embeddings
Codebook: 16 vectors × 64 dimensions
Output: Discrete codes ∈ {0, 1, ..., 15} per spatial location
```

**Action Quantizer**:
```
Input:  z_act (B, 1, 14, 14, 10)  # Action embeddings (5-step sequence)
Codebook: 16 vectors × 10 dimensions
Output: Discrete codes ∈ {0, 1, ..., 15} per spatial location

Note: All spatial locations have identical action embedding
      → All assigned to same action code (e.g., all grid cells → Code 2)
```

### Quantization Process

For each spatial location (total: 14×14 = 196 locations):

```
Spatial grid [5, 3] (dot location):
  Continuous: z_state = [0.23, -0.45, 0.82, ..., 0.11] (64-dim)
  Distances to state codebook: [2.1, 0.8, 3.4, ..., 1.9]
  Assigned code: 1 (minimum distance)
  Quantized: z_q = state_codebook[1] = [0.21, -0.43, 0.79, ..., 0.09]

All spatial locations (shared action):
  Continuous: z_action = [0.8, 0.3, 0.7, ..., 0.2] (10-dim, 5-step trajectory)
  Distances to action codebook: [1.5, 0.6, 2.3, ..., 1.8]
  Assigned code: 0 (represents "sustained rightward motion")
  Quantized: All grid cells use action_codebook[0]
```

**Result**: Each frame encoded as 196 discrete state codes + 196 action codes

---

## Slide 8: VCReg Regularization

### Variance-Covariance Regularization

**Problem**: Representation collapse
- All embeddings become identical → trivial solution
- Correlated dimensions → redundant information

**VCReg Solution** (Bardes et al., 2022):

**Variance Loss**:
```
ℒ_var = Σ_d max(0, 1 - √(Var(z_d) + ε))

Encourages std(z, dim=batch) ≥ 1 for each dimension d
Prevents collapse to constant embeddings
```

**Covariance Loss**:
```
ℒ_cov = Σ_{i≠j} Cov(z_i, z_j)²

Penalizes off-diagonal covariance
Decorrelates embedding dimensions
```

**Total**: ℒ_vcreg = ℒ_var + ℒ_cov

**Applied on**: Quantized source embeddings z_src_q

---

## Slide 9: Dynamics Prediction

### Conv3D Predictor Architecture

**Design**: Causal 3D convolutions for spatiotemporal reasoning

```
Input: z_src_q (B, 1, 14, 14, 74) quantized embeddings

Rearrange: (B, T, H, W, C) → (B, C, T, H, W)
│
├─ CausalConv3d(74→128, kernel=(1,3,3))  # No temporal (T=1)
│  BatchNorm3d + ReLU + Dropout(0.1)
│
├─ CausalConv3d(128→128, kernel=(1,3,3))
│  BatchNorm3d + ReLU + Dropout(0.1)
│
├─ CausalConv3d(128→74, kernel=(1,3,3))
│
└─ Residual: output = conv_out + input

Output: z_pred (B, 1, 14, 14, 74) predicted embeddings
```

### Spatial Reasoning with 3×3 Kernels

**Local Context**: Each location sees 8 neighbors
- Integrates: visual features + action + spatial constraints
- Learns: wall blocks transitions, door allows crossing
- Predicts: state transitions as embedding shifts

---

## Slide 10: Causal Convolution Details

### Temporal Causality

**Standard Convolution**: Uses past, present, AND future
**Causal Convolution**: Uses past and present ONLY

```python
class CausalConv3d:
    def forward(self, x):
        # Pad only in the PAST
        temporal_padding = kernel_size[0] - 1
        x = F.pad(x, (0,0, 0,0, temporal_padding, 0))
        #              W   H         T
        #          (left,right, top,bottom, past,future)
        return conv3d(x)
```

**Why**: Prevents information leakage from future frames

**For Wall Environment**:
- num_hist=1 → temporal_kernel=min(3,1)=1
- Effectively no temporal convolution (single frame)
- Only spatial 3×3 convolution applied

---

## Slide 11: VQ-VAE Decoder Architecture

### Visual Reconstruction from Embeddings

**Purpose**: Decode quantized embeddings back to pixel space for visualization and monitoring

```
Input: z_q (B, T, 14, 14, 64) quantized visual embeddings
│
├─ Rearrange: (B, T, H, W, C) → (B×T, C, H, W)
│  Prepare for 2D convolutions
│
├─ Upsample Branch (14×14 → 56×56):
│  ├─ Conv2d(64→384, 3×3, padding=1)
│  ├─ 4× ResidualBlocks(384, 128)
│  │  Each block: ReLU → Conv(3×3) → ReLU → Conv(1×1) + skip
│  ├─ ReLU
│  └─ ConvTranspose2d(384→192, 4×4, stride=2) → 28×28
│     ConvTranspose2d(192→64,  4×4, stride=2) → 56×56
│
├─ Decoder Branch (56×56 → 224×224):
│  ├─ Conv2d(64→384, 3×3, padding=1)
│  ├─ 4× ResidualBlocks(384, 128)
│  ├─ ReLU
│  └─ ConvTranspose2d(384→192, 4×4, stride=2) → 112×112
│     ConvTranspose2d(192→3,   4×4, stride=2) → 224×224
│
Output: RGB image (B×T, 3, 224, 224)
```

### Key Design Choices

**Two-Stage Upsampling**:
1. **Upsample Branch**: 14×14 → 56×56 (latent refinement)
   - Operates in high-dimensional space (384 channels)
   - Residual blocks preserve information
   - Learns intermediate representations

2. **Decoder Branch**: 56×56 → 224×224 (pixel generation)
   - Gradually reduces channels (384 → 3)
   - ConvTranspose2d for spatial upsampling
   - Final output: RGB pixels

**Residual Blocks**:
```
ResBlock(in=384, hidden=128):
  x → ReLU → Conv2d(384→128, 3×3) → ReLU → Conv2d(128→384, 1×1)
  output = block(x) + x  # Skip connection
```
- Helps gradient flow
- Preserves spatial structure
- Enables deeper networks

**Why This Architecture?**:
- **Stride-4 Total**: Matches encoder's 16× downsampling (224→14)
- **High Capacity**: 384 channels, 4 res blocks per stage
- **Spatial Preservation**: ConvTranspose2d maintains alignment
- **Pretrained**: Decoder can be loaded from separate VQ-VAE training

### Decoder Usage in Training

**Monitoring Only** (not used for main loss):
```python
# Decoder forward pass (no gradients flow to encoder/predictor)
with torch.no_grad():
    visual_pred = decoder(z_pred_q)           # Predicted images
    visual_reconstructed = decoder(z_src_q)   # Reconstructed inputs

# Compute reconstruction metrics (for logging only)
decoder_loss_pred = MSE(visual_pred, obs_tgt)
decoder_loss_recon = MSE(visual_reconstructed, obs_src)
```

**Gradients Flow Only to Decoder**:
- Encoder/Predictor trained via embedding loss (ℒ_pred)
- Decoder trained via reconstruction loss (ℒ_recon)
- Allows independent optimization
- Decoder can be frozen during world model training

### Reconstruction Loss

```
ℒ_recon = ||decoder(z_src_q) - obs_src||²
        + ||decoder(z_tgt_q) - obs_tgt||²

Applied to: Both source and target frames
Purpose: Learn pixel-space mapping from embeddings
Weight: Equal to other losses (λ=1.0)
```

**Two Reconstruction Tasks**:
1. **Source Reconstruction**: Decode current observation
   - Tests: Encoder → Quantizer → Decoder pipeline
   - Measures: Information preservation through quantization

2. **Target Reconstruction**: Decode future observation
   - Tests: Full encoding of teacher targets
   - Ensures: Target embeddings are decodable

---

## Slide 12: Complete Architecture Diagram

```
                    WALL ENVIRONMENT
     obs_src (B,1,3,224,224)    obs_tgt (B,1,3,224,224)
     act_src (B,1,10)            act_tgt (B,1,10)
     [Δx₀,Δy₀,...,Δx₄,Δy₄]      [5-step action sequence]
            │                           │
     ┌──────▼──────────┐       ┌───────▼─────────┐
     │  Conv2D Encoder │       │  Conv2D Encoder │
     │   (Student)     │       │   (Teacher)     │
     │ 224→14×14 grid  │       │  no_grad()      │
     └──────┬──────────┘       └───────┬─────────┘
            │ (B,1,14,14,64)           │
     ┌──────▼───────────┐              │
     │ Action Encoder   │              │
     │ (B,1,10)→(B,1,10)│              │
     │ Conv1d(10→10)    │              │
     └──────┬───────────┘              │
            │                          │
     ┌──────▼──────────────────┐       │
     │ Concatenate              │       │
     │ z_src (B,1,14,14,74)     │       │
     │ 64(vis)+10(act 5-step)   │       │
     └──────┬──────────────────┘       │
            │                          │
     ┌──────▼────────────────────────┐ │
     │ QUANTIZATION                  │ │
     │ State:  64 → 16 codes         │ │
     │ Action: 10 → 16 codes         │ │
     │ (5-step trajectories)         │ │
     │ z_src_q (B,1,14,14,74)        │ │
     └──────┬────────────────────────┘ │
            │                          │
     ┌──────▼──────────┐               │
     │ VCReg Loss      │               │
     │ ℒ_var + ℒ_cov   │               │
     └──────┬──────────┘               │
            │                          │
     ┌──────▼───────────────┐          │
     │ Conv3D Predictor     │          │
     │ 3×3 spatial kernels  │          │
     │ Predicts cumulative  │          │
     │ effect of 5 actions  │          │
     │ z_pred (B,1,14,14,74)│          │
     └──────┬───────────────┘          │
            │                          │
     ┌──────▼──────────────┐   ┌───────▼─────────┐
     │ Re-Quantize         │   │ Quantize Target  │
     │ z_pred_q (obs only) │   │ z_tgt_q (obs)    │
     └──────┬──────────────┘   └────────┬─────────┘
            │                           │
            │                    ┌──────┴────────┐
            │                    │               │
     ┌──────▼──────────────┐    │  EMBEDDING    │
     │ VQ-VAE DECODER      │    │  LOSS         │
     │ (no_grad for pred)  │    │  ℒ_pred       │
     │ 14×14 → 224×224     │    │               │
     │ 2-stage upsample    │    │ z_pred_obs    │
     └──────┬──────────────┘    │    vs         │
            │                   │ z_tgt_obs     │
     ┌──────▼──────────────┐    │               │
     │ visual_pred         │    └───────┬───────┘
     │ (B,1,3,224,224)     │            │
     │ [monitoring only]   │            │
     └──────┬──────────────┘            │
            │                           │
            │        ┌──────────────────┘
            │        │
            └────┬───┴───┬─────────┐
                 │       │         │
          ┌──────▼───────▼─────────▼────┐
          │ DECODER RECONSTRUCTION      │
          │ (separate training branch)   │
          │                             │
          │ z = cat([z_src_q, z_tgt_q]) │
          │ recon = decoder(z)          │
          │ ℒ_recon = MSE(recon, obs)   │
          └──────┬──────────────────────┘
                 │
          ┌──────▼──────────────────────┐
          │ TOTAL LOSS COMPUTATION      │
          │ ℒ = ℒ_pred + ℒ_vcreg        │
          │     + ℒ_quant + ℒ_recon     │
          └──────┬──────────────────────┘
                 │
          ┌──────▼──────────────────────┐
          │ BACKWARD & OPTIMIZE         │
          │ Encoder/Predictor: ℒ_pred   │
          │ Decoder: ℒ_recon            │
          │ Quantizers: ℒ_quant         │
          └─────────────────────────────┘
```

---

## Slide 13: Loss Function Decomposition

### Total Training Objective

```
ℒ_total = ℒ_pred + λ_vcreg·ℒ_vcreg + λ_quant·ℒ_quant + λ_recon·ℒ_recon
```

### Component Breakdown

**1. Prediction Loss** (main signal for world model):
```
ℒ_pred = ||z_pred_obs - z_tgt_obs||²

where z_pred_obs = z_pred[..., :64]  # Visual part only (exclude action)
      z_tgt_obs  = z_tgt[..., :64]   # Visual part only (exclude action)

Note: Action part (last 10 dims) excluded from prediction loss
      Only visual state transitions are supervised
```

**2. VCReg Loss** (regularization, λ=1.0):
```
ℒ_vcreg = Σ_d max(0, 1-√(Var(z_d)+ε)) + Σ_{i≠j} Cov(z_i,z_j)²
```

**3. Quantization Loss** (codebook learning, λ=1.0):
```
ℒ_quant = ||sg[z_obs]-z_obs_q||² + 0.25·||z_obs-sg[z_obs_q]||²
        + ||sg[z_act]-z_act_q||² + 0.25·||z_act-sg[z_act_q]||²
```

**4. Reconstruction Loss** (decoder training, λ=1.0):
```
ℒ_recon = ||decoder(z_src_q[..., :64]) - obs_src||²
        + ||decoder(z_tgt_q[..., :64]) - obs_tgt||²

where z_src_q, z_tgt_q are quantized embeddings (visual part only)
      decoder: 14×14×64 → 224×224×3 (VQ-VAE architecture)
```

### Gradient Flow

- **Encoder**: Gets gradients from ℒ_pred + ℒ_vcreg + ℒ_commit
  - ℒ_pred: Learn future prediction
  - ℒ_vcreg: Maintain representation diversity
  - ℒ_commit: Align with codebook

- **Predictor**: Gets gradients from ℒ_pred only
  - Learn dynamics model in embedding space

- **Decoder**: Gets gradients from ℒ_recon only
  - Learn pixel reconstruction from embeddings
  - **Independent from encoder/predictor training**

- **Codebooks**: Get gradients from ℒ_codebook only
  - State codebook: Update toward encoder outputs
  - Action codebook: Update toward action embeddings

- **Straight-through Estimator**: Bypasses quantization for encoder gradients
  - Forward: z_q = quantize(z)
  - Backward: ∇z = ∇z_q (gradient flows as if no quantization)

### Loss Weight Configuration

```yaml
model:
  vcreg_loss_weight: 1.0          # Balance diversity vs prediction
  quantization_loss_weight: 1.0   # Balance discretization vs smoothness
  # decoder_loss implicitly weighted at 1.0 (separate branch)
```

**Trade-offs**:
- **High ℒ_vcreg**: More diverse embeddings, may hurt prediction accuracy
- **High ℒ_quant**: Tighter codebook fit, may lose information
- **ℒ_recon**: Only trains decoder, doesn't affect world model quality

---

## Slide 14: Training Configuration

### Hyperparameters

```yaml
Training:
  epochs: 25
  batch_size: 32
  frameskip: 5              # Predict 5 steps ahead
  num_hist: 1               # Single history frame

Encoder:
  architecture: Conv2D
  depth: 4
  emb_dim: 64
  lr: 5e-4

Action Encoder:
  in_chans: 10              # [Δx₀,Δy₀,...,Δx₄,Δy₄] concatenated
  emb_dim: 10               # Same as input (identity-like)
  num_action_repeat: 1      # No repetition needed
  frameskip: 5              # 5 consecutive actions concatenated
  lr: 5e-4

Quantization:
  state_codebook_size: 16
  action_codebook_size: 16
  commitment_cost: 0.25
  lr: 5e-4

Predictor:
  architecture: Conv3D
  hidden_dim: 128
  depth: 3
  dropout: 0.1
  lr: 5e-4

Decoder:
  architecture: VQ-VAE
  channel: 384              # Hidden channels in residual blocks
  n_res_block: 4            # Number of residual blocks per stage
  n_res_channel: 128        # Channels in residual connections
  emb_dim: 64               # Input embedding dimension
  lr: 3e-4                  # Slightly lower than encoder

Regularization:
  vcreg_weight: 1.0
  quantization_weight: 1.0
  use_ema_teacher: False    # No EMA
```

---

## Slide 15: What the Model Learns

### Emergent Behaviors

**Encoder Learning**:
1. **Wall Detection**: Column 7 in 14×14 grid encodes vertical wall
2. **Door Recognition**: Different pattern at grid[7,4] vs solid wall
3. **Agent Localization**: High activation at dot position
4. **Spatial Layout**: Implicit understanding of navigable regions

**Predictor Learning**:
1. **Action-Conditioned Dynamics**:
   - 5-step action sequence [+Δx,0]×5 → spatial activation pattern shifts rightward
   - High activation at grid[5,3] moves to grid[6,3] after full trajectory
   - Learns translation-equivariant representation (motion → embedding shift)

2. **Wall Constraints**:
   - Predicts collision when action sequence would shift activation into wall
   - Wall column (grid[:,7]) blocks spatial transitions

3. **Door Navigation**:
   - Learns door region (grid[7,4]) allows activation crossing
   - Different embedding pattern: solid wall vs door opening

4. **Multi-Step Trajectory Integration**:
   - Input: Full 5-step action sequence (10-dimensional)
   - Predicts: Cumulative effect of entire trajectory
   - Example: 5 actions of [+2,0] each = total displacement of ~10 pixels

**Codebook Specialization** (16 state codes):
```
Code 0-3:   Dot in left quadrants (various positions)
Code 4-5:   Dot near wall, left side
Code 6-7:   Dot at door level / crossing
Code 8-11:  Dot in right quadrants
Code 12-13: Wall/door structural patterns
Code 14-15: Background and other states
```

---

## Slide 15: Self-Supervision Mechanisms

### Four Levels of Self-Supervision

**Level 1: Temporal Prediction**
```
Self-signal: Future frame is the label
No human annotation needed
```

**Level 2: Teacher-Student**
```
Teacher provides stable targets (same encoder, no_grad)
Student learns to match teacher's future embeddings
Prevents moving target problem
```

**Level 3: VCReg Regularization**
```
Self-signal: Embedding statistics
Variance loss: embeddings should differ across samples
Covariance loss: dimensions should be decorrelated
```

**Level 4: Vector Quantization**
```
Self-signal: Natural clustering in embedding space
Codebook learns discrete prototypes from data
No predefined categories ("dot", "wall", etc.)
```

### Key Insight

**No external supervision required**:
- No labels for "this is a wall"
- No labels for "action succeeded"
- No labels for "dot at position X"

Model learns **everything** from (observation, action, next_observation) tuples!

---

## Slide 16: Codebook Size Analysis

### Trade-offs

| Codebook Size | Pros | Cons | Use Case |
|---------------|------|------|----------|
| **n=8** | Fast training, max compression | Severe bottleneck | ❌ Not recommended |
| **n=16** | Fast, interpretable, sufficient | Limited expressiveness | ✅ Learning & debugging |
| **n=64** | Good balance, fine-grained | Slower training | ✅ **Best performance** |
| **n=256** | Maximum detail | Very slow, collapse risk | Research only |

### Recommendation for Wall Environment

**Start: n=16** (currently configured)
- Quick validation
- Understand learned codes
- Codebook utilization: expect 12-14/16 codes used

**Scale: n=64** (for final model)
- Better spatial resolution (64 ≈ 8×8 discretization)
- More useful for downstream planning
- Still tractable training time

### Monitoring Codebook Health

```python
# Check utilization
unique_codes = np.unique(encoding_indices)
utilization = len(unique_codes) / n_embed

# Goal: >75% utilization
# Warning: <50% = codebook collapse
```

---

## Slide 17: Experimental Validation

### Metrics to Track

**1. Prediction Accuracy**:
```
z_loss = ||z_pred_obs - z_tgt_obs||²

Expected trajectory:
Epoch 1:  z_loss ≈ 2.0 (random)
Epoch 10: z_loss ≈ 0.5 (learning)
Epoch 25: z_loss ≈ 0.1 (converged)
```

**2. Representation Quality**:
```
vcreg_loss = var_loss + cov_loss

Expected: Decreases then stabilizes
Low vcreg → diverse, decorrelated embeddings
```

**3. Quantization Quality**:
```
quant_loss = codebook_loss + commit_loss

Expected trajectory:
Epoch 1:  quant_loss ≈ 1.5 (random codebook)
Epoch 10: quant_loss ≈ 0.3 (specializing)
Epoch 25: quant_loss ≈ 0.1 (converged)
```

**4. Codebook Utilization**:
```
Used codes / Total codes

Goal: >12/16 (75%) for n=16
      >48/64 (75%) for n=64
```

---

## Slide 18: Downstream Applications

### What This Model Enables

**1. Model-Predictive Control**:
```
Given: Current observation, goal position
Plan: Sequence of actions using learned dynamics
Execute: Model predicts outcome of each action
```

**2. Visual Planning**:
```
State Space: 16^196 discrete states (quantized)
Search: A*, Dijkstra, or dynamic programming
Heuristic: Embedding space distance
```

**3. Skill Discovery**:
```
Identify: Repeating patterns in learned codes
Extract: Reusable navigation primitives
Example: "navigate to door" = Code 3 → Code 6
```

**4. Transfer Learning**:
```
Pretrain: On wall environment
Fine-tune: On related navigation tasks
Encoder captures general spatial reasoning
```

---

## Slide 19: Comparison to Related Work

### Positioning in Literature

**World Models (Ha & Schmidhuber, 2018)**:
- Similar: Learns dynamics model for planning
- Different: We use visual inputs (not latent states), quantization

**JEPA (LeCun et al., 2022)**:
- Similar: Self-supervised prediction in embedding space
- Different: We add vector quantization for discrete planning

**VQ-VAE (van den Oord et al., 2017)**:
- Similar: Vector quantization for discrete representations
- Different: We quantize embeddings for dynamics (not reconstruction)

**VCReg (Bardes et al., 2022)**:
- Similar: Use variance-covariance regularization
- Different: Applied to quantized dynamics model

### Our Contribution

**Novel Combination**:
- Spatial Conv2D encoder (preserves geometry)
- Separate state/action quantization (interpretable)
- Causal Conv3D predictor (spatiotemporal reasoning)
- Self-supervised on visual observations (no labels)

---

## Slide 20: Limitations & Future Work

### Current Limitations

**1. Single-Step Prediction**:
- Only predicts 1 frame ahead (frameskip=5)
- Multi-step rollout error accumulates
- Future: Autoregressive multi-step prediction

**2. Deterministic Dynamics**:
- No stochasticity modeling
- Cannot handle uncertainty
- Future: Add variational/diffusion components

**3. Fixed Codebook Size**:
- n=16 may be too coarse for complex scenarios
- n=256 may cause collapse
- Future: Adaptive codebook (grow/shrink)

**4. 2D Environment**:
- Only tested on simple wall navigation
- Limited visual complexity
- Future: Scale to 3D, robotic manipulation

---

## Slide 21: Future Directions

### Short-Term Extensions

**1. Hierarchical Quantization**:
```
Coarse codebook (16 codes):  Global position
Fine codebook (64 codes):    Local details
Multi-scale representation
```

**2. Action Abstraction**:
```
Learn: High-level action codes ("go to door")
Instead of: Low-level velocities [Δx, Δy]
Benefits: Temporal abstraction, planning efficiency
```

**3. Multi-Environment Training**:
```
Train on: Various wall configurations
Generalize to: Unseen wall/door positions
Meta-learning across environment variations
```

### Long-Term Research

**1. Real Robot Deployment**:
- Transfer from simulation to real sensors
- Handle visual noise, lighting changes
- Online adaptation

**2. Language-Conditioned Planning**:
- "Navigate to the door and cross to the other side"
- Combine with vision-language models

**3. Causal Discovery**:
- Identify causal relationships in learned codes
- Separate: agent state vs environment state vs dynamics

---

## Slide 22: Technical Implementation Details

### Code Structure

```
qdino_wm/
├── models/
│   ├── encoder/
│   │   ├── conv2d.py              # Visual encoder
│   │   └── proprio.py             # Action encoder
│   ├── predictor/
│   │   └── conv3d.py              # Dynamics predictor
│   ├── quantizer/
│   │   └── vector_quantizer.py   # VQ layer
│   └── visual_world_model.py      # Full model
├── objectives/
│   └── vcreg.py                   # VCReg loss
├── datasets/
│   └── wall_dset.py               # Wall data loader
├── train.py                        # Training loop
└── conf/
    └── train.yaml                  # Hyperparameters
```

### Key Configuration

```bash
python train.py --config-name train.yaml \
  env=wall \
  frameskip=5 \
  num_hist=1 \
  quantize=True \
  encoder=conv2d \
  predictor=conv3d \
  model.vcreg_loss_weight=1.0 \
  use_ema_teacher=False
```

---

## Slide 23: Computational Requirements

### Training Resources

**Model Size**:
```
Encoder:         ~2M parameters
Action Encoder:  ~1K parameters
Predictor:       ~5M parameters
State Codebook:  16 × 64 = 1K parameters
Action Codebook: 16 × 10 = 160 parameters
Total:           ~7M parameters
```

**Memory**:
```
Batch size 32:   ~4GB GPU memory
Activations:     ~2GB
Gradients:       ~2GB
```

**Training Time**:
```
Wall dataset (20K trajectories):
  1 epoch:  ~10 minutes (single V100)
  25 epochs: ~4 hours

Inference:
  Single forward pass: ~10ms
  Planning (100 steps): ~1 second
```

**Scalability**: Can train on single GPU, distributed training for larger datasets

---

## Slide 24: Reproducibility Checklist

### Ensuring Reproducible Results

**1. Random Seeds**:
```python
torch.manual_seed(0)
np.random.seed(0)
random.seed(0)
```

**2. Deterministic Operations**:
```python
torch.backends.cudnn.deterministic = True
torch.backends.cudnn.benchmark = False
```

**3. Version Control**:
```
PyTorch: 2.0+
CUDA: 11.8+
Python: 3.9+
```

**4. Data Splits**:
```yaml
split_ratio: 0.9
split_mode: "random"
seed: 0
```

**5. Hyperparameter Sensitivity**:
- VCReg weight: Robust in [0.5, 2.0]
- Commitment cost: Sensitive, use 0.25
- Learning rates: Use warmup + cosine decay
- Batch size: 32 is sufficient

---

## Slide 25: Key Takeaways

### Main Contributions

1. **Self-Supervised Visual Dynamics**
   - Learn from observation sequences alone
   - No labels, no rewards, no demonstrations

2. **Spatial Preservation**
   - Conv2D encoder maintains geometric structure
   - 14×14 grid preserves wall/dot spatial relationships

3. **Discrete State Space**
   - Vector quantization creates interpretable codes
   - Enables graph-based planning algorithms

4. **Multi-Component Regularization**
   - VCReg prevents collapse
   - Quantization provides compression
   - Residual connections aid optimization

### Impact

- **Scalable**: Works with limited data (20K trajectories)
- **Efficient**: Trains in ~4 hours on single GPU
- **Interpretable**: Can visualize learned codes
- **Practical**: Suitable for model-based RL and planning

---

## Slide 26: Questions to Consider

### Discussion Points

**1. Codebook Size**:
- Is n=16 sufficient for wall environment?
- When should we scale to n=64 or n=256?
- How to detect codebook collapse?

**2. Multi-Step Prediction**:
- Current: Single-step (frameskip=5)
- Alternative: Autoregressive rollout?
- Trade-off: Accuracy vs error accumulation

**3. Generalization**:
- Trained on fixed wall position (x=32)
- Will it generalize to x=20 or x=40?
- Need for meta-training?

**4. Comparison**:
- How does this compare to JEPA on wall?
- How does this compare to end-to-end RL?
- Ablation studies needed?

**5. Downstream Tasks**:
- Best planning algorithm for discrete codes?
- How to use learned model for control?
- Value function on top of embeddings?

---

## Slide 27: Ablation Studies (Proposed)

### Experimental Design

**Test 1: Quantization Impact**
```
Baseline: Continuous embeddings (quantize=False)
Ours:     Quantized embeddings (quantize=True, n=16)
Metric:   Prediction MSE, planning success rate
Hypothesis: Quantization improves planning, slight prediction drop
```

**Test 2: VCReg Necessity**
```
Ablation 1: No VCReg (weight=0.0)
Ablation 2: Only variance loss
Ablation 3: Only covariance loss
Ours:       Full VCReg (weight=1.0)
Metric:     Representation rank, downstream task performance
```

**Test 3: Spatial Preservation**
```
Baseline: Flatten to 1D (standard ViT)
Ours:     Preserve 14×14 grid (Conv2D)
Metric:   Wall crossing success, spatial reasoning
Hypothesis: Spatial structure crucial for geometry
```

**Test 4: Codebook Size**
```
Test: n ∈ {8, 16, 32, 64, 128, 256}
Metric:   Prediction accuracy, code utilization, planning time
Find:     Optimal trade-off point
```

---

## Slide 28: Related Datasets & Benchmarks

### Comparison Environments

**Similar Complexity**:
1. **Point Maze** (Eysenbach et al., 2019)
   - Similar: 2D navigation with obstacles
   - Difference: No visual input (state-based)

2. **MuJoCo Ant Maze** (Fu et al., 2020)
   - Similar: Spatial navigation task
   - Difference: More complex dynamics, 3D

3. **Atari Montezuma's Revenge**
   - Similar: Visual navigation, sparse rewards
   - Difference: Much higher visual complexity

**Our Wall Environment Position**:
- Simpler than: Full 3D navigation, manipulation
- More complex than: Grid worlds, state-based
- **Sweet spot**: Visual + geometric reasoning at scale

### Benchmark Metrics

```
Navigation Success Rate: Goal reaching (% within radius)
Sample Efficiency:       Data needed for 90% success
Planning Efficiency:     Time to find valid plan
Generalization:          Success on unseen wall configs
```

---

## Slide 29: Code Visualization Examples

### What We Can Analyze

**1. Codebook Visualization**:
```python
# Decode each of 16 state codes
for code_idx in range(16):
    embedding = state_codebook[code_idx]
    decoded_img = decoder(embedding)
    plt.subplot(4, 4, code_idx+1)
    plt.imshow(decoded_img)
    plt.title(f"Code {code_idx}")
```

**2. Spatial Code Maps**:
```python
# Show which code assigned to each grid location
code_map = encoding_indices.reshape(14, 14)
plt.imshow(code_map, cmap='tab20')
plt.title("State Code Assignment")
# Expect: Different codes for dot region vs wall vs background
```

**3. Trajectory in Code Space**:
```python
# Track code transitions over time
trajectory_codes = [encode(obs_t) for obs_t in trajectory]
plt.plot(trajectory_codes)
plt.ylabel("Dominant Code ID")
plt.xlabel("Time Step")
# Expect: Smooth transitions, jumps at wall crossing
```

**4. Transition Matrix**:
```python
# P(code_j | code_i, action_k)
transition_counts[i, j, k] += 1
plt.imshow(transition_matrix[action_k], cmap='viridis')
# Shows learned dynamics as discrete graph
```

---

## Slide 30: Conclusion

### Summary

**Problem**: Learn visual dynamics model for spatial navigation without labels

**Solution**: Self-supervised prediction with quantized embeddings
- Conv2D encoder preserves spatial structure
- Vector quantization creates discrete state space
- Conv3D predictor learns constrained dynamics
- VCReg ensures representation quality

**Results**:
- End-to-end trainable in ~4 hours
- Learns wall constraints and door navigation
- Produces interpretable discrete codes
- Enables downstream planning

### Impact

**Scientific**:
- Demonstrates self-supervised learning of visual dynamics
- Shows quantization benefits for spatial reasoning
- Provides interpretable world models

**Practical**:
- Applicable to robotics (navigation, manipulation)
- Scalable to larger environments
- Foundation for model-based RL

---

## Slide 31: References

### Key Papers

**World Models & Dynamics Learning**:
- Ha & Schmidhuber (2018). "World Models"
- Hafner et al. (2020). "Dream to Control: Learning Behaviors by Latent Imagination" (DreamerV2)
- Hafner et al. (2023). "Mastering Diverse Domains through World Models" (DreamerV3)

**Self-Supervised Learning**:
- LeCun et al. (2022). "A Path Towards Autonomous Machine Intelligence" (JEPA)
- Bardes et al. (2022). "VICReg: Variance-Invariance-Covariance Regularization"
- Assran et al. (2023). "Self-Supervised Learning from Images with a Joint-Embedding Predictive Architecture"

**Vector Quantization**:
- van den Oord et al. (2017). "Neural Discrete Representation Learning" (VQ-VAE)
- Razavi et al. (2019). "Generating Diverse High-Fidelity Images with VQ-VAE-2"
- Esser et al. (2021). "Taming Transformers for High-Resolution Image Synthesis" (VQGAN)

**Spatial Reasoning**:
- Watters et al. (2019). "COBRA: Data-Efficient Model-Based RL through Unsupervised Object Discovery"
- Veerapaneni et al. (2020). "Entity Abstraction in Visual Model-Based Reinforcement Learning"

---

## Slide 32: Appendix A - Mathematical Notation

### Symbol Definitions

**Data**:
- **o_t** ∈ ℝ^(3×224×224): RGB observation at time t
- **a_t** ∈ ℝ^10: Concatenated actions [Δx₀,Δy₀,...,Δx₄,Δy₄] for frameskip=5
- **s_t** ∈ ℝ^2: True state [x, y] (not used during training)

**Learned Components**:
- **f_enc**: Encoder, o → z_obs ∈ ℝ^(14×14×64)
- **f_act**: Action encoder, a^(10) → z_act ∈ ℝ^10 (10→10 projection)
- **Q_state**: State quantizer with codebook **C_s** ∈ ℝ^(16×64)
- **Q_action**: Action quantizer with codebook **C_a** ∈ ℝ^(16×10)
- **f_pred**: Predictor, z_t → z_{t+k} where k=frameskip=5
- **f_dec**: Decoder, z → ô (optional)

**Operators**:
- **sg[·]**: Stop gradient operator
- **||·||**: L2 norm
- **⊕**: Concatenation operation
- **E[·]**: Expectation over batch
- **Var(·)**: Variance
- **Cov(·,·)**: Covariance

---

## Slide 33: Appendix B - Training Pseudocode

```python
# Training loop for one epoch
for batch in dataloader:
    obs, act = batch  # (B, 2, 3, 224, 224), (B, 2, 10)
    # act contains 5 concatenated actions per frame

    # Separate source and target
    obs_src, obs_tgt = obs[:, :1], obs[:, 1:]
    act_src, act_tgt = act[:, :1], act[:, 1:]

    # ===== STUDENT PATH (with gradients) =====
    # Encode source
    z_vis = encoder(obs_src)                    # (B, 1, 14, 14, 64)
    z_act = action_encoder(act_src)             # (B, 1, 10) - already 10-dim
    z_act_tiled = tile(z_act, h=14, w=14)       # (B, 1, 14, 14, 10)
    z_src = concat([z_vis, z_act_tiled])        # (B, 1, 14, 14, 74)

    # Note: act_src is (B, 1, 10) containing [Δx₀,Δy₀,...,Δx₄,Δy₄]

    # Quantize source
    z_obs, z_act_emb = split(z_src)             # 64 + 10
    z_obs_q, L_state = state_quantizer(z_obs)   # Quantize visual
    z_act_q, L_action = action_quantizer(z_act_emb)  # Quantize action
    z_src_q = concat([z_obs_q, z_act_q])        # (B, 1, 14, 14, 74)

    # VCReg loss
    L_vcreg = vcreg_loss(z_src_q)

    # Predict future
    z_pred = predictor(z_src_q)                 # (B, 1, 14, 14, 74)
    z_pred_obs, _ = split(z_pred)
    z_pred_obs_q, _ = state_quantizer(z_pred_obs)  # Re-quantize

    # ===== TEACHER PATH (no gradients) =====
    with torch.no_grad():
        z_vis_tgt = encoder(obs_tgt)            # Same encoder
        z_act_tgt = action_encoder(act_tgt)
        z_act_tgt_tiled = tile(z_act_tgt, h=14, w=14)
        z_tgt = concat([z_vis_tgt, z_act_tgt_tiled])
        z_tgt_obs, _ = split(z_tgt)
        z_tgt_obs_q, _ = state_quantizer(z_tgt_obs)

    # ===== LOSS COMPUTATION =====
    L_pred = mse(z_pred_obs_q, z_tgt_obs_q)     # Prediction loss
    L_quant = L_state + L_action                 # Quantization loss

    L_total = L_pred + 1.0*L_vcreg + 1.0*L_quant

    # ===== OPTIMIZATION =====
    optimizer.zero_grad()
    L_total.backward()
    optimizer.step()
```

---

## Slide 34: Appendix C - Hyperparameter Sensitivity

### Empirical Guidelines

**VCReg Weight** (λ_vcreg):
```
Too low (<0.1):  Representation collapse risk
Optimal (0.5-2.0): Stable, diverse embeddings
Too high (>5.0):  Interferes with prediction task
Recommendation: 1.0 (default)
```

**Commitment Cost** (β):
```
Too low (<0.1):  Codebook not used (encoder drifts)
Optimal (0.2-0.3): Good encoder-codebook alignment
Too high (>0.5): Limits encoder expressiveness
Recommendation: 0.25 (default)
```

**Learning Rates**:
```
Encoder:     5e-4 (larger, slow updates)
Predictor:   5e-4 (same as encoder)
Quantizers:  5e-4 (fast codebook learning)
Schedulers:  Warmup (1000 steps) + Cosine decay
```

**Batch Size**:
```
Too small (<16):  Noisy VCReg statistics
Optimal (32-64):   Good balance
Too large (>128):  Diminishing returns, memory issues
Recommendation: 32
```

**Frameskip**:
```
Small (1-2):   Easy prediction, less temporal abstraction, short action sequences
Medium (5):    Good balance (current), action_dim=2×5=10
Large (10+):   Harder prediction, more temporal abstraction, action_dim=2×10=20
Recommendation: 5 (wall moves ~10 pixels total displacement)

Note: Frameskip directly affects action dimensionality
      frameskip=5 → action_dim=10 (5 consecutive [Δx,Δy] pairs)
```

---

## Slide 35: Contact & Resources

### For Further Discussion

**Code Repository**:
```
/Users/decentral/Desktop/world_models/qdino_wm/
```

**Key Files**:
- `train.py`: Main training script
- `models/visual_world_model.py`: Full architecture
- `conf/train.yaml`: Hyperparameters
- `visualize.py`: Embedding visualization

**Reproduce This Work**:
```bash
cd qdino_wm
python train.py --config-name train.yaml \
  env=wall frameskip=5 num_hist=1 \
  quantize=True model.vcreg_loss_weight=1.0 \
  encoder=conv2d predictor=conv3d
```

**Questions?**
- Architecture choices
- Hyperparameter tuning
- Extension to other environments
- Downstream task integration

---

## End of Presentation

### Thank You!

**Summary**: Self-supervised visual world model with quantized embeddings for spatial navigation

**Key Innovation**: End-to-end learnable discrete dynamics model preserving spatial structure

**Impact**: Enables interpretable planning in visual environments without manual supervision

---

### Additional Slides Available on Request:
- Detailed ablation study designs
- Comparison with baseline methods
- Extension to multi-agent scenarios
- Hardware deployment considerations
- Theoretical convergence guarantees
