# Training Guide

This document provides comprehensive information on training models using the BlueDisc framework to overcome amplitude suppression in seismic phase picking.

## Overview

The BlueDisc framework addresses a fundamental challenge: **amplitude suppression**, where S-phase predictions remain trapped at ~0.5 amplitude despite being temporally accurate. The training script (`01_training.py`) implements three training modes to diagnose and solve this problem.

## The Amplitude Suppression Problem

**Symptoms:**
- S-phase peaks suppressed at ~0.5 amplitude (below 0.7 detection threshold)
- Predictions systematically shifted toward high-amplitude boundaries
- Temporally accurate but ineffective detections

**Root Causes:**
1. **Temporal Uncertainty**: S-wave onset separation from high-amplitude boundaries varies with epicentral distance
2. **CNN Anchoring**: Models anchor predictions to sharp amplitude changes, not subtle onsets
3. **Geometric Trap**: Pointwise BCE loss lacks lateral gradients to correct temporal misalignment

## Training Modes

### Mode 1: Hybrid Loss (Recommended) - Shape-then-Align Framework

Combines adversarial shape learning with temporal anchoring to break amplitude suppression.

**When to use**: Production models requiring optimal S-phase detection

```bash
python 01_training.py \
    --label D \
    --dataset INSTANCE \
    --g-lr 0.001 \
    --d-lr 0.001 \
    --data-weight 1.0 \
    --batch-size 100 \
    --max-steps 10000
```

**How it works:**
1. **GAN Loss**: Discriminator enforces Gaussian shape constraints
2. **BCE Loss**: Anchors predictions to correct temporal positions  
3. **Emergent Lateral Gradients**: Shape constraint channels pointwise errors into lateral movement
4. **Result**: 64% increase in effective S-phase detections

### Mode 2: Data-Only (Baseline) - Conventional Training

Traditional supervised learning with pointwise BCE loss only. Exhibits amplitude suppression.

**When to use**: Baseline comparison, P-wave focused tasks

```bash
python 01_training.py \
    --label D \
    --dataset INSTANCE \
    --g-lr 0.001 \
    --batch-size 100 \
    --max-steps 10000
```

**Characteristics:**
- No discriminator, no adversarial training
- Suffers from amplitude suppression on S-phases
- Predictions drift toward high-amplitude boundaries
- Dense horizontal band at ~0.5 amplitude in S-phase distributions

### Mode 3: Pure GAN (Experimental) - Adversarial Only

Trains with GAN loss only (λ=0), no BCE temporal anchoring.

**When to use**: Research into shape learning mechanisms, not for production

```bash
python 01_training.py \
    --label D \
    --dataset INSTANCE \
    --g-lr 0.001 \
    --d-lr 0.001 \
    --data-weight 0.0 \
    --batch-size 100 \
    --max-steps 10000
```

**Characteristics:**
- Shape learning without temporal constraints
- Prone to mode collapse (repetitive predictions)
- High temporal instability (unbounded drift)
- Demonstrates necessity of BCE as stability anchor

## Command-Line Arguments

### Required Arguments

| Argument | Type | Description |
|----------|------|-------------|
| `--label` | str | Label type to train on. Choices: `D` (detection), `N` (noise) |
| `--dataset` | str | SeisBench dataset name (e.g., `ETHZ`, `InstanceCount`) |

### Optional Arguments

| Argument | Type | Default | Description |
|----------|------|---------|-------------|
| `--g-lr` | float | 0.0001 | Generator learning rate |
| `--d-lr` | float | 0.0001 | Discriminator learning rate (GAN mode only) |
| `--data-weight` | float | None | Data loss weight. If specified, enables GAN mode |
| `--sample-size` | int | 1 | Number of dev samples for tracking during training |
| `--batch-size` | int | 100 | Training batch size |
| `--max-steps` | int | 10000 | Maximum training steps |
| `--device` | str | auto | Device: `cpu`, `cuda`, or `auto` |

## Training Examples

### Example 1: P-wave Detection with GAN

```bash
python 01_training.py \
    --label D \
    --dataset ETHZ \
    --g-lr 0.0001 \
    --d-lr 0.0001 \
    --data-weight 1.0 \
    --batch-size 128 \
    --max-steps 20000 \
    --device cuda
```

### Example 2: Noise Detection

```bash
python 01_training.py \
    --label N \
    --dataset ETHZ \
    --g-lr 0.0001 \
    --d-lr 0.0001 \
    --data-weight 1.0 \
    --batch-size 100 \
    --max-steps 10000
```

### Example 3: Data-Only Training (Baseline)

```bash
python 01_training.py \
    --label D \
    --dataset ETHZ \
    --g-lr 0.0001 \
    --batch-size 100 \
    --max-steps 10000
```

### Example 4: Different Data Weight

```bash
python 01_training.py \
    --label D \
    --dataset ETHZ \
    --g-lr 0.0001 \
    --d-lr 0.0001 \
    --data-weight 0.5 \
    --batch-size 100 \
    --max-steps 10000
```

## Experiment Naming Convention

The experiment name is automatically generated based on training configuration:

- Data-only: `PN_D` or `PN_N`
- GAN mode: `PN_D_GAN_Data1.0` or `PN_D_GAN_Data0.5`
- Zero data weight: `PN_D_GAN_Data0`

## Model Architecture

### Generator: PhaseNet (U-Net Based)

**Architecture**: U-Net style encoder-decoder with skip connections

**Specifications**:
- **Input**: 3-channel seismogram (3001 samples @ 100 Hz = 30.01 seconds)
  - Channel order: Z, N, E (vertical, north, east components)
- **Output**: 3-channel probability predictions (3001 samples)
  - P-wave probability
  - S-wave probability  
  - Noise probability
- **Activation**: Logits (raw scores), Sigmoid applied before discriminator
- **Learning Rate**: 1×10⁻³
- **Optimizer**: Adam with β₁=0.0 (no momentum for GAN stability)

**Role in Framework**:
- Learns to generate geometrically coherent Gaussian curves under discriminator guidance
- Balances shape constraints (GAN) with temporal accuracy (BCE)
- In pure data-only mode: exhibits amplitude suppression on S-phases

### Discriminator: BlueDisc (Lightweight CNN)

**Architecture**: 3-block CNN with batch normalization

**Specifications**:
- **Input**: 6-channel concatenated tensor
  - 3 channels: label (P, S, N probabilities)
  - 3 channels: waveform condition (Z, N, E components)
  - Total: 6 channels × 3001 samples
- **Block 1**: Conv1d(6→64, k=11, s=2) + BatchNorm + LeakyReLU
- **Block 2**: Conv1d(64→64, k=11, s=2) + BatchNorm + LeakyReLU  
- **Block 3**: Conv1d(64→128, k=5, s=2) + BatchNorm + LeakyReLU
- **Output**: Linear(47,616→1) → Logits (real/fake score)
- **Learning Rate**: 1×10⁻³
- **Optimizer**: Adam with β₁=0.0

**Role in Framework**:
- **Adaptive Shape Critic**: Learns authentic label geometry without a priori assumptions
- **Holistic Evaluator**: Assesses entire prediction shape, not individual points
- **Gradient Provider**: Supplies lateral corrective forces via shape constraints
- **Stability Anchor**: When balanced with BCE, prevents mode collapse

### Design Rationale

**Why PhaseNet?**
- Established baseline for phase picking
- U-Net architecture suitable for segmentation tasks
- Allows direct comparison with conventional methods

**Why Lightweight Discriminator?**
- Prevents overwhelming generator
- Focuses on geometric shape recognition
- Fast training with minimal computational overhead
- Sufficient capacity to learn Gaussian geometry

## Loss Functions and Mathematical Framework

### Overall Objective

The framework solves a minimax game:

```
min_G max_D L(θ, ψ) = L_cGAN(θ, ψ) + λ * L_BCE(θ)
```

Where:
- `θ`: Generator parameters (PhaseNet)
- `ψ`: Discriminator parameters (BlueDisc)
- `λ`: Hyperparameter balancing shape vs. temporal forces (default: 1.0, internally scaled by 4000)

### Discriminator Loss (Maximize Classification Accuracy)

```python
L_D(θ, ψ) = L_Fake + L_Real

L_Fake = BCELogits(D_ψ(G_θ(x), x), 0)  # Classify generated labels as fake
L_Real = BCELogits(D_ψ(y, x), 1)        # Classify real labels as real
```

**Physical Interpretation:**
- Learns to distinguish authentic Gaussian curves from malformed predictions
- Provides holistic shape critique across entire prediction
- Strong gradients early in training enforce geometric coherence

### Generator Loss (Fool Discriminator + Match Ground Truth)

```python
L_G(θ, ψ) = L_GAN + λ * L_Data

L_GAN = BCELogits(D_ψ(G_θ(x), x), 1)   # Adversarial: fool discriminator
L_Data = BCELogits(G_θ(x), y)           # Data fidelity: match ground truth
```

**Component Roles:**

1. **L_GAN (Shape Constraint)**:
   - Enforces geometrically coherent predictions
   - Creates rigid template that moves as single unit
   - Provides lateral corrective force when combined with L_Data

2. **L_Data (Temporal Anchoring)**:
   - Pointwise BCE between prediction and ground truth
   - Enforces instance-wise pairing (prevents mode collapse)
   - Provides vertical gradients at each timepoint

3. **Emergent Lateral Gradient**:
   - Shape rigidity resists vertical deformation from BCE
   - Collective pointwise errors channel into net lateral gradient  
   - Enables temporal alignment without losing shape

### The Role of λ (Hyperparameter)

λ controls the balance between two competing forces:

| λ Value | Dominant Force | Behavior | Use Case |
|---------|---------------|----------|----------|
| **1.0** (recommended) | Balanced | Stable convergence, effective detections | Production |
| **>5.0** | BCE dominates | Rightward bias, suppression returns | Too temporal-focused |
| **0.1-0.5** | GAN dominates | Temporal instability, good shape | Research |
| **0.0** | Pure GAN | Mode collapse, unbounded drift | Not recommended |

**Finding λ** (see visualization in docs):
- Too high: BCE overpowers lateral correction → suppression persists
- Too low: GAN causes temporal instability → mode collapse risk
- Optimal: Forces balance → elliptical convergence at label apex

### Training Dynamics

**Phase 1: Shape Learning (Steps 0-3000)**
- Steep GAN gradients enforce Gaussian geometry
- BCE prevents unbounded drift
- Predictions form proper curves but drift temporally

**Phase 2: Temporal Calibration (Steps 3000-7000)**
- Shape stabilizes as GAN gradients flatten
- BCE fine-tunes temporal position
- Lateral gradients guide predictions to correct location

**Phase 3: Refinement (Steps 7000-10000)**
- Both losses in equilibrium
- Small oscillations around optimal point
- Peak amplitudes exceed detection threshold

## Experiment Tracking with MLflow

All experiments are tracked using MLflow. The training script automatically logs:

### Metrics
- `g_loss`: Generator total loss
- `d_loss`: Discriminator total loss (GAN mode)
- `g_data_loss`: Generator data loss component
- `g_SGAN_loss`: Generator adversarial loss (GAN mode)
- `d_SGAN_loss`: Discriminator adversarial loss (GAN mode)
- `sample_XXX_loss`: Loss on tracked samples
- `sample_XXX_Dfake`: Discriminator score on fake samples
- `sample_XXX_Dreal`: Discriminator score on real samples

### Parameters
All command-line arguments are automatically logged

### Artifacts
- Model checkpoints (every 100 steps)
- Tracked sample predictions
- Training configuration

## Monitoring Training

### View MLflow UI

```bash
mlflow ui --host 127.0.0.1 --port 5000
```

Then open `http://127.0.0.1:5000` in your browser.

### Key Metrics to Monitor

1. **Generator Data Loss**: Should decrease steadily
2. **Discriminator Scores**: 
   - `D_real` should approach 1
   - `D_fake` should approach 0 initially, then increase as generator improves
3. **Sample Losses**: Track performance on validation samples

## Best Practices

### Learning Rates

- **Generator**: Start with 0.0001
- **Discriminator**: Start with 0.0001
- Both should be adjusted based on training stability

### Batch Size

- Larger batches (128-256) for stable gradient estimates
- Smaller batches (32-64) if memory is limited
- Default of 100 is a good balance

### Training Steps

- Start with 10,000 steps for quick experiments
- Use 20,000-50,000 steps for final models
- Monitor validation metrics to avoid overfitting

### Data Weight

- `1.0`: Equal importance to data and adversarial loss
- `> 1.0`: Prioritize data fidelity
- `< 1.0`: Prioritize adversarial training
- `0.0`: Pure adversarial training (not recommended)

## Reproducibility

The training script ensures reproducibility through:

1. **Fixed Random Seed**: Set to 42
2. **Deterministic Operations**: When possible on CUDA
3. **Worker Seed Management**: Consistent data loading across runs

To reproduce results, use the exact same:
- Command-line arguments
- Dataset version
- Hardware (CPU/GPU differences may cause minor variations)
- Software versions (PyTorch, CUDA)

## Troubleshooting

### Out of Memory

- Reduce `--batch-size`
- Use gradient accumulation (requires code modification)
- Use smaller model architecture

### Training Instability

- Reduce learning rates (especially discriminator)
- Increase data weight
- Use gradient clipping (requires code modification)

### Poor Performance

- Increase training steps
- Adjust data weight
- Check data preprocessing pipeline
- Verify label quality in dataset

## Advanced Options

For advanced users, additional customization can be done by modifying:

- `module/generator.py`: Generator architecture
- `module/discriminator.py`: Discriminator architecture
- `module/gan_model.py`: Loss functions and training logic
- `module/pipeline.py`: Data augmentation pipeline

## Next Steps

After training:
1. [Run inference](inference_evaluation.md#inference) on test data
2. [Evaluate performance](inference_evaluation.md#evaluation)
3. [Visualize results](visualization.md)

