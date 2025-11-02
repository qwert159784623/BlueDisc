# Methodology: Diagnosing and Breaking Amplitude Suppression

This document provides a comprehensive technical overview of the amplitude suppression problem and the shape-then-align solution implemented in the BlueDisc framework.

## Table of Contents

1. [The Amplitude Suppression Problem](#the-amplitude-suppression-problem)
2. [Geometric Diagnosis](#geometric-diagnosis)
3. [The Shape-then-Align Solution](#the-shape-then-align-solution)
4. [Implementation Details](#implementation-details)
5. [Experimental Validation](#experimental-validation)

---

## The Amplitude Suppression Problem

### Phenomenon Description

Amplitude suppression is a previously unexplained training bottleneck in deep learning-based seismic phase picking. It manifests as two distinct symptoms:

1. **Amplitude Suppression**: S-phase probability peaks consistently trapped at ~0.5 amplitude
2. **Temporal Misalignment**: Predictions systematically shifted toward high-amplitude boundaries

**Critical Impact**: Many predictions are temporally accurate (within ±0.1s) but fail to cross the 0.7 detection threshold, rendering them ineffective for seismological applications.

### Physical Origins

The problem emerges from the physical characteristics of S-wave arrivals:

#### 1. Temporal Uncertainty from Epicentral Distance

S-wave onsets are separated from their high-amplitude wave packets by a gap that:
- Varies systematically with epicentral distance (longer paths → larger gaps)
- Creates high temporal uncertainty in the target distribution
- Is compounded by inconsistent manual labeling of subtle onsets

**Mathematical representation**: Target distribution has large temporal variance σ_temporal

#### 2. CNN Anchoring Bias

Convolutional neural networks naturally anchor predictions to regions of sharp amplitude change:
- CNNs use correlation-based feature detection
- High-amplitude boundaries have strongest gradients
- Subtle S-wave onsets provide weaker signals
- Result: Systematic rightward bias in predictions

**Hypothesis**: Models develop shared P-S templates early in training, anchoring both to amplitude boundaries

#### 3. Loss Landscape Geometry

The pointwise Binary Cross-Entropy (BCE) loss creates a geometric trap:

```
L_BCE = -[y·log(ŷ) + (1-y)·log(1-ŷ)]  # Applied at each timepoint independently
```

**Key limitation**: Provides only vertical gradients (amplitude adjustment) at each point, with no lateral gradients (temporal shift) to resolve misalignment.

### The Dynamic Suppression Process

Analysis of training histories reveals suppression as a dynamic process, not a static outcome:

**Steps 0-2000**: 
- Both P and S phases form half-Gaussian curves with plateaus
- Predictions anchored at amplitude boundaries
- Initial template development

**Steps 2000-5000**:
- Model begins differentiating P and S features  
- **Critical divergence**: P-phase peaks grow in amplitude, S-phase development arrested
- S-phase trapped as optimizer repeatedly enters/exits label core region

**Steps 5000-10000**:
- P-phase successfully converges (high amplitude, correct position)
- S-phase oscillates around 0.5 amplitude, never escaping
- Persistent rightward shift toward amplitude boundary

---

## Geometric Diagnosis

### Loss Landscape Analysis

Theoretical visualization of the pointwise BCE loss landscape reveals the geometric origin of the trap.

#### Standard Pointwise BCE (Data-Only Training)

**Characteristics**:
- Low-loss region at label peak (target location)
- **Critical flaw**: Additional low-loss regions on both sides when amplitude ≈ 0
- Non-global minimum structure
- Vertical gradients only (suppress amplitude when temporally misaligned)
- No lateral guidance to correct temporal offset

**Consequence**: When prediction is displaced from label center, gradient directions:
1. Suppress amplitude (vertical force)
2. Fail to shift prediction laterally (no horizontal force)
3. Trap prediction in suboptimal local minimum

#### Holistic Gaussian Constraint (Shape-then-Align)

**Characteristics**:
- Constrains prediction to perfect Gaussian template
- Computes holistic BCE loss as template moves across amplitude-time plane
- **Single global minimum** at correct time and full amplitude
- Strong lateral gradients guide template to correct position

**Validation**: Numerical simulation confirms predictions maintain:
- Substantially higher peak amplitudes (>0.5)
- Anchoring at center of temporally uncertain targets
- No drift toward amplitude boundaries

### Numerical Simulation

To isolate core mechanisms, we removed the neural network entirely, leaving only trainable prediction values optimized by gradient descent.

**Experimental Design**:
- **Columns**: Temporal uncertainty (σ = 0.1, 0.2, 0.3, 0.5)
- **Rows**: Systematic bias (skewness = 0 vs. -10)
- **Optimizers**: Pointwise BCE vs. Gaussian-constrained

**Key Findings**:

1. **Temporal uncertainty controls amplitude suppression**:
   - Wider distributions → lower final amplitudes
   - Optimizer repeatedly enters/exits label core
   - Prevents consistent amplitude growth

2. **Systematic bias causes temporal offset only when combined with uncertainty**:
   - Low uncertainty: High amplitude despite bias
   - High uncertainty + bias: Both suppression and offset
   - Reproduces observed symptoms (bottom-right condition)

3. **Gaussian constraint provides robustness**:
   - Maintains amplitude >0.5 across all conditions
   - Anchors at distribution center despite uncertainty
   - Prevents drift under systematic bias

---

## The Shape-then-Align Solution

### Conceptual Framework

The key insight: **Shape must stabilize before temporal alignment can succeed**.

**Rationale**:
1. Large loss fluctuations from temporal uncertainty prevent fine-grained learning
2. Pointwise optimization with unstable targets leads to amplitude suppression
3. Holistic shape constraints create stable training environment
4. Temporal alignment succeeds only after geometric stabilization

**Strategy**: Decouple optimization into two sequential objectives:
1. **Shape Learning**: Enforce geometric coherence (Gaussian curves)
2. **Temporal Alignment**: Fine-tune position with stable templates

### Mathematical Formulation

We implement this through a conditional GAN (cGAN) framework with hybrid loss:

```
min_G max_D L(θ, ψ) = L_cGAN(θ, ψ) + λ · L_BCE(θ)
```

Where:
- `θ`: Generator parameters (PhaseNet)
- `ψ`: Discriminator parameters (BlueDisc)
- `λ`: Balance parameter (shape freedom vs. temporal anchoring)

#### Conditional Adversarial Loss

Based on f-divergence framework:

```
L_cGAN(θ, ψ) = E_{x~pD}[f(D_ψ(G_θ(x), x))] + E_{(x,y)~pD}[f(-D_ψ(y, x))]
```

For standard GAN: `f(t) = -log(1 + e^(-t))` (BCEWithLogitsLoss)

#### Discriminator Objective (Maximize Classification)

```
L_D(θ, ψ) = L_Fake + L_Real

L_Fake = BCELogits(D_ψ(G_θ(x), x), 0)  # Generated labels → 0
L_Real = BCELogits(D_ψ(y, x), 1)        # Real labels → 1
```

**Role**: Learn to distinguish geometrically valid from invalid predictions

#### Generator Objective (Fool Discriminator + Match Data)

```
L_G(θ, ψ) = L_GAN + λ · L_Data

L_GAN = BCELogits(D_ψ(G_θ(x), x), 1)   # Fool discriminator
L_Data = BCELogits(G_θ(x), y)           # Match ground truth
```

**Dual Role**:
1. **L_GAN**: Enforces shape constraints through adversarial pressure
2. **L_Data**: Anchors predictions temporally via instance-wise pairing

### Emergent Lateral Gradients

The hybrid loss creates an emergent phenomenon absent in pointwise optimization:

**Mechanism**:
1. **BCE Loss**: Applies vertical gradients at each timepoint
2. **Shape Constraint**: Discriminator resists vertical deformation
3. **Rigidity**: Prediction moves as single unit, preserving peak
4. **Channeling**: Collective pointwise errors → net lateral gradient
5. **Alignment**: Entire rigid prediction shifts horizontally

**Mathematical Intuition**:
- Pointwise BCE: ∇L_BCE ≈ [∂L/∂ŷ₁, ∂L/∂ŷ₂, ..., ∂L/∂ŷₙ] (vertical only)
- Shape constraint: Couples all timepoints via D's holistic evaluation
- Result: ∇L_total has lateral component enabling temporal shift

### Training Dynamics

The framework exhibits a predictable three-phase training sequence:

#### Phase 1: Shape Learning (Steps 0-3000)

**Dominant Force**: GAN loss (steep discriminator gradients)

**Behavior**:
- Predictions form proper Gaussian curves
- High temporal variability (drifting in time)
- Shape enforced but position unstable
- BCE prevents unbounded drift

**Interpretation**: Shape template established as foundation

#### Phase 2: Temporal Convergence (Steps 3000-7000)

**Dominant Force**: Transition (GAN gradients flatten, BCE strengthens)

**Behavior**:
- Shape stabilizes (discriminator satisfied)
- Lateral gradients guide temporal alignment
- Predictions drift toward correct location
- Amplitude continues growing

**Interpretation**: With stable shape, temporal fine-tuning succeeds

#### Phase 3: Refinement (Steps 7000-10000)

**Dominant Force**: Equilibrium (both losses balanced)

**Behavior**:
- Small oscillations around optimal point
- Peak amplitudes exceed detection threshold
- Both shape and position converged
- Final performance plateau

**Interpretation**: Successful convergence at label apex

---

## Implementation Details

### Model Architecture

#### Generator: PhaseNet (Encoder-Decoder)

**Base Architecture**: U-Net with skip connections

**Specifications**:
- **Input**: 3 channels × 3001 samples (Z, N, E components @ 100 Hz)
- **Output**: 3 channels × 3001 samples (P, S, N probabilities)
- **Encoder**: 5 convolutional blocks with downsampling
- **Decoder**: 5 transposed convolutional blocks with upsampling
- **Skip Connections**: Concatenate encoder features to decoder
- **Final Activation**: Logits (Sigmoid applied before discriminator)

**Training Configuration**:
- Optimizer: Adam(β₁=0.0, β₂=0.9, lr=1×10⁻³)
- β₁=0.0: Removes momentum for GAN stability
- Gradient clipping: None (not required with hybrid loss)

#### Discriminator: BlueDisc (Lightweight CNN)

**Design Philosophy**: Adaptive shape critic with minimal parameters

**Architecture**:
```
Input: Concatenate[label, waveform] → 6 channels × 3001 samples

Block 1: Conv1d(in=6, out=64, kernel=11, stride=2)
         → BatchNorm1d(64)
         → LeakyReLU(0.2)
         → Output: 64 × 1496

Block 2: Conv1d(in=64, out=64, kernel=11, stride=2)
         → BatchNorm1d(64)
         → LeakyReLU(0.2)
         → Output: 64 × 743

Block 3: Conv1d(in=64, out=128, kernel=5, stride=2)
         → BatchNorm1d(128)
         → LeakyReLU(0.2)
         → Output: 128 × 370

Flatten: 128 × 370 → 47,360
Output:  Linear(47,360 → 1) → Logits
```

**Training Configuration**:
- Optimizer: Adam(β₁=0.0, β₂=0.9, lr=1×10⁻³)
- Weight initialization: Default PyTorch (kaiming_uniform)
- Batch normalization: Stabilizes discriminator training

**Design Rationale**:
- **Lightweight**: ~3M parameters (vs. PhaseNet's ~3.5M)
- **Conditional**: Waveform context prevents pure memorization
- **Holistic**: Large receptive field evaluates entire prediction
- **Adaptive**: No hardcoded Gaussian templates

### Data Pipeline

#### Dataset: INSTANCE

**Characteristics**:
- **Source**: European seismology networks
- **Size**: 699,980 train / 115,027 dev / 244,242 test samples
- **Sampling**: 100 Hz (resampled from original)
- **Quality**: Deliberately includes noise, missing labels, picking errors
- **Rationale**: Realistic evaluation environment exposes training bottlenecks

#### Preprocessing Workflow

**1. Stratified Sampling**:
- 2/3 signal samples: 60s windows centered on P or S arrivals
- 1/3 noise samples: Random 30.01s windows from full recordings
- Balances phase detection and noise discrimination

**2. Window Extraction**:
- Random 30.01s (3001 samples @ 100 Hz) from signal windows
- Ensures phase arrival appears at variable positions

**3. Label Generation**:
```python
# Gaussian peaks at phase arrivals
P_label = gaussian(center=P_arrival, sigma=0.2s)
S_label = gaussian(center=S_arrival, sigma=0.2s)
N_label = 1 - (P_label + S_label)  # Noise = complement
```

**4. Normalization**:
- Demean: Remove DC offset
- Detrend: Remove linear trends
- Peak normalization: Scale to ±1 range

### Training Configuration

#### Hyperparameters

| Parameter | Value | Rationale |
|-----------|-------|-----------|
| **λ (data_weight)** | 1.0 → 4000 (internal) | Balances shape vs. temporal forces |
| **Batch size** | 100 | Stable gradients, efficient GPU usage |
| **Learning rate** | 1×10⁻³ | Both G and D |
| **β₁ (momentum)** | 0.0 | Removes momentum for GAN stability |
| **β₂ (RMSprop)** | 0.9 | Adaptive learning rates |
| **Max steps** | 10,000 | Sufficient for convergence |
| **Random seed** | 42 | Reproducibility |

#### Training Procedure

**Alternating Optimization** (each step):

1. **Train Discriminator**:
   ```python
   # Sample batch (x, y)
   fake = G(x).detach()  # Detach to prevent G gradients
   D_fake = D(fake, x)
   D_real = D(y, x)
   L_D = BCELogits(D_fake, 0) + BCELogits(D_real, 1)
   L_D.backward()
   optimizer_D.step()
   ```

2. **Train Generator**:
   ```python
   fake = G(x)
   D_fake = D(fake, x)
   L_GAN = BCELogits(D_fake, 1)
   L_Data = BCELogits(fake, y)
   L_G = L_GAN + λ * L_Data
   L_G.backward()
   optimizer_G.step()
   ```

**Normalization**: Total losses scaled to maintain effective learning rates independent of λ:
- Discriminator: L_D / 2
- Generator: L_G / (1 + λ)

#### Checkpoint Strategy

- **Frequency**: Every 100 steps
- **Storage**: MLflow artifacts directory
- **Contents**: 
  - Generator state_dict
  - Discriminator state_dict (if GAN mode)
  - Optimizer states
  - Training step counter
  - Random number generator state

---

## Experimental Validation

### Quantitative Results

#### Effective Detection Rate (Strict Criteria)

**Definition**: `(peak > 0.7) AND (|time_error| < 0.1s)`

| Method | S-phase Effective Detections | Improvement |
|--------|----------------------------|-------------|
| **Data-Only (Baseline)** | N_base | — |
| **BlueDisc (Hybrid)** | 1.64 × N_base | **+64%** |

**Significance**: Previously sub-threshold but temporally accurate predictions elevated into detectable range.

#### Suppression Band Analysis

**Definition**: `(0.4 < peak < 0.6) AND (|time_error| < 0.1s)`

| Method | Suppressed Predictions | Interpretation |
|--------|----------------------|----------------|
| **Data-Only** | 20-30% of S-phases | Trapped in geometric trap |
| **BlueDisc** | <5% of S-phases | Trap eliminated |

**Visualization**: Scatter plot shows dense horizontal band → elliptical convergence

### Qualitative Observations

#### Training History Analysis

**Representative Event** (Fig. 1 in paper):

**Data-Only Training**:
- Steps 0-2000: Half-Gaussian with plateau
- Steps 2000-5000: P-phase peak grows, S-phase arrested
- Steps 5000-10000: S-phase oscillates at ~0.5, never escapes

**Hybrid Training**:
- Steps 0-3000: Gaussian shapes form, temporal drift visible
- Steps 3000-7000: Shape stable, lateral alignment begins
- Steps 7000-10000: Converged at (time=0, amplitude=0.8-0.9)

#### Case Study: S-phase Rich Event

**Waveform Characteristics**:
- Clear S-wave onset at t=0
- High-amplitude wave packet begins at t+0.5s
- Large temporal gap tests suppression mechanism

**Predictions**:
- **Ground Truth**: Gaussian centered at onset (t=0, peak=1.0)
- **Data-Only**: Flattened peak (t+0.3s, peak=0.48) ✗
- **BlueDisc**: Sharp Gaussian (t+0.05s, peak=0.85) ✓

**Diagnostic Interpretation**:
- Data-only exhibits both symptoms (suppression + rightward shift)
- BlueDisc overcomes suppression (>0.7) and corrects alignment (<0.1s error)

### Ablation Studies

#### Effect of λ on Convergence

| λ Value | Shape Quality | Temporal Stability | Outcome |
|---------|--------------|-------------------|---------|
| **0.0** | ✓ Good | ✗ Unbounded drift | Mode collapse |
| **0.1-0.5** | ✓ Excellent | △ Moderate | Research use |
| **1.0** | ✓ Good | ✓ Stable | **Optimal** |
| **5.0+** | △ Degrades | ✓ Over-constrained | Returns to suppression |

**Visualization**: Convergence trajectories in peak-time scatter plot
- λ too low: Vertical convergence, horizontal drift
- λ optimal: Elliptical convergence to apex
- λ too high: Horizontal band (suppression returns)

#### Pure GAN Training (λ=0)

**Observations**:
- Shape learning successful (Gaussian curves)
- Temporal position completely unstable
- Mode collapse after ~4000 steps
- P-S separation becomes fixed (ignores input waveform)

**Conclusion**: BCE essential for instance-wise pairing, not just temporal anchoring

### Generalization Experiment

#### Tapered Boxcar Target

**Experimental Design**:
- Replace Gaussian labels with boxcar functions
- Start: P-wave arrival
- End: S + (S-P) × 1.7 (simulates variable signal duration)
- Edges: Tapered with Gaussian (σ=20 samples)

**Results**:
- Framework successfully learned non-Gaussian shape (validates adaptive critic)
- Required extremely low λ=10 (shape more complex than Gaussian)
- Training became fragile, collapsed after step 6000
- Demonstrates potential as general shape learner
- Reveals critical dependence on λ tuning

**Interpretation**: Discriminator autonomously discovers target geometry without a priori assumptions, but balance becomes more delicate for complex shapes.

---

## Summary

### Key Contributions

1. **Diagnosis**: Identified amplitude suppression as geometric trap in pointwise BCE loss landscape
2. **Mechanism**: Explained via temporal uncertainty + CNN bias + lack of lateral gradients
3. **Solution**: Shape-then-align framework via conditional GAN with hybrid loss
4. **Validation**: 64% increase in effective S-phase detections, suppression band eliminated

### Broader Implications

**For Seismology**:
- Improved S-phase detection without architecture changes
- Principled solution to previously unexplained bottleneck
- Applicable to other subtle-feature detection tasks

**For Machine Learning**:
- Demonstrates lateral gradient deficiency in pointwise losses
- Shape-then-align strategy for geometric fitting problems
- Hybrid loss as stability mechanism for GANs
- Diagnostic framework for analyzing training failures

### Future Directions

1. **Stable GAN Variants**: Spectral normalization, progressive growing for improved robustness
2. **Automatic λ Tuning**: Adaptive balance based on training dynamics
3. **Higher Dimensions**: Extension to 2D/3D segmentation tasks
4. **Theoretical Analysis**: Formal proof of lateral gradient emergence
5. **Broader Applications**: Image segmentation with small adjacent objects

---

## References

For detailed mathematical derivations, architectural specifications, and additional experiments, see:

- Paper: "Diagnosing and Breaking Amplitude Suppression in Seismic Phase Picking Through Adversarial Shape Learning"
- [Training Guide](training.md): Practical training instructions
- [Architecture Documentation](architecture.md): Module-level details
- [Inference & Evaluation](inference_evaluation.md): Performance assessment
- [Visualization Tools](visualization.md): Diagnostic plotting

