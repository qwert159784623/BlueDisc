# Architecture Documentation

This document provides a comprehensive technical overview of the BlueDisc framework architecture, addressing the amplitude suppression problem in seismic phase picking through adversarial shape learning.

## Scientific Context

### The Amplitude Suppression Problem

Deep learning phase pickers consistently exhibit a paradoxical failure: high signal-to-noise S-wave predictions remain trapped at suppressed amplitudes (~0.5), failing to cross detection thresholds (>0.7) even when temporally accurate.

**Root Causes:**
1. **Temporal Uncertainty**: S-wave onsets exhibit high temporal uncertainty relative to high-amplitude boundaries
2. **CNN Bias**: Correlation-based CNNs anchor predictions to sharp amplitude changes rather than subtle onsets  
3. **Geometric Trap**: Pointwise BCE loss provides only vertical gradients, lacking lateral corrective forces

### Our Solution: Shape-then-Align Framework

The BlueDisc framework implements a conditional GAN (cGAN) that decouples shape learning from temporal alignment:

1. **Shape Stabilization**: Discriminator enforces geometric coherence (Gaussian curves)
2. **Temporal Alignment**: BCE loss anchors predictions to correct temporal positions
3. **Emergent Lateral Gradients**: Hybrid loss creates lateral corrective forces absent in pointwise optimization

## System Architecture

```
┌─────────────────────────────────────────────────────────────────┐
│                    BlueDisc cGAN Framework                       │
├─────────────────────────────────────────────────────────────────┤
│                                                                   │
│  ┌─────────────────┐                      ┌─────────────────┐   │
│  │   Generator     │◄──── GAN Loss ──────►│ Discriminator   │   │
│  │   (PhaseNet)    │                      │   (BlueDisc)    │   │
│  └────────┬────────┘                      └────────▲────────┘   │
│           │                                        │            │
│           ├────────── BCE Loss ────────────────────┘            │
│           │          (Data Fidelity)                            │
│           │                                                     │
│  ┌────────▼──────────────────────────────────────────────────┐ │
│  │              Hybrid Loss Function                          │ │
│  │    L(θ,ψ) = L_cGAN(θ,ψ) + λ * L_BCE(θ)                   │ │
│  │    • λ controls shape vs. temporal balance                │ │
│  │    • GAN: holistic shape constraints                      │ │
│  │    • BCE: instance-wise temporal anchoring                │ │
│  └───────────────────────────────────────────────────────────┘ │
│                                                                   │
│  ┌──────────────┐  ┌──────────────┐  ┌──────────────┐          │
│  │   Training   │  │  Inference   │  │  Evaluation  │          │
│  │ (01_*.py)    │  │  (02_*.py)   │  │  (03_*.py)   │          │
│  └──────────────┘  └──────────────┘  └──────────────┘          │
│                                                                   │
│  ┌─────────────────────────────────────────────────────────┐    │
│  │              MLflow Experiment Tracking                   │    │
│  │  • Metrics: G/D losses, sample predictions               │    │
│  │  • Artifacts: Checkpoints, predictions, visualizations   │    │
│  └─────────────────────────────────────────────────────────┘    │
│                                                                   │
└─────────────────────────────────────────────────────────────────┘
```

## Directory Structure

```
BlueDisc/
├── module/                      # Core modules (library code)
│   ├── __init__.py
│   ├── device_manager.py       # GPU/CPU device management
│   ├── discriminator.py        # Discriminator models
│   ├── evaluator.py            # Evaluation metrics
│   ├── gan_model.py            # GAN training logic
│   ├── generator.py            # Generator models
│   ├── labeler.py              # Label generation utilities
│   ├── logger.py               # MLflow logging wrapper
│   ├── pipeline.py             # Data preprocessing pipeline
│   └── random_seed.py          # Reproducibility utilities
│
├── 01_training.py              # Main training script
├── 02_inference.py             # Inference script
├── 03_evaluation.py            # Evaluation script
│
├── plot_compare_runs.py        # Compare multiple runs
├── plot_compare_time.py        # Temporal evolution plots
├── plot_compare_peak.py        # Peak detection analysis
├── plot_compare_phase.py       # Phase comparison plots
├── plot_compare_shape.py       # Prediction shape analysis
│
├── loss_landscape_analysis.py  # Loss landscape visualization
├── no_model_bce_test.py        # BCE loss testing
│
├── requirements.txt            # Python dependencies
├── README.md                   # Main documentation
│
├── docs/                       # Detailed documentation
│   ├── training.md
│   ├── inference_evaluation.md
│   ├── visualization.md
│   └── architecture.md
│
└── mlruns/                     # MLflow tracking directory
    ├── 0/                      # Default experiment
    ├── {experiment_id}/        # Experiment folders
    │   └── {run_id}/          # Individual run data
    │       ├── artifacts/      # Model checkpoints, predictions
    │       ├── metrics/        # Training metrics
    │       ├── params/         # Hyperparameters
    │       └── tags/           # Run metadata
    └── models/                 # Registered models
```

## Core Modules

### 1. `gan_model.py` - cGAN Training Logic

**Purpose**: Implements the conditional GAN minimax game between generator and discriminator.

**Key Classes**:
- `GANModel`: Orchestrates the hybrid loss training framework

**Key Loss Functions**:

**Discriminator Loss** (maximize classification accuracy):
```
L_D = L_Fake + L_Real
L_Fake = BCELogits(D(G(x), x), 0)  # Fake labels should be classified as 0
L_Real = BCELogits(D(y, x), 1)     # Real labels should be classified as 1
```

**Generator Loss** (fool discriminator + match ground truth):
```
L_G = L_GAN + λ * L_Data
L_GAN = BCELogits(D(G(x), x), 1)   # Generator wants discriminator to classify as real
L_Data = BCELogits(G(x), y)         # Generator must match ground truth
```

**Core Mechanisms**:
1. **Shape Enforcement**: Discriminator penalizes geometrically invalid predictions
2. **Temporal Anchoring**: BCE loss enforces instance-wise pairing (waveform ↔ label)
3. **Emergent Lateral Gradients**: Shape constraint + pointwise BCE = lateral corrective force
4. **Dynamic Equilibrium**: λ balances shape freedom vs. temporal stability

**Three Training Modes**:
- **Hybrid Loss (λ > 0)**: Recommended - combines GAN shape learning with BCE temporal anchoring
- **Data Only (λ → ∞)**: Baseline - conventional supervised training, suffers from amplitude suppression
- **Pure GAN (λ = 0)**: Experimental - shape learning without temporal constraints, prone to mode collapse

**Usage**:
```python
from module.gan_model import GANModel

# Recommended: Hybrid loss with λ=1.0 (internally scaled by 4000)
model = GANModel(
    gan_type="SGAN",
    generator=g_model,
    discriminator=d_model,
    g_data_weight=1.0,      # λ parameter
    gan_loss_weight=1.0,
    logger=logger
)
```

### 2. `generator.py` - Generator Models

**Purpose**: Implements generator architectures for seismic waveform generation.

**Key Classes**:
- `BaseGenerator`: Base class with common functionality
- `WrappedPhaseNet`: PhaseNet-based generator

**Features**:
- PhaseNet architecture integration
- Configurable phase ordering (P, S, N)
- Built-in optimizer and scheduler
- Batch loss computation

**Architecture**:
- **Input**: 3-channel seismogram (3000 samples)
- **Output**: 3-channel phase probabilities (P, S, N)
- **Structure**: U-Net style encoder-decoder

### 3. `discriminator.py` - BlueDisc Discriminator

**Purpose**: Implements the adaptive shape critic that enforces geometric coherence.

**Key Classes**:
- `Discriminator`: BlueDisc - lightweight CNN discriminator
- `DBuilder`: Factory for building discriminators

**Function**: 
BlueDisc distinguishes real labels (from dataset) from fake labels (generated by PhaseNet), conditioned on the corresponding seismic waveform. By learning to recognize authentic label geometry, it provides holistic shape constraints that guide the generator toward producing valid Gaussian curves.

**Design Philosophy**:
- **Adaptive Critic**: Autonomously learns target geometry without a priori assumptions
- **Lightweight Architecture**: Minimal parameters to avoid overpowering generator
- **Conditional Design**: Evaluates label authenticity in context of input waveform
- **Holistic Evaluation**: Assesses entire prediction shape, not individual points

**Architecture**:
```
Input: [label (3 channels) + waveform (3 channels)] = 6 channels
       (concatenated along channel dimension)

Block 1: Conv1d (6 → 64, kernel=11, stride=2) + BatchNorm + LeakyReLU
Block 2: Conv1d (64 → 64, kernel=11, stride=2) + BatchNorm + LeakyReLU  
Block 3: Conv1d (64 → 128, kernel=5, stride=2) + BatchNorm + LeakyReLU
Flatten: (128 × 372) → 47,616
Output:  Linear (47,616 → 1) → Logits (real/fake score)
```

**Training Characteristics**:
- **Learning Rate**: 1×10⁻³ (same as generator)
- **Optimizer**: Adam with β₁=0.0 (removes momentum for stability)
- **Activation**: LeakyReLU with negative slope 0.2
- **Output**: Raw logits (BCEWithLogitsLoss combines sigmoid + BCE)

**Role in Framework**:
1. **Shape Learning Phase**: Strong gradients enforce Gaussian geometry
2. **Convergence Phase**: Gradients weaken as generator improves, allowing BCE to fine-tune
3. **Stability Anchor**: When combined with BCE, prevents mode collapse and unbounded drift

### 4. `pipeline.py` - Data Preprocessing

**Purpose**: Builds data augmentation and preprocessing pipelines.

**Key Classes**:
- `AugmentationsBuilder`: Constructs preprocessing pipeline

**Features**:
- Automatic phase detection from dataset metadata
- Window extraction around phase arrivals
- Gaussian labeling for phase picks
- Detection labeling (tapered)
- Normalization (demean + peak normalization)

**Pipeline Stages**:
1. **Window Selection**: Around phases or random
2. **Random Windowing**: Extract 3001-sample windows
3. **Type Conversion**: Convert to float32
4. **Probabilistic Labeling**: Gaussian peaks for phases
5. **Detection Labeling**: Binary detection labels
6. **Normalization**: Demean and peak normalization
7. **Metadata**: Add trace names

### 5. `evaluator.py` - Evaluation Metrics

**Purpose**: Computes evaluation metrics for model predictions.

**Key Functions**:
- `get_picks()`: Extract phase picks from predictions
- `match_peaks_and_calculate_errors()`: Match predictions to ground truth

**Metrics**:
- Precision, Recall, F1-score
- Timing errors (MAE, std)
- Peak matching at different thresholds

### 6. `logger.py` - MLflow Integration

**Purpose**: Provides logging utilities for experiment tracking.

**Key Classes**:
- `MLFlowLogger`: Wrapper for MLflow logging

**Features**:
- Automatic metric logging
- Parameter tracking
- Artifact management
- Model checkpointing

### 7. `device_manager.py` - Device Management

**Purpose**: Handles CPU/GPU device allocation.

**Key Classes**:
- `DeviceManager`: Manages device selection and transfers

**Features**:
- Automatic device detection
- Manual device specification
- Model transfer utilities
- Multi-GPU support preparation

### 8. `random_seed.py` - Reproducibility

**Purpose**: Ensures reproducible training results.

**Key Classes**:
- `RandomSeedManager`: Manages random seeds across libraries

**Features**:
- Sets seeds for Python, NumPy, PyTorch
- Worker initialization for DataLoader
- Deterministic operations (when possible)

### 9. `labeler.py` - Label Generation

**Purpose**: Custom label generation for detection tasks.

**Key Classes**:
- `TaperedDetectionLabeller`: Generates tapered detection labels

**Features**:
- Gaussian-shaped detection labels
- Configurable sigma (width)
- Integration with SeisBench pipeline

## Main Scripts

### `01_training.py`

**Purpose**: Train GAN or data-only models.

**Workflow**:
1. Parse command-line arguments
2. Initialize device and random seeds
3. Load dataset from SeisBench
4. Build generator and discriminator
5. Create data loaders with augmentations
6. Initialize MLflow experiment
7. Train model with checkpointing
8. Log metrics and artifacts

**Key Features**:
- Flexible training modes
- Automatic experiment naming
- Periodic evaluation on dev set
- Model checkpointing every 100 steps

### `02_inference.py`

**Purpose**: Generate predictions using trained models.

**Workflow**:
1. Load trained model from MLflow
2. Load test dataset
3. Generate predictions batch-by-batch
4. Save predictions to HDF5 files

**Output Format** (HDF5):
```
prediction_{step}.h5
├── predictions: [N, 3, 3000]  # Model outputs
├── labels: [N, 3, 3000]       # Ground truth
└── trace_names: [N]            # Sample identifiers
```

### `03_evaluation.py`

**Purpose**: Evaluate model predictions and compute metrics.

**Workflow**:
1. Load predictions from inference
2. Extract phase picks using peak detection
3. Match predictions to ground truth
4. Compute metrics at different thresholds
5. Save detailed results

**Output**:
- Matching statistics (JSON)
- Detailed matching results per sample
- Evaluation summary

## Plotting Scripts

### Common Features

All plotting scripts share:
- MLflow integration for loading data
- Consistent color schemes
- Publication-quality output
- Flexible command-line interfaces

### Script Overview

| Script | Purpose | Key Visualizations |
|--------|---------|-------------------|
| `plot_compare_runs.py` | Compare models | Waveforms + multi-model predictions |
| `plot_compare_time.py` | Training evolution | Predictions over training steps |
| `plot_compare_peak.py` | Peak analysis | Timing errors, histograms |
| `plot_compare_phase.py` | Phase comparison | P-wave vs S-wave performance |
| `plot_compare_shape.py` | Shape analysis | Prediction waveform characteristics |

## Data Flow

### Training Data Flow

```
SeisBench Dataset
        ↓
Augmentation Pipeline
        ↓
DataLoader (batched)
        ↓
Generator (forward)
        ↓
Loss Computation ← Discriminator (if GAN mode)
        ↓
Backpropagation
        ↓
MLflow Logging
```

### Inference Data Flow

```
MLflow Artifacts (checkpoint)
        ↓
Load Model
        ↓
Test Dataset
        ↓
DataLoader
        ↓
Batch Prediction
        ↓
HDF5 Storage
        ↓
Evaluation Script
        ↓
Metrics & Visualizations
```

## Model Training Loop

```python
for epoch in range(max_epochs):
    for batch in train_loader:
        # Forward pass
        predictions = generator(batch['X'])
        
        if use_gan:
            # Train discriminator
            d_real = discriminator(batch['y'], batch['X'])
            d_fake = discriminator(predictions.detach(), batch['X'])
            d_loss = discriminator_loss(d_real, d_fake)
            d_loss.backward()
            d_optimizer.step()
            
            # Train generator (adversarial + data)
            g_fake = discriminator(predictions, batch['X'])
            g_loss = (data_weight * data_loss(predictions, batch['y']) + 
                     gan_weight * generator_loss(g_fake))
        else:
            # Data-only training
            g_loss = data_loss(predictions, batch['y'])
        
        g_loss.backward()
        g_optimizer.step()
        
        # Log metrics
        logger.log_metrics(step, g_loss, d_loss)
        
        # Save checkpoint
        if step % 100 == 0:
            save_checkpoint(generator, discriminator, step)
```

## Extension Points

### Adding New Generator

1. Create new class in `generator.py`:
```python
class MyGenerator(BaseGenerator):
    def __init__(self):
        super().__init__()
        # Your architecture
```

2. Add to `GBuilder.model_dict`

3. Use in training: `--g-model MyGenerator`

### Adding New Discriminator

1. Create new class in `discriminator.py`
2. Add to `DBuilder.model_dict`
3. Use in training: `--d-model MyDiscriminator`

### Adding New Augmentation

1. Modify `AugmentationsBuilder` in `pipeline.py`:
```python
self.augmentations.append(
    sbg.YourNewAugmentation(...)
)
```

### Adding New Metric

1. Add function to `evaluator.py`:
```python
def compute_new_metric(predictions, labels):
    # Your metric computation
    return metric_value
```

2. Call in `03_evaluation.py`

## Dependencies

### Core Dependencies

- **PyTorch**: Deep learning framework
- **SeisBench**: Seismic dataset and model library
- **MLflow**: Experiment tracking and model management

### Data Processing

- **NumPy**: Numerical operations
- **h5py**: HDF5 file I/O
- **pandas**: Metadata handling

### Visualization

- **Matplotlib**: Plotting
- **Seaborn**: Statistical visualizations
- **SciPy**: Signal processing (peak detection)

### Utilities

- **tqdm**: Progress bars
- **ObsPy**: Seismology utilities

## Best Practices

### Code Organization

- Keep modules focused and single-purpose
- Use builder patterns for model construction
- Centralize configuration in argument parsing
- Log extensively for debugging

### Experiment Management

- Use descriptive experiment names
- Log all hyperparameters
- Save model checkpoints frequently
- Track evaluation metrics

### Data Handling

- Use deterministic data loading
- Validate data shapes and types
- Handle edge cases (empty batches, etc.)
- Normalize consistently

### Performance

- Use DataLoader workers for faster loading
- Enable pin_memory for GPU training
- Use mixed precision training for large models
- Profile code to identify bottlenecks

## Troubleshooting

Common issues and solutions are documented in individual guides:
- [Training issues](training.md#troubleshooting)
- [Inference issues](inference_evaluation.md#troubleshooting)
- [Visualization issues](visualization.md#troubleshooting)

## Contributing

When extending this codebase:

1. Follow existing code style
2. Add docstrings to new functions
3. Update relevant documentation
4. Test with small dataset first
5. Log new metrics to MLflow
6. Create example scripts for new features

