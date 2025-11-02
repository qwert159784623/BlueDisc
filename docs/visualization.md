# Visualization Tools

This document describes the visualization tools for diagnosing amplitude suppression and validating the shape-then-align framework.

## Overview

Visualization is critical for understanding the amplitude suppression phenomenon and validating that the BlueDisc framework successfully overcomes it. The repository provides specialized plotting tools for:

- **Diagnosing Suppression**: Identify suppression bands, temporal shifts, and geometric traps
- **Training Dynamics**: Observe shape-then-align process across training steps
- **Model Comparison**: Quantify improvements over baseline (data-only) methods
- **Statistical Analysis**: Visualize peak-time distributions revealing suppression patterns
- **Case Studies**: Examine individual waveforms showing success and failure modes

## Key Diagnostic Visualizations

### The Suppression Band (Critical Diagnostic)

**What to look for**: Dense horizontal cluster of predictions at amplitude ~0.5

- **In scatter plots**: Horizontal line in S-phase peak distributions
- **Physical meaning**: Temporally accurate but sub-threshold predictions
- **Conventional method**: 20-30% of S-phase predictions trapped here
- **BlueDisc framework**: Should be eliminated (near-zero suppressed predictions)

### The Elliptical Convergence (Success Indicator)

**What to look for**: Predictions clustering at label apex (time=0, amplitude=0.8-0.9)

- **Shape**: Elliptical distribution, narrow in both dimensions
- **Physical meaning**: Shape stabilized + temporal alignment achieved
- **Indicates**: Successful shape-then-align optimization

### Training Evolution (Shape-then-Align Process)

**What to observe across steps**:
1. **Early (0-3000 steps)**: Shape forms but drifts temporally
2. **Middle (3000-7000 steps)**: Shape stabilizes, temporal convergence begins
3. **Late (7000-10000 steps)**: Final refinement at correct location

## Available Plotting Scripts

## Available Plotting Scripts

### 1. `plot_compare_runs.py` - Visual Diagnosis of Amplitude Suppression

**Purpose**: Compare predictions from multiple models on identical waveforms to visually diagnose suppression and validate corrections.

#### Usage

```bash
python plot_compare_runs.py \
    --run-id <hybrid-loss-run> \
    --run-id2 <data-only-run> \
    --run-id3 <pure-gan-run> \
    --dataset INSTANCE \
    --step 10000 \
    --sample-id 0
```

#### Arguments

| Argument | Type | Description |
|----------|------|-------------|
| `--run-id` | str | First run ID (required) - typically hybrid loss |
| `--run-id2` | str | Second run ID (optional) - typically data-only baseline |
| `--run-id3` | str | Third run ID (optional) - typically pure GAN |
| `--dataset` | str | Dataset name (must match training) |
| `--step` | int | Training step checkpoint to visualize |
| `--sample-id` | int | Sample index in test/dev set |
| `--data-split` | str | Data split: `test`, `dev`, `track` |

#### Output Components

1. **Waveform Panel**:
   - 3-component seismogram (Z, N, E)
   - Ground truth arrival markers (vertical lines)
   - Color coding: P-wave (blue), S-wave (orange), Noise (green)

2. **Prediction Panels** (one per model):
   - Light colors: Ground truth labels (Gaussian curves)
   - Dark colors: Model predictions
   - Detection threshold line (horizontal at 0.7)

3. **Diagnostic Markers**:
   - Longest line: Reference label (ground truth)
   - Upper short line: Conventional prediction (data-only)
   - Lower short line: BlueDisc prediction (hybrid loss)

#### Interpreting Results

**Signs of Amplitude Suppression** (data-only model):
- ✗ S-phase peak below 0.7 threshold (suppressed amplitude)
- ✗ Peak shifted right toward high-amplitude boundary (temporal misalignment)
- ✗ Flattened-top appearance (geometric trap)
- ✗ Peak at ~0.5 amplitude even with clear S-wave signal

**Signs of Successful Correction** (BlueDisc model):
- ✓ S-phase peak >0.7 threshold (overcomes suppression)
- ✓ Peak aligned with subtle onset, not amplitude boundary
- ✓ Sharp Gaussian shape (proper geometry)
- ✓ Peak at ~0.8-0.9 amplitude

**Signs of Mode Collapse** (pure GAN):
- ✗ Repetitive predictions regardless of input
- ✗ High amplitude but wrong timing
- ✗ P and S predictions with fixed separation

#### Example Usage

```bash
# Critical diagnostic: Hybrid vs Data-only on S-phase rich event
python plot_compare_runs.py \
    --run-id <hybrid-loss-run-id> \
    --run-id2 <data-only-run-id> \
    --dataset INSTANCE \
    --step 10000 \
    --sample-id 42

# Three-way comparison: All training modes
python plot_compare_runs.py \
    --run-id <hybrid-run-id> \
    --run-id2 <data-only-run-id> \
    --run-id3 <pure-gan-run-id> \
    --dataset INSTANCE \
    --step 10000 \
    --sample-id 10
```

#### Recommended Samples to Examine

1. **S-phase rich events**: Clearly separated onset and amplitude boundary
2. **Near-field earthquakes**: Sharp signals, tests basic capability
3. **Regional events**: Longer propagation, high temporal uncertainty  
4. **Double earthquakes**: Tests robustness and mode collapse
5. **Pure noise**: Validates noise detection (should stay near zero)

### 2. `plot_compare_time.py` - Shape-then-Align Dynamics

**Purpose**: Visualize the shape-then-align training process by showing how predictions evolve across training steps.

#### Usage

```bash
python plot_compare_time.py \
    --run-id <run-id> \
    --dataset INSTANCE \
    --sample-id 0 \
    --start-step 0 \
    --end-step 10000 \
    --step-interval 500
```

#### Arguments

| Argument | Type | Description |
|----------|------|-------------|
| `--run-id` | str | MLflow run ID |
| `--dataset` | str | Dataset name |
| `--sample-id` | int | Sample to track (from track/dev set) |
| `--start-step` | int | Starting training step |
| `--end-step` | int | Ending training step |
| `--step-interval` | int | Step interval for snapshots (e.g., 500, 1000) |

#### Output

Multi-panel figure showing prediction snapshots at specified intervals, revealing the training dynamics.

#### Interpreting Training Dynamics

**Phase 1: Shape Formation (0-3000 steps)**
- Predictions form Gaussian curves (shape constraint active)
- High temporal variability (drifting in time)
- Amplitude grows but position unstable
- **Key observation**: Shape appears before position locks

**Phase 2: Temporal Convergence (3000-7000 steps)**
- Shape stabilizes (GAN gradients flatten)
- Predictions drift toward correct temporal location
- Lateral gradients guide alignment
- **Key observation**: Shape rigidity enables lateral movement

**Phase 3: Refinement (7000-10000 steps)**
- Small oscillations around optimal point
- Both shape and position stable
- Amplitude exceeds detection threshold
- **Key observation**: Successful convergence at label apex

**Compare with Data-Only Training**:
- Initial similarity: both form basic curves
- **Critical divergence** (~3000-5000 steps):
  - Data-only: Amplitude plateaus at ~0.5, shifts rightward
  - Hybrid: Amplitude continues growing, aligns correctly
- **Final state**:
  - Data-only: Suppressed at 0.5, misaligned
  - Hybrid: Peak at 0.8-0.9, properly aligned

#### Recommended Analysis

```bash
# Track critical suppression transition (data-only)
python plot_compare_time.py \
    --run-id <data-only-run-id> \
    --dataset INSTANCE \
    --sample-id 0 \
    --start-step 2000 \
    --end-step 6000 \
    --step-interval 500

# Observe shape-then-align process (hybrid loss)
python plot_compare_time.py \
    --run-id <hybrid-run-id> \
    --dataset INSTANCE \
    --sample-id 0 \
    --start-step 0 \
    --end-step 10000 \
    --step-interval 1000
```

#### What to Look For

**Healthy Training (Hybrid Loss)**:
- ✓ Smooth shape forms early
- ✓ Temporal drift visible but controlled
- ✓ Gradual convergence to correct position
- ✓ Final amplitude >0.7

**Suppression Developing (Data-Only)**:
- ✗ Initial growth followed by plateau
- ✗ Persistent oscillation around 0.5
- ✗ Rightward drift toward amplitude boundary
- ✗ Never escapes suppression trap

**Instability (Pure GAN, λ too low)**:
- ✗ Erratic jumping between positions
- ✗ Shape maintained but unbounded drift
- ✗ No convergence even at late steps

### 3. `plot_compare_peak.py` - Peak Detection Analysis

Analyze peak detection performance and timing errors.

#### Usage

```bash
python plot_compare_peak.py \
    --run-id <run-id> \
    --dataset ETHZ \
    --step 10000
```

#### Output

- Peak detection histograms
- Timing error distributions
- Precision/Recall curves
- Confidence threshold analysis

### 4. `plot_compare_phase.py` - Phase Comparison

Compare P-wave and S-wave detection performance.

#### Usage

```bash
python plot_compare_phase.py \
    --run-id <run-id> \
    --dataset ETHZ \
    --step 10000
```

#### Output

- Phase-specific performance metrics
- P-wave vs S-wave comparison
- Phase confusion analysis

### 5. `plot_compare_shape.py` - Prediction Shape Analysis

Analyze the shape characteristics of predicted phase arrivals.

#### Usage

```bash
python plot_compare_shape.py \
    --run-id <run-id> \
    --dataset ETHZ \
    --step 10000 \
    --sample-id 0
```

#### Output

- Prediction waveform shapes
- Width and amplitude analysis
- Shape similarity metrics

## Visualization Examples

### Example 1: Quick Comparison of Two Models

```bash
# Compare GAN model vs baseline
python plot_compare_runs.py \
    --run-id <gan-run-id> \
    --run-id2 <baseline-run-id> \
    --dataset ETHZ \
    --step 10000 \
    --sample-id 0
```

### Example 2: Training Progress Visualization

```bash
# Watch how predictions improve during training
python plot_compare_time.py \
    --run-id <run-id> \
    --dataset ETHZ \
    --sample-id 0 \
    --start-step 0 \
    --end-step 10000 \
    --step-interval 500
```

### Example 3: Performance Analysis

```bash
# Analyze peak detection performance
python plot_compare_peak.py \
    --run-id <run-id> \
    --dataset ETHZ \
    --step 10000
```

### Example 4: Multi-Model Comparison

```bash
# Compare three different training configurations
python plot_compare_runs.py \
    --run-id <data-weight-1.0> \
    --run-id2 <data-weight-0.5> \
    --run-id3 <data-only> \
    --dataset ETHZ \
    --step 10000 \
    --sample-id 10
```

## Customization

### Color Palette

The plotting scripts use a built-in color palette function:

```python
def _color_palette(color=1, shade=1):
    """
    color: 0=Blue, 1=Deep Orange, 2=Green, 3=Purple
    shade: 0=light, 1=light medium, 2=regular, 3=dark
    """
```

To change colors, modify this function in the plotting scripts.

### Plot Settings

Common customization options:

```python
dpi = 150                    # Resolution
figsize = (10, 8)           # Figure size in inches
linewidth = 0.5             # Line width
confidence = 0.3            # Confidence threshold for peaks
```

### Adding Custom Metrics

To add custom metrics to plots:

1. Modify the plotting script
2. Load additional data from HDF5 files or MLflow
3. Add new subplots or overlays

Example:

```python
# Add custom metric to plot_compare_runs.py
custom_metric = calculate_custom_metric(predictions, labels)
ax.text(0.95, 0.05, f'Custom: {custom_metric:.3f}', 
        transform=ax.transAxes, ha='right')
```

## Working with Plot Output

### Saving Figures

By default, figures are displayed interactively. To save:

```python
# Add to any plotting script
plt.savefig('output_figure.png', dpi=300, bbox_inches='tight')
plt.savefig('output_figure.pdf', bbox_inches='tight')  # Vector format
```

### Batch Plotting

Create plots for multiple samples:

```bash
#!/bin/bash
RUN_ID="your-run-id"
DATASET="ETHZ"
STEP=10000

for i in {0..10}; do
    python plot_compare_runs.py \
        --run-id $RUN_ID \
        --dataset $DATASET \
        --step $STEP \
        --sample-id $i
done
```

### Creating Publication Figures

For paper-quality figures:

```python
import matplotlib.pyplot as plt

# Set style
plt.style.use('seaborn-v0_8-paper')
plt.rcParams['font.family'] = 'serif'
plt.rcParams['font.size'] = 10
plt.rcParams['figure.dpi'] = 300

# Create figure
fig, ax = plt.subplots(figsize=(3.5, 2.5))  # Typical column width

# ... plotting code ...

# Save
plt.savefig('figure.pdf', bbox_inches='tight', pad_inches=0.05)
plt.savefig('figure.png', dpi=300, bbox_inches='tight')
```

## Understanding Plot Elements

### Waveform Plots

- **Black line**: Z component (vertical)
- **Gray lines**: N and E components (horizontal)
- **Vertical lines**: Phase picks (P-wave in blue, S-wave in orange)
- **Shaded regions**: Prediction confidence

### Prediction Plots

- **Solid lines**: Model predictions
- **Dashed lines**: Ground truth labels
- **Filled areas**: Confidence bands
- **Markers**: Detected peaks

### Color Coding

- **Blue**: P-wave related
- **Orange**: S-wave related  
- **Green**: Noise or additional metrics
- **Purple**: Comparison or alternative methods

## Performance Visualization Tips

### 1. Identifying Good Predictions

Good predictions show:
- Sharp peaks at correct locations
- Low noise between events
- Consistent confidence levels
- Minimal false positives

### 2. Common Failure Patterns

Watch for:
- **Smeared peaks**: Poor temporal resolution
- **Multiple peaks**: Uncertainty in arrival time
- **Missing peaks**: Missed detections
- **False peaks**: False positives

### 3. Comparing Models

When comparing models, focus on:
- Peak sharpness and accuracy
- Consistency across samples
- False positive/negative rates
- Timing precision

## Integration with MLflow

All plots can access MLflow metadata:

```python
import mlflow

client = mlflow.MlflowClient()
run = client.get_run(run_id)

# Get parameters
batch_size = run.data.params['batch_size']
data_weight = run.data.params.get('data_weight', 'N/A')

# Get metrics
final_loss = client.get_metric_history(run_id, 'g_loss')[-1].value

# Add to plot title
plt.title(f'Model: data_weight={data_weight}, loss={final_loss:.4f}')
```

## Troubleshooting

### Plot Not Showing

```python
# Add at end of script
plt.show()
```

### Memory Issues with Large Datasets

```python
# Plot subset of data
data = data[::10]  # Plot every 10th point
```

### Font Issues

```python
# Use default fonts
plt.rcParams['font.family'] = 'sans-serif'
```

### Export Issues

```bash
# Install additional backends
pip install pillow
```

## Advanced Visualization

### Interactive Plots

```python
%matplotlib notebook  # In Jupyter
# or
import plotly.graph_objects as go
```

### Animation

```python
from matplotlib.animation import FuncAnimation

# Create animation of training progress
fig, ax = plt.subplots()
animation = FuncAnimation(fig, update_frame, frames=num_steps, 
                         interval=100, repeat=True)
animation.save('training_progress.mp4')
```

### 3D Visualization

```python
from mpl_toolkits.mplot3d import Axes3D

fig = plt.figure()
ax = fig.add_subplot(111, projection='3d')
# Plot loss landscape or parameter space
```

## Next Steps

- Combine visualizations into comprehensive reports
- Create automated visualization pipelines
- Generate comparison tables from plot data
- Export metrics for statistical analysis

