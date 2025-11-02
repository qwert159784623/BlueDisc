# Inference & Evaluation Guide

This document describes how to perform inference on trained models and evaluate their performance.

## Inference

The inference script (`02_inference.py`) generates predictions on any data split using a trained model.

### Basic Usage

```bash
python 02_inference.py \
    --run-id <your-run-id> \
    --dataset ETHZ \
    --data-split test
```

### Command-Line Arguments

| Argument | Type | Default | Description |
|----------|------|---------|-------------|
| `--run-id` | str | Required | MLflow run ID from training |
| `--dataset` | str | Required | Dataset name (must match training) |
| `--step` | int | None | Specific checkpoint step to use |
| `--epoch` | int | None | Specific epoch to use |
| `--device` | str | auto | Device: `cpu`, `cuda`, or `auto` |
| `--data-split` | str | test | Data split: `track`, `train`, `dev`, or `test` |

### Inference Examples

#### Example 1: Test Set Inference

```bash
python 02_inference.py \
    --run-id 3e85bee71bd648feacdcb5494ba66f04 \
    --dataset ETHZ \
    --data-split test
```

#### Example 2: Specific Checkpoint

```bash
python 02_inference.py \
    --run-id 3e85bee71bd648feacdcb5494ba66f04 \
    --dataset ETHZ \
    --data-split test \
    --step 5000
```

#### Example 3: Development Set

```bash
python 02_inference.py \
    --run-id 3e85bee71bd648feacdcb5494ba66f04 \
    --dataset ETHZ \
    --data-split dev
```

### Output

Predictions are saved as HDF5 files in the MLflow artifacts directory:

```
mlruns/{experiment_id}/{run_id}/artifacts/{data_split}/prediction/
├── prediction_0.h5
├── prediction_100.h5
├── prediction_200.h5
└── ...
```

Each HDF5 file contains:
- `predictions`: Model output predictions (N × 3 × 3000)
- `labels`: Ground truth labels (N × 3 × 3000)
- `trace_names`: Sample identifiers

### Finding Your Run ID

#### Method 1: MLflow UI

1. Start MLflow UI: `mlflow ui --host 127.0.0.1 --port 5000`
2. Open `http://127.0.0.1:5000` in browser
3. Click on your experiment
4. Find your run and copy the Run ID

#### Method 2: Terminal Output

The Run ID is printed during training:
```
Run ID: 3e85bee71bd648feacdcb5494ba66f04
```

#### Method 3: Directory Structure

```bash
ls mlruns/{experiment_id}/
```

## Evaluation

The evaluation script (`03_evaluation.py`) computes performance metrics on inference results.

### Basic Usage

```bash
python 03_evaluation.py \
    --run-id <your-run-id> \
    --data-split test
```

### Command-Line Arguments

| Argument | Type | Default | Description |
|----------|------|---------|-------------|
| `--run-id` | str | Required | MLflow run ID |
| `--max-step` | int | None | Maximum step to evaluate (evaluates all if not specified) |
| `--data-split` | str | test | Data split to evaluate: `track`, `train`, `dev`, or `test` |

### Evaluation Examples

#### Example 1: Full Test Set Evaluation

```bash
python 03_evaluation.py \
    --run-id 3e85bee71bd648feacdcb5494ba66f04 \
    --data-split test
```

#### Example 2: Evaluate Up to Specific Step

```bash
python 03_evaluation.py \
    --run-id 3e85bee71bd648feacdcb5494ba66f04 \
    --data-split test \
    --max-step 10000
```

#### Example 3: Development Set

```bash
python 03_evaluation.py \
    --run-id 3e85bee71bd648feacdcb5494ba66f04 \
    --data-split dev
```

### Evaluation Metrics

The evaluation script computes performance metrics that specifically assess the framework's success in overcoming amplitude suppression.

#### Dual-Threshold Evaluation Strategy

**1. Lenient Thresholds (Distribution Visualization)**
- **Confidence**: >0.1 (low threshold)
- **Temporal Tolerance**: ±3000 samples (30 seconds @ 100 Hz)
- **Purpose**: Visualize full spectrum of predictions including suppressed peaks
- **Use**: Statistical analysis (Fig. 3 in paper showing suppression band)

**2. Strict Thresholds (Effective Detection)**
- **Confidence**: >0.7 (high threshold)
- **Temporal Accuracy**: ±10 samples (0.1 seconds @ 100 Hz)  
- **Purpose**: Define truly useful detections for seismological applications
- **Use**: Final performance metric (64% improvement calculation)

#### Key Performance Indicators

**1. Effective Detection Rate**
```
Effective Detection = (peak > 0.7) AND (|time_error| < 0.1s)
```
- **Conventional Method**: Dense suppression band at ~0.5 amplitude
- **BlueDisc Framework**: Suppression band eliminated, 64% more effective detections

**2. Amplitude Distribution**
- **S-phase**: Critical indicator of suppression
  - Conventional: Horizontal band at 0.5 (many temporally accurate but sub-threshold)
  - BlueDisc: Elliptical distribution centered at label apex (peak ~0.8-0.9)
- **P-phase**: Both methods perform well (less affected by suppression)

**3. Temporal Accuracy**
- **Mean Absolute Error (MAE)**: Average timing offset from catalog picks
- **Standard Deviation**: Consistency of timing precision
- **Residual Distribution**: Histogram of time errors
  - Goal: Centered at zero with narrow spread

#### Diagnostic Metrics

**4. Suppression Band Analysis**
```python
suppressed_predictions = (0.4 < peak < 0.6) & (|time_error| < 0.1s)
```
- Identifies temporally accurate but amplitude-suppressed predictions
- High count indicates optimization trap
- BlueDisc should reduce this to near-zero

**5. Peak-Time Scatter Plot**
- **X-axis**: Temporal offset from label (seconds)
- **Y-axis**: Peak amplitude  
- **Visualization**:
  - Conventional: Dense horizontal band + rightward shift
  - BlueDisc: Elliptical cluster at (0, 0.8-0.9)

#### Timing Error Statistics

**Matched Predictions Analysis**:
- **MAE**: Mean absolute temporal error
- **Std Dev**: Timing consistency
- **Distribution**: Should be Gaussian centered at zero

**Systematic Bias Detection**:
- **Rightward bias**: Predictions shifted toward high-amplitude boundaries
- **Conventional method**: Systematic +0.2 to +0.5s shift on S-phases
- **BlueDisc**: Centered near zero offset

### Output

Results are saved in the MLflow artifacts directory:

```
mlruns/{experiment_id}/{run_id}/artifacts/{data_split}/
├── matching_results/
│   ├── matching_stats_{step}.json      # Per-step statistics
│   └── matching_details_{step}.json    # Detailed matching results
└── evaluation_summary.json              # Overall summary
```

### Evaluation Thresholds

The evaluation uses two sets of thresholds:

#### Matching Thresholds (Lenient)
```python
match_confidence = 0.1      # Low confidence threshold
match_tolerance = 3000      # 30 seconds tolerance (3000 samples @ 100Hz)
```

Purpose: Assess overall detection capability

#### Precise Thresholds (Strict)
```python
precise_confidence = 0.7    # High confidence threshold
precise_tolerance = 10      # 0.1 seconds tolerance (10 samples @ 100Hz)
```

Purpose: Assess precise picking accuracy

### Understanding Results

#### Diagnosing Amplitude Suppression

**Visual Indicators (from scatter plots)**:

1. **Suppression Band** (Bad)
   - Dense horizontal line at amplitude ~0.5
   - Many points at correct time (±0.1s) but wrong amplitude
   - Indicates geometric trap in optimization

2. **Elliptical Convergence** (Good)
   - Points cluster around (time=0, amplitude=0.8-0.9)
   - Narrow spread in both dimensions
   - Indicates successful shape-then-align optimization

3. **Rightward Shift** (Bad)
   - Systematic temporal offset (predictions too late)
   - Caused by CNN anchoring to high-amplitude boundaries
   - Common in S-phases with large onset-boundary gap

#### Training Success Indicators

**From Training Curves (MLflow)**:

1. **Generator Data Loss**: Should decrease steadily
   - Hybrid: Reaches ~0.15-0.20 (balances with GAN)
   - Data-only: May reach ~0.10 but suffers suppression
   - Lower ≠ better if amplitude suppression persists

2. **Discriminator Scores**:
   - **D_real**: Should start high (~0.7-0.9) and slightly decrease
   - **D_fake**: Should start low (~0.1-0.3) and gradually increase
   - **Convergence**: Both approach ~0.5 indicates generator learning

3. **Sample Tracking** (track set):
   - Watch individual waveform predictions evolve
   - Should see: shape forms → drifts → converges
   - Red flag: persistent oscillation or plateau at 0.5

#### Good Performance Indicators

**Quantitative Benchmarks**:

| Metric | Data-Only (Baseline) | BlueDisc (Target) |
|--------|---------------------|-------------------|
| **S-phase F1 (strict)** | ~0.55-0.60 | ~0.75-0.80 |
| **Effective S-detections** | Baseline | +64% improvement |
| **Suppressed peaks** | 20-30% of predictions | <5% |
| **Temporal MAE** | ~0.3-0.5s (biased) | ~0.1-0.2s (centered) |
| **Peak amplitude (S)** | ~0.5 (suppressed) | ~0.8-0.9 (healthy) |

**Qualitative Indicators**:

1. **Gaussian Shape**: Predictions should be smooth, symmetric curves
2. **Single Peak**: One clear maximum per phase (not plateaus or double peaks)
3. **Appropriate Width**: σ ≈ 0.2s (consistent with labels)
4. **Correct Positioning**: Aligned with subtle onset, not amplitude boundary

#### Common Failure Modes

**1. Amplitude Suppression (Data-Only Training)**
- **Symptom**: Peaks at ~0.5, horizontal band in scatter plot
- **Cause**: Pointwise BCE lacks lateral corrective forces
- **Solution**: Use hybrid loss with appropriate λ

**2. Mode Collapse (Pure GAN, λ too low)**
- **Symptom**: Repetitive predictions, ignore input waveform
- **Cause**: Discriminator only checks plausibility, not correctness
- **Solution**: Increase λ to strengthen BCE anchoring

**3. Temporal Instability (λ too low)**
- **Symptom**: High amplitude but poor timing, wide temporal spread
- **Cause**: GAN dominates, BCE can't anchor predictions
- **Solution**: Increase λ for stronger temporal constraint

**4. Persistent Suppression (λ too high)**
- **Symptom**: Similar to data-only, rightward bias
- **Cause**: BCE overpowers GAN, loses lateral gradients
- **Solution**: Decrease λ to allow shape learning

#### Debugging Checklist

If results are poor:

1. ✓ Check λ value (recommended: 1.0, internally scaled to 4000)
2. ✓ Verify training completed (10,000 steps minimum)
3. ✓ Examine discriminator scores (should converge)
4. ✓ Plot sample predictions (check for shape quality)
5. ✓ Analyze scatter plots (look for suppression band)
6. ✓ Compare with data-only baseline (quantify improvement)

- **Matching F1-score > 0.9**: Model detects most events
- **Precise F1-score > 0.7**: Model provides accurate timing
- **MAE < 50 samples**: Timing error < 0.5 seconds
- **Std < 100 samples**: Consistent predictions

#### Common Issues

1. **High Matching F1, Low Precise F1**
   - Model detects events but timing is imprecise
   - Solution: Adjust training or use more training data

2. **Low Matching F1**
   - Model misses many events
   - Solution: Increase training steps, adjust thresholds

3. **High False Positives**
   - Low precision
   - Solution: Train with more negative examples

4. **High False Negatives**
   - Low recall
   - Solution: Increase model sensitivity, check data balance

## Complete Workflow

### Step 1: Train Model

```bash
python 01_training.py \
    --label D \
    --dataset ETHZ \
    --g-lr 0.0001 \
    --d-lr 0.0001 \
    --data-weight 1.0 \
    --batch-size 100 \
    --max-steps 10000
```

Save the Run ID from output.

### Step 2: Run Inference

```bash
python 02_inference.py \
    --run-id <run-id> \
    --dataset ETHZ \
    --data-split test
```

### Step 3: Evaluate Results

```bash
python 03_evaluation.py \
    --run-id <run-id> \
    --data-split test
```

### Step 4: Visualize (Optional)

See [Visualization Guide](visualization.md) for plotting tools.

## Advanced Usage

### Batch Processing Multiple Runs

```bash
#!/bin/bash
RUNS=("run-id-1" "run-id-2" "run-id-3")

for run_id in "${RUNS[@]}"; do
    echo "Processing $run_id"
    python 02_inference.py --run-id $run_id --dataset ETHZ --data-split test
    python 03_evaluation.py --run-id $run_id --data-split test
done
```

### Custom Evaluation Thresholds

To use custom thresholds, modify `03_evaluation.py`:

```python
match_confidence = 0.2      # Your custom value
match_tolerance = 2000      # Your custom value
precise_confidence = 0.8    # Your custom value
precise_tolerance = 5       # Your custom value
```

### Analyzing Specific Checkpoints

```bash
# Inference at specific checkpoint
python 02_inference.py \
    --run-id <run-id> \
    --dataset ETHZ \
    --step 5000 \
    --data-split test

# Evaluate that checkpoint
python 03_evaluation.py \
    --run-id <run-id> \
    --max-step 5000 \
    --data-split test
```

## Performance Benchmarks

Expected performance on ETHZ dataset:

| Metric | Data-Only | GAN (data_weight=1.0) |
|--------|-----------|----------------------|
| Matching F1 | 0.92-0.95 | 0.93-0.96 |
| Precise F1 | 0.75-0.80 | 0.78-0.83 |
| MAE (samples) | 30-50 | 25-45 |

*Note: Actual results may vary depending on training configuration and dataset version.*

## Troubleshooting

### Run ID Not Found

```
Error: Run <run-id> not found
```

Solution:
- Verify Run ID in MLflow UI
- Check MLflow tracking URI matches training

### No Predictions Found

```
Error: No prediction files found
```

Solution:
- Ensure inference completed successfully
- Check data split name matches
- Verify checkpoint exists for specified step

### Memory Issues During Inference

Solution:
- Process in smaller batches (requires code modification)
- Use CPU instead of GPU for inference
- Close other applications

### Evaluation Errors

```
Error: Missing prediction file
```

Solution:
- Run inference before evaluation
- Check file permissions in mlruns directory

## Next Steps

After evaluation:
1. [Visualize results](visualization.md) with plotting tools
2. Compare multiple runs using comparison scripts
3. Analyze specific predictions for error patterns
4. Fine-tune training parameters based on results

