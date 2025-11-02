# BlueDisc: Breaking Amplitude Suppression in Seismic Phase Picking

[![Python 3.8+](https://img.shields.io/badge/python-3.8+-blue.svg)](https://www.python.org/downloads/)
[![License](https://img.shields.io/badge/license-MIT-green.svg)](LICENSE)

This repository contains the implementation code for our paper: **"Diagnosing and Breaking Amplitude Suppression in Seismic Phase Picking Through Adversarial Shape Learning"**

## Overview

BlueDisc addresses a fundamental challenge in deep learning-based seismic phase picking: **amplitude suppression**. High signal-to-noise S-wave predictions consistently fail to cross detection thresholds, trapped at suppressed amplitudes (~0.5) even when temporally accurate. 

### The Problem: Amplitude Suppression

Conventional PhaseNet models trained with pointwise Binary Cross-Entropy (BCE) loss suffer from:
- **Amplitude Suppression**: S-phase peaks trapped at ~0.5 amplitude (below 0.7 detection threshold)
- **Temporal Misalignment**: Predictions systematically shifted toward high-amplitude boundaries
- **Geometric Trap**: Pointwise BCE loss lacks lateral corrective forces to resolve temporal misalignment

### Our Solution: Shape-then-Align Framework

We implement a conditional GAN (cGAN) framework that:
1. **Enforces Shape Constraints**: BlueDisc discriminator learns target geometry (Gaussian curves)
2. **Decouples Optimization**: Shape stabilizes first, then temporal alignment succeeds
3. **Generates Lateral Gradients**: Hybrid loss creates emergent lateral corrective forces

**Result**: 64% increase in effective S-phase detections (peak >0.7, time error <0.1s)

### Key Features

- **Adversarial Shape Learning**: cGAN framework with BlueDisc discriminator
- **Hybrid Loss Function**: Balances BCE (temporal anchoring) and GAN (shape constraints)
- **Multiple Training Modes**: GAN-based, data-only, and purely adversarial training
- **Dataset Flexibility**: Compatible with all SeisBench datasets (INSTANCE, ETHZ, etc.)
- **Comprehensive Tracking**: Integrated MLflow for experiment management
- **Full Pipeline**: Training → Inference → Evaluation → Visualization

## Quick Start

### Installation

```bash
pip install -r requirements.txt
```

### Training

**Recommended: Hybrid Loss (Shape-then-Align Framework)**

Train with both GAN and BCE loss for optimal S-phase detection:

```bash
python 01_training.py \
    --label D \
    --dataset InstanceCount \
    --g-lr 0.001 \
    --d-lr 0.001 \
    --data-weight 1.0 \
    --batch-size 100 \
    --max-steps 10000
```

**Baseline: Data Loss Only**

Train with conventional BCE loss only (no discriminator):

```bash
python 01_training.py \
    --label D \
    --dataset InstanceCount \
    --g-lr 0.001 \
    --batch-size 100 \
    --max-steps 10000
```

**Experimental: Purely Adversarial**

Train with GAN loss only (no BCE):

```bash
python 01_training.py \
    --label D \
    --dataset InstanceCount \
    --g-lr 0.001 \
    --d-lr 0.001 \
    --data-weight 0.0 \
    --batch-size 100 \
    --max-steps 10000
```

### Inference

Generate predictions on test set:

```bash
python 02_inference.py \
    --run-id <your-run-id> \
    --dataset INSTANCE \
    --data-split test
```

### Evaluation

Compute performance metrics:

```bash
python 03_evaluation.py \
    --run-id <your-run-id> \
    --data-split test
```

### Visualization

Compare predictions across different models:

```bash
python plot_compare_runs.py \
    --run-id <hybrid-loss-run-id> \
    --run-id2 <data-only-run-id> \
    --dataset INSTANCE \
    --step 10000 \
    --sample-id 0
```

## Documentation

Detailed documentation is available in the `docs/` directory:

- [Training Guide](docs/training.md) - Complete training parameters and examples
- [Inference & Evaluation](docs/inference_evaluation.md) - How to perform inference and evaluation
- [Visualization Tools](docs/visualization.md) - Plotting and analysis tools
- [Architecture](docs/architecture.md) - Project structure and module descriptions

## Repository Structure

```
BlueDisc/
├── 01_training.py              # Training script
├── 02_inference.py             # Inference script
├── 03_evaluation.py            # Evaluation script
├── plot_*.py                   # Visualization tools
├── module/                     # Core modules
│   ├── gan_model.py           # GAN model implementation
│   ├── generator.py           # Generator (PhaseNet-based)
│   ├── discriminator.py       # Discriminator (BlueDisc)
│   ├── pipeline.py            # Data preprocessing pipeline
│   ├── evaluator.py           # Evaluation metrics
│   └── ...
├── mlruns/                     # MLflow experiment logs
├── docs/                       # Detailed documentation
└── requirements.txt            # Python dependencies
```

## Scientific Background

### The Problem

Deep learning phase pickers trained with pointwise Binary Cross-Entropy (BCE) loss suffer from **amplitude suppression**: S-phase predictions consistently fail to cross detection thresholds despite being temporally accurate. This occurs because:

1. **Temporal Uncertainty**: S-wave onset separation from high-amplitude boundaries varies with epicentral distance
2. **CNN Anchoring**: Models anchor predictions to sharp amplitude changes, not subtle onsets
3. **Geometric Trap**: Pointwise BCE lacks lateral gradients to correct temporal misalignment

### The Solution

BlueDisc implements a **shape-then-align framework** via conditional GAN:

1. **BlueDisc Discriminator**: Enforces geometric coherence (Gaussian curves) through adversarial training
2. **Hybrid Loss Function**: Combines GAN (shape) + BCE (temporal anchoring)
3. **Emergent Lateral Gradients**: Shape constraint channels pointwise errors into lateral movement
4. **Result**: Shape stabilizes → temporal alignment succeeds → amplitude suppression eliminated

**Mathematical Framework**:
```
L(θ, ψ) = L_cGAN(θ, ψ) + λ × L_BCE(θ)
       = [GAN shape constraints] + [BCE temporal anchoring]
```

## Key Results

- **64% increase** in effective S-phase detections (peak >0.7, time error <0.1s)
- **Elimination** of suppression band (predictions trapped at ~0.5 amplitude)
- **Elliptical convergence** at label apex (time=0, amplitude=0.8-0.9)
- **Robust performance** across diverse seismic datasets

## Documentation

Comprehensive documentation is available in the `docs/` directory:

- **[Methodology](docs/methodology.md)** - Comprehensive scientific background, mathematical formulation, and experimental validation
- **[Training Guide](docs/training.md)** - Complete training parameters, loss functions, and training modes
- **[Inference & Evaluation](docs/inference_evaluation.md)** - Inference procedures and diagnostic metrics
- **[Visualization Tools](docs/visualization.md)** - Diagnostic plotting to identify suppression and validate corrections
- **[Architecture](docs/architecture.md)** - Project structure, module descriptions, and technical specifications

## Citation

If you use this code in your research, please cite our paper:

```bibtex
@article{huang2024bluedisc,
  title={Diagnosing and Breaking Amplitude Suppression in Seismic Phase Picking Through Adversarial Shape Learning},
  author={Huang, Chun-Ming and Chang, Li-Heng and Chang, I-Hsin and Lee, An-Sheng and Kuo-Chen, Hao},
  journal={[Journal Name]},
  year={2024},
  note={Manuscript in preparation}
}
```

## Requirements

```
torch>=2.0.0
seisbench>=0.4.0
mlflow>=2.0.0
numpy>=1.20.0
scipy>=1.7.0
matplotlib>=3.5.0
h5py>=3.6.0
tqdm>=4.60.0
```

See `requirements.txt` for complete dependencies.

## License

This project is released under the MIT License. See LICENSE file for details.

## Acknowledgments

- **PhaseNet**: Base architecture from [Zhu & Beroza (2018)](https://doi.org/10.1093/gji/ggy423)
- **SeisBench**: Dataset management and benchmarking framework
- **INSTANCE Dataset**: European seismology networks via [Michelini et al. (2021)](https://doi.org/10.5281/zenodo.5641031)
- **pix2pix**: Conditional GAN architecture inspiration from [Isola et al. (2017)](https://arxiv.org/abs/1611.07004)

## Contact

For questions, issues, or collaborations:

- **Chun-Ming Huang**: jimmy60504@gmail.com
- **Hao Kuo-Chen**: kuochenhao@ntu.edu.tw (Corresponding author)
- **GitHub Issues**: [Report bugs or request features](https://github.com/SeisBlue/BlueDisc/issues)

## Related Projects

- **[SeisBlue](https://github.com/SeisBlue/SeisBlue)**: Deep learning earthquake seismology framework
- **[SeisBench](https://github.com/seisbench/seisbench)**: Machine learning benchmark for seismology
- **[Pick-Benchmark](https://github.com/seisbench/pick-benchmark)**: Standardized phase picking evaluation

