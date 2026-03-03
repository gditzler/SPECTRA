# SPECTRA vs TorchSig Benchmark

Comprehensive comparison of SPECTRA and TorchSig across dataset generation speed and ML classification performance.

## Overview

This benchmark evaluates two RF signal processing frameworks on a common 8-class modulation recognition task:

| Idx | Class | SPECTRA | TorchSig |
|-----|-------|---------|----------|
| 0 | BPSK | `BPSK()` | `"bpsk"` |
| 1 | QPSK | `QPSK()` | `"qpsk"` |
| 2 | 8PSK | `PSK8()` | `"8psk"` |
| 3 | 16QAM | `QAM16()` | `"16qam"` |
| 4 | 64QAM | `QAM64()` | `"64qam"` |
| 5 | OOK | `OOK()` | `"ook"` |
| 6 | FSK | `FSK(order=2)` | `"2fsk"` |
| 7 | OFDM | `OFDM()` | `"ofdm-64"` |

Two classifiers are evaluated across a 2x2 cross-framework train/test matrix:

- **CNN**: ResNet-18 on magnitude spectrograms
- **CSP**: CyclostationaryAMC with cumulant features + random forest

## Prerequisites

```bash
# Core dependencies (already installed with SPECTRA)
pip install torch numpy pyyaml matplotlib

# torchvision for ResNet-18
pip install torchvision

# scikit-learn for CSP classifier
pip install scikit-learn
```

## Installing TorchSig (optional)

TorchSig is only needed for cross-framework comparisons. Without it, benchmarks run the SPECTRA-only subset.

```bash
python benchmarks/torchsig_compat/install.py
```

## Running Benchmarks

### Speed Benchmark

Measures `__getitem__` latency and DataLoader throughput:

```bash
# SPECTRA only
python benchmarks/comparison/speed_benchmark.py --skip-torchsig

# Both frameworks
python benchmarks/comparison/speed_benchmark.py
```

### ML Benchmark

Runs the 2x2 cross-framework classification matrix:

```bash
# SPECTRA only (1x1 matrix)
python benchmarks/comparison/ml_benchmark.py --skip-torchsig

# Both frameworks (2x2 matrix)
python benchmarks/comparison/ml_benchmark.py

# Skip specific classifiers
python benchmarks/comparison/ml_benchmark.py --skip-cnn
python benchmarks/comparison/ml_benchmark.py --skip-csp
```

### Visualization

Generates PNG plots from saved results:

```bash
python benchmarks/comparison/visualize.py
```

Outputs (saved to `benchmarks/comparison/results/`):
- `speed_comparison.png` — Latency and throughput bar charts
- `cnn_confusion_matrices.png` — CNN confusion matrices
- `csp_confusion_matrices.png` — CSP confusion matrices
- `accuracy_matrix.png` — Heatmap of all accuracy values

## Configuration

All parameters are in `benchmarks/comparison/config.yaml`:

- Dataset sizes (8000 train / 2000 test)
- IQ sample length (1024)
- SNR range (0–20 dB)
- CNN hyperparameters (epochs, LR, batch size, STFT params)
- CSP feature set and classifier type
- Speed benchmark parameters (warmup, workers)

## Full Run (Both Frameworks)

```bash
python benchmarks/torchsig_compat/install.py
python benchmarks/comparison/speed_benchmark.py
python benchmarks/comparison/ml_benchmark.py
python benchmarks/comparison/visualize.py
```

## Interpretation

The cross-framework matrix reveals how well models generalize between data generators:

- **Diagonal** (train/test same framework): Within-framework baseline
- **Off-diagonal** (train on A, test on B): Cross-framework generalization

Large diagonal-vs-off-diagonal gaps indicate the frameworks generate systematically different waveforms or impairments.
