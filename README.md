# SPECTRA

Realistic radar and communication waveform generation with PyTorch integration.

SPECTRA generates synthetic RF signals on-the-fly for training machine learning models. A Rust backend (via PyO3) handles compute-intensive DSP primitives while Python provides composition, impairments, and native PyTorch `Dataset` classes.

## Features

- **60+ waveform generators** — PSK, QAM, FSK, OFDM, ASK, AM, FM, chirp, polyphase codes, Zadoff-Chu, Barker, and more
- **22 channel impairments** — AWGN, frequency offset, phase noise, IQ imbalance, fading, TDL, MIMO, PA nonlinearity, timing, and others composable like torchvision transforms
- **MIMO multi-antenna support** — flat Rayleigh and TDL channels, spatial correlation (Kronecker model), ULA steering vectors, seamless `NarrowbandDataset` integration
- **Cyclostationary signal processing** — Rust-accelerated SCD, SCF, CAF, cumulants, PSD, and energy detection transforms for signal analysis and feature extraction
- **Time-frequency analysis** — Wigner-Ville Distribution and Ambiguity Function transforms for radar and comms research
- **Data augmentations** — CutOut, MixUp, CutMix, PatchShuffle, TimeReversal with dataset-level wrappers for soft-label training
- **Class balancing** — built-in `class_weights` parameter and `balanced_sampler()` utility for imbalanced datasets
- **AMC classifiers** — `CyclostationaryAMC` with cumulant, cyclic-peak, or combined feature sets and scikit-learn tree-based backends
- **Wideband scene composition** — overlay multiple signals at different frequencies and times with physically correct complex-domain summation
- **Detection-ready labels** — ground truth in physical units (seconds, Hz) with automatic conversion to COCO-style bounding boxes on spectrograms
- **Benchmark configs** — reproducible `spectra-18`, `spectra-18-wideband`, and `spectra-40` benchmarks loadable with one function call
- **Curriculum learning** — `CurriculumSchedule` ramps SNR, signal count, and impairment severity over training epochs
- **Streaming DataLoader** — `StreamingDataLoader` generates fresh data per epoch with deterministic seeding and curriculum integration
- **Deterministic generation** — every sample is reproducible from `(seed, index)`, safe across DataLoader workers
- **No disk I/O** — infinite variability generated on-the-fly in `__getitem__()`

## Installation

Requires Python 3.10+ and Rust 1.83+.

```bash
uv venv --python 3.12 .venv
source .venv/bin/activate
uv pip install maturin numpy pytest
uv pip install torch --index-url https://download.pytorch.org/whl/cpu
maturin develop --release
```

For the AMC classifiers (requires scikit-learn):

```bash
pip install 'spectra[classifiers]'
```

## Quick Start

### Narrowband (Automatic Modulation Classification)

```python
from torch.utils.data import DataLoader
from spectra import BPSK, QPSK, NarrowbandDataset, AWGN, Compose

dataset = NarrowbandDataset(
    waveform_pool=[QPSK(), BPSK()],
    num_samples=1000,
    num_iq_samples=1024,
    sample_rate=1e6,
    impairments=Compose([AWGN(snr_range=(5, 30))]),
    seed=42,
)

loader = DataLoader(dataset, batch_size=32, num_workers=4)
for iq_tensor, labels in loader:
    # iq_tensor: [B, 2, 1024] (I and Q channels)
    # labels: [B] (waveform class index)
    ...
```

### Wideband (Signal Detection)

```python
from spectra import (
    QPSK, BPSK, SceneConfig, WidebandDataset,
    AWGN, Compose, STFT, collate_fn,
)

config = SceneConfig(
    capture_duration=1e-3,
    capture_bandwidth=1e6,
    sample_rate=1e6,
    num_signals=(1, 5),
    signal_pool=[QPSK(), BPSK()],
    snr_range=(5, 30),
)

dataset = WidebandDataset(
    scene_config=config,
    num_samples=500,
    impairments=Compose([AWGN(snr_range=(10, 30))]),
    transform=STFT(nfft=256, hop_length=64),
    seed=42,
)

loader = DataLoader(dataset, batch_size=8, collate_fn=collate_fn)
for spectrograms, targets in loader:
    # spectrograms: [B, 1, freq_bins, time_bins]
    # targets: list of dicts with "boxes" [N, 4] and "labels" [N]
    ...
```

### CSP Classification

```python
from spectra import (
    BPSK, QPSK, PSK8, QAM16,
    CyclostationaryDataset, CyclostationaryAMC,
    SCD, Cumulants, AWGN, Compose,
)

# Multi-representation dataset with SCD + cumulant features
dataset = CyclostationaryDataset(
    waveform_pool=[BPSK(), QPSK(), PSK8(), QAM16()],
    num_samples=400,
    num_iq_samples=4096,
    sample_rate=1e6,
    representations={"scd": SCD(nfft=64, n_alpha=64), "cum": Cumulants()},
    impairments=Compose([AWGN(snr_range=(10, 25))]),
    seed=42,
)

# Train a random-forest AMC from cumulant features
amc = CyclostationaryAMC(feature_set="cumulants", classifier="random_forest")
amc.fit_from_dataset(dataset)
```

## API Overview

| Module | Key Classes / Functions | Purpose |
|---|---|---|
| `spectra.waveforms` | `BPSK`, `QPSK`, `QAM16`..`QAM1024`, `PSK8`..`PSK64`, `FSK`, `GMSK`, `OFDM`, `LFM`, `OOK`, `ASK4`..`ASK64`, `Tone`, `Noise`, `FM`, `AMDSB`, ... | 60+ baseband waveform generators |
| `spectra.impairments` | `AWGN`, `FrequencyOffset`, `PhaseNoise`, `IQImbalance`, `RayleighFading`, `TDLChannel`, `MIMOChannel`, `RappPA`, `SalehPA`, `Compose`, ... | 22 composable channel impairments |
| `spectra.scene` | `Composer`, `SceneConfig`, `SignalDescription`, `STFTParams`, `to_coco` | Multi-signal scene composition and labeling |
| `spectra.transforms` | `STFT`, `Spectrogram`, `SCD`, `SCF`, `CAF`, `Cumulants`, `WVD`, `AmbiguityFunction`, `MixUp`, `CutMix`, `CutOut`, ... | Spectral transforms, CSP features, time-frequency, augmentations |
| `spectra.datasets` | `NarrowbandDataset`, `WidebandDataset`, `CyclostationaryDataset`, `MixUpDataset`, `CutMixDataset`, `balanced_sampler`, ... | PyTorch dataset classes with balancing and augmentation wrappers |
| `spectra.classifiers` | `CyclostationaryAMC` | Traditional AMC with cumulant/cyclic-peak features |
| `spectra.benchmarks` | `load_benchmark` | Reproducible benchmark dataset loader |
| `spectra.curriculum` | `CurriculumSchedule` | Progressive difficulty scheduling |
| `spectra.streaming` | `StreamingDataLoader` | Epoch-aware DataLoader with curriculum |
| `spectra.utils` | `frequency_shift`, `srrc_taps`, `low_pass`, `DatasetWriter`, ... | DSP utilities and I/O |

## Benchmarks

Load a reproducible benchmark with a single call:

```python
from spectra import load_benchmark

train_ds, val_ds, test_ds = load_benchmark("spectra-18", split="all")
# 18-class narrowband AMC: 8000 train / 2000 val / 2000 test
```

Available benchmarks: `spectra-18` (narrowband AMC), `spectra-18-wideband` (signal detection).

## Examples

See [`examples/`](examples/) for 8 runnable scripts (with matching Jupyter notebooks):

| # | Topic | Level |
|---|---|---|
| 01 | Basic Waveform Generation | Novice |
| 02 | Impairments and Channel Effects | Intermediate |
| 03 | Transforms and Spectrograms | Intermediate |
| 04 | Narrowband Classification Dataset | Advanced |
| 05 | Wideband Scene Composition | Pro |
| 06 | Full Pipeline: Dataset to Classifier | Pro |
| 07 | CSP Feature Visualization | Intermediate |
| 08 | CSP Classification | Advanced |

## Running Tests

```bash
# All tests
pytest

# Rust FFI tests only
pytest -m rust

# Cyclostationary signal processing tests
pytest -m csp

# Single test
pytest tests/test_waveforms_psk.py::TestQPSKWaveform::test_bandwidth -v
```

## License

MIT
