# SPECTRA

Realistic radar and communication waveform generation with PyTorch integration.

SPECTRA generates synthetic RF signals on-the-fly for training machine learning models. A Rust backend (via PyO3) handles compute-intensive DSP primitives while Python provides composition, impairments, and native PyTorch `Dataset` classes.

## Features

- **Rust-accelerated DSP** — symbol generation, pulse-shaping filters, and oscillators run in compiled Rust
- **Composable impairments** — chain AWGN, frequency offset, and other channel effects like torchvision transforms
- **Wideband scene composition** — overlay multiple signals at different frequencies and times with physically correct complex-domain summation
- **Detection-ready labels** — ground truth in physical units (seconds, Hz) with automatic conversion to COCO-style bounding boxes on spectrograms
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

## API Overview

| Module | Classes / Functions | Purpose |
|---|---|---|
| `spectra.waveforms` | `QPSK`, `BPSK` | Baseband waveform generators |
| `spectra.impairments` | `AWGN`, `FrequencyOffset`, `Compose` | Composable channel impairments |
| `spectra.scene` | `Composer`, `SceneConfig`, `SignalDescription`, `STFTParams`, `to_coco` | Multi-signal scene composition and labeling |
| `spectra.transforms` | `STFT` | Spectrogram transform |
| `spectra.datasets` | `NarrowbandDataset`, `WidebandDataset`, `collate_fn` | PyTorch dataset classes |

## Running Tests

```bash
# All tests
pytest

# Rust FFI tests only
pytest -m rust

# Single test
pytest tests/test_waveforms_psk.py::TestQPSKWaveform::test_bandwidth -v
```

## License

MIT

