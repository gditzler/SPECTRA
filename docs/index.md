# SPECTRA

**Realistic RF waveform generation for machine learning.**

SPECTRA generates synthetic radio-frequency signals on-the-fly for training
and evaluating AMC, signal detection, and cyclostationary feature classifiers.
A Rust backend (via PyO3) handles compute-intensive DSP; Python orchestrates
composition, impairments, and PyTorch DataLoader integration.

---

## Features

- **60+ waveform generators** — PSK, QAM, FSK, OFDM, AM/FM, chirp, polyphase codes, and more
- **16 channel impairments** — AWGN, fading, phase noise, IQ imbalance, Doppler, and more
- **Wideband scene composition** — mix multiple signals with COCO-format bounding-box labels
- **Cyclostationary processing** — SCD (SSCA/FAM/S3CA), SCF, CAF, cumulants, PSD via Rust
- **Reproducible benchmarks** — `spectra-18` (18-class AMC) and `spectra-18-wideband` (detection)
- **Curriculum learning** — progressive difficulty ramps with `CurriculumSchedule`
- **Pluggable file I/O** — SigMF, HDF5, raw IQ, SQLite, Zarr, NumPy

## Quick Start

```python
import spectra
from spectra import NarrowbandDataset, QPSK, BPSK, AWGN, Compose
from torch.utils.data import DataLoader

dataset = NarrowbandDataset(
    waveform_pool=[QPSK(), BPSK()],
    num_samples=10_000,
    num_iq_samples=1024,
    sample_rate=1e6,
    impairments=Compose([AWGN(snr_range=(-5.0, 20.0))]),
    seed=42,
)
loader = DataLoader(dataset, batch_size=64, num_workers=4)
iq, labels = next(iter(loader))  # iq: [64, 2, 1024]
```

[Get started](getting-started/installation.md){ .md-button .md-button--primary }
[API Reference](api/waveforms.md){ .md-button }
