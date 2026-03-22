# SPECTRA

**Realistic RF waveform generation for machine learning.**

SPECTRA generates synthetic radio-frequency signals on-the-fly for training
and evaluating AMC, signal detection, and cyclostationary feature classifiers.
A Rust backend (via PyO3) handles compute-intensive DSP; Python orchestrates
composition, impairments, and PyTorch DataLoader integration.

---

## Features

- **80+ waveform generators** — PSK, QAM, FSK/MSK/GMSK, OFDM/SC-FDMA, 5G NR, AM/FM, radar, spread spectrum, aviation/maritime protocols, chirp, polyphase and Barker/Costas/Zadoff-Chu codes, and more
- **24 composable channel impairments** — AWGN, fading, phase noise, IQ imbalance, Doppler, TDL, MIMO, PA nonlinearity, timing, spectral effects, and more (plus **RadarClutter** for 2-D radar matrices)
- **MIMO multi-antenna support** — flat/TDL channels, spatial correlation, steering vectors, seamless dataset integration
- **Wideband scene composition** — mix multiple signals with COCO-format bounding-box labels
- **Cyclostationary processing** — SCD (SSCA/FAM/S3CA), SCF, CAF, cumulants, PSD via Rust
- **Time-frequency analysis** — Wigner-Ville Distribution, Ambiguity Function
- **Data augmentations** — CutOut, MixUp, CutMix, PatchShuffle, TimeReversal, and more
- **Class balancing** — built-in `class_weights` and `balanced_sampler` for imbalanced datasets
- **Reproducible benchmarks** — 13 built-in YAML configs (AMC, wideband scenes, 5G, radar, spread spectrum, direction finding, channel robustness, SNR sweeps); load with `load_benchmark()`, `load_channel_benchmark()`, or `load_snr_sweep()`
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
