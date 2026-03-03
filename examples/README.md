# SPECTRA Examples

Example notebooks and scripts demonstrating SPECTRA's capabilities for RF waveform generation, impairment simulation, and machine learning dataset construction. Each example is available as both a Jupyter notebook (`.ipynb`) and a standalone Python script (`.py`).

## Prerequisites

```bash
# Build SPECTRA (requires Python 3.10+, Rust 1.83+)
maturin develop --release

# Install dependencies
pip install numpy matplotlib torch --index-url https://download.pytorch.org/whl/cpu
```

## Examples

### 01 — Basic Waveform Generation (Novice)

Generate and visualize digital and analog modulation waveforms.

- Generate QPSK, BPSK, QAM, PSK, OOK, FSK, GMSK, OFDM signals
- Plot IQ time-domain, constellation diagrams, and power spectral density
- Compare digital vs. analog modulations (AM-DSB, AM-USB, FM, Tone)

```bash
cd examples && python 01_basic_waveforms.py
```

### 02 — Impairments and Channel Effects (Intermediate)

Apply realistic channel impairments and visualize their effects.

- AWGN at varying SNR levels (0–20 dB)
- Frequency offset, phase noise, and IQ imbalance
- Multipath fading: Rayleigh and Rician channels
- Chain impairments with `Compose`

```bash
cd examples && python 02_impairments_and_channels.py
```

### 03 — Transforms and Spectrograms (Intermediate)

Compute spectrograms, apply data augmentations, and use DSP utilities.

- STFT spectrograms for different modulation types
- `ComplexTo2D` and `Normalize` transforms
- Data augmentations: CutOut, TimeReversal, PatchShuffle, RandomDropSamples, AddSlope
- DSP utilities: filter design (low-pass, SRRC, Gaussian) and frequency shifting

```bash
cd examples && python 03_transforms_and_spectrograms.py
```

### 04 — Narrowband Classification Dataset (Advanced)

Build a PyTorch dataset for automatic modulation classification (AMC).

- `NarrowbandDataset` with 8 modulation classes
- PyTorch `DataLoader` integration
- Per-class spectrogram visualization
- Family-level grouping with `FamilyName`
- SNR-dependent constellation analysis

```bash
cd examples && python 04_narrowband_dataset.py
```

### 05 — Wideband Scene Composition (Pro)

Generate multi-signal wideband RF captures with detection labels.

- `SceneConfig` and `Composer` for wideband scene generation
- Spectrogram visualization with COCO-format bounding boxes
- Multiple scene generation with varying signal compositions
- `WidebandDataset` with `DataLoader` for object detection tasks

```bash
cd examples && python 05_wideband_scenes.py
```

### 06 — Full Pipeline: Dataset to Classifier (Pro)

End-to-end workflow from dataset generation to trained classifier.

- Generate a reproducible 8-class AMC dataset with impairments
- Train a CNN classifier on STFT spectrograms
- Evaluate with training curves, per-class accuracy, and confusion matrix
- Save and reload dataset metadata via YAML

```bash
cd examples && python 06_full_pipeline.py
```

## Output

All figures are saved to `examples/outputs/`. Running all six scripts produces 27 PNG figures and 1 YAML metadata file.

## File Structure

```
examples/
  plot_helpers.py                       # Shared plotting utilities
  01_basic_waveforms.ipynb / .py        # Novice
  02_impairments_and_channels.ipynb / .py   # Intermediate
  03_transforms_and_spectrograms.ipynb / .py # Intermediate
  04_narrowband_dataset.ipynb / .py     # Advanced
  05_wideband_scenes.ipynb / .py        # Pro
  06_full_pipeline.ipynb / .py          # Pro
  outputs/                              # Generated figures
```
