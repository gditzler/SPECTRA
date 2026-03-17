# SPECTRA Examples

Example notebooks and scripts demonstrating SPECTRA's capabilities for RF waveform generation, impairment simulation, and machine learning dataset construction. Each example is available as both a Jupyter notebook (`.ipynb`) and a standalone Python script (`.py`).

## Prerequisites

```bash
# Build SPECTRA (requires Python 3.10+, Rust 1.83+)
maturin develop --release

# Install dependencies
pip install numpy matplotlib torch --index-url https://download.pytorch.org/whl/cpu

# For CSP classification examples (08)
pip install 'spectra[classifiers]'
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

### 07 — CSP Feature Visualization (Intermediate)

Explore cyclostationary signal processing transforms for RF signal analysis.

- Spectral Correlation Density (SCD) comparison across modulations
- Spectral Coherence Function (SCF) for BPSK vs QPSK
- Cyclic Autocorrelation Function (CAF) heatmaps
- Higher-order cumulant feature comparison
- Rust-backed Welch PSD and energy detection

```bash
cd examples && python 07_csp_features.py
```

### 08 — CSP Classification (Advanced)

Build and evaluate a cyclostationary AMC classifier.

- `CyclostationaryDataset` with SCD, cumulant, and PSD representations
- `CyclostationaryAMC` with random forest classifier
- Confusion matrix evaluation
- Feature set comparison (cumulants vs cyclic_peaks vs combined)
- Accuracy vs SNR sweep

```bash
cd examples && python 08_csp_classification.py
```

### 15 — Radar Range Profile Processing (Intermediate)

Build a `RadarDataset` with LFM, Barker-coded, and P4 polyphase-coded radar
waveforms. Apply a matched filter to detect point targets, then compare
CA-CFAR and OS-CFAR adaptive threshold detectors against range-bin ground truth.

- Build a `RadarDataset` with point-scatterer targets at random range bins
- Apply `matched_filter` to maximise SNR for a known pulse shape
- Apply **CA-CFAR** and **OS-CFAR** adaptive threshold detectors
- Visualise and compare detection results against ground truth

```bash
cd examples && python 15_radar_processing.py
```

### 14 — Beamforming with a ULA (Intermediate)

Apply Delay-and-Sum, MVDR (Capon), and LCMV beamformers to a two-source scenario with a
desired signal and a strong interferer. Visualises and compares normalised beam patterns
showing where each algorithm places its main lobe and nulls.

- Apply **Delay-and-Sum** (DAS) conventional beamforming
- Apply **MVDR** (Capon) minimum-variance distortionless-response beamforming
- Apply **LCMV** beamforming with simultaneous gain and null constraints
- Compare normalised beam patterns with `compute_beam_pattern()`

```bash
cd examples && python 14_beamforming.py
```

### 13 — Direction-Finding Dataset with MUSIC and ESPRIT (Intermediate)

Build a `DirectionFindingDataset` with multiple co-channel sources at known azimuths, then
estimate those azimuths using the MUSIC pseudospectrum and ESPRIT subspace algorithms.
Compares estimated angles to ground truth and reports RMSE over 100 samples.

- Configure a ULA with `spectra.arrays.ula()`
- Build a `DirectionFindingDataset` with multiple concurrent sources
- Apply **MUSIC** (noise-subspace pseudospectrum) and **ESPRIT** (rotational invariance)
- Compute RMSE and compare algorithm performance

```bash
cd examples && python 13_direction_finding.py
```

### 12 — Choi-Williams Distribution (Intermediate)

Compare the Choi-Williams Distribution against the WVD for cross-term suppression.

- Side-by-side WVD vs CWD on multi-component signals
- Sigma parameter sweep showing resolution vs. cross-term trade-off
- Linear chirp instantaneous frequency tracking
- Output format demonstration (magnitude, mag_phase, real_imag)

```bash
cd examples && python 12_cwd_cross_term_suppression.py
```

## Output

All figures are saved to `examples/outputs/`. Running all scripts produces 37+ PNG figures and 1 YAML metadata file.

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
  07_csp_features.ipynb / .py           # Intermediate
  08_csp_classification.ipynb / .py     # Advanced
  12_cwd_cross_term_suppression.ipynb / .py  # Intermediate
  13_direction_finding.ipynb / .py      # Intermediate
  14_beamforming.ipynb / .py            # Intermediate
  15_radar_processing.ipynb / .py       # Intermediate
  outputs/                              # Generated figures
```
