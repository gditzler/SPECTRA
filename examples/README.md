# SPECTRA Examples

Example scripts demonstrating SPECTRA's capabilities for RF waveform generation,
impairment simulation, signal processing, and machine learning dataset construction.
Examples are organized into domain-specific sub-folders. Most examples are available
as both standalone Python scripts (`.py`) and Jupyter notebooks (`.ipynb`).

## Prerequisites

```bash
# Build SPECTRA (requires Python 3.10+, Rust 1.83+)
maturin develop --release

# Install dependencies
pip install numpy matplotlib torch --index-url https://download.pytorch.org/whl/cpu

# For CSP classification examples
pip install 'spectra[classifiers]'
```

## Running Examples

All scripts are designed to be run from the repository root:

```bash
python examples/getting_started/basic_waveforms.py
python examples/radar/mti_doppler.py
```

## Directory Structure

```
examples/
├── plot_helpers.py                          # Shared plotting utilities
├── outputs/                                 # Generated figures
│
├── getting_started/                         # Introductory examples
│   ├── basic_waveforms.py                   # Novice
│   ├── impairments_and_channels.py          # Intermediate
│   └── transforms_and_spectrograms.py       # Intermediate
│
├── waveforms/                               # Advanced signal generation
│   ├── spread_spectrum.py                   # Intermediate
│   ├── protocol_signals.py                  # Intermediate
│   └── nr_5g_signals.py                     # Advanced
│
├── impairments/                             # Channel effects & distortions
│   └── advanced_impairments.py              # Intermediate
│
├── transforms/                              # Signal processing & features
│   ├── alignment_transforms.py              # Intermediate
│   └── time_frequency_analysis.py           # Intermediate
│
├── datasets/                                # Dataset construction & I/O
│   ├── narrowband_dataset.py                # Advanced
│   ├── wideband_scenes.py                   # Pro
│   ├── folder_and_manifest.py               # Intermediate
│   ├── snr_sweep_evaluation.py              # Advanced
│   ├── augmentation_wrappers.py             # Intermediate
│   ├── dataset_io.py                        # Intermediate
│   └── streaming_curriculum.py              # Advanced
│
├── classification/                          # AMC training & evaluation
│   ├── full_pipeline.py                     # Pro
│   ├── csp_classification.py                # Advanced
│   ├── train_narrowband_cnn.py              # Advanced
│   └── resnet_amc.py                        # Advanced
│
├── cyclostationary/                         # CSP features & time-frequency
│   ├── csp_features.py                      # Intermediate
│   ├── s3ca_vs_ssca.py                      # Advanced
│   ├── cwd_cross_term_suppression.py        # Intermediate
│   └── wvd_time_frequency.py                # Intermediate
│
├── antenna_arrays/                          # Arrays, DoA, beamforming
│   ├── direction_finding.py                 # Intermediate
│   ├── beamforming.py                       # Intermediate
│   ├── wideband_direction_finding.py        # Intermediate–Advanced
│   ├── antenna_elements.py                  # Intermediate
│   └── array_geometries.py                  # Intermediate
│
├── radar/                                   # Radar processing & tracking
│   ├── radar_processing.py                  # Intermediate
│   ├── radar_pipeline.py                    # Advanced
│   └── mti_doppler.py                       # Intermediate
│
├── communications/                          # Link simulation & receivers
│   └── link_simulator.py                    # Advanced
│
├── environment/                             # Propagation & link budgets
│   └── propagation_and_links.py             # Intermediate
│
└── benchmarks/                              # Reproducible evaluation
    └── benchmark_evaluation.py              # Advanced
```

## Examples by Category

### Getting Started

| Example | Level | Description |
|---------|-------|-------------|
| `basic_waveforms.py` | Novice | Generate BPSK, QPSK, QAM, FSK; plot IQ, constellation, PSD |
| `impairments_and_channels.py` | Intermediate | AWGN, frequency offset, phase noise, Rayleigh/Rician fading, Compose |
| `transforms_and_spectrograms.py` | Intermediate | STFT, ComplexTo2D, data augmentations, DSP filter design |

### Waveforms

| Example | Level | Description |
|---------|-------|-------------|
| `spread_spectrum.py` | Intermediate | DSSS-BPSK/QPSK, FHSS, THSS, CDMA Forward/Reverse, ChirpSS |
| `protocol_signals.py` | Intermediate | ADS-B, Mode S, AIS, ACARS, DME, ILS Localizer |
| `nr_5g_signals.py` | Advanced | NR OFDM, PDSCH, PUSCH, PRACH, SSB signals |

### Impairments

| Example | Level | Description |
|---------|-------|-------------|
| `advanced_impairments.py` | Intermediate | ColoredNoise, DopplerShift, IQImbalance, TDL channels, PA models, quantization |

### Transforms

| Example | Level | Description |
|---------|-------|-------------|
| `alignment_transforms.py` | Intermediate | DCRemove, PowerNormalize, AGC, ClipNormalize, SpectralWhitening |
| `time_frequency_analysis.py` | Intermediate | ReassignedGabor, InstantaneousFrequency, AmbiguityFunction |

### Datasets

| Example | Level | Description |
|---------|-------|-------------|
| `narrowband_dataset.py` | Advanced | NarrowbandDataset, PyTorch DataLoader, per-class spectrograms |
| `wideband_scenes.py` | Pro | SceneConfig, Composer, COCO bounding boxes, WidebandDataset |
| `folder_and_manifest.py` | Intermediate | SignalFolderDataset, ManifestDataset for pre-existing recordings |
| `snr_sweep_evaluation.py` | Advanced | SNRSweepDataset for structured (SNR × class) evaluation |
| `augmentation_wrappers.py` | Intermediate | MixUpDataset, CutMixDataset cross-sample augmentation |
| `dataset_io.py` | Intermediate | DatasetWriter, export_dataset_to_folder, NumpyWriter |
| `streaming_curriculum.py` | Advanced | StreamingDataLoader, CurriculumSchedule, progressive difficulty |

### Classification

| Example | Level | Description |
|---------|-------|-------------|
| `full_pipeline.py` | Pro | End-to-end: dataset → CNN training → evaluation |
| `csp_classification.py` | Advanced | CyclostationaryAMC with SCD/cumulant features, random forest |
| `train_narrowband_cnn.py` | Advanced | CNNAMC benchmark training with confusion matrix |
| `resnet_amc.py` | Advanced | ResNetAMC on spectrogram features |

### Cyclostationary & Time-Frequency

| Example | Level | Description |
|---------|-------|-------------|
| `csp_features.py` | Intermediate | SCD, SCF, CAF heatmaps, cumulant features, PSD |
| `s3ca_vs_ssca.py` | Advanced | S³CA vs SSCA SCD estimators |
| `cwd_cross_term_suppression.py` | Intermediate | Choi-Williams Distribution vs WVD, sigma sweep |
| `wvd_time_frequency.py` | Intermediate | Wigner-Ville Distribution analysis |

### Antenna Arrays

| Example | Level | Description |
|---------|-------|-------------|
| `direction_finding.py` | Intermediate | ULA, DirectionFindingDataset, MUSIC, ESPRIT |
| `beamforming.py` | Intermediate | Delay-and-Sum, MVDR, LCMV beam patterns |
| `wideband_direction_finding.py` | Intermediate–Advanced | WidebandDirectionFindingDataset, sub-band MUSIC |
| `antenna_elements.py` | Intermediate | Isotropic, dipole, Yagi, cosine-power radiation patterns |
| `array_geometries.py` | Intermediate | ULA, UCA, rectangular arrays, CalibrationErrors |

### Radar

| Example | Level | Description |
|---------|-------|-------------|
| `radar_processing.py` | Intermediate | Matched filter, CA-CFAR, OS-CFAR on LFM/coded pulses |
| `radar_pipeline.py` | Advanced | Target trajectories, Swerling RCS, clutter, MTI, Kalman tracking |
| `mti_doppler.py` | Intermediate | Single/double pulse canceller, Doppler filter bank, range-Doppler map |

### Communications

| Example | Level | Description |
|---------|-------|-------------|
| `link_simulator.py` | Advanced | CoherentReceiver, BER/SER/PER curves, LinkSimulator |

### Environment

| Example | Level | Description |
|---------|-------|-------------|
| `propagation_and_links.py` | Intermediate | Free-space/log-distance/COST231 propagation, multi-emitter link budget |

### Benchmarks

| Example | Level | Description |
|---------|-------|-------------|
| `benchmark_evaluation.py` | Advanced | Load and evaluate built-in benchmarks (spectra-18, spectra-spread, etc.) |

## Output

All figures are saved to `examples/outputs/`. Running all scripts produces 60+ PNG figures.
