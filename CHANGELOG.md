# Changelog

All notable changes to this project will be documented in this file.

The format is based on [Keep a Changelog](https://keepachangelog.com/en/1.1.0/),
and this project adheres to [Semantic Versioning](https://semver.org/spec/v2.0.0.html).

## [Unreleased]

### Changed (BREAKING)

- 16-QAM (and all square M-QAM ≥ 16) constellation labelling is now Gray-coded. The prior row-major mapping produced a BER↔SER mismatch of up to log₂(M) at moderate-to-high SNR. Saved datasets and classifiers trained on the prior mapping must be regenerated. See `examples/verification/verify_qam16.py::P3` and `rust/src/modulators.rs::qam_constellation_gray_adjacency`.

### Fixed

- `GMSK` modulation index restored to h = 0.5; was previously h_eff = 0.5/sps = 0.0625 due to zero-insertion upsampling combined with a sum-normalised Gaussian filter. Affects spectral occupancy and BER curves for `sp.GMSK`. Regression guarded by `tests/test_waveforms_fsk.py::TestGMSKModulationIndex`.

### Added

- `spectra.models` package with `CNNAMC` (1D CNN for raw IQ) and `ResNetAMC` (lightweight ResNet for spectrograms, no torchvision)
- `spectra.metrics` module with `confusion_matrix`, `classification_report`, `per_snr_accuracy`, `accuracy`
- End-to-end training example (`examples/11_train_narrowband_cnn.py`)
- `TDLChannel` impairment with 3GPP TS 38.901 profiles (TDL-A/B/C/D/E) and ITU profiles (Pedestrian A/B, Vehicular A/B)
- `RappPA` and `SalehPA` power amplifier nonlinearity impairments
- `FractionalDelay` and `SamplingJitter` timing impairments
- SC-FDMA (DFT-spread OFDM) waveform
- Enhanced OFDM with guard bands, DC null, pilots, and multi-modulation support (BPSK, QPSK, QAM16, QAM64)
- PyPI wheel publishing workflow via GitHub Actions
- Python linting CI job with Ruff
- Pre-commit configuration for Ruff and cargo fmt

## [0.1.0] - 2026-03-05

### Added

- InstantaneousFrequency transform
- MkDocs documentation site with Material theme, user guide, and API reference
- Google-style docstrings for priority API classes
- DopplerShift impairment with constant and linear profiles
- SNR sweep evaluation utilities (`evaluate_snr_sweep`, `evaluate_channel_conditions`)
- Channel benchmark (`spectra-channel`) and SNR benchmark (`spectra-snr`)
- `SNRSweepDataset` for structured SNR-sweep evaluation
- `spectra-40` 40-class narrowband benchmark
- S3CA (Strip Spectral Correlation Analyzer) method for SCD computation
- Sparse FFT implementation for S3CA
- TorchSig feature parity: waveforms, impairments, transforms, DSP utils, metadata, Zarr persistence
- 60+ waveform implementations: PSK (BPSK, QPSK, 8PSK, 16PSK, 32PSK, 64PSK), QAM (16-1024), FSK/MSK/GMSK/GFSK families, OFDM (72-2048 subcarriers), ASK/OOK, AM (DSB, DSB-SC, USB, LSB), FM, LFM chirp, ChirpSS, DSSS-BPSK, Barker/Costas/Zadoff-Chu codes, polyphase codes (Frank, P1-P4), Noise, Tone
- 16 RF impairments: AWGN, FrequencyOffset, FrequencyDrift, PhaseOffset, PhaseNoise, IQImbalance, DCOffset, SampleRateOffset, RayleighFading, RicianFading, Quantization, PassbandRipple, SpectralInversion, ColoredNoise, AdjacentChannelInterference, IntermodulationProducts
- Wideband scene compositor (`Composer`) with multi-signal overlap and COCO-format labels
- `NarrowbandDataset` and `WidebandDataset` with PyTorch DataLoader integration
- `CyclostationaryDataset` for multi-representation CSP features
- `SignalFolderDataset` and `ManifestDataset` for file-based datasets
- Transforms: STFT, Spectrogram, SCD, SCF, CAF, Cumulants, PSD, EnergyDetector, and data augmentations (CutOut, TimeReversal, PatchShuffle, etc.)
- File I/O: SigMF, HDF5, NumPy, RawIQ, SQLite readers and writers with auto-detection
- `CurriculumSchedule` for progressive difficulty ramps
- `StreamingDataLoader` for epoch-aware data generation
- Reproducible benchmarks (`spectra-18`, `spectra-18-wideband`)
- Rust DSP backend via PyO3: QPSK/BPSK symbol generation, RRC pulse shaping, chirp/tone oscillators, SCD (SSCA + FAM + S3CA), SCF, CAF, cumulants, PSD, energy detection
- Example Jupyter notebooks (basic waveforms through full pipeline)
- GitHub Actions CI for Rust checks and Python test matrix
