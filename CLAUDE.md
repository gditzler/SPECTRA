# CLAUDE.md

This file provides guidance to Claude Code (claude.ai/code) when working with code in this repository.

## What is SPECTRA?

SPECTRA is a Rust-backed Python library for realistic RF waveform generation with PyTorch DataLoader integration. Rust (PyO3) handles compute-intensive DSP primitives; Python orchestrates composition, impairments, and dataset construction.

## Build & Development

```bash
# Set up environment (requires Python 3.10+, Rust 1.83+)
uv venv --python 3.12 .venv
source .venv/bin/activate
uv pip install maturin pytest numpy
uv pip install torch --index-url https://download.pytorch.org/whl/cpu

# Build Rust extension into the venv
maturin develop --release

# Run all tests
pytest tests/ -v

# Run a single test file or test
pytest tests/test_rust_modulators.py -v
pytest tests/test_waveforms_psk.py::TestQPSKWaveform::test_bandwidth -v

# Run only Rust FFI tests, slow tests, or CSP tests
pytest -m rust -v
pytest -m slow -v
pytest -m csp -v

# Rust-only checks (no Python needed)
cargo fmt --manifest-path rust/Cargo.toml --all -- --check
cargo clippy --manifest-path rust/Cargo.toml --all-targets -- -D warnings
cargo test --manifest-path rust/Cargo.toml

# Build documentation (requires spectra[docs] optional dep)
mkdocs serve   # local preview
mkdocs build   # build static site to site/
```

After any Rust code change, re-run `maturin develop --release` before running Python tests.

## Architecture

**Rust layer** (`rust/src/`) — stateless functions that accept and return NumPy arrays via PyO3:
- `modulators.rs` — BPSK/QPSK/8PSK/M-PSK/M-QAM/M-ASK/M-FSK symbol generation (xorshift64 PRNG with splitmix64 seeding), `_with_indices` variants for link simulation, `get_*_constellation()` accessors
- `filters.rs` — Root-Raised-Cosine pulse-shaping filter (upsample + convolve), Gaussian FIR taps, windowed-sinc lowpass taps, generic complex convolution
- `oscillators.rs` — chirp and tone complex sinusoid generation
- `codes.rs` — polyphase code generators (Frank, P1–P4, Costas), spread-spectrum codes (Gold, Kasami, Walsh-Hadamard)
- `radar.rs` — pulse train generation, FMCW sweep, stepped-frequency waveform, nonlinear FM sweep
- `protocols.rs` — protocol frame generators (ADS-B, Mode S, AIS, ACARS) with CRC
- `nr.rs` — 5G NR primitives: OFDM symbol (IFFT + CP), PSS, SSS, DMRS generation
- `cyclo_spectral.rs` — SCD (SSCA, FAM), PSD (Welch), channelizer
- `cyclo_temporal.rs` — cumulants, CAF
- `s3ca.rs` — S³CA-based SCD estimation (**work in progress** — algorithm produces incorrect output; full implementation tracked on `feature/s3ca` branch)
- `cwd.rs` — Choi-Williams Distribution
- `reassigned_gabor.rs` — Reassigned Gabor (spectrogram) Transform
- `sfft.rs` — sliding FFT utilities
- `lib.rs` — PyO3 module entry point, registers all Rust functions as `spectra._rust`

**Python layer** (`python/spectra/`) — orchestration and PyTorch integration:
- `waveforms/` — `Waveform` ABC with `generate()`, `bandwidth()`, `label`. Implementations across PSK, QAM, FSK, OFDM (incl. SC-FDMA), ASK, AM, FM, chirp, polyphase code, 5G NR (OFDM, PDSCH, PUSCH, PRACH, SSB), radar (pulsed, pulse-Doppler, FMCW, stepped-frequency, NLFM, Barker-coded, polyphase-coded), spread spectrum (DSSS-BPSK/QPSK, FHSS, THSS, CDMA forward/reverse, chirp-SS), aviation/maritime protocols (ADS-B, Mode S, AIS, ACARS, DME, ILS Localizer), tone, noise, LFM, Barker code, Costas code, and Zadoff-Chu families.
- `impairments/` — `Transform` ABC with `__call__(iq, desc, **kwargs) -> (iq, desc)`. Impairments: AWGN, colored noise, frequency offset, frequency drift, phase offset, phase noise, IQ imbalance, DC offset, sample rate offset, quantization, Rayleigh/Rician fading, TDL channel, MIMO channel, Doppler shift, power amplifiers (Rapp, Saleh), fractional delay, sampling jitter, radar clutter, adjacent-channel interference, intermodulation products, passband ripple, spectral inversion. Composable via `Compose([...])`.
- `antennas/` — `AntennaElement` ABC with `pattern()` method. Implementations: `IsotropicElement`, `ShortDipoleElement`, `HalfWaveDipoleElement`, `CosinePowerElement`, `YagiElement`, `MSIAntennaElement` (with MSI file parser).
- `arrays/` — `AntennaArray` with steering vector computation and factory functions (`ula`, `uca`, `rectangular`). `CalibrationErrors` dataclass for per-element gain/phase offsets.
- `scene/` — `Composer` generates wideband scenes: multiple signals frequency-shifted into a shared capture bandwidth. `SceneConfig` for declarative scene setup. `SignalDescription` dataclass holds physical-unit ground truth. `to_coco()` converts physical labels to pixel-space bounding boxes.
- `transforms/` — `STFT`, `Spectrogram`, `SCD`, `SCF`, `CAF`, `Cumulants`, `PSD`, `EnergyDetector`, `AmbiguityFunction`, `CWD` (Choi-Williams), `WVD` (Wigner-Ville), `ReassignedGabor`, `InstantaneousFrequency`, `ComplexTo2D`, `Normalize`, `SpectrogramNormalize`, `ToSnapshotMatrix`, plus data augmentations (`CutOut`, `CutMix`, `MixUp`, `TimeReversal`, `PatchShuffle`, `AGC`, `AddSlope`, `ChannelSwap`, `RandomDropSamples`, `RandomMagRescale`), alignment transforms (`DCRemove`, `Resample`, `PowerNormalize`, `AGCNormalize`, `ClipNormalize`, `BandpassAlign`, `NoiseFloorMatch`, `NoiseProfileTransfer`, `SpectralWhitening`, `ReceiverEQ`), and target transforms (`ClassIndex`, `FamilyIndex`, `FamilyName`, `BoxesNormalize`, `YOLOLabel`).
- `utils/dsp.py` — DSP helpers: `low_pass`, `srrc_taps`, `gaussian_taps`, `convolve`, `upsample`, `frequency_shift`, `noise_generator`, `compute_spectrogram`, `polyphase_decimator`, `polyphase_interpolator`, `multistage_resampler`, `bandwidth_from_bounds`, `center_freq_from_bounds`.
- `utils/rrc_cache.py` — Cached RRC tap computation.
- `utils/writer.py` — `DatasetWriter` for batch dataset export (Zarr backend).
- `utils/file_handlers/` — Pluggable RF file readers and writers with auto-detection registry. Readers: `SigMFReader`, `RawIQReader`, `HDF5Reader`, `NumpyReader`, `SQLiteReader`. Writers: `SigMFWriter`, `RawIQWriter`, `HDF5Writer`, `NumpyWriter`, `SQLiteWriter`. Also `ZarrHandler` for chunked array I/O and `export_dataset_to_folder` utility.
- `datasets/narrowband.py` — `NarrowbandDataset`: single signal → class label.
- `datasets/wideband.py` — `WidebandDataset`: multi-signal scene → COCO-format detection targets.
- `datasets/cyclo.py` — `CyclostationaryDataset`: multi-representation CSP features.
- `datasets/radar_pipeline.py` — `RadarPipelineDataset`: end-to-end waveform → channel → receiver → tracker.
- `datasets/radar.py` — `RadarDataset`: on-the-fly range profile generation for radar target detection.
- `datasets/direction_finding.py` — `DirectionFindingDataset`: snapshot matrix + DoA labels using antenna arrays.
- `datasets/wideband_df.py` — `WidebandDirectionFindingDataset`: joint wideband spectrum + DoA dataset.
- `datasets/snr_sweep.py` — `SNRSweepDataset`: structured (SNR × class × sample) grid for sweep evaluation.
- `datasets/df_snr_sweep.py` — `DirectionFindingSNRSweepDataset`: SNR sweep variant for direction finding.
- `datasets/folder.py` — `SignalFolderDataset`: ImageFolder-style dataset loading IQ recordings from class-per-directory structure.
- `datasets/manifest.py` — `ManifestDataset`: CSV/JSON manifest-based dataset for flat-file layouts.
- `datasets/mixing.py` — `MixUpDataset`, `CutMixDataset`: cross-sample augmentation wrappers.
- `datasets/sampler.py` — `balanced_sampler`: inverse-frequency `WeightedRandomSampler` factory.
- `datasets/metadata.py` — `DatasetMetadata`, `NarrowbandMetadata`, `WidebandMetadata` dataclasses with YAML serialization.
- `datasets/iq_utils.py` — IQ tensor utilities (truncate/pad, conversion).
- All datasets use deterministic `(seed, idx)` seeding for DataLoader worker safety.
- `targets/` — `ConstantVelocity`, `ConstantTurnRate` trajectory models and `SwerlingRCS` (cases 0–IV), `NonFluctuatingRCS` for radar target simulation.
- `tracking/` — `KalmanFilter` (generic linear KF), `ConstantVelocityKF` and `RangeDopplerKF` factory functions for radar tracking.
- `algorithms/` — `music`, `esprit`, `root_music`, `capon`, `find_peaks_doa` (DoA estimation), `delay_and_sum`, `mvdr`, `lcmv`, `compute_beam_pattern` (beamforming), `matched_filter`, `ca_cfar`, `os_cfar` (radar), `single_pulse_canceller`, `double_pulse_canceller`, `doppler_filter_bank` (MTI).
- `receivers/` — `Receiver` ABC, `CoherentReceiver` (matched filter + slicer + demapper), `Decoder` ABC with `PassthroughDecoder`, `ViterbiDecoder`/`LDPCDecoder` stubs.
- `link/` — `LinkSimulator` for BER/SER/PER vs. Eb/N0 curves with `LinkResults` dataclass.
- `classifiers/` — `CyclostationaryAMC` for traditional AMC with cumulant/cyclic-peak features and scikit-learn backends. Requires `spectra[classifiers]` optional dep.
- `benchmarks/` — `load_benchmark()` (narrowband, wideband, `direction_finding`), `load_channel_benchmark()`, `load_snr_sweep()` load reproducible configs. `evaluate_snr_sweep()`, `evaluate_channel_conditions()` for structured evaluation. 13 built-in YAML configs: `spectra-18`, `spectra-18-wideband`, `spectra-40`, `spectra-5g`, `spectra-airport`, `spectra-channel`, `spectra-congested-ism`, `spectra-df`, `spectra-maritime-vhf`, `spectra-protocol`, `spectra-radar`, `spectra-snr`, `spectra-spread`.
- `curriculum.py` — `CurriculumSchedule` for progressive difficulty ramps.
- `streaming.py` — `StreamingDataLoader` for epoch-aware data generation with curriculum.
- `studio/` — Gradio-based UI (Gradio 6.0+) for interactive waveform generation, visualization, and SigMF export. Modules: `app.py` (main app factory), `generate_tab.py`, `visualize_tab.py`, `export_tab.py`, `params.py` (UI parameter mappings), `plotting.py` (plot renderers), `theme.py` (theme/CSS applied at `launch()`). Requires `spectra[ui]` optional dep.
- `cli/` — Unified CLI: `spectra studio` (launch UI), `spectra generate` (headless batch from YAML config), `spectra viz` (quick IQ file plots), `spectra build` (interactive config wizard via `config_builder.py`). Also `spectra-build` entry point via `signal_builder.py`.
- `models/` — PyTorch model architectures for AMC: `CNNAMC` (1D CNN), `ResNetAMC`.
- `metrics.py` — `confusion_matrix`, `accuracy`, `classification_report`, `per_snr_accuracy`, `per_snr_rmse`, `bit_error_rate`, `symbol_error_rate`, `packet_error_rate`.

**Key design rule:** Rust functions are pure/stateless. All state, composition, and randomness management lives in Python.

## Build System

Maturin bridges Rust and Python. The Rust crate (`rust/Cargo.toml`, name `spectra-rs`) builds a `cdylib` that maturin installs as `spectra._rust`. Configuration in `pyproject.toml` points to `rust/Cargo.toml` via `manifest-path` and sets `python-source = "python"`.

**Optional dependency groups** (in `pyproject.toml`):
- `spectra[classifiers]` — scikit-learn
- `spectra[io]` — sigmf, h5py
- `spectra[alignment]` — scipy
- `spectra[ui]` — gradio, scipy
- `spectra[yaml]` — pyyaml
- `spectra[zarr]` — zarr
- `spectra[docs]` — mkdocs, mkdocs-material, mkdocstrings
- `spectra[dev]` — maturin, pytest, pytest-cov, pyyaml, zarr, h5py
- `spectra[all]` — all optional deps combined

## Testing

Tests live in `tests/` with shared fixtures in `conftest.py` (`rng`, `sample_rate`, `assert_valid_iq`). Test markers: `@pytest.mark.rust` for Rust FFI tests, `@pytest.mark.slow` for slow tests, `@pytest.mark.csp` for cyclostationary signal processing tests, `@pytest.mark.io` for file I/O tests, `@pytest.mark.benchmark` for benchmark output format and logic tests. Strict markers are enforced.

## Documentation

MkDocs with Material theme. Config in `mkdocs.yml`, source in `docs/`. API docs auto-generated via mkdocstrings. Sections: Getting Started, User Guide, API Reference, Contributing.

## Pre-commit

`.pre-commit-config.yaml` runs Ruff (lint + format) for Python and `cargo fmt --check` for Rust.

## Linting

Ruff is configured in `pyproject.toml` (`line-length = 100`, `target-version = "py310"`, selects E/F/I/W rules).

## CI

No CI workflow is configured yet. When added, it should live at `.github/workflows/ci.yml` and cover:
- Rust checks: `cargo fmt --check`, `cargo clippy`, `cargo test`
- Python tests: matrix of (Ubuntu, macOS) × (Python 3.10, 3.11, 3.12) with CPU-only PyTorch
- Lint: Ruff check + format