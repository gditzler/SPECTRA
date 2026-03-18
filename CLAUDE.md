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

# Run only Rust FFI tests or slow tests
pytest -m rust -v
pytest -m slow -v

# Rust-only checks (no Python needed)
cargo fmt --manifest-path rust/Cargo.toml --all -- --check
cargo clippy --manifest-path rust/Cargo.toml --all-targets -- -D warnings
cargo test --manifest-path rust/Cargo.toml
```

After any Rust code change, re-run `maturin develop --release` before running Python tests.

## Architecture

**Rust layer** (`rust/src/`) — stateless functions that accept and return NumPy arrays via PyO3:
- `modulators.rs` — QPSK/BPSK symbol generation (xorshift64 PRNG with splitmix64 seeding), `_with_indices` variants for link simulation, `get_*_constellation()` accessors
- `filters.rs` — Root-Raised-Cosine pulse-shaping filter (upsample + convolve)
- `oscillators.rs` — chirp and tone complex sinusoid generation
- `codes.rs` — polyphase code generators (Frank, P1–P4, Costas)
- `cyclo_spectral.rs` — SCD (SSCA, FAM), PSD (Welch), channelizer
- `cyclo_temporal.rs` — cumulants, CAF
- `s3ca.rs` — S³CA-based SCD estimation
- `sfft.rs` — sliding FFT utilities
- `lib.rs` — PyO3 module entry point, registers all Rust functions as `spectra._rust`

**Python layer** (`python/spectra/`) — orchestration and PyTorch integration:
- `waveforms/` — `Waveform` ABC with `generate()`, `bandwidth()`, `label`. 86+ implementations across PSK, QAM, FSK, OFDM, ASK, AM, FM, chirp, polyphase code, 5G NR, and protocol families.
- `impairments/` — `Transform` ABC with `__call__(iq, desc, **kwargs) -> (iq, desc)`. 23 impairments (including `MIMOChannel`, `RadarClutter`) composable via `Compose([AWGN(), FrequencyOffset()])`.
- `scene/` — `Composer` generates wideband scenes: multiple signals frequency-shifted into a shared capture bandwidth. `SignalDescription` dataclass holds physical-unit ground truth. `to_coco()` converts physical labels to pixel-space bounding boxes.
- `transforms/` — `STFT`, `Spectrogram`, `SCD`, `SCF`, `CAF`, `Cumulants`, `PSD`, `EnergyDetector`, `AmbiguityFunction`, plus data augmentations (`CutOut`, `TimeReversal`, `PatchShuffle`, etc.).
- `utils/file_handlers/` — Pluggable RF file readers (`SigMFReader`, `RawIQReader`, `HDF5Reader`, `NumpyReader`) with auto-detection registry, plus `SigMFWriter` for export.
- `datasets/folder.py` — `SignalFolderDataset`: ImageFolder-style dataset loading IQ recordings from class-per-directory structure.
- `datasets/manifest.py` — `ManifestDataset`: CSV/JSON manifest-based dataset for flat-file layouts.
- `datasets/` — `NarrowbandDataset` (single signal → class label), `WidebandDataset` (multi-signal scene → COCO-format detection targets), `CyclostationaryDataset` (multi-representation CSP features), `RadarPipelineDataset` (end-to-end waveform → channel → receiver → tracker). All use deterministic `(seed, idx)` seeding for DataLoader worker safety.
- `targets/` — `ConstantVelocity`, `ConstantTurnRate` trajectory models and `SwerlingRCS` (cases 0–IV) for radar target simulation.
- `tracking/` — `KalmanFilter` (generic linear KF), `ConstantVelocityKF` and `RangeDopplerKF` factory functions for radar tracking.
- `algorithms/` — `music`, `esprit`, `root_music`, `capon` (DoA estimation), `delay_and_sum`, `mvdr`, `lcmv` (beamforming), `matched_filter`, `ca_cfar`, `os_cfar` (radar), `single_pulse_canceller`, `double_pulse_canceller`, `doppler_filter_bank` (MTI).
- `receivers/` — `CoherentReceiver` (matched filter + slicer + demapper), `Decoder` ABC with `PassthroughDecoder`, `ViterbiDecoder`/`LDPCDecoder` stubs.
- `link/` — `LinkSimulator` for BER/SER/PER vs. Eb/N0 curves with `LinkResults` dataclass.
- `classifiers/` — `CyclostationaryAMC` for traditional AMC with cumulant/cyclic-peak features and scikit-learn backends. Requires `spectra[classifiers]` optional dep.
- `benchmarks/` — `load_benchmark()` loads reproducible configs (`spectra-18`, `spectra-18-wideband`).
- `curriculum.py` — `CurriculumSchedule` for progressive difficulty ramps.
- `streaming.py` — `StreamingDataLoader` for epoch-aware data generation with curriculum.
- `studio/` — Gradio-based UI for interactive waveform generation, visualization (7 plot types), and SigMF export. Requires `spectra[ui]` optional dep.
- `cli/` — Unified CLI: `spectra studio` (launch UI), `spectra generate` (headless batch), `spectra viz` (quick plots), `spectra build` (interactive wizard).
- `models/` — PyTorch model architectures for AMC (`CNN1D`, `ResNetAMC`).
- `metrics.py` — Classification metrics plus `bit_error_rate`, `symbol_error_rate`, `packet_error_rate`.

**Key design rule:** Rust functions are pure/stateless. All state, composition, and randomness management lives in Python.

## Build System

Maturin bridges Rust and Python. The Rust crate (`rust/Cargo.toml`, name `spectra-rs`) builds a `cdylib` that maturin installs as `spectra._rust`. Configuration in `pyproject.toml` points to `rust/Cargo.toml` via `manifest-path` and sets `python-source = "python"`.

## Testing

Tests live in `tests/` with shared fixtures in `conftest.py` (`rng`, `sample_rate`, `assert_valid_iq`). Test markers: `@pytest.mark.rust` for Rust FFI tests, `@pytest.mark.slow` for slow tests, `@pytest.mark.csp` for cyclostationary signal processing tests. Strict markers are enforced.

## CI

GitHub Actions (`.github/workflows/ci.yml`): Rust checks (fmt, clippy, test) on Ubuntu; Python tests on matrix of (Ubuntu, macOS) × (Python 3.10, 3.11, 3.12) with CPU-only PyTorch.


