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
- `modulators.rs` — QPSK/BPSK symbol generation (xorshift64 PRNG with splitmix64 seeding)
- `filters.rs` — Root-Raised-Cosine pulse-shaping filter (upsample + convolve)
- `oscillators.rs` — chirp and tone complex sinusoid generation
- `lib.rs` — PyO3 module entry point, registers all Rust functions as `spectra._rust`

**Python layer** (`python/spectra/`) — orchestration and PyTorch integration:
- `waveforms/` — `Waveform` ABC with `generate()`, `bandwidth()`, `label`. Implementations (QPSK, BPSK) call Rust for symbols + filtering.
- `impairments/` — `Transform` ABC with `__call__(iq, desc, **kwargs) -> (iq, desc)`. Composable via `Compose([AWGN(), FrequencyOffset()])`.
- `scene/` — `Composer` generates wideband scenes: multiple signals frequency-shifted into a shared capture bandwidth. `SignalDescription` dataclass holds physical-unit ground truth. `to_coco()` converts physical labels to pixel-space bounding boxes.
- `transforms/` — `STFT` wraps `torch.stft` producing `[1, freq, time]` magnitude spectrograms.
- `datasets/` — `NarrowbandDataset` (single signal → class label) and `WidebandDataset` (multi-signal scene → COCO-format detection targets). Both use deterministic `(seed, idx)` seeding for DataLoader worker safety.

**Key design rule:** Rust functions are pure/stateless. All state, composition, and randomness management lives in Python.

## Build System

Maturin bridges Rust and Python. The Rust crate (`rust/Cargo.toml`, name `spectra-rs`) builds a `cdylib` that maturin installs as `spectra._rust`. Configuration in `pyproject.toml` points to `rust/Cargo.toml` via `manifest-path` and sets `python-source = "python"`.

## Testing

Tests live in `tests/` with shared fixtures in `conftest.py` (`rng`, `sample_rate`, `assert_valid_iq`). Test markers: `@pytest.mark.rust` for Rust FFI tests, `@pytest.mark.slow` for slow tests. Strict markers are enforced.

## CI

GitHub Actions (`.github/workflows/ci.yml`): Rust checks (fmt, clippy, test) on Ubuntu; Python tests on matrix of (Ubuntu, macOS) × (Python 3.10, 3.11, 3.12) with CPU-only PyTorch.
