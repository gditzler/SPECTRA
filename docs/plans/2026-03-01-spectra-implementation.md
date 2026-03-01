# SPECTRA Implementation Plan

> **For Claude:** REQUIRED SUB-SKILL: Use superpowers:executing-plans to implement this plan task-by-task.

**Goal:** Build a Rust-backed Python library for realistic RF waveform generation with PyTorch DataLoader integration, supporting both narrowband AMC and wideband signal detection with STFT bounding box labels.

**Architecture:** Rust engine (PyO3) exports stateless DSP primitives (oscillators, modulators, filters). Python orchestration layer composes these into waveforms, impairment pipelines, wideband scenes, and PyTorch datasets. On-the-fly generation via `__getitem__()` with deterministic seeding.

**Tech Stack:** Rust (PyO3 0.28, numpy 0.28, rustfft 6.4, num-complex 0.4), Python (NumPy, PyTorch, pytest), maturin build system.

**Design doc:** `docs/plans/2026-03-01-spectra-architecture-design.md`

---

## Task 1: Project Scaffolding & Build System

**Files:**
- Create: `pyproject.toml`
- Create: `rust/Cargo.toml`
- Create: `rust/src/lib.rs`
- Create: `python/spectra/__init__.py`
- Create: `tests/conftest.py`
- Create: `.gitignore`

**Step 1: Create `.gitignore`**

```gitignore
# Rust
rust/target/
**/*.rs.bk

# Python
__pycache__/
*.py[cod]
*.egg-info/
dist/
build/
.venv/
*.so
*.dylib
*.pyd

# IDE
.vscode/
.idea/
```

**Step 2: Create `pyproject.toml`**

```toml
[build-system]
requires = ["maturin>=1.9,<2"]
build-backend = "maturin"

[project]
name = "spectra"
version = "0.1.0"
description = "Realistic radar and communication waveform generation with PyTorch integration"
requires-python = ">=3.10"
license = {text = "MIT"}
dependencies = [
    "numpy>=1.24",
    "torch>=2.0",
]

[project.optional-dependencies]
dev = [
    "maturin>=1.9,<2",
    "pytest>=8.0",
    "pytest-cov>=5.0",
]

[tool.maturin]
manifest-path = "rust/Cargo.toml"
python-source = "python"
module-name = "spectra._rust"
bindings = "pyo3"

[tool.pytest.ini_options]
testpaths = ["tests"]
python_files = "test_*.py"
addopts = ["--tb=short", "-v", "--strict-markers"]
markers = [
    "slow: marks tests as slow",
    "rust: marks tests that exercise Rust FFI directly",
]
```

**Step 3: Create `rust/Cargo.toml`**

```toml
[package]
name = "spectra-rs"
version = "0.1.0"
edition = "2021"
rust-version = "1.83"
publish = false

[lib]
name = "_rust"
crate-type = ["cdylib"]

[dependencies]
pyo3 = { version = "0.28", features = ["extension-module"] }
numpy = "0.28"
num-complex = "0.4"
rustfft = "6.4"
```

**Step 4: Create `rust/src/lib.rs` with a minimal PyO3 module**

```rust
use pyo3::prelude::*;

/// SPECTRA Rust backend for high-performance DSP primitives.
#[pymodule]
fn _rust(m: &Bound<'_, PyModule>) -> PyResult<()> {
    m.add("__version__", "0.1.0")?;
    Ok(())
}
```

**Step 5: Create `python/spectra/__init__.py`**

```python
from spectra._rust import __version__

__all__ = ["__version__"]
```

**Step 6: Create `tests/conftest.py`**

```python
import numpy as np
import pytest


@pytest.fixture
def rng():
    """Seeded random number generator for reproducible tests."""
    return np.random.default_rng(seed=42)


@pytest.fixture
def sample_rate():
    """Default sample rate for tests."""
    return 1e6


@pytest.fixture
def assert_valid_iq():
    """Assert that an IQ array is well-formed."""
    def _check(iq, expected_length=None):
        assert isinstance(iq, np.ndarray), f"Expected ndarray, got {type(iq)}"
        assert iq.dtype == np.complex64, f"Expected complex64, got {iq.dtype}"
        assert iq.ndim == 1, f"Expected 1D array, got {iq.ndim}D"
        assert not np.any(np.isnan(iq)), "Array contains NaN values"
        assert not np.any(np.isinf(iq)), "Array contains Inf values"
        if expected_length is not None:
            assert len(iq) == expected_length, (
                f"Expected length {expected_length}, got {len(iq)}"
            )
    return _check
```

**Step 7: Create venv, build, and run smoke test**

```bash
cd /path/to/spectra
python -m venv .venv
source .venv/bin/activate
pip install maturin pytest numpy torch
maturin develop
```

**Step 8: Write the smoke test**

Create `tests/test_smoke.py`:

```python
def test_import():
    import spectra
    assert hasattr(spectra, "__version__")
    assert spectra.__version__ == "0.1.0"
```

**Step 9: Run the smoke test**

Run: `pytest tests/test_smoke.py -v`
Expected: PASS

**Step 10: Commit**

```bash
git add .gitignore pyproject.toml rust/ python/ tests/
git commit -m "feat: scaffold project with maturin + PyO3 build system"
```

---

## Task 2: Rust Modulators — QPSK Symbol Generation

**Files:**
- Create: `rust/src/modulators.rs`
- Modify: `rust/src/lib.rs`
- Create: `tests/test_rust_modulators.py`

**Step 1: Write failing tests for QPSK symbol generation**

Create `tests/test_rust_modulators.py`:

```python
import numpy as np
import numpy.testing as npt
import pytest


class TestGenerateQpskSymbols:
    def test_returns_complex64_ndarray(self):
        from spectra._rust import generate_qpsk_symbols
        symbols = generate_qpsk_symbols(64, seed=0)
        assert isinstance(symbols, np.ndarray)
        assert symbols.dtype == np.complex64

    def test_correct_length(self):
        from spectra._rust import generate_qpsk_symbols
        for n in [1, 10, 100, 1024]:
            symbols = generate_qpsk_symbols(n, seed=0)
            assert symbols.shape == (n,)

    def test_constellation_points_on_unit_circle(self):
        from spectra._rust import generate_qpsk_symbols
        symbols = generate_qpsk_symbols(1000, seed=0)
        magnitudes = np.abs(symbols)
        npt.assert_allclose(magnitudes, 1.0, atol=1e-6)

    def test_four_constellation_points(self):
        from spectra._rust import generate_qpsk_symbols
        symbols = generate_qpsk_symbols(10000, seed=0)
        expected_angles = [np.pi / 4, 3 * np.pi / 4, -3 * np.pi / 4, -np.pi / 4]
        for angle in expected_angles:
            point = np.exp(1j * angle)
            distances = np.abs(symbols - point)
            assert np.any(distances < 1e-5), f"Missing constellation point at angle {angle}"

    def test_deterministic_with_seed(self):
        from spectra._rust import generate_qpsk_symbols
        s1 = generate_qpsk_symbols(256, seed=42)
        s2 = generate_qpsk_symbols(256, seed=42)
        npt.assert_array_equal(s1, s2)

    def test_different_seeds_differ(self):
        from spectra._rust import generate_qpsk_symbols
        s1 = generate_qpsk_symbols(256, seed=0)
        s2 = generate_qpsk_symbols(256, seed=1)
        assert not np.array_equal(s1, s2)

    def test_zero_symbols(self):
        from spectra._rust import generate_qpsk_symbols
        symbols = generate_qpsk_symbols(0, seed=0)
        assert symbols.shape == (0,)
```

**Step 2: Run tests to verify they fail**

Run: `maturin develop && pytest tests/test_rust_modulators.py -v`
Expected: FAIL with `ImportError: cannot import name 'generate_qpsk_symbols'`

**Step 3: Implement `rust/src/modulators.rs`**

```rust
use numpy::ndarray::Array1;
use numpy::{IntoPyArray, PyArray1};
use num_complex::Complex32;
use pyo3::prelude::*;

/// QPSK constellation: points at pi/4, 3pi/4, -3pi/4, -pi/4
const QPSK_CONSTELLATION: [Complex32; 4] = [
    Complex32::new(std::f32::consts::FRAC_1_SQRT_2, std::f32::consts::FRAC_1_SQRT_2),   // pi/4
    Complex32::new(-std::f32::consts::FRAC_1_SQRT_2, std::f32::consts::FRAC_1_SQRT_2),  // 3pi/4
    Complex32::new(-std::f32::consts::FRAC_1_SQRT_2, -std::f32::consts::FRAC_1_SQRT_2), // -3pi/4
    Complex32::new(std::f32::consts::FRAC_1_SQRT_2, -std::f32::consts::FRAC_1_SQRT_2),  // -pi/4
];

/// Simple seeded PRNG (xorshift64) for deterministic symbol generation.
struct Xorshift64 {
    state: u64,
}

impl Xorshift64 {
    fn new(seed: u64) -> Self {
        Self { state: if seed == 0 { 1 } else { seed } }
    }

    fn next(&mut self) -> u64 {
        self.state ^= self.state << 13;
        self.state ^= self.state >> 7;
        self.state ^= self.state << 17;
        self.state
    }
}

#[pyfunction]
pub fn generate_qpsk_symbols<'py>(
    py: Python<'py>,
    num_symbols: usize,
    seed: u64,
) -> Bound<'py, PyArray1<Complex32>> {
    let mut rng = Xorshift64::new(seed);
    let symbols = Array1::from_shape_fn(num_symbols, |_| {
        let idx = (rng.next() % 4) as usize;
        QPSK_CONSTELLATION[idx]
    });
    symbols.into_pyarray(py)
}
```

**Step 4: Register module in `rust/src/lib.rs`**

```rust
use pyo3::prelude::*;

mod modulators;

#[pymodule]
fn _rust(m: &Bound<'_, PyModule>) -> PyResult<()> {
    m.add("__version__", "0.1.0")?;
    m.add_function(wrap_pyfunction!(modulators::generate_qpsk_symbols, m)?)?;
    Ok(())
}
```

**Step 5: Build and run tests**

Run: `maturin develop && pytest tests/test_rust_modulators.py -v`
Expected: All PASS

**Step 6: Commit**

```bash
git add rust/src/modulators.rs rust/src/lib.rs tests/test_rust_modulators.py
git commit -m "feat: add Rust QPSK symbol generation with deterministic seeding"
```

---

## Task 3: Rust Filters — Root Raised Cosine

**Files:**
- Create: `rust/src/filters.rs`
- Modify: `rust/src/lib.rs`
- Create: `tests/test_rust_filters.py`

**Step 1: Write failing tests for RRC filter**

Create `tests/test_rust_filters.py`:

```python
import numpy as np
import numpy.testing as npt
import pytest


class TestApplyRrcFilter:
    def test_returns_complex64_ndarray(self):
        from spectra._rust import generate_qpsk_symbols, apply_rrc_filter
        symbols = generate_qpsk_symbols(64, seed=0)
        filtered = apply_rrc_filter(symbols, rolloff=0.35, span=10, sps=8)
        assert isinstance(filtered, np.ndarray)
        assert filtered.dtype == np.complex64

    def test_output_is_upsampled(self):
        from spectra._rust import generate_qpsk_symbols, apply_rrc_filter
        symbols = generate_qpsk_symbols(64, seed=0)
        sps = 8
        span = 10
        filtered = apply_rrc_filter(symbols, rolloff=0.35, span=span, sps=sps)
        expected_len = len(symbols) * sps + span * sps
        assert len(filtered) == expected_len

    def test_no_nan_or_inf(self):
        from spectra._rust import generate_qpsk_symbols, apply_rrc_filter
        symbols = generate_qpsk_symbols(128, seed=0)
        for rolloff in [0.0, 0.25, 0.35, 0.5, 1.0]:
            filtered = apply_rrc_filter(symbols, rolloff=rolloff, span=6, sps=4)
            assert not np.any(np.isnan(filtered)), f"NaN with rolloff={rolloff}"
            assert not np.any(np.isinf(filtered)), f"Inf with rolloff={rolloff}"

    def test_energy_preservation(self):
        from spectra._rust import generate_qpsk_symbols, apply_rrc_filter
        symbols = generate_qpsk_symbols(256, seed=0)
        sps = 4
        filtered = apply_rrc_filter(symbols, rolloff=0.35, span=10, sps=sps)
        input_energy = np.sum(np.abs(symbols) ** 2)
        output_energy = np.sum(np.abs(filtered) ** 2) / sps
        ratio = output_energy / input_energy
        assert 0.5 < ratio < 2.0, f"Energy ratio {ratio} outside acceptable range"

    def test_deterministic(self):
        from spectra._rust import generate_qpsk_symbols, apply_rrc_filter
        symbols = generate_qpsk_symbols(64, seed=0)
        f1 = apply_rrc_filter(symbols, rolloff=0.35, span=10, sps=8)
        f2 = apply_rrc_filter(symbols, rolloff=0.35, span=10, sps=8)
        npt.assert_array_equal(f1, f2)
```

**Step 2: Run tests to verify they fail**

Run: `maturin develop && pytest tests/test_rust_filters.py -v`
Expected: FAIL with `ImportError`

**Step 3: Implement `rust/src/filters.rs`**

```rust
use numpy::ndarray::Array1;
use numpy::{IntoPyArray, PyArray1, PyReadonlyArray1};
use num_complex::Complex32;
use pyo3::prelude::*;

/// Generate root-raised-cosine filter taps.
fn rrc_taps(rolloff: f32, span: usize, sps: usize) -> Vec<f32> {
    let num_taps = span * sps + 1;
    let half = (num_taps / 2) as f32;
    let sps_f = sps as f32;
    let mut taps: Vec<f32> = (0..num_taps)
        .map(|i| {
            let t = (i as f32 - half) / sps_f;
            if t.abs() < 1e-12 {
                (1.0 - rolloff + 4.0 * rolloff / std::f32::consts::PI) / sps_f.sqrt()
            } else if (t.abs() - 1.0 / (4.0 * rolloff)).abs() < 1e-12 && rolloff > 0.0 {
                let sqrt2 = std::f32::consts::SQRT_2;
                rolloff / (sqrt2 * sps_f.sqrt())
                    * ((1.0 + 2.0 / std::f32::consts::PI) * (std::f32::consts::PI / (4.0 * rolloff)).sin()
                        + (1.0 - 2.0 / std::f32::consts::PI) * (std::f32::consts::PI / (4.0 * rolloff)).cos())
            } else {
                let pi_t = std::f32::consts::PI * t;
                let num = (pi_t * (1.0 - rolloff)).sin()
                    + 4.0 * rolloff * t * (pi_t * (1.0 + rolloff)).cos();
                let den = pi_t * (1.0 - (4.0 * rolloff * t).powi(2));
                if den.abs() < 1e-12 {
                    0.0
                } else {
                    num / (den * sps_f.sqrt())
                }
            }
        })
        .collect();

    // Normalize to unit energy
    let energy: f32 = taps.iter().map(|t| t * t).sum();
    if energy > 0.0 {
        let norm = energy.sqrt();
        for t in &mut taps {
            *t /= norm;
        }
    }
    taps
}

/// Apply RRC pulse-shaping filter: upsample by sps, then convolve with RRC taps.
#[pyfunction]
pub fn apply_rrc_filter<'py>(
    py: Python<'py>,
    symbols: PyReadonlyArray1<'py, Complex32>,
    rolloff: f32,
    span: usize,
    sps: usize,
) -> Bound<'py, PyArray1<Complex32>> {
    let symbols = symbols.as_array();
    let taps = rrc_taps(rolloff, span, sps);

    // Upsample: insert sps-1 zeros between each symbol
    let upsampled_len = symbols.len() * sps;
    let mut upsampled = vec![Complex32::new(0.0, 0.0); upsampled_len];
    for (i, &s) in symbols.iter().enumerate() {
        upsampled[i * sps] = s;
    }

    // Convolve
    let output_len = upsampled_len + taps.len() - 1;
    let output = Array1::from_shape_fn(output_len, |n| {
        let mut sum = Complex32::new(0.0, 0.0);
        for (k, &tap) in taps.iter().enumerate() {
            if n >= k && (n - k) < upsampled_len {
                sum += upsampled[n - k] * tap;
            }
        }
        sum
    });
    output.into_pyarray(py)
}
```

**Step 4: Register in `rust/src/lib.rs`**

Add `mod filters;` and register `filters::apply_rrc_filter` in the `_rust` pymodule.

**Step 5: Build and run tests**

Run: `maturin develop && pytest tests/test_rust_filters.py -v`
Expected: All PASS

**Step 6: Commit**

```bash
git add rust/src/filters.rs rust/src/lib.rs tests/test_rust_filters.py
git commit -m "feat: add Rust RRC pulse-shaping filter"
```

---

## Task 4: Rust Oscillators — Chirp & Tone Generation

**Files:**
- Create: `rust/src/oscillators.rs`
- Modify: `rust/src/lib.rs`
- Create: `tests/test_rust_oscillators.py`

**Step 1: Write failing tests**

Create `tests/test_rust_oscillators.py`:

```python
import numpy as np
import numpy.testing as npt
import pytest


class TestGenerateChirp:
    def test_output_shape_and_dtype(self):
        from spectra._rust import generate_chirp
        fs = 1e6
        duration = 0.001
        signal = generate_chirp(duration=duration, fs=fs, f0=1e3, f1=1e5)
        assert signal.shape == (int(duration * fs),)
        assert signal.dtype == np.complex64

    def test_unit_magnitude(self):
        from spectra._rust import generate_chirp
        signal = generate_chirp(duration=0.01, fs=1e6, f0=0.0, f1=1e5)
        npt.assert_allclose(np.abs(signal), 1.0, atol=1e-5)

    def test_frequency_sweep(self):
        from spectra._rust import generate_chirp
        fs = 1e6
        f0, f1 = 1e3, 1e5
        signal = generate_chirp(duration=0.01, fs=fs, f0=f0, f1=f1)
        phase = np.unwrap(np.angle(signal))
        inst_freq = np.diff(phase) / (2.0 * np.pi) * fs
        margin = int(0.05 * len(inst_freq))
        npt.assert_allclose(np.mean(inst_freq[:margin]), f0, rtol=0.2)
        npt.assert_allclose(np.mean(inst_freq[-margin:]), f1, rtol=0.2)


class TestGenerateTone:
    def test_output_shape_and_dtype(self):
        from spectra._rust import generate_tone
        signal = generate_tone(frequency=1e3, duration=0.01, fs=1e6)
        assert signal.shape == (int(0.01 * 1e6),)
        assert signal.dtype == np.complex64

    def test_unit_magnitude(self):
        from spectra._rust import generate_tone
        signal = generate_tone(frequency=1e3, duration=0.01, fs=1e6)
        npt.assert_allclose(np.abs(signal), 1.0, atol=1e-5)

    def test_correct_frequency(self):
        from spectra._rust import generate_tone
        freq = 1000.0
        fs = 1e6
        signal = generate_tone(frequency=freq, duration=0.01, fs=fs)
        phase = np.unwrap(np.angle(signal))
        inst_freq = np.diff(phase) / (2.0 * np.pi) * fs
        npt.assert_allclose(inst_freq, freq, atol=1.0)
```

**Step 2: Run tests to verify they fail**

Run: `maturin develop && pytest tests/test_rust_oscillators.py -v`
Expected: FAIL with `ImportError`

**Step 3: Implement `rust/src/oscillators.rs`**

```rust
use numpy::ndarray::Array1;
use numpy::{IntoPyArray, PyArray1};
use num_complex::Complex32;
use pyo3::prelude::*;

/// Generate a linear frequency modulated (chirp) signal.
#[pyfunction]
pub fn generate_chirp<'py>(
    py: Python<'py>,
    duration: f64,
    fs: f64,
    f0: f64,
    f1: f64,
) -> Bound<'py, PyArray1<Complex32>> {
    let num_samples = (duration * fs) as usize;
    let chirp_rate = (f1 - f0) / duration;
    let signal = Array1::from_shape_fn(num_samples, |i| {
        let t = i as f64 / fs;
        let phase = 2.0 * std::f64::consts::PI * (f0 * t + 0.5 * chirp_rate * t * t);
        Complex32::new(phase.cos() as f32, phase.sin() as f32)
    });
    signal.into_pyarray(py)
}

/// Generate a complex sinusoidal tone.
#[pyfunction]
pub fn generate_tone<'py>(
    py: Python<'py>,
    frequency: f64,
    duration: f64,
    fs: f64,
) -> Bound<'py, PyArray1<Complex32>> {
    let num_samples = (duration * fs) as usize;
    let signal = Array1::from_shape_fn(num_samples, |i| {
        let t = i as f64 / fs;
        let phase = 2.0 * std::f64::consts::PI * frequency * t;
        Complex32::new(phase.cos() as f32, phase.sin() as f32)
    });
    signal.into_pyarray(py)
}
```

**Step 4: Register in `rust/src/lib.rs`**

Add `mod oscillators;` and register both `generate_chirp` and `generate_tone`.

**Step 5: Build and run tests**

Run: `maturin develop && pytest tests/test_rust_oscillators.py -v`
Expected: All PASS

**Step 6: Commit**

```bash
git add rust/src/oscillators.rs rust/src/lib.rs tests/test_rust_oscillators.py
git commit -m "feat: add Rust chirp and tone oscillator generation"
```

---

## Task 5: Python Waveform Base Class & QPSK Waveform

**Files:**
- Create: `python/spectra/waveforms/__init__.py`
- Create: `python/spectra/waveforms/base.py`
- Create: `python/spectra/waveforms/psk.py`
- Create: `tests/test_waveforms_psk.py`

**Step 1: Write failing tests**

Create `tests/test_waveforms_psk.py`:

```python
import numpy as np
import numpy.testing as npt
import pytest


class TestQPSKWaveform:
    def test_generate_returns_valid_iq(self, assert_valid_iq, sample_rate):
        from spectra.waveforms import QPSK
        waveform = QPSK()
        iq = waveform.generate(num_symbols=128, sample_rate=sample_rate)
        assert_valid_iq(iq)

    def test_default_parameters(self):
        from spectra.waveforms import QPSK
        waveform = QPSK()
        assert waveform.samples_per_symbol == 8
        assert waveform.pulse_shape == "rrc"
        assert waveform.rolloff == 0.35

    def test_label(self):
        from spectra.waveforms import QPSK
        assert QPSK().label == "QPSK"

    def test_bandwidth(self, sample_rate):
        from spectra.waveforms import QPSK
        sps = 8
        waveform = QPSK(samples_per_symbol=sps)
        expected_bw = sample_rate / sps
        assert waveform.bandwidth(sample_rate) == pytest.approx(expected_bw)

    def test_custom_parameters(self, assert_valid_iq, sample_rate):
        from spectra.waveforms import QPSK
        waveform = QPSK(samples_per_symbol=4, rolloff=0.5, filter_span=6)
        iq = waveform.generate(num_symbols=64, sample_rate=sample_rate)
        assert_valid_iq(iq)

    def test_deterministic_with_seed(self, sample_rate):
        from spectra.waveforms import QPSK
        waveform = QPSK()
        iq1 = waveform.generate(num_symbols=64, sample_rate=sample_rate, seed=42)
        iq2 = waveform.generate(num_symbols=64, sample_rate=sample_rate, seed=42)
        npt.assert_array_equal(iq1, iq2)

    @pytest.mark.parametrize("num_symbols", [1, 16, 256])
    def test_various_lengths(self, num_symbols, assert_valid_iq, sample_rate):
        from spectra.waveforms import QPSK
        waveform = QPSK(samples_per_symbol=4)
        iq = waveform.generate(num_symbols=num_symbols, sample_rate=sample_rate)
        assert_valid_iq(iq)


class TestBPSKWaveform:
    def test_generate_returns_valid_iq(self, assert_valid_iq, sample_rate):
        from spectra.waveforms import BPSK
        waveform = BPSK()
        iq = waveform.generate(num_symbols=128, sample_rate=sample_rate)
        assert_valid_iq(iq)

    def test_label(self):
        from spectra.waveforms import BPSK
        assert BPSK().label == "BPSK"

    def test_constellation_is_real_axis(self, sample_rate):
        """BPSK symbols should lie on the real axis before filtering."""
        from spectra._rust import generate_bpsk_symbols
        symbols = generate_bpsk_symbols(1000, seed=0)
        npt.assert_allclose(symbols.imag, 0.0, atol=1e-6)
```

**Step 2: Run tests to verify they fail**

Run: `maturin develop && pytest tests/test_waveforms_psk.py -v`
Expected: FAIL

**Step 3: Add `generate_bpsk_symbols` to `rust/src/modulators.rs`**

```rust
#[pyfunction]
pub fn generate_bpsk_symbols<'py>(
    py: Python<'py>,
    num_symbols: usize,
    seed: u64,
) -> Bound<'py, PyArray1<Complex32>> {
    let mut rng = Xorshift64::new(seed);
    let symbols = Array1::from_shape_fn(num_symbols, |_| {
        if rng.next() % 2 == 0 {
            Complex32::new(1.0, 0.0)
        } else {
            Complex32::new(-1.0, 0.0)
        }
    });
    symbols.into_pyarray(py)
}
```

Register `generate_bpsk_symbols` in `lib.rs`.

**Step 4: Implement `python/spectra/waveforms/base.py`**

```python
from abc import ABC, abstractmethod
from typing import Optional

import numpy as np


class Waveform(ABC):
    """Abstract base class for all waveform generators."""

    @abstractmethod
    def generate(
        self,
        num_symbols: int,
        sample_rate: float,
        seed: Optional[int] = None,
    ) -> np.ndarray:
        """Generate complex baseband IQ samples."""
        ...

    @abstractmethod
    def bandwidth(self, sample_rate: float) -> float:
        """Signal bandwidth in Hz given a sample rate."""
        ...

    @property
    @abstractmethod
    def label(self) -> str:
        """Classification label string."""
        ...
```

**Step 5: Implement `python/spectra/waveforms/psk.py`**

```python
from typing import Optional

import numpy as np

from spectra._rust import (
    apply_rrc_filter,
    generate_bpsk_symbols,
    generate_qpsk_symbols,
)
from spectra.waveforms.base import Waveform


class QPSK(Waveform):
    def __init__(
        self,
        samples_per_symbol: int = 8,
        pulse_shape: str = "rrc",
        rolloff: float = 0.35,
        filter_span: int = 10,
    ):
        self.samples_per_symbol = samples_per_symbol
        self.pulse_shape = pulse_shape
        self.rolloff = rolloff
        self.filter_span = filter_span

    def generate(
        self,
        num_symbols: int,
        sample_rate: float,
        seed: Optional[int] = None,
    ) -> np.ndarray:
        s = seed if seed is not None else np.random.randint(0, 2**32)
        symbols = generate_qpsk_symbols(num_symbols, seed=s)
        filtered = apply_rrc_filter(
            symbols, self.rolloff, self.filter_span, self.samples_per_symbol
        )
        return filtered

    def bandwidth(self, sample_rate: float) -> float:
        symbol_rate = sample_rate / self.samples_per_symbol
        return symbol_rate * (1.0 + self.rolloff)

    @property
    def label(self) -> str:
        return "QPSK"


class BPSK(Waveform):
    def __init__(
        self,
        samples_per_symbol: int = 8,
        pulse_shape: str = "rrc",
        rolloff: float = 0.35,
        filter_span: int = 10,
    ):
        self.samples_per_symbol = samples_per_symbol
        self.pulse_shape = pulse_shape
        self.rolloff = rolloff
        self.filter_span = filter_span

    def generate(
        self,
        num_symbols: int,
        sample_rate: float,
        seed: Optional[int] = None,
    ) -> np.ndarray:
        s = seed if seed is not None else np.random.randint(0, 2**32)
        symbols = generate_bpsk_symbols(num_symbols, seed=s)
        filtered = apply_rrc_filter(
            symbols, self.rolloff, self.filter_span, self.samples_per_symbol
        )
        return filtered

    def bandwidth(self, sample_rate: float) -> float:
        symbol_rate = sample_rate / self.samples_per_symbol
        return symbol_rate * (1.0 + self.rolloff)

    @property
    def label(self) -> str:
        return "BPSK"
```

**Step 6: Create `python/spectra/waveforms/__init__.py`**

```python
from spectra.waveforms.psk import BPSK, QPSK

__all__ = ["BPSK", "QPSK"]
```

**Step 7: Build and run tests**

Run: `maturin develop && pytest tests/test_waveforms_psk.py -v`
Expected: All PASS

**Step 8: Commit**

```bash
git add python/spectra/waveforms/ rust/src/modulators.rs rust/src/lib.rs tests/test_waveforms_psk.py
git commit -m "feat: add BPSK and QPSK waveform generators with RRC pulse shaping"
```

---

## Task 6: Impairments Pipeline — Base, AWGN, FrequencyOffset, Compose

**Files:**
- Create: `python/spectra/impairments/__init__.py`
- Create: `python/spectra/impairments/base.py`
- Create: `python/spectra/impairments/awgn.py`
- Create: `python/spectra/impairments/frequency_offset.py`
- Create: `python/spectra/impairments/compose.py`
- Create: `python/spectra/scene/__init__.py`
- Create: `python/spectra/scene/signal_desc.py`
- Create: `tests/test_impairments.py`

**Step 1: Write failing tests**

Create `tests/test_impairments.py`:

```python
import numpy as np
import numpy.testing as npt
import pytest


class TestSignalDescription:
    def test_creation(self):
        from spectra.scene.signal_desc import SignalDescription
        desc = SignalDescription(
            t_start=0.0, t_stop=0.001,
            f_low=-5e3, f_high=5e3,
            label="QPSK", snr=20.0,
        )
        assert desc.t_start == 0.0
        assert desc.label == "QPSK"

    def test_f_center(self):
        from spectra.scene.signal_desc import SignalDescription
        desc = SignalDescription(
            t_start=0.0, t_stop=0.001,
            f_low=-5e3, f_high=5e3,
            label="QPSK", snr=20.0,
        )
        assert desc.f_center == pytest.approx(0.0)

    def test_bandwidth_property(self):
        from spectra.scene.signal_desc import SignalDescription
        desc = SignalDescription(
            t_start=0.0, t_stop=0.001,
            f_low=-5e3, f_high=5e3,
            label="QPSK", snr=20.0,
        )
        assert desc.bandwidth == pytest.approx(10e3)


class TestAWGN:
    def test_adds_noise(self, sample_rate):
        from spectra.impairments import AWGN
        from spectra.scene.signal_desc import SignalDescription
        iq = np.ones(1024, dtype=np.complex64)
        desc = SignalDescription(0.0, 0.001, -5e3, 5e3, "QPSK", 20.0)
        noisy_iq, _ = AWGN(snr=10.0)(iq, desc)
        assert not np.array_equal(iq, noisy_iq)

    def test_output_shape_preserved(self):
        from spectra.impairments import AWGN
        from spectra.scene.signal_desc import SignalDescription
        iq = np.ones(512, dtype=np.complex64)
        desc = SignalDescription(0.0, 0.001, -5e3, 5e3, "QPSK", 20.0)
        noisy_iq, _ = AWGN(snr=20.0)(iq, desc)
        assert noisy_iq.shape == iq.shape
        assert noisy_iq.dtype == np.complex64

    def test_snr_range_randomizes(self):
        from spectra.impairments import AWGN
        from spectra.scene.signal_desc import SignalDescription
        iq = np.ones(1024, dtype=np.complex64)
        desc = SignalDescription(0.0, 0.001, -5e3, 5e3, "QPSK", 20.0)
        awgn = AWGN(snr_range=(0, 30))
        results = [awgn(iq.copy(), desc)[0] for _ in range(10)]
        # Different noise levels should produce different outputs
        diffs = [np.sum(np.abs(results[i] - results[i+1])) for i in range(9)]
        assert not all(d == 0 for d in diffs)

    def test_high_snr_preserves_signal(self):
        from spectra.impairments import AWGN
        from spectra.scene.signal_desc import SignalDescription
        iq = np.ones(4096, dtype=np.complex64)
        desc = SignalDescription(0.0, 0.001, -5e3, 5e3, "QPSK", 20.0)
        noisy_iq, _ = AWGN(snr=60.0)(iq, desc)
        npt.assert_allclose(iq, noisy_iq, atol=0.01)


class TestFrequencyOffset:
    def test_applies_offset(self, sample_rate):
        from spectra.impairments import FrequencyOffset
        from spectra.scene.signal_desc import SignalDescription
        iq = np.ones(1024, dtype=np.complex64)
        desc = SignalDescription(0.0, 1024 / sample_rate, -5e3, 5e3, "QPSK", 20.0)
        offset_iq, new_desc = FrequencyOffset(offset=100.0)(iq, desc, sample_rate=sample_rate)
        # Magnitude should be preserved
        npt.assert_allclose(np.abs(offset_iq), 1.0, atol=1e-5)
        # Frequency content should shift
        assert not np.allclose(iq, offset_iq)

    def test_updates_signal_desc(self, sample_rate):
        from spectra.impairments import FrequencyOffset
        from spectra.scene.signal_desc import SignalDescription
        iq = np.ones(1024, dtype=np.complex64)
        desc = SignalDescription(0.0, 0.001, -5e3, 5e3, "QPSK", 20.0)
        _, new_desc = FrequencyOffset(offset=100.0)(iq, desc, sample_rate=sample_rate)
        assert new_desc.f_low == pytest.approx(-5e3 + 100.0)
        assert new_desc.f_high == pytest.approx(5e3 + 100.0)


class TestCompose:
    def test_chains_transforms(self, sample_rate):
        from spectra.impairments import AWGN, FrequencyOffset, Compose
        from spectra.scene.signal_desc import SignalDescription
        iq = np.ones(1024, dtype=np.complex64)
        desc = SignalDescription(0.0, 1024 / sample_rate, -5e3, 5e3, "QPSK", 20.0)
        chain = Compose([
            FrequencyOffset(offset=100.0),
            AWGN(snr=20.0),
        ])
        result_iq, result_desc = chain(iq, desc, sample_rate=sample_rate)
        assert result_iq.shape == iq.shape
        assert result_desc.f_low == pytest.approx(-5e3 + 100.0)

    def test_empty_compose(self):
        from spectra.impairments import Compose
        from spectra.scene.signal_desc import SignalDescription
        iq = np.ones(512, dtype=np.complex64)
        desc = SignalDescription(0.0, 0.001, -5e3, 5e3, "QPSK", 20.0)
        chain = Compose([])
        result_iq, result_desc = chain(iq, desc)
        npt.assert_array_equal(iq, result_iq)
```

**Step 2: Run tests to verify they fail**

Run: `pytest tests/test_impairments.py -v`
Expected: FAIL with `ModuleNotFoundError`

**Step 3: Implement `python/spectra/scene/signal_desc.py`**

```python
from dataclasses import dataclass, field
from typing import Dict


@dataclass
class SignalDescription:
    t_start: float
    t_stop: float
    f_low: float
    f_high: float
    label: str
    snr: float
    modulation_params: Dict = field(default_factory=dict)

    @property
    def f_center(self) -> float:
        return (self.f_low + self.f_high) / 2.0

    @property
    def bandwidth(self) -> float:
        return self.f_high - self.f_low

    @property
    def duration(self) -> float:
        return self.t_stop - self.t_start
```

**Step 4: Implement `python/spectra/impairments/base.py`**

```python
from abc import ABC, abstractmethod
from typing import Tuple

import numpy as np

from spectra.scene.signal_desc import SignalDescription


class Transform(ABC):
    @abstractmethod
    def __call__(
        self, iq: np.ndarray, desc: SignalDescription, **kwargs
    ) -> Tuple[np.ndarray, SignalDescription]:
        ...
```

**Step 5: Implement `python/spectra/impairments/awgn.py`**

```python
from typing import Optional, Tuple, Union

import numpy as np

from spectra.impairments.base import Transform
from spectra.scene.signal_desc import SignalDescription


class AWGN(Transform):
    def __init__(
        self,
        snr: Optional[float] = None,
        snr_range: Optional[Tuple[float, float]] = None,
    ):
        if snr is None and snr_range is None:
            raise ValueError("Must provide either snr or snr_range")
        self.snr = snr
        self.snr_range = snr_range

    def __call__(
        self, iq: np.ndarray, desc: SignalDescription, **kwargs
    ) -> Tuple[np.ndarray, SignalDescription]:
        if self.snr_range is not None:
            snr_db = np.random.uniform(*self.snr_range)
        else:
            snr_db = self.snr

        signal_power = np.mean(np.abs(iq) ** 2)
        snr_linear = 10.0 ** (snr_db / 10.0)
        noise_power = signal_power / snr_linear

        noise = np.sqrt(noise_power / 2.0) * (
            np.random.randn(len(iq)) + 1j * np.random.randn(len(iq))
        ).astype(np.complex64)

        return iq + noise, desc
```

**Step 6: Implement `python/spectra/impairments/frequency_offset.py`**

```python
from typing import Optional, Tuple

import numpy as np

from spectra.impairments.base import Transform
from spectra.scene.signal_desc import SignalDescription


class FrequencyOffset(Transform):
    def __init__(
        self,
        offset: Optional[float] = None,
        max_offset: Optional[float] = None,
    ):
        if offset is None and max_offset is None:
            raise ValueError("Must provide either offset or max_offset")
        self.offset = offset
        self.max_offset = max_offset

    def __call__(
        self, iq: np.ndarray, desc: SignalDescription, **kwargs
    ) -> Tuple[np.ndarray, SignalDescription]:
        sample_rate = kwargs.get("sample_rate")
        if sample_rate is None:
            raise ValueError("FrequencyOffset requires sample_rate kwarg")

        if self.max_offset is not None:
            fo = np.random.uniform(-self.max_offset, self.max_offset)
        else:
            fo = self.offset

        t = np.arange(len(iq)) / sample_rate
        shift = np.exp(1j * 2.0 * np.pi * fo * t).astype(np.complex64)
        shifted_iq = iq * shift

        from dataclasses import replace
        new_desc = replace(desc, f_low=desc.f_low + fo, f_high=desc.f_high + fo)
        return shifted_iq, new_desc
```

**Step 7: Implement `python/spectra/impairments/compose.py`**

```python
from typing import List, Tuple

import numpy as np

from spectra.impairments.base import Transform
from spectra.scene.signal_desc import SignalDescription


class Compose(Transform):
    def __init__(self, transforms: List[Transform]):
        self.transforms = transforms

    def __call__(
        self, iq: np.ndarray, desc: SignalDescription, **kwargs
    ) -> Tuple[np.ndarray, SignalDescription]:
        for t in self.transforms:
            iq, desc = t(iq, desc, **kwargs)
        return iq, desc
```

**Step 8: Create `__init__.py` files**

`python/spectra/impairments/__init__.py`:
```python
from spectra.impairments.awgn import AWGN
from spectra.impairments.compose import Compose
from spectra.impairments.frequency_offset import FrequencyOffset

__all__ = ["AWGN", "Compose", "FrequencyOffset"]
```

`python/spectra/scene/__init__.py`:
```python
from spectra.scene.signal_desc import SignalDescription

__all__ = ["SignalDescription"]
```

**Step 9: Run tests**

Run: `pytest tests/test_impairments.py -v`
Expected: All PASS

**Step 10: Commit**

```bash
git add python/spectra/impairments/ python/spectra/scene/ tests/test_impairments.py
git commit -m "feat: add impairments pipeline with AWGN, FrequencyOffset, and Compose"
```

---

## Task 7: Wideband Scene Compositor

**Files:**
- Create: `python/spectra/scene/composer.py`
- Create: `tests/test_scene_composer.py`

**Step 1: Write failing tests**

Create `tests/test_scene_composer.py`:

```python
import numpy as np
import pytest


class TestSceneConfig:
    def test_creation(self):
        from spectra.scene.composer import SceneConfig
        from spectra.waveforms import QPSK, BPSK
        config = SceneConfig(
            capture_duration=1e-3,
            capture_bandwidth=20e6,
            sample_rate=40e6,
            num_signals=(2, 5),
            signal_pool=[QPSK(), BPSK()],
            snr_range=(5, 25),
            allow_overlap=True,
        )
        assert config.capture_duration == 1e-3
        assert config.sample_rate == 40e6


class TestComposer:
    @pytest.fixture
    def basic_config(self):
        from spectra.scene.composer import SceneConfig
        from spectra.waveforms import QPSK, BPSK
        return SceneConfig(
            capture_duration=1e-3,
            capture_bandwidth=1e6,
            sample_rate=2e6,
            num_signals=(2, 4),
            signal_pool=[QPSK(samples_per_symbol=4), BPSK(samples_per_symbol=4)],
            snr_range=(10, 20),
            allow_overlap=True,
        )

    def test_generate_returns_iq_and_descs(self, basic_config):
        from spectra.scene.composer import Composer
        composer = Composer(basic_config)
        iq, descs = composer.generate(seed=42)
        assert isinstance(iq, np.ndarray)
        assert iq.dtype == np.complex64
        assert isinstance(descs, list)
        assert len(descs) >= 2

    def test_iq_length_matches_config(self, basic_config):
        from spectra.scene.composer import Composer
        composer = Composer(basic_config)
        iq, _ = composer.generate(seed=42)
        expected_len = int(basic_config.capture_duration * basic_config.sample_rate)
        assert len(iq) == expected_len

    def test_signal_descs_have_required_fields(self, basic_config):
        from spectra.scene.composer import Composer
        composer = Composer(basic_config)
        _, descs = composer.generate(seed=42)
        for desc in descs:
            assert hasattr(desc, "t_start")
            assert hasattr(desc, "t_stop")
            assert hasattr(desc, "f_low")
            assert hasattr(desc, "f_high")
            assert hasattr(desc, "label")
            assert hasattr(desc, "snr")

    def test_signals_within_capture_bounds(self, basic_config):
        from spectra.scene.composer import Composer
        composer = Composer(basic_config)
        _, descs = composer.generate(seed=42)
        half_bw = basic_config.capture_bandwidth / 2
        for desc in descs:
            assert desc.t_start >= 0.0
            assert desc.t_stop <= basic_config.capture_duration
            assert desc.f_low >= -half_bw
            assert desc.f_high <= half_bw

    def test_deterministic_with_seed(self, basic_config):
        from spectra.scene.composer import Composer
        composer = Composer(basic_config)
        iq1, descs1 = composer.generate(seed=42)
        iq2, descs2 = composer.generate(seed=42)
        np.testing.assert_array_equal(iq1, iq2)
        assert len(descs1) == len(descs2)
        for d1, d2 in zip(descs1, descs2):
            assert d1.label == d2.label
            assert d1.t_start == d2.t_start

    def test_multiple_signals_present(self, basic_config):
        from spectra.scene.composer import Composer
        composer = Composer(basic_config)
        _, descs = composer.generate(seed=42)
        assert len(descs) >= 2

    def test_fixed_num_signals(self):
        from spectra.scene.composer import SceneConfig, Composer
        from spectra.waveforms import QPSK
        config = SceneConfig(
            capture_duration=1e-3,
            capture_bandwidth=1e6,
            sample_rate=2e6,
            num_signals=3,
            signal_pool=[QPSK(samples_per_symbol=4)],
            snr_range=(10, 20),
            allow_overlap=True,
        )
        composer = Composer(config)
        _, descs = composer.generate(seed=42)
        assert len(descs) == 3
```

**Step 2: Run tests to verify they fail**

Run: `pytest tests/test_scene_composer.py -v`
Expected: FAIL

**Step 3: Implement `python/spectra/scene/composer.py`**

```python
from dataclasses import dataclass
from typing import List, Optional, Tuple, Union

import numpy as np

from spectra.scene.signal_desc import SignalDescription
from spectra.waveforms.base import Waveform


@dataclass
class SceneConfig:
    capture_duration: float
    capture_bandwidth: float
    sample_rate: float
    num_signals: Union[int, Tuple[int, int]]
    signal_pool: List[Waveform]
    snr_range: Tuple[float, float]
    allow_overlap: bool = True


class Composer:
    def __init__(self, config: SceneConfig):
        self.config = config

    def generate(
        self, seed: int, impairments=None
    ) -> Tuple[np.ndarray, List[SignalDescription]]:
        rng = np.random.default_rng(seed)
        cfg = self.config

        num_capture_samples = int(cfg.capture_duration * cfg.sample_rate)
        composite = np.zeros(num_capture_samples, dtype=np.complex64)
        descriptions: List[SignalDescription] = []

        # Determine number of signals
        if isinstance(cfg.num_signals, tuple):
            n_signals = rng.integers(cfg.num_signals[0], cfg.num_signals[1] + 1)
        else:
            n_signals = cfg.num_signals

        half_bw = cfg.capture_bandwidth / 2.0

        for i in range(n_signals):
            # Pick a waveform from the pool
            waveform = cfg.signal_pool[rng.integers(0, len(cfg.signal_pool))]

            # Determine signal bandwidth
            sig_bw = waveform.bandwidth(cfg.sample_rate)

            # Random center frequency within capture bandwidth
            max_center = half_bw - sig_bw / 2.0
            if max_center <= -half_bw + sig_bw / 2.0:
                f_center = 0.0
            else:
                f_center = rng.uniform(-max_center, max_center)

            # Random SNR
            snr_db = rng.uniform(*cfg.snr_range)
            snr_linear = 10.0 ** (snr_db / 10.0)

            # Determine number of symbols to fill the capture
            sps = getattr(waveform, "samples_per_symbol", 8)
            num_symbols = num_capture_samples // sps

            # Generate baseband IQ
            sig_seed = int(rng.integers(0, 2**32))
            iq = waveform.generate(
                num_symbols=num_symbols,
                sample_rate=cfg.sample_rate,
                seed=sig_seed,
            )

            # Truncate or pad to fit capture window
            if len(iq) >= num_capture_samples:
                iq = iq[:num_capture_samples]
            else:
                padded = np.zeros(num_capture_samples, dtype=np.complex64)
                # Random start time
                max_start = num_capture_samples - len(iq)
                start_idx = int(rng.integers(0, max(1, max_start)))
                padded[start_idx : start_idx + len(iq)] = iq
                iq = padded

            # Apply per-signal impairments if provided
            if impairments is not None:
                desc_temp = SignalDescription(
                    t_start=0.0,
                    t_stop=cfg.capture_duration,
                    f_low=f_center - sig_bw / 2.0,
                    f_high=f_center + sig_bw / 2.0,
                    label=waveform.label,
                    snr=snr_db,
                )
                iq, desc_temp = impairments(iq, desc_temp, sample_rate=cfg.sample_rate)

            # Frequency-shift to center frequency
            t = np.arange(len(iq)) / cfg.sample_rate
            iq = iq * np.exp(1j * 2.0 * np.pi * f_center * t).astype(np.complex64)

            # Scale to target SNR (relative to unit noise)
            sig_power = np.mean(np.abs(iq) ** 2)
            if sig_power > 0:
                iq = iq * np.sqrt(snr_linear / sig_power).astype(np.float32)

            # Find actual time extent (nonzero samples)
            nonzero = np.nonzero(np.abs(iq) > 1e-10)[0]
            if len(nonzero) > 0:
                t_start = nonzero[0] / cfg.sample_rate
                t_stop = (nonzero[-1] + 1) / cfg.sample_rate
            else:
                t_start = 0.0
                t_stop = cfg.capture_duration

            composite += iq

            descriptions.append(
                SignalDescription(
                    t_start=t_start,
                    t_stop=t_stop,
                    f_low=f_center - sig_bw / 2.0,
                    f_high=f_center + sig_bw / 2.0,
                    label=waveform.label,
                    snr=snr_db,
                )
            )

        return composite, descriptions
```

**Step 4: Update `python/spectra/scene/__init__.py`**

```python
from spectra.scene.composer import Composer, SceneConfig
from spectra.scene.signal_desc import SignalDescription

__all__ = ["Composer", "SceneConfig", "SignalDescription"]
```

**Step 5: Run tests**

Run: `pytest tests/test_scene_composer.py -v`
Expected: All PASS

**Step 6: Commit**

```bash
git add python/spectra/scene/ tests/test_scene_composer.py
git commit -m "feat: add wideband scene compositor with multi-signal overlap"
```

---

## Task 8: STFT Transform & Bounding Box Labels

**Files:**
- Create: `python/spectra/transforms/__init__.py`
- Create: `python/spectra/transforms/stft.py`
- Create: `python/spectra/scene/labels.py`
- Create: `tests/test_labels.py`

**Step 1: Write failing tests**

Create `tests/test_labels.py`:

```python
import numpy as np
import torch
import pytest


class TestSTFT:
    def test_output_shape(self):
        from spectra.transforms.stft import STFT
        stft = STFT(nfft=256, hop_length=64)
        iq = np.random.randn(4096).astype(np.float32) + \
             1j * np.random.randn(4096).astype(np.float32)
        iq = iq.astype(np.complex64)
        spectrogram = stft(iq)
        assert isinstance(spectrogram, torch.Tensor)
        assert spectrogram.ndim == 3  # [channels, freq, time]
        assert spectrogram.shape[0] == 1  # single channel (magnitude)
        assert spectrogram.shape[1] == 256  # nfft freq bins

    def test_output_is_real(self):
        from spectra.transforms.stft import STFT
        stft = STFT(nfft=128, hop_length=32)
        iq = (np.random.randn(1024) + 1j * np.random.randn(1024)).astype(np.complex64)
        spectrogram = stft(iq)
        assert spectrogram.dtype == torch.float32


class TestToCoco:
    def test_basic_conversion(self):
        from spectra.scene.signal_desc import SignalDescription
        from spectra.scene.labels import to_coco, STFTParams
        descs = [
            SignalDescription(
                t_start=0.0, t_stop=0.0005,
                f_low=-100e3, f_high=100e3,
                label="QPSK", snr=20.0,
            ),
        ]
        params = STFTParams(
            nfft=256, hop_length=64,
            sample_rate=1e6, num_samples=1000,
        )
        result = to_coco(descs, params, class_list=["QPSK", "BPSK"])
        assert "boxes" in result
        assert "labels" in result
        assert isinstance(result["boxes"], torch.Tensor)
        assert result["boxes"].shape == (1, 4)
        assert isinstance(result["labels"], torch.Tensor)
        assert result["labels"].shape == (1,)

    def test_multiple_signals(self):
        from spectra.scene.signal_desc import SignalDescription
        from spectra.scene.labels import to_coco, STFTParams
        descs = [
            SignalDescription(0.0, 0.0005, -100e3, 100e3, "QPSK", 20.0),
            SignalDescription(0.0002, 0.0008, -300e3, -100e3, "BPSK", 15.0),
        ]
        params = STFTParams(nfft=256, hop_length=64, sample_rate=1e6, num_samples=1000)
        result = to_coco(descs, params, class_list=["QPSK", "BPSK"])
        assert result["boxes"].shape == (2, 4)
        assert result["labels"].shape == (2,)

    def test_boxes_are_valid(self):
        from spectra.scene.signal_desc import SignalDescription
        from spectra.scene.labels import to_coco, STFTParams
        descs = [
            SignalDescription(0.0, 0.0005, -100e3, 100e3, "QPSK", 20.0),
        ]
        params = STFTParams(nfft=256, hop_length=64, sample_rate=1e6, num_samples=1000)
        result = to_coco(descs, params, class_list=["QPSK"])
        box = result["boxes"][0]
        # x_min < x_max, y_min < y_max
        assert box[0] < box[2]
        assert box[1] < box[3]
        # All coords non-negative
        assert torch.all(box >= 0)

    def test_class_label_indices(self):
        from spectra.scene.signal_desc import SignalDescription
        from spectra.scene.labels import to_coco, STFTParams
        descs = [
            SignalDescription(0.0, 0.0005, -100e3, 100e3, "BPSK", 20.0),
            SignalDescription(0.0, 0.0005, -300e3, -100e3, "QPSK", 15.0),
        ]
        params = STFTParams(nfft=256, hop_length=64, sample_rate=1e6, num_samples=1000)
        class_list = ["QPSK", "BPSK"]
        result = to_coco(descs, params, class_list=class_list)
        assert result["labels"][0].item() == 1  # BPSK is index 1
        assert result["labels"][1].item() == 0  # QPSK is index 0

    def test_empty_descriptions(self):
        from spectra.scene.labels import to_coco, STFTParams
        params = STFTParams(nfft=256, hop_length=64, sample_rate=1e6, num_samples=1000)
        result = to_coco([], params, class_list=["QPSK"])
        assert result["boxes"].shape == (0, 4)
        assert result["labels"].shape == (0,)
```

**Step 2: Run tests to verify they fail**

Run: `pytest tests/test_labels.py -v`
Expected: FAIL

**Step 3: Implement `python/spectra/transforms/stft.py`**

```python
import numpy as np
import torch


class STFT:
    def __init__(self, nfft: int = 256, hop_length: int = 64):
        self.nfft = nfft
        self.hop_length = hop_length

    def __call__(self, iq: np.ndarray) -> torch.Tensor:
        iq_tensor = torch.from_numpy(iq)
        window = torch.hann_window(self.nfft)
        stft_result = torch.stft(
            iq_tensor,
            n_fft=self.nfft,
            hop_length=self.hop_length,
            win_length=self.nfft,
            window=window,
            return_complex=True,
        )
        # fftshift along frequency axis so DC is centered
        stft_result = torch.fft.fftshift(stft_result, dim=0)
        magnitude = torch.abs(stft_result)
        # Return as [1, freq, time] for compatibility with image-based models
        return magnitude.unsqueeze(0).float()
```

**Step 4: Implement `python/spectra/scene/labels.py`**

```python
from dataclasses import dataclass
from typing import Dict, List

import torch

from spectra.scene.signal_desc import SignalDescription


@dataclass
class STFTParams:
    nfft: int
    hop_length: int
    sample_rate: float
    num_samples: int

    @property
    def num_time_bins(self) -> int:
        return (self.num_samples - self.nfft) // self.hop_length + 1

    @property
    def num_freq_bins(self) -> int:
        return self.nfft

    @property
    def freq_resolution(self) -> float:
        return self.sample_rate / self.nfft

    @property
    def time_resolution(self) -> float:
        return self.hop_length / self.sample_rate


def to_coco(
    signal_descs: List[SignalDescription],
    stft_params: STFTParams,
    class_list: List[str],
) -> Dict[str, torch.Tensor]:
    if len(signal_descs) == 0:
        return {
            "boxes": torch.zeros((0, 4), dtype=torch.float32),
            "labels": torch.zeros((0,), dtype=torch.int64),
            "signal_descs": [],
        }

    boxes = []
    labels = []
    p = stft_params

    for desc in signal_descs:
        # Time -> pixel coords (x axis)
        x_min = desc.t_start / p.time_resolution
        x_max = desc.t_stop / p.time_resolution

        # Frequency -> pixel coords (y axis)
        # After fftshift, frequency axis is [-fs/2, fs/2] mapped to [0, nfft]
        half_fs = p.sample_rate / 2.0
        y_min = (desc.f_low + half_fs) / p.sample_rate * p.nfft
        y_max = (desc.f_high + half_fs) / p.sample_rate * p.nfft

        # Clamp to spectrogram bounds
        x_min = max(0.0, x_min)
        x_max = min(float(p.num_time_bins), x_max)
        y_min = max(0.0, y_min)
        y_max = min(float(p.num_freq_bins), y_max)

        boxes.append([x_min, y_min, x_max, y_max])
        labels.append(class_list.index(desc.label))

    return {
        "boxes": torch.tensor(boxes, dtype=torch.float32),
        "labels": torch.tensor(labels, dtype=torch.int64),
        "signal_descs": signal_descs,
    }
```

**Step 5: Create `python/spectra/transforms/__init__.py`**

```python
from spectra.transforms.stft import STFT

__all__ = ["STFT"]
```

**Step 6: Update `python/spectra/scene/__init__.py`** to export labels

```python
from spectra.scene.composer import Composer, SceneConfig
from spectra.scene.labels import STFTParams, to_coco
from spectra.scene.signal_desc import SignalDescription

__all__ = ["Composer", "SceneConfig", "SignalDescription", "STFTParams", "to_coco"]
```

**Step 7: Run tests**

Run: `pytest tests/test_labels.py -v`
Expected: All PASS

**Step 8: Commit**

```bash
git add python/spectra/transforms/ python/spectra/scene/labels.py tests/test_labels.py
git commit -m "feat: add STFT transform and bounding box label conversion"
```

---

## Task 9: PyTorch Datasets — Narrowband & Wideband

**Files:**
- Create: `python/spectra/datasets/__init__.py`
- Create: `python/spectra/datasets/narrowband.py`
- Create: `python/spectra/datasets/wideband.py`
- Create: `tests/test_datasets.py`

**Step 1: Write failing tests**

Create `tests/test_datasets.py`:

```python
import numpy as np
import torch
import pytest


class TestNarrowbandDataset:
    def test_len(self):
        from spectra.datasets import NarrowbandDataset
        from spectra.waveforms import QPSK, BPSK
        ds = NarrowbandDataset(
            waveform_pool=[QPSK(samples_per_symbol=4), BPSK(samples_per_symbol=4)],
            num_samples=100,
            num_iq_samples=1024,
            sample_rate=1e6,
            seed=42,
        )
        assert len(ds) == 100

    def test_getitem_returns_tensor_and_label(self):
        from spectra.datasets import NarrowbandDataset
        from spectra.waveforms import QPSK, BPSK
        ds = NarrowbandDataset(
            waveform_pool=[QPSK(samples_per_symbol=4), BPSK(samples_per_symbol=4)],
            num_samples=100,
            num_iq_samples=1024,
            sample_rate=1e6,
            seed=42,
        )
        data, label = ds[0]
        assert isinstance(data, torch.Tensor)
        assert isinstance(label, int)
        assert data.shape == (2, 1024)  # I and Q channels
        assert data.dtype == torch.float32

    def test_deterministic(self):
        from spectra.datasets import NarrowbandDataset
        from spectra.waveforms import QPSK
        ds = NarrowbandDataset(
            waveform_pool=[QPSK(samples_per_symbol=4)],
            num_samples=10,
            num_iq_samples=512,
            sample_rate=1e6,
            seed=42,
        )
        d1, l1 = ds[0]
        d2, l2 = ds[0]
        torch.testing.assert_close(d1, d2)
        assert l1 == l2

    def test_different_indices_differ(self):
        from spectra.datasets import NarrowbandDataset
        from spectra.waveforms import QPSK, BPSK
        ds = NarrowbandDataset(
            waveform_pool=[QPSK(samples_per_symbol=4), BPSK(samples_per_symbol=4)],
            num_samples=100,
            num_iq_samples=512,
            sample_rate=1e6,
            seed=42,
        )
        d0, _ = ds[0]
        d1, _ = ds[1]
        assert not torch.equal(d0, d1)

    def test_with_dataloader(self):
        from spectra.datasets import NarrowbandDataset
        from spectra.waveforms import QPSK
        from torch.utils.data import DataLoader
        ds = NarrowbandDataset(
            waveform_pool=[QPSK(samples_per_symbol=4)],
            num_samples=32,
            num_iq_samples=256,
            sample_rate=1e6,
            seed=42,
        )
        loader = DataLoader(ds, batch_size=8)
        batch_data, batch_labels = next(iter(loader))
        assert batch_data.shape == (8, 2, 256)
        assert batch_labels.shape == (8,)


class TestWidebandDataset:
    @pytest.fixture
    def wideband_ds(self):
        from spectra.datasets import WidebandDataset
        from spectra.scene.composer import SceneConfig
        from spectra.waveforms import QPSK, BPSK
        from spectra.transforms.stft import STFT
        config = SceneConfig(
            capture_duration=1e-3,
            capture_bandwidth=1e6,
            sample_rate=2e6,
            num_signals=(2, 4),
            signal_pool=[QPSK(samples_per_symbol=4), BPSK(samples_per_symbol=4)],
            snr_range=(10, 20),
            allow_overlap=True,
        )
        return WidebandDataset(
            scene_config=config,
            num_samples=50,
            transform=STFT(nfft=128, hop_length=32),
            seed=42,
        )

    def test_len(self, wideband_ds):
        assert len(wideband_ds) == 50

    def test_getitem_returns_tensor_and_targets(self, wideband_ds):
        data, targets = wideband_ds[0]
        assert isinstance(data, torch.Tensor)
        assert data.ndim == 3  # [C, freq, time]
        assert isinstance(targets, dict)
        assert "boxes" in targets
        assert "labels" in targets
        assert "signal_descs" in targets

    def test_deterministic(self, wideband_ds):
        d1, t1 = wideband_ds[0]
        d2, t2 = wideband_ds[0]
        torch.testing.assert_close(d1, d2)
        torch.testing.assert_close(t1["boxes"], t2["boxes"])

    def test_boxes_match_signal_count(self, wideband_ds):
        _, targets = wideband_ds[0]
        num_signals = len(targets["signal_descs"])
        assert targets["boxes"].shape[0] == num_signals
        assert targets["labels"].shape[0] == num_signals


class TestCollate:
    def test_collate_fn(self):
        from spectra.datasets import collate_fn
        # Simulate two samples with different numbers of boxes
        batch = [
            (
                torch.randn(1, 64, 32),
                {"boxes": torch.randn(3, 4), "labels": torch.tensor([0, 1, 0]), "signal_descs": []},
            ),
            (
                torch.randn(1, 64, 32),
                {"boxes": torch.randn(2, 4), "labels": torch.tensor([1, 1]), "signal_descs": []},
            ),
        ]
        data, targets = collate_fn(batch)
        assert data.shape == (2, 1, 64, 32)
        assert isinstance(targets, list)
        assert len(targets) == 2
        assert targets[0]["boxes"].shape == (3, 4)
        assert targets[1]["boxes"].shape == (2, 4)
```

**Step 2: Run tests to verify they fail**

Run: `pytest tests/test_datasets.py -v`
Expected: FAIL

**Step 3: Implement `python/spectra/datasets/narrowband.py`**

```python
from typing import Callable, List, Optional, Tuple

import numpy as np
import torch
from torch.utils.data import Dataset

from spectra.impairments.compose import Compose
from spectra.waveforms.base import Waveform


class NarrowbandDataset(Dataset):
    def __init__(
        self,
        waveform_pool: List[Waveform],
        num_samples: int,
        num_iq_samples: int,
        sample_rate: float,
        impairments: Optional[Compose] = None,
        transform: Optional[Callable] = None,
        target_transform: Optional[Callable] = None,
        seed: Optional[int] = None,
    ):
        self.waveform_pool = waveform_pool
        self.num_samples = num_samples
        self.num_iq_samples = num_iq_samples
        self.sample_rate = sample_rate
        self.impairments = impairments
        self.transform = transform
        self.target_transform = target_transform
        self.seed = seed if seed is not None else 0

    def __len__(self) -> int:
        return self.num_samples

    def __getitem__(self, idx: int) -> Tuple[torch.Tensor, int]:
        rng = np.random.default_rng(seed=(self.seed, idx))

        # Pick a waveform class
        waveform_idx = int(rng.integers(0, len(self.waveform_pool)))
        waveform = self.waveform_pool[waveform_idx]

        # Generate enough symbols
        sps = getattr(waveform, "samples_per_symbol", 8)
        num_symbols = self.num_iq_samples // sps + 1
        sig_seed = int(rng.integers(0, 2**32))

        iq = waveform.generate(
            num_symbols=num_symbols,
            sample_rate=self.sample_rate,
            seed=sig_seed,
        )

        # Truncate to requested length
        iq = iq[: self.num_iq_samples]
        if len(iq) < self.num_iq_samples:
            padded = np.zeros(self.num_iq_samples, dtype=np.complex64)
            padded[: len(iq)] = iq
            iq = padded

        # Apply impairments
        if self.impairments is not None:
            from spectra.scene.signal_desc import SignalDescription
            desc = SignalDescription(
                t_start=0.0,
                t_stop=self.num_iq_samples / self.sample_rate,
                f_low=-waveform.bandwidth(self.sample_rate) / 2,
                f_high=waveform.bandwidth(self.sample_rate) / 2,
                label=waveform.label,
                snr=0.0,
            )
            iq, _ = self.impairments(iq, desc, sample_rate=self.sample_rate)

        # Convert to tensor: [2, num_iq_samples] (I and Q channels)
        if self.transform is not None:
            data = self.transform(iq)
        else:
            data = torch.tensor(
                np.stack([iq.real, iq.imag]), dtype=torch.float32
            )

        label = waveform_idx
        if self.target_transform is not None:
            label = self.target_transform(label)

        return data, label
```

**Step 4: Implement `python/spectra/datasets/wideband.py`**

```python
from typing import Callable, Dict, List, Optional, Tuple

import numpy as np
import torch
from torch.utils.data import Dataset

from spectra.impairments.compose import Compose
from spectra.scene.composer import Composer, SceneConfig
from spectra.scene.labels import STFTParams, to_coco


class WidebandDataset(Dataset):
    def __init__(
        self,
        scene_config: SceneConfig,
        num_samples: int,
        impairments: Optional[Compose] = None,
        transform: Optional[Callable] = None,
        target_transform: Optional[Callable] = None,
        seed: Optional[int] = None,
    ):
        self.scene_config = scene_config
        self.num_samples = num_samples
        self.impairments = impairments
        self.transform = transform
        self.target_transform = target_transform
        self.seed = seed if seed is not None else 0
        self.composer = Composer(scene_config)

        # Build class list from signal pool
        self.class_list = sorted(set(w.label for w in scene_config.signal_pool))

    def __len__(self) -> int:
        return self.num_samples

    def __getitem__(self, idx: int) -> Tuple[torch.Tensor, Dict]:
        # Deterministic seed from (base_seed, idx)
        rng = np.random.default_rng(seed=(self.seed, idx))
        scene_seed = int(rng.integers(0, 2**32))

        iq, signal_descs = self.composer.generate(seed=scene_seed)

        # Apply scene-level impairments
        if self.impairments is not None:
            from spectra.scene.signal_desc import SignalDescription
            scene_desc = SignalDescription(
                t_start=0.0,
                t_stop=self.scene_config.capture_duration,
                f_low=-self.scene_config.capture_bandwidth / 2,
                f_high=self.scene_config.capture_bandwidth / 2,
                label="scene",
                snr=0.0,
            )
            iq, _ = self.impairments(iq, scene_desc, sample_rate=self.scene_config.sample_rate)

        # Apply transform (e.g., STFT)
        if self.transform is not None:
            data = self.transform(iq)
            # Build STFT params for label conversion
            stft = self.transform
            stft_params = STFTParams(
                nfft=stft.nfft,
                hop_length=stft.hop_length,
                sample_rate=self.scene_config.sample_rate,
                num_samples=len(iq),
            )
            targets = to_coco(signal_descs, stft_params, self.class_list)
        else:
            data = torch.tensor(
                np.stack([iq.real, iq.imag]), dtype=torch.float32
            )
            targets = {"boxes": torch.zeros((0, 4)), "labels": torch.zeros((0,), dtype=torch.int64), "signal_descs": signal_descs}

        targets["signal_descs"] = signal_descs

        if self.target_transform is not None:
            targets = self.target_transform(targets)

        return data, targets
```

**Step 5: Create `python/spectra/datasets/__init__.py`**

```python
from typing import Dict, List, Tuple

import torch

from spectra.datasets.narrowband import NarrowbandDataset
from spectra.datasets.wideband import WidebandDataset


def collate_fn(
    batch: List[Tuple[torch.Tensor, Dict]],
) -> Tuple[torch.Tensor, List[Dict]]:
    """Custom collate for variable-length detection targets."""
    data = torch.stack([item[0] for item in batch])
    targets = [item[1] for item in batch]
    return data, targets


__all__ = ["NarrowbandDataset", "WidebandDataset", "collate_fn"]
```

**Step 6: Run tests**

Run: `maturin develop && pytest tests/test_datasets.py -v`
Expected: All PASS

**Step 7: Commit**

```bash
git add python/spectra/datasets/ tests/test_datasets.py
git commit -m "feat: add NarrowbandDataset and WidebandDataset with DataLoader support"
```

---

## Task 10: Update Package `__init__.py` & Run Full Test Suite

**Files:**
- Modify: `python/spectra/__init__.py`

**Step 1: Update `python/spectra/__init__.py` with public API exports**

```python
from spectra._rust import __version__
from spectra.waveforms import BPSK, QPSK
from spectra.scene import Composer, SceneConfig, SignalDescription, STFTParams, to_coco
from spectra.impairments import AWGN, Compose, FrequencyOffset
from spectra.datasets import NarrowbandDataset, WidebandDataset, collate_fn
from spectra.transforms import STFT

__all__ = [
    "__version__",
    "BPSK",
    "QPSK",
    "Composer",
    "SceneConfig",
    "SignalDescription",
    "STFTParams",
    "to_coco",
    "AWGN",
    "Compose",
    "FrequencyOffset",
    "NarrowbandDataset",
    "WidebandDataset",
    "collate_fn",
    "STFT",
]
```

**Step 2: Run the full test suite**

Run: `maturin develop && pytest tests/ -v`
Expected: All tests PASS

**Step 3: Commit**

```bash
git add python/spectra/__init__.py
git commit -m "feat: export full public API from spectra package"
```

---

## Task 11: GitHub Actions CI

**Files:**
- Create: `.github/workflows/ci.yml`

**Step 1: Create CI workflow**

```yaml
name: CI

on:
  push:
    branches: [main]
  pull_request:
    branches: [main]

env:
  CARGO_TERM_COLOR: always

jobs:
  rust-checks:
    runs-on: ubuntu-latest
    steps:
      - uses: actions/checkout@v4
      - uses: dtolnay/rust-toolchain@stable
        with:
          components: clippy, rustfmt
      - uses: actions/cache@v4
        with:
          path: |
            ~/.cargo/registry
            ~/.cargo/git
            rust/target
          key: ${{ runner.os }}-cargo-${{ hashFiles('rust/Cargo.lock') }}
      - run: cargo fmt --manifest-path rust/Cargo.toml --all -- --check
      - run: cargo clippy --manifest-path rust/Cargo.toml --all-targets -- -D warnings
      - run: cargo test --manifest-path rust/Cargo.toml

  python-tests:
    runs-on: ${{ matrix.os }}
    strategy:
      fail-fast: false
      matrix:
        os: [ubuntu-latest, macos-latest]
        python-version: ["3.10", "3.11", "3.12"]
    steps:
      - uses: actions/checkout@v4
      - uses: actions/setup-python@v5
        with:
          python-version: ${{ matrix.python-version }}
      - uses: dtolnay/rust-toolchain@stable
      - uses: actions/cache@v4
        with:
          path: |
            ~/.cargo/registry
            ~/.cargo/git
            rust/target
          key: ${{ runner.os }}-${{ matrix.python-version }}-cargo-${{ hashFiles('rust/Cargo.lock') }}
      - run: |
          python -m pip install --upgrade pip
          pip install maturin pytest numpy
          pip install torch --index-url https://download.pytorch.org/whl/cpu
      - run: maturin develop --release
      - run: pytest tests/ -v
```

**Step 2: Commit**

```bash
git add .github/workflows/ci.yml
git commit -m "ci: add GitHub Actions for Rust checks and Python tests"
```

---

## Summary of Tasks

| Task | Component | Key Deliverable |
|------|-----------|-----------------|
| 1 | Scaffolding | pyproject.toml, Cargo.toml, PyO3 module, project structure |
| 2 | Rust modulators | QPSK symbol generation with deterministic seeding |
| 3 | Rust filters | RRC pulse-shaping filter (upsample + convolve) |
| 4 | Rust oscillators | Chirp and tone generation |
| 5 | Python waveforms | Waveform base class, BPSK, QPSK with layered API |
| 6 | Impairments | Transform base, AWGN, FrequencyOffset, Compose pipeline |
| 7 | Scene compositor | Multi-signal wideband scene generation with overlap |
| 8 | Labels | STFT transform, physical-to-pixel bounding box conversion |
| 9 | Datasets | NarrowbandDataset, WidebandDataset, DataLoader integration |
| 10 | Package API | Public exports, full test suite verification |
| 11 | CI | GitHub Actions for Rust and Python testing |

Each task follows TDD: write failing tests, implement, verify pass, commit.
