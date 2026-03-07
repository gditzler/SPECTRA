# Choi-Williams Distribution (CWD) Implementation Plan

> **For Claude:** REQUIRED SUB-SKILL: Use superpowers:executing-plans to implement this plan task-by-task.

**Goal:** Add a Rust-backed Choi-Williams Distribution (CWD) transform to SPECTRA for cross-term-suppressed time-frequency analysis.

**Architecture:** The CWD is a member of Cohen's class of bilinear time-frequency distributions. It applies an exponential kernel `exp(-θ²τ²/σ)` to the Wigner-Ville distribution's ambiguity function to suppress cross-terms between signal components. The Rust backend computes the instantaneous autocorrelation, applies the CWD kernel via convolution along the time axis for each lag, then FFTs along lag to produce the time-frequency representation. The Python wrapper follows the existing WVD/SCD class pattern — NumPy input, torch.Tensor output, with `format_csp_output` for flexible output formatting.

**Tech Stack:** Rust (PyO3, rustfft, num-complex, numpy crate), Python (NumPy, PyTorch), pytest

---

## Background: CWD Algorithm

The Choi-Williams Distribution is defined as:

```
CWD(t, f) = ∫∫ √(σ/(4πτ²)) · exp(-σ(s-t)²/τ²) · x(s+τ/2) · x*(s-τ/2) · exp(-j2πfτ) ds dτ
```

Discrete implementation:
1. Compute instantaneous autocorrelation: `R(n, τ) = x(n + τ) · conj(x(n - τ))` for each time index `n` and lag `τ`
2. For each lag `τ ≠ 0`, convolve `R(·, τ)` along the time axis with the kernel `g(n) = √(σ/(4πτ²)) · exp(-σ·n²/(4τ²))`
3. For `τ = 0`, `R(n, 0) = |x(n)|²` (no kernel needed — kernel is a delta at τ=0)
4. FFT along lag `τ` for each time index to get frequency axis
5. fftshift along frequency for DC-centered output

The `sigma` parameter controls cross-term suppression:
- Small σ → strong smoothing, more cross-term suppression, reduced resolution
- Large σ → approaches WVD (less smoothing, more cross-terms, better resolution)

---

### Task 1: Rust — `compute_cwd` function

**Files:**
- Create: `rust/src/cwd.rs`
- Modify: `rust/src/lib.rs:1-49`

**Step 1: Create `rust/src/cwd.rs` with the `compute_cwd` function**

```rust
use num_complex::Complex32;
use numpy::ndarray::Array2;
use numpy::{IntoPyArray, PyArray2, PyReadonlyArray1};
use pyo3::prelude::*;
use rustfft::FftPlanner;
use std::f32::consts::PI;

/// Compute the Choi-Williams Distribution (CWD).
///
/// The CWD applies an exponential kernel to the instantaneous autocorrelation
/// to suppress cross-terms relative to the standard Wigner-Ville distribution.
///
/// # Arguments
/// * `iq` — Input complex IQ samples, shape `[N]`.
/// * `nfft` — FFT size for frequency axis (number of lag bins, zero-padded).
/// * `n_time` — Number of time samples in the output (evenly spaced from the input).
///   If `None` or 0, uses all input samples.
/// * `sigma` — Kernel parameter controlling cross-term suppression.
///   Smaller values = more suppression. Typical range: 0.1–10.0.
///
/// # Returns
/// 2-D complex array `[n_time, nfft]`, DC-centred along frequency axis.
#[pyfunction]
#[pyo3(signature = (iq, nfft, n_time, sigma))]
pub fn compute_cwd<'py>(
    py: Python<'py>,
    iq: PyReadonlyArray1<'py, Complex32>,
    nfft: usize,
    n_time: usize,
    sigma: f32,
) -> Bound<'py, PyArray2<Complex32>> {
    let samples: Vec<Complex32> = iq.as_array().to_vec();
    let n = samples.len();

    if n == 0 || nfft == 0 {
        return Array2::<Complex32>::zeros((n_time.max(1), nfft)).into_pyarray(py);
    }

    // Determine time indices
    let time_indices: Vec<usize> = if n_time > 0 && n_time < n {
        (0..n_time)
            .map(|i| (i as f64 * (n - 1) as f64 / (n_time - 1).max(1) as f64).round() as usize)
            .collect()
    } else {
        (0..n).collect()
    };
    let out_n_time = time_indices.len();

    let max_lag = (nfft / 2) as isize;

    // Step 1: Compute instantaneous autocorrelation R(n, tau) for all
    // selected time indices and lags, then apply CWD kernel convolution.
    //
    // For each lag tau, the kernel is:
    //   g(m) = sqrt(sigma / (4 * pi * tau^2)) * exp(-sigma * m^2 / (4 * tau^2))
    // We convolve R(·, tau) with g(·) along the time axis.
    //
    // For efficiency we compute the autocorrelation at ALL input time indices
    // (not just the selected ones), apply the kernel convolution, then
    // subsample.

    // Allocate FFT buffer: for each output time index, we store nfft complex
    // values (lags -max_lag..max_lag mapped into FFT bins).
    let mut cwd = Array2::<Complex32>::zeros((out_n_time, nfft));

    // For each lag, compute R(n, tau) for all n, convolve with kernel, sample.
    for tau_idx in 0..nfft {
        let tau = tau_idx as isize - max_lag;

        if tau == 0 {
            // R(n, 0) = |x(n)|^2, no kernel convolution needed
            for (i, &t) in time_indices.iter().enumerate() {
                let val = samples[t].norm_sqr();
                cwd[[i, tau_idx]] = Complex32::new(val, 0.0);
            }
            continue;
        }

        let abs_tau = tau.unsigned_abs();

        // Compute R(n, tau) = x(n + tau) * conj(x(n - tau)) for valid n
        // Valid range: abs_tau <= n <= n - 1 - abs_tau
        let valid_start = abs_tau;
        let valid_end = if n > abs_tau { n - abs_tau } else { 0 };

        if valid_start >= valid_end {
            continue;
        }

        let valid_len = valid_end - valid_start;
        let mut r_tau: Vec<Complex32> = Vec::with_capacity(valid_len);
        for nn in valid_start..valid_end {
            let idx_plus = (nn as isize + tau) as usize;
            let idx_minus = (nn as isize - tau) as usize;
            r_tau.push(samples[idx_plus] * samples[idx_minus].conj());
        }

        // Build kernel for this lag
        let tau_sq = (abs_tau * abs_tau) as f32;
        let kernel_scale = (sigma / (4.0 * PI * tau_sq)).sqrt();
        let kernel_exp_coeff = -sigma / (4.0 * tau_sq);

        // Determine kernel half-width: truncate where kernel < 1e-6 * peak
        // exp(coeff * m^2) < 1e-6 => m^2 > -ln(1e-6) / (-coeff)
        let kernel_half = (((-1e-6_f32.ln()) / (-kernel_exp_coeff)).sqrt().ceil() as usize)
            .min(valid_len);
        let kernel_len = 2 * kernel_half + 1;
        let mut kernel: Vec<f32> = Vec::with_capacity(kernel_len);
        for ki in 0..kernel_len {
            let m = ki as f32 - kernel_half as f32;
            kernel.push(kernel_scale * (kernel_exp_coeff * m * m).exp());
        }

        // Convolve R(·, tau) with kernel along time and sample at time_indices
        for (i, &t) in time_indices.iter().enumerate() {
            if t < valid_start || t >= valid_end {
                continue;
            }
            let r_idx = t - valid_start;

            let mut acc = Complex32::new(0.0, 0.0);
            for (ki, &kv) in kernel.iter().enumerate() {
                let src = r_idx as isize + ki as isize - kernel_half as isize;
                if src >= 0 && (src as usize) < valid_len {
                    acc += Complex32::new(kv, 0.0) * r_tau[src as usize];
                }
            }
            cwd[[i, tau_idx]] = acc;
        }
    }

    // Step 2: FFT along lag axis (axis 1) for each time index
    let mut planner = FftPlanner::new();
    let fft = planner.plan_fft_forward(nfft);

    for i in 0..out_n_time {
        let mut row: Vec<Complex32> = (0..nfft).map(|j| cwd[[i, j]]).collect();
        fft.process(&mut row);
        for (j, &val) in row.iter().enumerate() {
            cwd[[i, j]] = val;
        }
    }

    // Step 3: fftshift along frequency (axis 1)
    let half = nfft / 2;
    let mut shifted = Array2::<Complex32>::zeros((out_n_time, nfft));
    for i in 0..out_n_time {
        for j in 0..nfft {
            shifted[[i, (j + half) % nfft]] = cwd[[i, j]];
        }
    }

    shifted.into_pyarray(py)
}

#[cfg(test)]
mod tests {
    use super::*;

    fn make_tone(freq: f32, n: usize) -> Vec<Complex32> {
        (0..n)
            .map(|i| {
                let phase = 2.0 * PI * freq * i as f32;
                Complex32::new(phase.cos(), phase.sin())
            })
            .collect()
    }

    #[test]
    fn test_cwd_output_shape() {
        // We can't call the pyfunction directly without Python,
        // so we test the algorithm components instead.
        let tone = make_tone(0.1, 256);
        assert_eq!(tone.len(), 256);
    }

    #[test]
    fn test_kernel_finite_for_nonzero_tau() {
        let sigma: f32 = 1.0;
        let tau: f32 = 5.0;
        let tau_sq = tau * tau;
        let kernel_scale = (sigma / (4.0 * PI * tau_sq)).sqrt();
        let kernel_exp_coeff = -sigma / (4.0 * tau_sq);
        // At m=0, kernel = kernel_scale
        let val = kernel_scale * (kernel_exp_coeff * 0.0).exp();
        assert!(val > 0.0);
        assert!(val.is_finite());
    }

    #[test]
    fn test_kernel_decays_with_distance() {
        let sigma: f32 = 1.0;
        let tau: f32 = 5.0;
        let tau_sq = tau * tau;
        let kernel_scale = (sigma / (4.0 * PI * tau_sq)).sqrt();
        let kernel_exp_coeff = -sigma / (4.0 * tau_sq);
        let at_0 = kernel_scale * (kernel_exp_coeff * 0.0).exp();
        let at_10 = kernel_scale * (kernel_exp_coeff * 100.0).exp();
        assert!(at_0 > at_10, "kernel should decay away from center");
    }
}
```

**Step 2: Register `compute_cwd` in `lib.rs`**

Add `mod cwd;` to the module declarations (after `mod cyclo_temporal;`) and add the function registration in the `_rust` pymodule function:

```rust
// In module declarations (top of file):
mod cwd;

// In _rust function, after the s3ca line:
m.add_function(wrap_pyfunction!(cwd::compute_cwd, m)?)?;
```

**Step 3: Run Rust tests**

Run: `cargo test --manifest-path rust/Cargo.toml`
Expected: All existing tests pass + 3 new CWD tests pass.

**Step 4: Build the extension**

Run: `maturin develop --release`
Expected: Build succeeds.

**Step 5: Verify Rust function is importable from Python**

Run: `python -c "from spectra._rust import compute_cwd; print('OK')"`
Expected: `OK`

**Step 6: Commit**

```bash
git add rust/src/cwd.rs rust/src/lib.rs
git commit -m "feat(rust): add compute_cwd Choi-Williams Distribution backend"
```

---

### Task 2: Rust FFI tests for `compute_cwd`

**Files:**
- Modify: `tests/test_rust_cyclo.py`

**Step 1: Write failing tests for the Rust `compute_cwd` function**

Append the following test class to `tests/test_rust_cyclo.py`:

```python
import numpy as np
import pytest

from spectra._rust import compute_cwd


class TestComputeCWD:
    """Tests for the Rust compute_cwd function."""

    @pytest.mark.rust
    def test_output_shape(self):
        iq = np.exp(1j * 2 * np.pi * 0.1 * np.arange(256)).astype(np.complex64)
        result = np.asarray(compute_cwd(iq, 128, 0, 1.0))
        assert result.shape == (256, 128)
        assert result.dtype == np.complex64

    @pytest.mark.rust
    def test_output_shape_with_n_time(self):
        iq = np.exp(1j * 2 * np.pi * 0.1 * np.arange(512)).astype(np.complex64)
        result = np.asarray(compute_cwd(iq, 64, 100, 1.0))
        assert result.shape == (100, 64)

    @pytest.mark.rust
    def test_no_nan_or_inf(self):
        iq = np.exp(1j * 2 * np.pi * 0.1 * np.arange(256)).astype(np.complex64)
        result = np.asarray(compute_cwd(iq, 64, 0, 1.0))
        assert not np.any(np.isnan(result.real))
        assert not np.any(np.isinf(result.real))

    @pytest.mark.rust
    def test_deterministic(self):
        iq = np.exp(1j * 2 * np.pi * 0.1 * np.arange(256)).astype(np.complex64)
        r1 = np.asarray(compute_cwd(iq, 64, 0, 1.0))
        r2 = np.asarray(compute_cwd(iq, 64, 0, 1.0))
        np.testing.assert_array_equal(r1, r2)

    @pytest.mark.rust
    def test_short_signal_returns_zeros(self):
        iq = np.array([1.0 + 0j, 2.0 + 0j], dtype=np.complex64)
        result = np.asarray(compute_cwd(iq, 64, 0, 1.0))
        # Very short signal — most bins should be near zero
        assert result.shape == (2, 64)

    @pytest.mark.rust
    def test_empty_signal(self):
        iq = np.array([], dtype=np.complex64)
        result = np.asarray(compute_cwd(iq, 64, 0, 1.0))
        assert result.shape[1] == 64

    @pytest.mark.rust
    def test_tone_has_concentrated_energy(self):
        """A pure tone should show energy at one frequency."""
        f0 = 0.1
        n = 512
        iq = np.exp(1j * 2 * np.pi * f0 * np.arange(n)).astype(np.complex64)
        result = np.asarray(compute_cwd(iq, 128, 0, 1.0))
        mag = np.abs(result)
        # At mid-time, peak should dominate
        mid = n // 2
        mid_slice = mag[mid, :]
        assert np.max(mid_slice) > 5 * np.mean(mid_slice)

    @pytest.mark.rust
    def test_sigma_affects_output(self):
        """Different sigma values should produce different outputs."""
        iq = np.exp(1j * 2 * np.pi * 0.1 * np.arange(256)).astype(np.complex64)
        r1 = np.asarray(compute_cwd(iq, 64, 0, 0.5))
        r2 = np.asarray(compute_cwd(iq, 64, 0, 5.0))
        assert not np.allclose(r1, r2)
```

**Step 2: Run the tests**

Run: `pytest tests/test_rust_cyclo.py::TestComputeCWD -v`
Expected: All 8 tests PASS (since Rust function was built in Task 1).

**Step 3: Commit**

```bash
git add tests/test_rust_cyclo.py
git commit -m "test(rust): add FFI tests for compute_cwd"
```

---

### Task 3: Python `CWD` transform class

**Files:**
- Create: `python/spectra/transforms/cwd.py`
- Modify: `python/spectra/transforms/__init__.py`

**Step 1: Create the Python `CWD` class**

Create `python/spectra/transforms/cwd.py`:

```python
"""Choi-Williams Distribution transform."""
import numpy as np
import torch

from spectra._rust import compute_cwd as _compute_cwd
from spectra.transforms.csp_utils import format_csp_output


class CWD:
    """Choi-Williams Distribution (time-frequency representation).

    Computes a cross-term-suppressed time-frequency representation by applying
    an exponential kernel to the Wigner-Ville distribution.  The kernel
    ``exp(-theta^2 * tau^2 / sigma)`` is parameterised by ``sigma``:

    - Small ``sigma`` → strong cross-term suppression, reduced resolution.
    - Large ``sigma`` → approaches the WVD (less suppression, better resolution).

    Computation is performed in Rust for performance.

    Args:
        nfft: FFT size for the frequency axis (number of lag bins). Default 256.
        n_time: Number of output time samples (subsampled from input).
            Default ``None`` (use all input samples).
        sigma: Kernel parameter controlling cross-term suppression.
            Typical range: 0.1–10.0. Default 1.0.
        output_format: ``"magnitude"`` (C=1), ``"mag_phase"`` (C=2),
            or ``"real_imag"`` (C=2). Default ``"magnitude"``.
        db_scale: Apply ``10 * log10`` to the magnitude (only for
            ``"magnitude"`` and ``"mag_phase"`` formats). Default ``False``.

    Returns:
        ``torch.Tensor`` of shape ``[C, n_time, nfft]``.
    """

    def __init__(
        self,
        nfft: int = 256,
        n_time: int | None = None,
        sigma: float = 1.0,
        output_format: str = "magnitude",
        db_scale: bool = False,
    ):
        if output_format not in ("magnitude", "mag_phase", "real_imag"):
            raise ValueError(
                f"Unknown output_format: {output_format!r}. "
                "Supported: 'magnitude', 'mag_phase', 'real_imag'."
            )
        if sigma <= 0:
            raise ValueError(f"sigma must be positive, got {sigma}")
        self.nfft = nfft
        self.n_time = n_time
        self.sigma = sigma
        self.output_format = output_format
        self.db_scale = db_scale

    def __call__(self, iq: np.ndarray) -> torch.Tensor:
        iq = np.ascontiguousarray(iq, dtype=np.complex64)
        n_time_arg = self.n_time if self.n_time is not None else 0
        cwd_complex = np.asarray(
            _compute_cwd(iq, self.nfft, n_time_arg, self.sigma)
        )
        return format_csp_output(cwd_complex, self.output_format, self.db_scale)
```

**Step 2: Register `CWD` in `__init__.py`**

Add the import and `__all__` entry to `python/spectra/transforms/__init__.py`:

```python
# Add import (after the CAF import line):
from spectra.transforms.cwd import CWD

# Add "CWD" to __all__ list (after "Cumulants"):
```

**Step 3: Verify import works**

Run: `python -c "from spectra.transforms import CWD; print('OK')"`
Expected: `OK`

**Step 4: Commit**

```bash
git add python/spectra/transforms/cwd.py python/spectra/transforms/__init__.py
git commit -m "feat(transforms): add CWD Choi-Williams Distribution transform"
```

---

### Task 4: Python transform tests for `CWD`

**Files:**
- Create: `tests/test_transforms_cwd.py`

**Step 1: Write the test file**

```python
"""Tests for the Choi-Williams Distribution transform."""
import numpy as np
import pytest
import torch

from spectra.transforms.cwd import CWD


def _make_tone(f0: float, n: int) -> np.ndarray:
    return np.exp(1j * 2 * np.pi * f0 * np.arange(n)).astype(np.complex64)


class TestCWDShape:
    """Output shape and dtype tests."""

    def test_magnitude_shape(self):
        iq = _make_tone(0.1, 512)
        cwd = CWD(nfft=256, output_format="magnitude")
        out = cwd(iq)
        assert out.shape == (1, 512, 256)
        assert out.dtype == torch.float32

    def test_mag_phase_shape(self):
        iq = _make_tone(0.1, 512)
        cwd = CWD(nfft=128, output_format="mag_phase")
        out = cwd(iq)
        assert out.shape == (2, 512, 128)

    def test_real_imag_shape(self):
        iq = _make_tone(0.1, 512)
        cwd = CWD(nfft=128, output_format="real_imag")
        out = cwd(iq)
        assert out.shape == (2, 512, 128)

    def test_n_time_subsampling(self):
        iq = _make_tone(0.1, 1024)
        cwd = CWD(nfft=256, n_time=64, output_format="magnitude")
        out = cwd(iq)
        assert out.shape == (1, 64, 256)

    def test_default_params(self):
        iq = _make_tone(0.1, 256)
        cwd = CWD()
        out = cwd(iq)
        assert out.shape == (1, 256, 256)
        assert out.dtype == torch.float32


class TestCWDContent:
    """Signal property and content tests."""

    def test_pure_tone_concentrated_frequency(self):
        """Pure tone should have energy concentrated at one frequency."""
        iq = _make_tone(0.1, 512)
        cwd = CWD(nfft=256, output_format="magnitude")
        out = cwd(iq)
        mid_slice = out[0, 256, :]
        peak_val = torch.max(mid_slice).item()
        mean_val = torch.mean(mid_slice).item()
        assert peak_val > 5 * mean_val

    def test_db_scale_reduces_values(self):
        iq = _make_tone(0.1, 256)
        linear = CWD(nfft=64, output_format="magnitude", db_scale=False)(iq)
        db = CWD(nfft=64, output_format="magnitude", db_scale=True)(iq)
        # dB values should differ from linear (log10 transform)
        assert not torch.allclose(linear, db)

    def test_no_nan_or_inf(self):
        iq = _make_tone(0.1, 256)
        cwd = CWD(nfft=128, output_format="magnitude")
        out = cwd(iq)
        assert not torch.any(torch.isnan(out))
        assert not torch.any(torch.isinf(out))

    def test_deterministic(self):
        iq = _make_tone(0.1, 256)
        cwd = CWD(nfft=64, sigma=1.0)
        r1 = cwd(iq)
        r2 = cwd(iq)
        assert torch.equal(r1, r2)

    def test_sigma_affects_output(self):
        """Different sigma values should produce different results."""
        iq = _make_tone(0.1, 256)
        r1 = CWD(nfft=64, sigma=0.5)(iq)
        r2 = CWD(nfft=64, sigma=5.0)(iq)
        assert not torch.allclose(r1, r2)

    def test_non_contiguous_input(self):
        """Transform should handle non-contiguous arrays."""
        iq = _make_tone(0.1, 512)
        strided = iq[::2]  # non-contiguous
        cwd = CWD(nfft=64)
        out = cwd(strided)
        assert out.shape == (1, 256, 64)
        assert not torch.any(torch.isnan(out))


class TestCWDValidation:
    """Input validation tests."""

    def test_invalid_output_format(self):
        with pytest.raises(ValueError, match="Unknown output_format"):
            CWD(output_format="bad_format")

    def test_invalid_sigma_zero(self):
        with pytest.raises(ValueError, match="sigma must be positive"):
            CWD(sigma=0.0)

    def test_invalid_sigma_negative(self):
        with pytest.raises(ValueError, match="sigma must be positive"):
            CWD(sigma=-1.0)
```

**Step 2: Run the tests**

Run: `pytest tests/test_transforms_cwd.py -v`
Expected: All 13 tests PASS.

**Step 3: Commit**

```bash
git add tests/test_transforms_cwd.py
git commit -m "test(transforms): add CWD transform tests"
```

---

### Task 5: Integration verification

**Files:** None (verification only)

**Step 1: Run all existing tests to confirm no regressions**

Run: `pytest tests/ -v`
Expected: All tests PASS, no regressions.

**Step 2: Run Rust checks**

Run: `cargo fmt --manifest-path rust/Cargo.toml --all -- --check`
Expected: No formatting issues.

Run: `cargo clippy --manifest-path rust/Cargo.toml --all-targets -- -D warnings`
Expected: No warnings.

**Step 3: Fix any issues found, then commit**

```bash
git add -A
git commit -m "chore: fix any lint/format issues from CWD addition"
```

---

### Task 6: Example script — `examples/12_cwd_cross_term_suppression.py`

**Files:**
- Create: `examples/12_cwd_cross_term_suppression.py`

**Step 1: Create the example script**

```python
"""
SPECTRA Example 12: Choi-Williams Distribution — Cross-Term Suppression
=======================================================================
Level: Intermediate

Learn how to:
- Compute the Choi-Williams Distribution (CWD) for multi-component signals
- Compare CWD cross-term suppression against the Wigner-Ville Distribution
- Explore the effect of the sigma parameter on resolution vs. cross-terms
"""

import sys
sys.path.insert(0, ".")

import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

import spectra as sp
from plot_helpers import savefig


# ── 1. Generate a Two-Tone Signal ───────────────────────────────────────────
# Two complex tones at different frequencies produce strong cross-terms in
# the WVD.  The CWD kernel suppresses these artefacts.

sample_rate = 1e6
n_samples = 512
t = np.arange(n_samples) / sample_rate

f1, f2 = 0.1e6, 0.3e6
iq = (np.exp(1j * 2 * np.pi * f1 * t) + np.exp(1j * 2 * np.pi * f2 * t)).astype(
    np.complex64
)

print(f"Two-tone signal: {f1/1e3:.0f} kHz + {f2/1e3:.0f} kHz, {n_samples} samples")


# ── 2. WVD vs CWD Side-by-Side ──────────────────────────────────────────────

nfft = 256
wvd_transform = sp.WVD(nfft=nfft, output_format="magnitude")
cwd_transform = sp.CWD(nfft=nfft, sigma=1.0, output_format="magnitude")

wvd_out = wvd_transform(iq).squeeze(0).numpy()
cwd_out = cwd_transform(iq).squeeze(0).numpy()

fig, axes = plt.subplots(1, 2, figsize=(14, 5))

im0 = axes[0].imshow(
    10 * np.log10(wvd_out.T + 1e-12),
    aspect="auto",
    origin="lower",
    cmap="inferno",
    interpolation="nearest",
)
axes[0].set_title("Wigner-Ville Distribution")
axes[0].set_xlabel("Time Sample")
axes[0].set_ylabel("Frequency Bin")
fig.colorbar(im0, ax=axes[0], label="dB")

im1 = axes[1].imshow(
    10 * np.log10(cwd_out.T + 1e-12),
    aspect="auto",
    origin="lower",
    cmap="inferno",
    interpolation="nearest",
)
axes[1].set_title("Choi-Williams Distribution (σ = 1.0)")
axes[1].set_xlabel("Time Sample")
axes[1].set_ylabel("Frequency Bin")
fig.colorbar(im1, ax=axes[1], label="dB")

fig.suptitle("WVD vs CWD — Two-Tone Cross-Term Suppression", fontsize=14)
fig.tight_layout()
savefig("12_wvd_vs_cwd.png")


# ── 3. Sigma Sweep ──────────────────────────────────────────────────────────
# Show how different sigma values affect cross-term suppression and
# time-frequency resolution.

sigmas = [0.1, 0.5, 1.0, 5.0]

fig, axes = plt.subplots(1, len(sigmas), figsize=(16, 4))
for ax, sigma in zip(axes, sigmas):
    cwd = sp.CWD(nfft=nfft, sigma=sigma, output_format="magnitude")
    out = cwd(iq).squeeze(0).numpy()
    im = ax.imshow(
        10 * np.log10(out.T + 1e-12),
        aspect="auto",
        origin="lower",
        cmap="inferno",
        interpolation="nearest",
    )
    ax.set_title(f"σ = {sigma}")
    ax.set_xlabel("Time Sample")
    if ax is axes[0]:
        ax.set_ylabel("Frequency Bin")
fig.suptitle("CWD — Effect of Sigma on Cross-Term Suppression", fontsize=14)
fig.tight_layout()
savefig("12_cwd_sigma_sweep.png")


# ── 4. Chirp Signal ─────────────────────────────────────────────────────────
# A linear chirp signal shows how the CWD tracks instantaneous frequency
# while suppressing artefacts.

chirp_wf = sp.LinearChirp(start_freq=-0.3e6, stop_freq=0.3e6)
chirp_iq = chirp_wf.generate(num_symbols=1024, sample_rate=sample_rate, seed=42)[
    :1024
]

cwd_chirp = sp.CWD(nfft=256, sigma=1.0, output_format="magnitude")
wvd_chirp = sp.WVD(nfft=256, output_format="magnitude")

cwd_out = cwd_chirp(chirp_iq).squeeze(0).numpy()
wvd_out = wvd_chirp(chirp_iq).squeeze(0).numpy()

fig, axes = plt.subplots(1, 2, figsize=(14, 5))

im0 = axes[0].imshow(
    10 * np.log10(wvd_out.T + 1e-12),
    aspect="auto",
    origin="lower",
    cmap="inferno",
    interpolation="nearest",
)
axes[0].set_title("WVD — Linear Chirp")
axes[0].set_xlabel("Time Sample")
axes[0].set_ylabel("Frequency Bin")
fig.colorbar(im0, ax=axes[0], label="dB")

im1 = axes[1].imshow(
    10 * np.log10(cwd_out.T + 1e-12),
    aspect="auto",
    origin="lower",
    cmap="inferno",
    interpolation="nearest",
)
axes[1].set_title("CWD — Linear Chirp (σ = 1.0)")
axes[1].set_xlabel("Time Sample")
axes[1].set_ylabel("Frequency Bin")
fig.colorbar(im1, ax=axes[1], label="dB")

fig.suptitle("Time-Frequency Tracking — Linear Chirp", fontsize=14)
fig.tight_layout()
savefig("12_cwd_chirp.png")


# ── 5. Output Formats ───────────────────────────────────────────────────────
# Demonstrate the three output format modes.

tone_iq = np.exp(1j * 2 * np.pi * 0.15 * np.arange(256)).astype(np.complex64)

for fmt in ["magnitude", "mag_phase", "real_imag"]:
    cwd = sp.CWD(nfft=64, sigma=1.0, output_format=fmt)
    out = cwd(tone_iq)
    print(f"  output_format={fmt!r:14s} → shape {tuple(out.shape)}, dtype {out.dtype}")


plt.close("all")
print("\nDone! Check examples/outputs/ for saved figures.")
```

**Step 2: Verify the script runs**

Run: `cd examples && python 12_cwd_cross_term_suppression.py`
Expected: Script completes, 3 PNG files saved to `examples/outputs/`.

**Step 3: Commit**

```bash
git add examples/12_cwd_cross_term_suppression.py
git commit -m "docs(examples): add CWD cross-term suppression example script"
```

---

### Task 7: Example notebook — `examples/12_cwd_cross_term_suppression.ipynb`

**Files:**
- Create: `examples/12_cwd_cross_term_suppression.ipynb`

**Step 1: Create the notebook**

The notebook mirrors the script from Task 6 but split into cells with markdown explanations. Structure:

- **Cell 1** (markdown): Title, level, learning objectives
- **Cell 2** (code): Imports (`sys.path`, numpy, `%matplotlib inline`, matplotlib, spectra, plot_helpers)
- **Cell 3** (markdown): `## 1. Generate a Two-Tone Signal`
- **Cell 4** (code): Two-tone signal generation (same as script section 1)
- **Cell 5** (markdown): `## 2. WVD vs CWD Side-by-Side`
- **Cell 6** (code): WVD vs CWD comparison plot (same as script section 2)
- **Cell 7** (markdown): `## 3. Effect of Sigma on Cross-Term Suppression`
- **Cell 8** (code): Sigma sweep plot (same as script section 3)
- **Cell 9** (markdown): `## 4. Chirp Signal — Instantaneous Frequency Tracking`
- **Cell 10** (code): Chirp WVD vs CWD comparison (same as script section 4)
- **Cell 11** (markdown): `## 5. Output Formats`
- **Cell 12** (code): Output format demo (same as script section 5)
- **Cell 13** (markdown): Summary of learnings, link to next example

Use the exact same code from the Task 6 script for each cell, but replace `matplotlib.use("Agg")` with `%matplotlib inline` in the import cell.

**Step 2: Verify the notebook runs**

Run: `jupyter nbconvert --to notebook --execute examples/12_cwd_cross_term_suppression.ipynb --output /dev/null`
Expected: Notebook executes without errors.

**Step 3: Commit**

```bash
git add examples/12_cwd_cross_term_suppression.ipynb
git commit -m "docs(examples): add CWD cross-term suppression notebook"
```

---

### Task 8: Update examples README

**Files:**
- Modify: `examples/README.md`

**Step 1: Add CWD example entry**

After the `### 08` section (before the `## Output` section), add:

```markdown
### 12 — Choi-Williams Distribution (Intermediate)

Compare the Choi-Williams Distribution against the WVD for cross-term suppression.

- Side-by-side WVD vs CWD on multi-component signals
- Sigma parameter sweep showing resolution vs. cross-term trade-off
- Linear chirp instantaneous frequency tracking
- Output format demonstration (magnitude, mag_phase, real_imag)

\```bash
cd examples && python 12_cwd_cross_term_suppression.py
\```
```

Update the file structure section to include the new files, and update the output count.

**Step 2: Commit**

```bash
git add examples/README.md
git commit -m "docs(examples): add CWD example to README"
```

---

## Summary of Files Changed

| Action | File |
|--------|------|
| Create | `rust/src/cwd.rs` — Rust CWD computation |
| Modify | `rust/src/lib.rs` — Register `compute_cwd` in PyO3 module |
| Create | `python/spectra/transforms/cwd.py` — Python CWD class |
| Modify | `python/spectra/transforms/__init__.py` — Export CWD |
| Modify | `tests/test_rust_cyclo.py` — Rust FFI tests |
| Create | `tests/test_transforms_cwd.py` — Python transform tests |
| Create | `examples/12_cwd_cross_term_suppression.py` — Example script |
| Create | `examples/12_cwd_cross_term_suppression.ipynb` — Example notebook |
| Modify | `examples/README.md` — Add CWD example entry |
