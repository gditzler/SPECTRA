# Domain Adaptation Transforms Implementation Plan

> **For agentic workers:** REQUIRED: Use superpowers:subagent-driven-development (if subagents available) or superpowers:executing-plans to implement this plan. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Add 10 composable signal alignment transforms to SPECTRA for normalizing IQ data across different capture sources, SDR hardware, and gain settings.

**Architecture:** Alignment transforms inherit from the existing `Transform` ABC (`spectra.impairments.base`), taking `(iq, desc, **kwargs)` and returning `(iq, desc)`. This makes them composable via `Compose` and usable in the existing impairment pipeline. They live in `python/spectra/transforms/alignment.py` as a single focused module. Only `Resample` requires scipy (lazy import); all others are pure NumPy.

**Tech Stack:** Python (NumPy, scipy for Resample only), pytest

---

## File Map

| File | Action | Responsibility |
|------|--------|---------------|
| `python/spectra/transforms/alignment.py` | Create | All 10 alignment transform classes |
| `python/spectra/transforms/__init__.py` | Modify | Export new classes |
| `pyproject.toml` | Modify | Add `alignment` optional dep group |
| `tests/test_transforms_alignment.py` | Create | All tests |

---

### Task 1: Optional dependency and test infrastructure

**Files:**
- Modify: `pyproject.toml:28-47`
- Create: `tests/test_transforms_alignment.py`

- [ ] **Step 1: Add `alignment` optional dep to `pyproject.toml`**

In `pyproject.toml`, add the `alignment` group after the `io` line and update `all`:

```toml
alignment = ["scipy>=1.10"]
```

Update `all` to include scipy:

```toml
all = ["pyyaml>=6.0", "zarr>=2.16", "scikit-learn>=1.0", "sigmf>=1.0", "h5py>=3.0", "scipy>=1.10"]
```

- [ ] **Step 2: Create test file with initial imports and fixtures**

Create `tests/test_transforms_alignment.py`:

```python
import numpy as np
import numpy.testing as npt
import pytest

from spectra.scene.signal_desc import SignalDescription


@pytest.fixture
def sample_iq():
    """Complex64 IQ signal with known properties."""
    rng = np.random.default_rng(42)
    n = 4096
    t = np.arange(n) / 1e6
    signal = np.exp(1j * 2 * np.pi * 50_000 * t).astype(np.complex64)
    noise = 0.1 * (rng.standard_normal(n) + 1j * rng.standard_normal(n)).astype(np.complex64)
    return signal + noise


@pytest.fixture
def sample_desc():
    """Minimal SignalDescription for testing."""
    return SignalDescription(
        t_start=0.0,
        t_stop=0.004096,
        f_low=-50_000.0,
        f_high=50_000.0,
        label="test",
        snr=20.0,
    )


@pytest.fixture
def sample_rate():
    return 1e6
```

- [ ] **Step 3: Verify test file is importable**

Run: `source .venv/bin/activate && python -c "from spectra.scene.signal_desc import SignalDescription; print('OK')"`
Expected: `OK`

- [ ] **Step 4: Commit**

```bash
git add pyproject.toml tests/test_transforms_alignment.py
git commit -m "chore: add alignment optional dep and test scaffold"
```

---

### Task 2: DCRemove and ClipNormalize (simplest transforms)

**Files:**
- Create: `python/spectra/transforms/alignment.py`
- Modify: `tests/test_transforms_alignment.py`

- [ ] **Step 1: Write failing tests for DCRemove and ClipNormalize**

Append to `tests/test_transforms_alignment.py`:

```python
from spectra.transforms.alignment import DCRemove, ClipNormalize


class TestDCRemove:
    def test_output_mean_near_zero(self, sample_iq, sample_desc):
        t = DCRemove()
        iq_out, desc_out = t(sample_iq, sample_desc)
        assert abs(np.mean(iq_out)) < 1e-6

    def test_preserves_dtype(self, sample_iq, sample_desc):
        t = DCRemove()
        iq_out, _ = t(sample_iq, sample_desc)
        assert iq_out.dtype == np.complex64

    def test_preserves_length(self, sample_iq, sample_desc):
        t = DCRemove()
        iq_out, _ = t(sample_iq, sample_desc)
        assert len(iq_out) == len(sample_iq)

    def test_desc_unchanged(self, sample_iq, sample_desc):
        t = DCRemove()
        _, desc_out = t(sample_iq, sample_desc)
        assert desc_out.f_low == sample_desc.f_low
        assert desc_out.f_high == sample_desc.f_high

    def test_with_dc_offset(self, sample_desc):
        iq = np.ones(1024, dtype=np.complex64) * (5.0 + 3.0j)
        t = DCRemove()
        iq_out, _ = t(iq, sample_desc)
        assert abs(np.mean(iq_out)) < 1e-6


class TestClipNormalize:
    def test_output_bounded(self, sample_iq, sample_desc):
        t = ClipNormalize(clip_sigma=3.0)
        iq_out, _ = t(sample_iq, sample_desc)
        assert np.max(np.abs(iq_out.real)) <= 1.0 + 1e-6
        assert np.max(np.abs(iq_out.imag)) <= 1.0 + 1e-6

    def test_preserves_dtype(self, sample_iq, sample_desc):
        t = ClipNormalize()
        iq_out, _ = t(sample_iq, sample_desc)
        assert iq_out.dtype == np.complex64

    def test_preserves_length(self, sample_iq, sample_desc):
        t = ClipNormalize()
        iq_out, _ = t(sample_iq, sample_desc)
        assert len(iq_out) == len(sample_iq)

    def test_outliers_clipped(self, sample_desc):
        iq = np.zeros(1000, dtype=np.complex64)
        iq[500] = 100.0 + 100.0j  # extreme outlier
        t = ClipNormalize(clip_sigma=2.0)
        iq_out, _ = t(iq, sample_desc)
        assert np.abs(iq_out[500]) < np.abs(iq[500])

    def test_custom_clip_sigma(self, sample_iq, sample_desc):
        t1 = ClipNormalize(clip_sigma=1.0)
        t2 = ClipNormalize(clip_sigma=5.0)
        iq1, _ = t1(sample_iq, sample_desc)
        iq2, _ = t2(sample_iq, sample_desc)
        # Tighter clipping should produce different result
        assert not np.array_equal(iq1, iq2)
```

- [ ] **Step 2: Run tests to verify they fail**

Run: `source .venv/bin/activate && pytest tests/test_transforms_alignment.py -v`
Expected: FAIL — `ImportError: cannot import name 'DCRemove' from 'spectra.transforms.alignment'`

- [ ] **Step 3: Implement DCRemove and ClipNormalize**

Create `python/spectra/transforms/alignment.py`:

```python
"""Domain adaptation transforms for cross-source IQ signal alignment.

These transforms normalize IQ signals across different capture sources,
SDR hardware, and gain settings. They inherit from the impairment
Transform ABC and are composable via Compose.

Statistical alignment (Tier 1): DCRemove, ClipNormalize, PowerNormalize,
AGCNormalize, Resample.

Spectral alignment (Tier 2): SpectralWhitening, NoiseFloorMatch,
BandpassAlign.

Reference-based (Tier 3 stubs): NoiseProfileTransfer, ReceiverEQ.
"""

from typing import Tuple

import numpy as np

from spectra.impairments.base import Transform
from spectra.scene.signal_desc import SignalDescription


class DCRemove(Transform):
    """Remove DC offset via mean subtraction.

    Many SDR receivers introduce a DC spur at the center frequency.
    This transform removes it by subtracting the mean of the signal.
    """

    def __call__(
        self, iq: np.ndarray, desc: SignalDescription, **kwargs
    ) -> Tuple[np.ndarray, SignalDescription]:
        return (iq - np.mean(iq)).astype(np.complex64), desc


class ClipNormalize(Transform):
    """Clip outlier samples beyond N sigma and scale to [-1, 1].

    Useful for taming signals with extreme amplitude spikes from
    ADC saturation or interference before feeding to a model.

    Args:
        clip_sigma: Clip threshold in standard deviations. Default 3.0.
    """

    def __init__(self, clip_sigma: float = 3.0):
        self._clip_sigma = clip_sigma

    def __call__(
        self, iq: np.ndarray, desc: SignalDescription, **kwargs
    ) -> Tuple[np.ndarray, SignalDescription]:
        # Clip real and imaginary independently
        re = iq.real.copy()
        im = iq.imag.copy()

        re_std = np.std(re)
        im_std = np.std(im)

        if re_std > 0:
            thresh_re = self._clip_sigma * re_std
            np.clip(re, -thresh_re, thresh_re, out=re)
        if im_std > 0:
            thresh_im = self._clip_sigma * im_std
            np.clip(im, -thresh_im, thresh_im, out=im)

        # Scale to [-1, 1]
        peak = max(np.max(np.abs(re)), np.max(np.abs(im)))
        if peak > 0:
            re /= peak
            im /= peak

        return (re + 1j * im).astype(np.complex64), desc
```

- [ ] **Step 4: Run tests to verify they pass**

Run: `source .venv/bin/activate && pytest tests/test_transforms_alignment.py -v`
Expected: All 10 tests PASS

- [ ] **Step 5: Commit**

```bash
git add python/spectra/transforms/alignment.py tests/test_transforms_alignment.py
git commit -m "feat(transforms): add DCRemove and ClipNormalize alignment transforms"
```

---

### Task 3: PowerNormalize and AGCNormalize

**Files:**
- Modify: `python/spectra/transforms/alignment.py`
- Modify: `tests/test_transforms_alignment.py`

- [ ] **Step 1: Write failing tests**

Append to `tests/test_transforms_alignment.py`:

```python
from spectra.transforms.alignment import PowerNormalize, AGCNormalize


class TestPowerNormalize:
    def test_output_rms_matches_target(self, sample_iq, sample_desc):
        target = -20.0
        t = PowerNormalize(target_power_dbfs=target)
        iq_out, _ = t(sample_iq, sample_desc)
        rms = np.sqrt(np.mean(np.abs(iq_out) ** 2))
        rms_db = 20.0 * np.log10(rms)
        assert abs(rms_db - target) < 0.1

    def test_preserves_dtype(self, sample_iq, sample_desc):
        t = PowerNormalize()
        iq_out, _ = t(sample_iq, sample_desc)
        assert iq_out.dtype == np.complex64

    def test_preserves_length(self, sample_iq, sample_desc):
        t = PowerNormalize()
        iq_out, _ = t(sample_iq, sample_desc)
        assert len(iq_out) == len(sample_iq)

    def test_zero_power_unchanged(self, sample_desc):
        iq = np.zeros(1024, dtype=np.complex64)
        t = PowerNormalize(target_power_dbfs=-20.0)
        iq_out, _ = t(iq, sample_desc)
        npt.assert_array_equal(iq_out, iq)

    def test_different_targets_differ(self, sample_iq, sample_desc):
        t1 = PowerNormalize(target_power_dbfs=-10.0)
        t2 = PowerNormalize(target_power_dbfs=-30.0)
        iq1, _ = t1(sample_iq, sample_desc)
        iq2, _ = t2(sample_iq, sample_desc)
        rms1 = np.sqrt(np.mean(np.abs(iq1) ** 2))
        rms2 = np.sqrt(np.mean(np.abs(iq2) ** 2))
        assert rms1 > rms2


class TestAGCNormalize:
    def test_rms_mode_unit_power(self, sample_iq, sample_desc):
        t = AGCNormalize(method="rms", target_level=1.0)
        iq_out, _ = t(sample_iq, sample_desc)
        rms = np.sqrt(np.mean(np.abs(iq_out) ** 2))
        assert abs(rms - 1.0) < 1e-5

    def test_peak_mode_bounded(self, sample_iq, sample_desc):
        t = AGCNormalize(method="peak", target_level=1.0)
        iq_out, _ = t(sample_iq, sample_desc)
        assert np.max(np.abs(iq_out)) <= 1.0 + 1e-6

    def test_preserves_dtype(self, sample_iq, sample_desc):
        t = AGCNormalize()
        iq_out, _ = t(sample_iq, sample_desc)
        assert iq_out.dtype == np.complex64

    def test_preserves_length(self, sample_iq, sample_desc):
        t = AGCNormalize()
        iq_out, _ = t(sample_iq, sample_desc)
        assert len(iq_out) == len(sample_iq)

    def test_zero_amplitude_safe(self, sample_desc):
        iq = np.zeros(1024, dtype=np.complex64)
        t = AGCNormalize(method="rms")
        iq_out, _ = t(iq, sample_desc)
        npt.assert_array_equal(iq_out, iq)

    def test_invalid_method_raises(self, sample_iq, sample_desc):
        with pytest.raises(ValueError):
            AGCNormalize(method="invalid")

    def test_custom_target_level(self, sample_iq, sample_desc):
        t = AGCNormalize(method="rms", target_level=0.5)
        iq_out, _ = t(sample_iq, sample_desc)
        rms = np.sqrt(np.mean(np.abs(iq_out) ** 2))
        assert abs(rms - 0.5) < 1e-5
```

- [ ] **Step 2: Run tests to verify they fail**

Run: `source .venv/bin/activate && pytest tests/test_transforms_alignment.py::TestPowerNormalize -v`
Expected: FAIL — `ImportError`

- [ ] **Step 3: Implement PowerNormalize and AGCNormalize**

Append to `python/spectra/transforms/alignment.py`:

```python
class PowerNormalize(Transform):
    """Scale IQ signal to a target RMS power level in dBFS.

    Useful for normalizing signals captured at different gain settings
    to a consistent power level before training.

    Args:
        target_power_dbfs: Target RMS power in dB relative to full scale.
            Default -20.0.
    """

    def __init__(self, target_power_dbfs: float = -20.0):
        self._target_dbfs = target_power_dbfs

    def __call__(
        self, iq: np.ndarray, desc: SignalDescription, **kwargs
    ) -> Tuple[np.ndarray, SignalDescription]:
        rms = np.sqrt(np.mean(np.abs(iq) ** 2))
        if rms == 0:
            return iq, desc
        target_linear = 10.0 ** (self._target_dbfs / 20.0)
        return (iq * (target_linear / rms)).astype(np.complex64), desc


class AGCNormalize(Transform):
    """Normalize gain to undo differences in hardware AGC settings.

    Two modes:
    - ``"rms"``: scale so RMS equals ``target_level`` (unit power default)
    - ``"peak"``: scale so max absolute value equals ``target_level``

    Args:
        method: ``"rms"`` or ``"peak"``. Default ``"rms"``.
        target_level: Target normalization level. Default 1.0.

    Raises:
        ValueError: If method is not ``"rms"`` or ``"peak"``.
    """

    def __init__(self, method: str = "rms", target_level: float = 1.0):
        if method not in ("rms", "peak"):
            raise ValueError(f"method must be 'rms' or 'peak', got '{method}'")
        self._method = method
        self._target = target_level

    def __call__(
        self, iq: np.ndarray, desc: SignalDescription, **kwargs
    ) -> Tuple[np.ndarray, SignalDescription]:
        if self._method == "rms":
            level = np.sqrt(np.mean(np.abs(iq) ** 2))
        else:
            level = np.max(np.abs(iq))

        if level == 0:
            return iq, desc
        return (iq * (self._target / level)).astype(np.complex64), desc
```

- [ ] **Step 4: Run tests to verify they pass**

Run: `source .venv/bin/activate && pytest tests/test_transforms_alignment.py -v`
Expected: All 22 tests PASS

- [ ] **Step 5: Commit**

```bash
git add python/spectra/transforms/alignment.py tests/test_transforms_alignment.py
git commit -m "feat(transforms): add PowerNormalize and AGCNormalize alignment transforms"
```

---

### Task 4: Resample (scipy-dependent)

**Files:**
- Modify: `python/spectra/transforms/alignment.py`
- Modify: `tests/test_transforms_alignment.py`

- [ ] **Step 1: Write failing tests**

Append to `tests/test_transforms_alignment.py`:

```python
from spectra.transforms.alignment import Resample

pytest.importorskip("scipy")


class TestResample:
    def test_upsample_length(self, sample_iq, sample_desc):
        t = Resample(target_sample_rate=2e6)
        iq_out, _ = t(sample_iq, sample_desc, sample_rate=1e6)
        expected_len = len(sample_iq) * 2
        assert abs(len(iq_out) - expected_len) <= 1

    def test_downsample_length(self, sample_iq, sample_desc):
        t = Resample(target_sample_rate=500_000)
        iq_out, _ = t(sample_iq, sample_desc, sample_rate=1e6)
        expected_len = len(sample_iq) // 2
        assert abs(len(iq_out) - expected_len) <= 1

    def test_preserves_dtype(self, sample_iq, sample_desc):
        t = Resample(target_sample_rate=2e6)
        iq_out, _ = t(sample_iq, sample_desc, sample_rate=1e6)
        assert iq_out.dtype == np.complex64

    def test_same_rate_unchanged(self, sample_iq, sample_desc):
        t = Resample(target_sample_rate=1e6)
        iq_out, _ = t(sample_iq, sample_desc, sample_rate=1e6)
        npt.assert_array_equal(iq_out, sample_iq)

    def test_round_trip(self, sample_desc):
        rng = np.random.default_rng(99)
        iq = (rng.standard_normal(1024) + 1j * rng.standard_normal(1024)).astype(
            np.complex64
        )
        up = Resample(target_sample_rate=2e6)
        down = Resample(target_sample_rate=1e6)
        iq_up, _ = up(iq, sample_desc, sample_rate=1e6)
        iq_back, _ = down(iq_up, sample_desc, sample_rate=2e6)
        # Round-trip should be close but not exact (filter effects)
        min_len = min(len(iq), len(iq_back))
        corr = np.abs(np.corrcoef(np.abs(iq[:min_len]), np.abs(iq_back[:min_len]))[0, 1])
        assert corr > 0.9

    def test_missing_sample_rate_raises(self, sample_iq, sample_desc):
        t = Resample(target_sample_rate=2e6)
        with pytest.raises(ValueError, match="sample_rate"):
            t(sample_iq, sample_desc)
```

- [ ] **Step 2: Run tests to verify they fail**

Run: `source .venv/bin/activate && pytest tests/test_transforms_alignment.py::TestResample -v`
Expected: FAIL — `ImportError`

- [ ] **Step 3: Implement Resample**

Append to `python/spectra/transforms/alignment.py`:

```python
class Resample(Transform):
    """Rational resampling to a target sample rate.

    Uses ``scipy.signal.resample_poly`` with rational approximation.
    Requires ``scipy`` — install with ``pip install spectra[alignment]``.

    The ``sample_rate`` keyword argument is required (forwarded by
    ``Compose`` or passed directly).

    Args:
        target_sample_rate: Target sample rate in Hz.

    Raises:
        ImportError: If scipy is not installed.
        ValueError: If ``sample_rate`` kwarg is missing.
    """

    def __init__(self, target_sample_rate: float):
        try:
            from scipy.signal import resample_poly  # noqa: F401
        except ImportError:
            raise ImportError(
                "Resample requires scipy. Install with: pip install spectra[alignment]"
            ) from None
        self._target_rate = target_sample_rate

    def __call__(
        self, iq: np.ndarray, desc: SignalDescription, **kwargs
    ) -> Tuple[np.ndarray, SignalDescription]:
        from fractions import Fraction

        from scipy.signal import resample_poly

        sample_rate = kwargs.get("sample_rate")
        if sample_rate is None:
            raise ValueError(
                "Resample requires sample_rate kwarg. "
                "Pass via Compose(...)(iq, desc, sample_rate=fs) or directly."
            )

        if sample_rate == self._target_rate:
            return iq, desc

        frac = Fraction(self._target_rate / sample_rate).limit_denominator(1000)
        up, down = frac.numerator, frac.denominator

        resampled = resample_poly(iq, up, down).astype(np.complex64)
        return resampled, desc
```

- [ ] **Step 4: Run tests to verify they pass**

Run: `source .venv/bin/activate && pytest tests/test_transforms_alignment.py -v`
Expected: All 28 tests PASS

- [ ] **Step 5: Commit**

```bash
git add python/spectra/transforms/alignment.py tests/test_transforms_alignment.py
git commit -m "feat(transforms): add Resample alignment transform with scipy backend"
```

---

### Task 5: SpectralWhitening, NoiseFloorMatch, BandpassAlign (Tier 2)

**Files:**
- Modify: `python/spectra/transforms/alignment.py`
- Modify: `tests/test_transforms_alignment.py`

- [ ] **Step 1: Write failing tests**

Append to `tests/test_transforms_alignment.py`:

```python
from spectra.transforms.alignment import (
    BandpassAlign,
    NoiseFloorMatch,
    SpectralWhitening,
)


class TestSpectralWhitening:
    def test_psd_flatter_after(self, sample_iq, sample_desc):
        t = SpectralWhitening(smoothing_window=64)
        iq_out, _ = t(sample_iq, sample_desc)
        # PSD variance should decrease (flatter spectrum)
        psd_before = np.abs(np.fft.fft(sample_iq)) ** 2
        psd_after = np.abs(np.fft.fft(iq_out)) ** 2
        assert np.var(psd_after) < np.var(psd_before)

    def test_preserves_dtype(self, sample_iq, sample_desc):
        t = SpectralWhitening()
        iq_out, _ = t(sample_iq, sample_desc)
        assert iq_out.dtype == np.complex64

    def test_preserves_length(self, sample_iq, sample_desc):
        t = SpectralWhitening()
        iq_out, _ = t(sample_iq, sample_desc)
        assert len(iq_out) == len(sample_iq)

    def test_energy_preserved(self, sample_iq, sample_desc):
        t = SpectralWhitening(smoothing_window=32)
        iq_out, _ = t(sample_iq, sample_desc)
        power_before = np.mean(np.abs(sample_iq) ** 2)
        power_after = np.mean(np.abs(iq_out) ** 2)
        # Energy should be in the same order of magnitude
        ratio = power_after / power_before
        assert 0.1 < ratio < 10.0


class TestNoiseFloorMatch:
    def test_preserves_dtype(self, sample_iq, sample_desc):
        t = NoiseFloorMatch(target_noise_floor_db=-40.0)
        iq_out, _ = t(sample_iq, sample_desc)
        assert iq_out.dtype == np.complex64

    def test_preserves_length(self, sample_iq, sample_desc):
        t = NoiseFloorMatch()
        iq_out, _ = t(sample_iq, sample_desc)
        assert len(iq_out) == len(sample_iq)

    def test_different_targets_scale_differently(self, sample_iq, sample_desc):
        t1 = NoiseFloorMatch(target_noise_floor_db=-30.0)
        t2 = NoiseFloorMatch(target_noise_floor_db=-50.0)
        iq1, _ = t1(sample_iq, sample_desc)
        iq2, _ = t2(sample_iq, sample_desc)
        power1 = np.mean(np.abs(iq1) ** 2)
        power2 = np.mean(np.abs(iq2) ** 2)
        assert power1 > power2

    def test_estimation_methods(self, sample_iq, sample_desc):
        t1 = NoiseFloorMatch(target_noise_floor_db=-40.0, estimation_method="median")
        t2 = NoiseFloorMatch(target_noise_floor_db=-40.0, estimation_method="minimum")
        iq1, _ = t1(sample_iq, sample_desc)
        iq2, _ = t2(sample_iq, sample_desc)
        # Different methods should give different scaling
        assert not np.array_equal(iq1, iq2)


class TestBandpassAlign:
    def test_preserves_dtype(self, sample_iq, sample_desc):
        t = BandpassAlign(center_freq=0.0, bandwidth=0.5)
        iq_out, _ = t(sample_iq, sample_desc, sample_rate=1e6)
        assert iq_out.dtype == np.complex64

    def test_preserves_length(self, sample_iq, sample_desc):
        t = BandpassAlign(center_freq=0.0, bandwidth=0.5)
        iq_out, _ = t(sample_iq, sample_desc, sample_rate=1e6)
        assert len(iq_out) == len(sample_iq)

    def test_out_of_band_suppressed(self, sample_desc):
        # Signal with energy at 0.25 * fs
        n = 4096
        t_arr = np.arange(n) / 1e6
        iq = np.exp(1j * 2 * np.pi * 250_000 * t_arr).astype(np.complex64)
        t = BandpassAlign(center_freq=0.0, bandwidth=0.1)  # narrow pass
        iq_out, _ = t(iq, sample_desc, sample_rate=1e6)
        # Energy should be reduced (signal is out of passband)
        power_before = np.mean(np.abs(iq) ** 2)
        power_after = np.mean(np.abs(iq_out) ** 2)
        assert power_after < power_before * 0.5

    def test_updates_desc_freq_bounds(self, sample_iq, sample_desc):
        t = BandpassAlign(center_freq=0.0, bandwidth=0.5)
        _, desc_out = t(sample_iq, sample_desc, sample_rate=1e6)
        assert desc_out.f_low == -250_000.0
        assert desc_out.f_high == 250_000.0
```

- [ ] **Step 2: Run tests to verify they fail**

Run: `source .venv/bin/activate && pytest tests/test_transforms_alignment.py::TestSpectralWhitening -v`
Expected: FAIL — `ImportError`

- [ ] **Step 3: Implement SpectralWhitening, NoiseFloorMatch, BandpassAlign**

Append to `python/spectra/transforms/alignment.py`:

```python
class SpectralWhitening(Transform):
    """Flatten PSD by dividing by the smoothed spectral envelope.

    Removes receiver-specific frequency coloring without needing a
    reference measurement. Preserves signal phase.

    Args:
        smoothing_window: Moving average window size for PSD smoothing.
            Larger values produce more aggressive flattening. Default 64.
    """

    def __init__(self, smoothing_window: int = 64):
        self._window = smoothing_window

    def __call__(
        self, iq: np.ndarray, desc: SignalDescription, **kwargs
    ) -> Tuple[np.ndarray, SignalDescription]:
        spectrum = np.fft.fft(iq)
        magnitude = np.abs(spectrum)

        # Smooth the magnitude envelope
        kernel = np.ones(self._window) / self._window
        smoothed = np.convolve(magnitude, kernel, mode="same")

        # Avoid division by zero
        floor = np.max(smoothed) * 1e-10
        smoothed = np.maximum(smoothed, floor)

        # Whiten: divide spectrum by envelope, preserving phase
        whitened_spectrum = spectrum / smoothed

        # Scale to preserve approximate energy
        original_power = np.mean(np.abs(iq) ** 2)
        result = np.fft.ifft(whitened_spectrum)
        result_power = np.mean(np.abs(result) ** 2)
        if result_power > 0:
            result *= np.sqrt(original_power / result_power)

        return result.astype(np.complex64), desc


class NoiseFloorMatch(Transform):
    """Estimate noise floor and scale to match a target level.

    Useful when combining captures with different noise figures
    or receiver sensitivities.

    Args:
        target_noise_floor_db: Target noise floor in dB. Default -40.0.
        estimation_method: ``"median"`` (robust) or ``"minimum"``
            (lower bound). Default ``"median"``.
    """

    def __init__(
        self, target_noise_floor_db: float = -40.0, estimation_method: str = "median"
    ):
        self._target_db = target_noise_floor_db
        self._method = estimation_method

    def __call__(
        self, iq: np.ndarray, desc: SignalDescription, **kwargs
    ) -> Tuple[np.ndarray, SignalDescription]:
        # Estimate PSD in dB
        spectrum = np.fft.fft(iq)
        psd = np.abs(spectrum) ** 2 / len(iq)
        psd_db = 10.0 * np.log10(np.maximum(psd, 1e-30))

        if self._method == "median":
            floor_db = float(np.median(psd_db))
        else:  # minimum
            floor_db = float(np.min(psd_db))

        # Scale factor in linear domain
        scale = 10.0 ** ((self._target_db - floor_db) / 20.0)
        return (iq * scale).astype(np.complex64), desc


class BandpassAlign(Transform):
    """Shift and filter signal to align to a target center frequency and bandwidth.

    Applies a frequency shift to center the signal energy, then a
    rectangular bandpass filter to limit bandwidth. Updates
    ``desc.f_low`` and ``desc.f_high``.

    The ``sample_rate`` keyword argument is required.

    Args:
        center_freq: Target center frequency in Hz (relative to baseband).
            Default 0.0.
        bandwidth: Target bandwidth as a fraction of sample rate (0, 1].
    """

    def __init__(self, center_freq: float = 0.0, bandwidth: float = 0.5):
        self._center = center_freq
        self._bw_frac = bandwidth

    def __call__(
        self, iq: np.ndarray, desc: SignalDescription, **kwargs
    ) -> Tuple[np.ndarray, SignalDescription]:
        from dataclasses import replace

        sample_rate = kwargs.get("sample_rate", 1.0)
        n = len(iq)

        # Frequency-domain bandpass filter
        freqs = np.fft.fftfreq(n, d=1.0 / sample_rate)
        spectrum = np.fft.fft(iq)

        half_bw = self._bw_frac * sample_rate / 2.0
        mask = np.abs(freqs - self._center) <= half_bw
        spectrum *= mask

        result = np.fft.ifft(spectrum).astype(np.complex64)

        new_desc = replace(
            desc,
            f_low=self._center - half_bw,
            f_high=self._center + half_bw,
        )
        return result, new_desc
```

- [ ] **Step 4: Run tests to verify they pass**

Run: `source .venv/bin/activate && pytest tests/test_transforms_alignment.py -v`
Expected: All 40 tests PASS

- [ ] **Step 5: Commit**

```bash
git add python/spectra/transforms/alignment.py tests/test_transforms_alignment.py
git commit -m "feat(transforms): add SpectralWhitening, NoiseFloorMatch, BandpassAlign"
```

---

### Task 6: Tier 3 stubs (NoiseProfileTransfer, ReceiverEQ)

**Files:**
- Modify: `python/spectra/transforms/alignment.py`
- Modify: `tests/test_transforms_alignment.py`

- [ ] **Step 1: Write failing tests**

Append to `tests/test_transforms_alignment.py`:

```python
from spectra.transforms.alignment import NoiseProfileTransfer, ReceiverEQ


class TestNoiseProfileTransfer:
    def test_raises_not_implemented(self, sample_iq, sample_desc):
        t = NoiseProfileTransfer(noise_source=np.zeros(100, dtype=np.complex64))
        with pytest.raises(NotImplementedError):
            t(sample_iq, sample_desc)


class TestReceiverEQ:
    def test_raises_not_implemented(self, sample_iq, sample_desc):
        t = ReceiverEQ(reference_psd=np.ones(256))
        with pytest.raises(NotImplementedError):
            t(sample_iq, sample_desc)
```

- [ ] **Step 2: Run tests to verify they fail**

Run: `source .venv/bin/activate && pytest tests/test_transforms_alignment.py::TestNoiseProfileTransfer -v`
Expected: FAIL — `ImportError`

- [ ] **Step 3: Implement stubs**

Append to `python/spectra/transforms/alignment.py`:

```python
class NoiseProfileTransfer(Transform):
    """Replace synthetic noise with noise characteristics from a real capture.

    .. note:: This is a research-grade transform. It is not yet implemented
       and will raise ``NotImplementedError`` when called.

    Intended approach:

    1. Estimate and subtract signal component from ``noise_source``
       (or use a known noise-only capture).
    2. Estimate noise PSD profile from the reference.
    3. Generate colored noise matching the reference PSD.
    4. Replace the synthetic noise component in the input signal.

    Args:
        noise_source: Path to noise capture file, or raw complex64 noise
            array from a real receiver.
    """

    def __init__(self, noise_source):
        self._source = noise_source

    def __call__(
        self, iq: np.ndarray, desc: SignalDescription, **kwargs
    ) -> Tuple[np.ndarray, SignalDescription]:
        raise NotImplementedError(
            "NoiseProfileTransfer is a planned research transform. "
            "Contributions welcome — see the design spec in "
            "docs/plans/2026-03-11-domain-adaptation-transforms.md"
        )


class ReceiverEQ(Transform):
    """Equalize receiver frequency response using a reference PSD profile.

    .. note:: This is a research-grade transform. It is not yet implemented
       and will raise ``NotImplementedError`` when called.

    Intended approach:

    1. Load or accept a reference PSD from a calibration capture of a
       known flat-spectrum signal.
    2. Compute the ratio ``current_psd / reference_psd``.
    3. Apply the inverse filter to equalize the receiver response.

    Args:
        reference_psd: Reference PSD array or path to file containing
            one. Should be from a flat-spectrum calibration signal.
    """

    def __init__(self, reference_psd):
        self._ref = reference_psd

    def __call__(
        self, iq: np.ndarray, desc: SignalDescription, **kwargs
    ) -> Tuple[np.ndarray, SignalDescription]:
        raise NotImplementedError(
            "ReceiverEQ is a planned research transform. "
            "Contributions welcome — see the design spec in "
            "docs/plans/2026-03-11-domain-adaptation-transforms.md"
        )
```

- [ ] **Step 4: Run tests to verify they pass**

Run: `source .venv/bin/activate && pytest tests/test_transforms_alignment.py -v`
Expected: All 42 tests PASS

- [ ] **Step 5: Commit**

```bash
git add python/spectra/transforms/alignment.py tests/test_transforms_alignment.py
git commit -m "feat(transforms): add NoiseProfileTransfer and ReceiverEQ stubs"
```

---

### Task 7: Compose integration test and exports

**Files:**
- Modify: `python/spectra/transforms/__init__.py`
- Modify: `tests/test_transforms_alignment.py`

- [ ] **Step 1: Write Compose integration test**

Append to `tests/test_transforms_alignment.py`:

```python
from spectra.impairments import Compose


class TestComposeIntegration:
    def test_chain_produces_valid_iq(self, sample_iq, sample_desc):
        chain = Compose([
            DCRemove(),
            AGCNormalize(method="rms"),
            SpectralWhitening(smoothing_window=32),
        ])
        iq_out, desc_out = chain(sample_iq, sample_desc, sample_rate=1e6)
        assert isinstance(iq_out, np.ndarray)
        assert iq_out.dtype == np.complex64
        assert len(iq_out) == len(sample_iq)
        assert not np.any(np.isnan(iq_out))
        assert not np.any(np.isinf(iq_out))

    def test_chain_all_statistical(self, sample_iq, sample_desc):
        chain = Compose([
            DCRemove(),
            ClipNormalize(clip_sigma=3.0),
            PowerNormalize(target_power_dbfs=-20.0),
        ])
        iq_out, _ = chain(sample_iq, sample_desc, sample_rate=1e6)
        rms = np.sqrt(np.mean(np.abs(iq_out) ** 2))
        rms_db = 20.0 * np.log10(rms)
        assert abs(rms_db - (-20.0)) < 0.1

    def test_all_deterministic(self, sample_iq, sample_desc):
        chain = Compose([
            DCRemove(),
            AGCNormalize(),
            SpectralWhitening(),
        ])
        iq1, _ = chain(sample_iq.copy(), sample_desc, sample_rate=1e6)
        iq2, _ = chain(sample_iq.copy(), sample_desc, sample_rate=1e6)
        npt.assert_array_equal(iq1, iq2)
```

- [ ] **Step 2: Update `python/spectra/transforms/__init__.py` with exports**

Add to the import section of `python/spectra/transforms/__init__.py`:

```python
from spectra.transforms.alignment import (
    AGCNormalize,
    BandpassAlign,
    ClipNormalize,
    DCRemove,
    NoiseFloorMatch,
    NoiseProfileTransfer,
    PowerNormalize,
    ReceiverEQ,
    Resample,
    SpectralWhitening,
)
```

Add to the `__all__` list:

```python
    "AGCNormalize",
    "BandpassAlign",
    "ClipNormalize",
    "DCRemove",
    "NoiseFloorMatch",
    "NoiseProfileTransfer",
    "PowerNormalize",
    "ReceiverEQ",
    "Resample",
    "SpectralWhitening",
```

- [ ] **Step 3: Run full test suite to verify no regressions**

Run: `source .venv/bin/activate && pytest tests/test_transforms_alignment.py -v`
Expected: All 45 tests PASS

Run: `source .venv/bin/activate && pytest tests/ -v`
Expected: All existing tests still pass

- [ ] **Step 4: Commit**

```bash
git add python/spectra/transforms/__init__.py python/spectra/transforms/alignment.py tests/test_transforms_alignment.py
git commit -m "feat(transforms): register alignment transforms and add Compose integration tests"
```

---

### Task 8: Final verification

- [ ] **Step 1: Run full test suite**

```bash
source .venv/bin/activate && pytest tests/ -v
```

Expected: All tests pass, zero regressions.

- [ ] **Step 2: Verify imports work**

```bash
source .venv/bin/activate && python -c "
from spectra.transforms import (
    DCRemove, ClipNormalize, PowerNormalize, AGCNormalize,
    Resample, SpectralWhitening, NoiseFloorMatch, BandpassAlign,
    NoiseProfileTransfer, ReceiverEQ,
)
from spectra.impairments import Compose
chain = Compose([DCRemove(), AGCNormalize(), SpectralWhitening()])
print(f'Loaded {len(chain.transforms)} transforms OK')
"
```

Expected: `Loaded 3 transforms OK`
