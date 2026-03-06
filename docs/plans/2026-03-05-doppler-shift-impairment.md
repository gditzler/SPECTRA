# Doppler Shift Impairment Implementation Plan

> **For Claude:** REQUIRED SUB-SKILL: Use superpowers:executing-plans to implement this plan task-by-task.

**Goal:** Add a `DopplerShift` impairment that applies a time-varying frequency shift to IQ signals to model relative motion between transmitter and receiver.

**Architecture:** A single `DopplerShift` class in `python/spectra/impairments/doppler.py` that accepts either direct Hz parameters or physical parameters (speed, angle, carrier frequency), supports two velocity profiles (`"constant"` and `"linear"` flyby), and updates `SignalDescription` to reflect the Doppler-shifted center frequency.

**Tech Stack:** Python, NumPy, existing `Transform` base class, `SignalDescription` dataclass, `pytest`.

---

## Design Rationale (read before implementing)

Doppler shift arises from relative radial motion. The physics:

```
f_d = (v_radial / c) * f_carrier
    = (v * cos(theta) / c) * f_carrier
```

Where:
- `v` = speed of transmitter/receiver relative to the other (m/s)
- `theta` = angle between velocity vector and line-of-sight (radians)
- `c = 3e8` m/s (speed of light)
- `f_carrier` = carrier/center frequency of the signal (Hz)

**What the impairment does in IQ domain:**
- Multiply the baseband IQ signal by a complex exponential `exp(j * 2*pi * phi(t))`
- For **constant velocity**: `phi(t) = fd * t` → constant frequency shift (like `FrequencyOffset`, but updates `desc` and is grounded in physical parameters)
- For **linear flyby**: `phi(t) = integral of fd(t) dt` where `fd(t)` goes from `+fd_max` to `-fd_max` linearly (approaching → broadside → receding). This is the unique value over `FrequencyOffset`.

**Environment:** Not needed. Terrain/sea affect multipath fading (use `RayleighFading`), not Doppler.
**Distance:** Not needed. Only radial velocity matters for frequency shift; distance only affects path loss.
**Angle:** Optionally useful via physical params. Default is to specify `fd_hz` directly.

---

### Task 1: Create the `DopplerShift` impairment class

**Files:**
- Create: `python/spectra/impairments/doppler.py`
- Reference: `python/spectra/impairments/frequency_offset.py` (pattern to follow)
- Reference: `python/spectra/impairments/frequency_drift.py` (time-varying phase pattern)
- Reference: `python/spectra/scene/signal_desc.py` (SignalDescription fields)

**Step 1: Write the failing tests first**

Create `tests/test_impairments_doppler.py`:

```python
import numpy as np
import numpy.testing as npt
import pytest

from spectra.scene.signal_desc import SignalDescription


def _make_desc(f_center=100e6):
    bw = 200e3
    return SignalDescription(0.0, 0.001, f_center - bw / 2, f_center + bw / 2, "QPSK", 20.0)


class TestDopplerShift:
    # --- Construction ---

    def test_requires_at_least_one_param(self):
        from spectra.impairments.doppler import DopplerShift
        with pytest.raises(ValueError, match="fd_hz"):
            DopplerShift()

    def test_construct_with_fd_hz(self):
        from spectra.impairments.doppler import DopplerShift
        d = DopplerShift(fd_hz=1000.0)
        assert d is not None

    def test_construct_with_max_fd_hz(self):
        from spectra.impairments.doppler import DopplerShift
        d = DopplerShift(max_fd_hz=5000.0)
        assert d is not None

    def test_construct_with_physical_params(self):
        from spectra.impairments.doppler import DopplerShift
        # 30 m/s (highway speed), head-on approach, 2.4 GHz carrier
        d = DopplerShift(speed_mps=30.0, carrier_hz=2.4e9, angle_deg=0.0)
        assert d is not None

    def test_physical_params_require_speed_and_carrier(self):
        from spectra.impairments.doppler import DopplerShift
        with pytest.raises(ValueError):
            DopplerShift(speed_mps=30.0)  # missing carrier_hz

    def test_invalid_profile_raises(self):
        from spectra.impairments.doppler import DopplerShift
        with pytest.raises(ValueError, match="profile"):
            DopplerShift(fd_hz=100.0, profile="random_walk")

    # --- Requires sample_rate ---

    def test_requires_sample_rate(self):
        from spectra.impairments.doppler import DopplerShift
        iq = np.ones(512, dtype=np.complex64)
        desc = _make_desc()
        with pytest.raises(ValueError, match="sample_rate"):
            DopplerShift(fd_hz=1000.0)(iq, desc)

    # --- Output shape and dtype ---

    def test_output_shape_and_dtype(self, sample_rate):
        from spectra.impairments.doppler import DopplerShift
        iq = np.ones(1024, dtype=np.complex64)
        desc = _make_desc()
        result, _ = DopplerShift(fd_hz=1000.0)(iq, desc, sample_rate=sample_rate)
        assert result.shape == iq.shape
        assert result.dtype == np.complex64

    # --- No NaNs or Infs ---

    def test_no_nans_or_infs_constant(self, sample_rate):
        from spectra.impairments.doppler import DopplerShift
        iq = np.ones(2048, dtype=np.complex64)
        desc = _make_desc()
        result, _ = DopplerShift(fd_hz=5000.0)(iq, desc, sample_rate=sample_rate)
        assert not np.any(np.isnan(result))
        assert not np.any(np.isinf(result))

    def test_no_nans_or_infs_linear(self, sample_rate):
        from spectra.impairments.doppler import DopplerShift
        iq = np.ones(2048, dtype=np.complex64)
        desc = _make_desc()
        result, _ = DopplerShift(fd_hz=5000.0, profile="linear")(
            iq, desc, sample_rate=sample_rate
        )
        assert not np.any(np.isnan(result))
        assert not np.any(np.isinf(result))

    # --- Constant Doppler: phase ramp correctness ---

    def test_constant_doppler_is_frequency_shift(self, sample_rate):
        """
        Constant Doppler on a pure tone at 0 Hz should produce a tone at fd Hz.
        Verify by measuring the instantaneous frequency.
        """
        from spectra.impairments.doppler import DopplerShift
        n = 4096
        fd = 1000.0
        iq = np.ones(n, dtype=np.complex64)
        desc = _make_desc()
        result, _ = DopplerShift(fd_hz=fd)(iq, desc, sample_rate=sample_rate)
        # Instantaneous phase should increase by 2*pi*fd/fs per sample
        phase = np.unwrap(np.angle(result))
        inst_freq = np.diff(phase) / (2 * np.pi) * sample_rate
        npt.assert_allclose(inst_freq, fd, atol=1.0)

    # --- Linear profile: zero net shift (symmetric flyby) ---

    def test_linear_doppler_zero_net_phase(self, sample_rate):
        """Linear profile: fd goes +fd to -fd, net phase change ~ 0."""
        from spectra.impairments.doppler import DopplerShift
        n = 4096
        fd = 1000.0
        iq = np.ones(n, dtype=np.complex64)
        desc = _make_desc()
        result, _ = DopplerShift(fd_hz=fd, profile="linear")(
            iq, desc, sample_rate=sample_rate
        )
        # Total phase accumulated = integral of fd(t) from 0 to T
        # fd(t) = fd*(1 - 2t/T), integral = fd*T*(1 - 1) = 0
        phase = np.unwrap(np.angle(result))
        total_phase = phase[-1] - phase[0]
        npt.assert_allclose(total_phase, 0.0, atol=0.1)

    # --- SignalDescription update ---

    def test_constant_desc_f_center_updated(self, sample_rate):
        """Constant Doppler: f_low and f_high shift by fd."""
        from spectra.impairments.doppler import DopplerShift
        iq = np.ones(1024, dtype=np.complex64)
        f_center = 100e6
        desc = _make_desc(f_center=f_center)
        fd = 2000.0
        _, new_desc = DopplerShift(fd_hz=fd)(iq, desc, sample_rate=sample_rate)
        npt.assert_allclose(new_desc.f_low, desc.f_low + fd, atol=1.0)
        npt.assert_allclose(new_desc.f_high, desc.f_high + fd, atol=1.0)

    def test_linear_desc_unchanged(self, sample_rate):
        """Linear flyby: f_center net shift is zero, desc unchanged."""
        from spectra.impairments.doppler import DopplerShift
        iq = np.ones(1024, dtype=np.complex64)
        desc = _make_desc()
        _, new_desc = DopplerShift(fd_hz=2000.0, profile="linear")(
            iq, desc, sample_rate=sample_rate
        )
        assert new_desc.f_low == desc.f_low
        assert new_desc.f_high == desc.f_high

    # --- Physical parameter construction computes correct fd ---

    def test_physical_params_head_on(self, sample_rate):
        """Head-on approach at 30 m/s at 1 GHz ~ 100 Hz Doppler."""
        from spectra.impairments.doppler import DopplerShift
        # f_d = v/c * f_c = 30/3e8 * 1e9 = 100 Hz
        d = DopplerShift(speed_mps=30.0, carrier_hz=1e9, angle_deg=0.0)
        iq = np.ones(4096, dtype=np.complex64)
        desc = _make_desc()
        result, new_desc = d(iq, desc, sample_rate=sample_rate)
        npt.assert_allclose(new_desc.f_low, desc.f_low + 100.0, atol=1.0)

    def test_physical_params_perpendicular_no_shift(self, sample_rate):
        """90-degree angle: radial velocity = 0, no Doppler shift."""
        from spectra.impairments.doppler import DopplerShift
        d = DopplerShift(speed_mps=100.0, carrier_hz=2.4e9, angle_deg=90.0)
        iq = np.ones(1024, dtype=np.complex64)
        desc = _make_desc()
        result, new_desc = d(iq, desc, sample_rate=sample_rate)
        npt.assert_allclose(result, iq, atol=1e-4)
        assert new_desc.f_low == desc.f_low

    # --- Randomized max_fd_hz produces variation ---

    def test_max_fd_hz_randomizes(self, sample_rate):
        from spectra.impairments.doppler import DopplerShift
        d = DopplerShift(max_fd_hz=5000.0)
        iq = np.ones(1024, dtype=np.complex64)
        desc = _make_desc()
        results = [d(iq.copy(), desc, sample_rate=sample_rate)[0] for _ in range(10)]
        diffs = [np.max(np.abs(results[i] - results[i + 1])) for i in range(9)]
        assert not all(d < 1e-6 for d in diffs)

    # --- Power preservation ---

    def test_power_preserved(self, sample_rate):
        """Doppler shift is a phase rotation: power must be preserved."""
        from spectra.impairments.doppler import DopplerShift
        iq = (np.random.randn(2048) + 1j * np.random.randn(2048)).astype(np.complex64)
        desc = _make_desc()
        for profile in ("constant", "linear"):
            result, _ = DopplerShift(fd_hz=3000.0, profile=profile)(
                iq, desc, sample_rate=sample_rate
            )
            npt.assert_allclose(
                np.mean(np.abs(result) ** 2),
                np.mean(np.abs(iq) ** 2),
                rtol=1e-4,
            )
```

**Step 2: Run tests to verify they all fail**

```bash
cd /path/to/SPECTRA && source .venv/bin/activate
pytest tests/test_impairments_doppler.py -v 2>&1 | head -30
```

Expected: `ImportError: cannot import name 'DopplerShift'`

**Step 3: Implement `DopplerShift`**

Create `python/spectra/impairments/doppler.py`:

```python
from dataclasses import replace
from typing import Optional, Tuple

import numpy as np

from spectra.impairments.base import Transform
from spectra.scene.signal_desc import SignalDescription

_C = 3e8  # speed of light (m/s)


class DopplerShift(Transform):
    """
    Doppler frequency shift from relative motion between transmitter and receiver.

    Parameterization options (exactly one group must be provided):
      - ``fd_hz``: fixed Doppler shift in Hz
      - ``max_fd_hz``: random Doppler shift drawn from Uniform(-max, +max) each call
      - ``speed_mps`` + ``carrier_hz``: physical parameters; angle_deg defaults to 0
        (head-on approach). Computes fd = speed * cos(angle) / c * carrier_hz.

    Profiles:
      - ``"constant"`` (default): constant radial velocity throughout the signal.
        Phase ramp = 2*pi*fd*t. SignalDescription f_low/f_high are shifted by fd.
      - ``"linear"``: velocity reverses linearly (flyby scenario: approaching then
        receding). fd varies from +fd to -fd. Net phase ~ 0; SignalDescription
        is unchanged.

    Parameters
    ----------
    fd_hz : float, optional
        Fixed Doppler shift in Hz. Positive = approaching.
    max_fd_hz : float, optional
        Maximum Doppler shift magnitude in Hz; actual fd drawn from Uniform(-max, max).
    speed_mps : float, optional
        Relative speed in m/s. Requires ``carrier_hz``.
    carrier_hz : float, optional
        Carrier/center frequency in Hz. Required with ``speed_mps``.
    angle_deg : float, optional
        Angle between velocity vector and line-of-sight in degrees (default 0 = head-on).
    profile : {"constant", "linear"}, optional
        Velocity profile over the signal duration. Default "constant".
    """

    _VALID_PROFILES = ("constant", "linear")

    def __init__(
        self,
        fd_hz: Optional[float] = None,
        max_fd_hz: Optional[float] = None,
        speed_mps: Optional[float] = None,
        carrier_hz: Optional[float] = None,
        angle_deg: float = 0.0,
        profile: str = "constant",
    ):
        if profile not in self._VALID_PROFILES:
            raise ValueError(
                f"profile must be one of {self._VALID_PROFILES}, got {profile!r}"
            )

        # Validate that exactly one parameterization is provided
        has_direct = fd_hz is not None or max_fd_hz is not None
        has_physical = speed_mps is not None

        if not has_direct and not has_physical:
            raise ValueError(
                "Provide fd_hz, max_fd_hz, or (speed_mps + carrier_hz)"
            )
        if has_physical and carrier_hz is None:
            raise ValueError("speed_mps requires carrier_hz")

        self._fd_hz = fd_hz
        self._max_fd_hz = max_fd_hz
        self._speed_mps = speed_mps
        self._carrier_hz = carrier_hz
        self._angle_deg = angle_deg
        self._profile = profile

    def _resolve_fd(self) -> float:
        """Compute the Doppler shift in Hz for this call."""
        if self._speed_mps is not None:
            return (
                self._speed_mps
                * np.cos(np.radians(self._angle_deg))
                / _C
                * self._carrier_hz
            )
        if self._max_fd_hz is not None:
            return float(np.random.uniform(-self._max_fd_hz, self._max_fd_hz))
        return float(self._fd_hz)

    def __call__(
        self, iq: np.ndarray, desc: SignalDescription, **kwargs
    ) -> Tuple[np.ndarray, SignalDescription]:
        sample_rate = kwargs.get("sample_rate")
        if sample_rate is None:
            raise ValueError("DopplerShift requires sample_rate kwarg")

        n = len(iq)
        fd = self._resolve_fd()
        t = np.arange(n) / sample_rate

        if self._profile == "constant":
            phase = 2.0 * np.pi * fd * t
            new_desc = replace(desc, f_low=desc.f_low + fd, f_high=desc.f_high + fd)
        else:  # "linear" flyby: fd goes from +fd to -fd
            fd_t = np.linspace(fd, -fd, n)
            phase = 2.0 * np.pi * np.cumsum(fd_t) / sample_rate
            new_desc = desc  # net shift is zero

        out = (iq * np.exp(1j * phase).astype(np.complex64)).astype(np.complex64)
        return out, new_desc
```

**Step 4: Run tests to verify they pass**

```bash
pytest tests/test_impairments_doppler.py -v
```

Expected: all tests PASS.

**Step 5: Commit**

```bash
git add python/spectra/impairments/doppler.py tests/test_impairments_doppler.py
git commit -m "feat(impairments): add DopplerShift impairment with constant and linear profiles"
```

---

### Task 2: Export `DopplerShift` from the impairments package

**Files:**
- Modify: `python/spectra/impairments/__init__.py`

**Step 1: Write a failing import test**

Add to `tests/test_impairments_doppler.py` at the top level (outside any class):

```python
def test_doppler_exported_from_package():
    from spectra.impairments import DopplerShift
    assert DopplerShift is not None
```

**Step 2: Run the test to see it fail**

```bash
pytest tests/test_impairments_doppler.py::test_doppler_exported_from_package -v
```

Expected: `ImportError: cannot import name 'DopplerShift' from 'spectra.impairments'`

**Step 3: Add the export**

Edit `python/spectra/impairments/__init__.py` — add import line after the `dc_offset` import:

```python
from spectra.impairments.doppler import DopplerShift
```

And add `"DopplerShift"` to `__all__` in alphabetical order (between `"DCOffset"` and `"FrequencyDrift"`).

**Step 4: Run all impairment tests**

```bash
pytest tests/test_impairments_doppler.py -v
pytest tests/test_impairments.py -v
```

Expected: all tests PASS.

**Step 5: Commit**

```bash
git add python/spectra/impairments/__init__.py tests/test_impairments_doppler.py
git commit -m "feat(impairments): export DopplerShift from spectra.impairments"
```

---

### Task 3: Run the full test suite to confirm no regressions

**Step 1: Run the full test suite**

```bash
pytest tests/ -v
```

Expected: all existing tests pass, all new Doppler tests pass.

**Step 2: If failures occur**

- If a test imports fail: check that `maturin develop --release` has been run after any Rust changes (Doppler is pure Python, so this should not be needed here).
- If unrelated tests fail: they were already broken before this change — check `git stash && pytest tests/ -v` to confirm.

**Step 3: Commit only if there were fixups needed**

If you had to make fixups, commit them:

```bash
git add <fixed files>
git commit -m "fix(impairments): resolve test suite regressions from DopplerShift addition"
```

---

## Verification Checklist

Before declaring done:

- [ ] `pytest tests/test_impairments_doppler.py -v` — all pass
- [ ] `pytest tests/test_impairments.py -v` — no regressions
- [ ] `from spectra.impairments import DopplerShift` works in a Python REPL
- [ ] `DopplerShift` appears in `spectra.impairments.__all__`
- [ ] Physical param test: `DopplerShift(speed_mps=30, carrier_hz=1e9, angle_deg=0)` gives ~100 Hz shift
- [ ] Perpendicular test: `angle_deg=90` gives zero shift
