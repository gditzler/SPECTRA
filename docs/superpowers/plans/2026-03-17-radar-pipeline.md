# Radar Processing Pipeline Implementation Plan

> **For agentic workers:** REQUIRED: Use superpowers:subagent-driven-development (if subagents available) or superpowers:executing-plans to implement this plan. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Add an end-to-end radar processing pipeline to SPECTRA: target kinematics → RCS → clutter → MTI → Kalman tracker → pipeline dataset.

**Architecture:** Five sub-projects built independently then integrated. `spectra/targets/` for kinematics and RCS, `spectra/impairments/clutter.py` for radar clutter (standalone callable, not Transform subclass), `spectra/algorithms/mti.py` for MTI processing, `spectra/tracking/` for Kalman filter, and `spectra/datasets/radar_pipeline.py` for the end-to-end dataset.

**Tech Stack:** Python 3.10+, NumPy, PyTorch, pytest. No new dependencies.

**Spec:** `docs/superpowers/specs/2026-03-17-radar-pipeline-design.md`

---

## File Structure

| File | Responsibility |
|------|---------------|
| `python/spectra/targets/__init__.py` | Package init, exports Trajectory protocol + all models |
| `python/spectra/targets/trajectory.py` | ConstantVelocity and ConstantTurnRate motion models |
| `python/spectra/targets/rcs.py` | NonFluctuatingRCS and SwerlingRCS amplitude models |
| `python/spectra/impairments/clutter.py` | RadarClutter callable with terrain presets |
| `python/spectra/impairments/__init__.py` | Modify: add RadarClutter export |
| `python/spectra/algorithms/mti.py` | Pulse cancellers and Doppler filter bank |
| `python/spectra/algorithms/__init__.py` | Modify: add MTI exports |
| `python/spectra/tracking/__init__.py` | Package init, exports KalmanFilter + factory |
| `python/spectra/tracking/kalman.py` | Generic KalmanFilter + ConstantVelocityKF factory |
| `python/spectra/datasets/radar_pipeline.py` | RadarPipelineDataset + RadarPipelineTarget |
| `python/spectra/datasets/__init__.py` | Modify: add pipeline exports |
| `tests/test_trajectory.py` | Tests for ConstantVelocity and ConstantTurnRate |
| `tests/test_rcs.py` | Tests for NonFluctuatingRCS and SwerlingRCS |
| `tests/test_clutter.py` | Tests for RadarClutter |
| `tests/test_mti.py` | Tests for MTI algorithms |
| `tests/test_kalman.py` | Tests for KalmanFilter |
| `tests/test_radar_pipeline.py` | Tests for RadarPipelineDataset |

---

## Task 1: Target Trajectories (`spectra/targets/trajectory.py`)

**Files:**
- Create: `python/spectra/targets/__init__.py`
- Create: `python/spectra/targets/trajectory.py`
- Create: `tests/test_trajectory.py`

- [ ] **Step 1: Write the failing tests**

```python
# tests/test_trajectory.py
"""Tests for target trajectory motion models."""
import numpy as np
import pytest


def test_cv_initial_state():
    from spectra.targets.trajectory import ConstantVelocity
    cv = ConstantVelocity(initial_range=1000.0, velocity=50.0, dt=1.0)
    state = cv.state_at(0)
    assert state.shape == (2,)
    assert state[0] == pytest.approx(1000.0)
    assert state[1] == pytest.approx(50.0)


def test_cv_propagation():
    from spectra.targets.trajectory import ConstantVelocity
    cv = ConstantVelocity(initial_range=1000.0, velocity=50.0, dt=1.0)
    state = cv.state_at(10)
    assert state[0] == pytest.approx(1500.0)  # 1000 + 50*10
    assert state[1] == pytest.approx(50.0)


def test_cv_range_at():
    from spectra.targets.trajectory import ConstantVelocity
    cv = ConstantVelocity(initial_range=500.0, velocity=-20.0, dt=0.5)
    r = cv.range_at(4)
    assert r == pytest.approx(500.0 + (-20.0) * 4 * 0.5)


def test_cv_states_shape():
    from spectra.targets.trajectory import ConstantVelocity
    cv = ConstantVelocity(initial_range=100.0, velocity=10.0, dt=0.1)
    states = cv.states(20)
    assert states.shape == (20, 2)
    # First state is initial
    assert states[0, 0] == pytest.approx(100.0)
    assert states[0, 1] == pytest.approx(10.0)
    # Last state
    assert states[19, 0] == pytest.approx(100.0 + 10.0 * 19 * 0.1)


def test_ct_initial_state():
    from spectra.targets.trajectory import ConstantTurnRate
    ct = ConstantTurnRate(initial_range=2000.0, velocity=100.0, turn_rate=0.1, dt=1.0)
    state = ct.state_at(0)
    assert state.shape == (2,)
    assert state[0] == pytest.approx(2000.0)


def test_ct_range_changes():
    """CT range should vary non-linearly (sinusoidal character)."""
    from spectra.targets.trajectory import ConstantTurnRate
    ct = ConstantTurnRate(initial_range=5000.0, velocity=200.0, turn_rate=0.05, dt=1.0)
    ranges = [ct.range_at(t) for t in range(50)]
    # Range should not be constant
    assert max(ranges) != min(ranges)
    # Range rate should vary (not constant like CV)
    states = ct.states(50)
    rates = states[:, 1]
    assert not np.allclose(rates, rates[0])


def test_ct_states_shape():
    from spectra.targets.trajectory import ConstantTurnRate
    ct = ConstantTurnRate(initial_range=1000.0, velocity=50.0, turn_rate=0.1, dt=0.5)
    states = ct.states(30)
    assert states.shape == (30, 2)


def test_trajectory_protocol():
    from spectra.targets.trajectory import ConstantVelocity, ConstantTurnRate, Trajectory
    cv = ConstantVelocity(initial_range=100.0, velocity=10.0, dt=1.0)
    ct = ConstantTurnRate(initial_range=100.0, velocity=10.0, turn_rate=0.1, dt=1.0)
    assert isinstance(cv, Trajectory)
    assert isinstance(ct, Trajectory)
```

- [ ] **Step 2: Run tests to verify they fail**

```bash
/Users/gditzler/.venvs/base/bin/pytest tests/test_trajectory.py -v
```
Expected: `ModuleNotFoundError: No module named 'spectra.targets'`

- [ ] **Step 3: Write the implementation**

```python
# python/spectra/targets/trajectory.py
"""Target trajectory motion models for radar simulation.

Two motion models are provided:

- :class:`ConstantVelocity` — linear range propagation at fixed velocity.
- :class:`ConstantTurnRate` — 1-D range projection of a 2-D circular arc.

Both satisfy the :class:`Trajectory` protocol and can be used interchangeably
in :class:`~spectra.datasets.radar_pipeline.RadarPipelineDataset`.
"""

from __future__ import annotations

from typing import Protocol, runtime_checkable

import numpy as np


@runtime_checkable
class Trajectory(Protocol):
    """Protocol for target trajectory models."""

    def state_at(self, step: int) -> np.ndarray: ...
    def states(self, num_steps: int) -> np.ndarray: ...
    def range_at(self, step: int) -> float: ...


class ConstantVelocity:
    """Constant-velocity (linear) trajectory.

    State vector: ``[range, range_rate]``.

    Args:
        initial_range: Starting range in metres.
        velocity: Constant range rate in m/s (positive = opening).
        dt: Time step between trajectory steps in seconds.
    """

    def __init__(self, initial_range: float, velocity: float, dt: float) -> None:
        self.initial_range = initial_range
        self.velocity = velocity
        self.dt = dt

    def state_at(self, step: int) -> np.ndarray:
        t = step * self.dt
        return np.array([self.initial_range + self.velocity * t, self.velocity])

    def states(self, num_steps: int) -> np.ndarray:
        out = np.empty((num_steps, 2))
        for i in range(num_steps):
            out[i] = self.state_at(i)
        return out

    def range_at(self, step: int) -> float:
        return float(self.state_at(step)[0])


class ConstantTurnRate:
    """Constant-turn-rate trajectory (1-D range projection of 2-D turning).

    Models a target moving at constant speed on a circular arc, observed
    from a fixed radar at the origin. The target starts at ``(initial_range, 0)``.

    Propagation::

        x(t) = x0 + (v / omega) * sin(omega * t)
        y(t) = (v / omega) * (1 - cos(omega * t))
        range(t) = sqrt(x(t)^2 + y(t)^2)

    State vector: ``[range, range_rate]``.

    Args:
        initial_range: Starting range in metres.
        velocity: Constant speed in m/s.
        turn_rate: Turn rate in rad/s.
        dt: Time step between trajectory steps in seconds.
    """

    def __init__(
        self, initial_range: float, velocity: float, turn_rate: float, dt: float
    ) -> None:
        self.initial_range = initial_range
        self.velocity = velocity
        self.turn_rate = turn_rate
        self.dt = dt

    def _xy(self, t: float):
        omega = self.turn_rate
        v = self.velocity
        if abs(omega) < 1e-12:
            return self.initial_range + v * t, 0.0
        x = self.initial_range + (v / omega) * np.sin(omega * t)
        y = (v / omega) * (1.0 - np.cos(omega * t))
        return float(x), float(y)

    def state_at(self, step: int) -> np.ndarray:
        t = step * self.dt
        x, y = self._xy(t)
        r = np.sqrt(x**2 + y**2)
        # Numerical range rate via finite difference
        eps = self.dt * 0.001
        x2, y2 = self._xy(t + eps)
        r2 = np.sqrt(x2**2 + y2**2)
        rdot = (r2 - r) / eps
        return np.array([r, rdot])

    def states(self, num_steps: int) -> np.ndarray:
        out = np.empty((num_steps, 2))
        for i in range(num_steps):
            out[i] = self.state_at(i)
        return out

    def range_at(self, step: int) -> float:
        return float(self.state_at(step)[0])
```

```python
# python/spectra/targets/__init__.py (minimal — only trajectory exports for now)
from spectra.targets.trajectory import ConstantTurnRate, ConstantVelocity, Trajectory

__all__ = [
    "ConstantTurnRate",
    "ConstantVelocity",
    "Trajectory",
]
```

- [ ] **Step 4: Run tests to verify they pass**

```bash
/Users/gditzler/.venvs/base/bin/pytest tests/test_trajectory.py -v
```
Expected: 8/8 PASS

- [ ] **Step 5: Commit**

```bash
git add python/spectra/targets/__init__.py python/spectra/targets/trajectory.py tests/test_trajectory.py
git commit -m "feat(targets): add ConstantVelocity and ConstantTurnRate trajectory models"
```

---

## Task 2: Swerling RCS Models (`spectra/targets/rcs.py`)

**Files:**
- Create: `python/spectra/targets/rcs.py`
- Modify: `python/spectra/targets/__init__.py` (add RCS exports)
- Create: `tests/test_rcs.py`

- [ ] **Step 1: Write the failing tests**

```python
# tests/test_rcs.py
"""Tests for Swerling RCS fluctuation models."""
import numpy as np
import pytest


def test_non_fluctuating_shape():
    from spectra.targets.rcs import NonFluctuatingRCS
    rcs = NonFluctuatingRCS(sigma=1.0)
    rng = np.random.default_rng(42)
    amps = rcs.amplitudes(num_dwells=5, num_pulses_per_dwell=10, rng=rng)
    assert amps.shape == (5, 10)


def test_non_fluctuating_constant():
    from spectra.targets.rcs import NonFluctuatingRCS
    rcs = NonFluctuatingRCS(sigma=4.0)
    rng = np.random.default_rng(0)
    amps = rcs.amplitudes(num_dwells=3, num_pulses_per_dwell=8, rng=rng)
    # All values should be sqrt(sigma) = 2.0
    assert np.allclose(amps, 2.0)


def test_swerling_i_constant_within_dwell():
    from spectra.targets.rcs import SwerlingRCS
    rcs = SwerlingRCS(case=1, sigma=1.0)
    rng = np.random.default_rng(42)
    amps = rcs.amplitudes(num_dwells=10, num_pulses_per_dwell=20, rng=rng)
    assert amps.shape == (10, 20)
    # Each row (dwell) should have constant value
    for row in range(10):
        assert np.allclose(amps[row, :], amps[row, 0])


def test_swerling_ii_varies_pulse_to_pulse():
    from spectra.targets.rcs import SwerlingRCS
    rcs = SwerlingRCS(case=2, sigma=1.0)
    rng = np.random.default_rng(42)
    amps = rcs.amplitudes(num_dwells=1, num_pulses_per_dwell=100, rng=rng)
    assert amps.shape == (1, 100)
    # Values should NOT all be equal (pulse-to-pulse fluctuation)
    assert not np.allclose(amps[0, :], amps[0, 0])


def test_swerling_iii_constant_within_dwell():
    from spectra.targets.rcs import SwerlingRCS
    rcs = SwerlingRCS(case=3, sigma=1.0)
    rng = np.random.default_rng(42)
    amps = rcs.amplitudes(num_dwells=5, num_pulses_per_dwell=16, rng=rng)
    for row in range(5):
        assert np.allclose(amps[row, :], amps[row, 0])


def test_swerling_iv_varies_pulse_to_pulse():
    from spectra.targets.rcs import SwerlingRCS
    rcs = SwerlingRCS(case=4, sigma=1.0)
    rng = np.random.default_rng(42)
    amps = rcs.amplitudes(num_dwells=1, num_pulses_per_dwell=100, rng=rng)
    assert not np.allclose(amps[0, :], amps[0, 0])


def test_swerling_mean_scales_with_sigma():
    """Mean RCS amplitude squared should approximate sigma."""
    from spectra.targets.rcs import SwerlingRCS
    sigma = 5.0
    rcs = SwerlingRCS(case=2, sigma=sigma)
    rng = np.random.default_rng(42)
    amps = rcs.amplitudes(num_dwells=1, num_pulses_per_dwell=10000, rng=rng)
    # amps are sqrt(rcs_draw / sigma) * sqrt(sigma) = sqrt(rcs_draw)
    # so amps**2 should have mean ~ sigma
    mean_power = np.mean(amps**2)
    assert abs(mean_power - sigma) / sigma < 0.1, f"Mean power {mean_power:.2f} != {sigma}"


def test_swerling_invalid_case():
    from spectra.targets.rcs import SwerlingRCS
    with pytest.raises(ValueError, match="case"):
        SwerlingRCS(case=5, sigma=1.0)
```

- [ ] **Step 2: Run tests to verify they fail**

```bash
/Users/gditzler/.venvs/base/bin/pytest tests/test_rcs.py -v
```
Expected: `ModuleNotFoundError: No module named 'spectra.targets.rcs'`

- [ ] **Step 3: Write the implementation**

```python
# python/spectra/targets/rcs.py
"""Swerling RCS fluctuation models for radar target simulation.

Provides :class:`NonFluctuatingRCS` (Swerling 0/V) and :class:`SwerlingRCS`
(cases I–IV) for generating per-pulse amplitude scale factors.
"""

from __future__ import annotations

import numpy as np


class NonFluctuatingRCS:
    """Non-fluctuating RCS (Swerling case 0/V).

    Returns constant amplitude ``sqrt(sigma)`` for every pulse.

    Args:
        sigma: Mean radar cross-section (linear scale).
    """

    def __init__(self, sigma: float) -> None:
        self.sigma = sigma

    def amplitudes(
        self, num_dwells: int, num_pulses_per_dwell: int, rng: np.random.Generator
    ) -> np.ndarray:
        """Return constant amplitude array, shape ``(num_dwells, num_pulses_per_dwell)``."""
        return np.full((num_dwells, num_pulses_per_dwell), np.sqrt(self.sigma))


class SwerlingRCS:
    """Swerling fluctuating RCS model (cases I–IV).

    Generates amplitude scale factors drawn from the appropriate chi-squared
    distribution:

    - **Cases I, II:** Chi-squared with 2 degrees of freedom (exponential).
    - **Cases III, IV:** Chi-squared with 4 degrees of freedom.
    - **Cases I, III:** Constant within each dwell (scan-to-scan fluctuation).
    - **Cases II, IV:** Independent pulse-to-pulse fluctuation.

    Args:
        case: Swerling case number (1, 2, 3, or 4).
        sigma: Mean radar cross-section (linear scale).
    """

    _VALID_CASES = {1, 2, 3, 4}

    def __init__(self, case: int, sigma: float) -> None:
        if case not in self._VALID_CASES:
            raise ValueError(
                f"Swerling case must be one of {self._VALID_CASES}, got {case}"
            )
        self.case = case
        self.sigma = sigma

    def amplitudes(
        self, num_dwells: int, num_pulses_per_dwell: int, rng: np.random.Generator
    ) -> np.ndarray:
        """Return amplitude scale factors, shape ``(num_dwells, num_pulses_per_dwell)``.

        Cases I/III broadcast a single draw per dwell across all pulses.
        Cases II/IV draw independently for each (dwell, pulse) entry.
        """
        scan_to_scan = self.case in (1, 3)
        dof = 2 if self.case in (1, 2) else 4

        if scan_to_scan:
            # One draw per dwell, broadcast across pulses
            rcs_draws = rng.chisquare(dof, size=num_dwells) * (self.sigma / dof)
            rcs_draws = np.repeat(rcs_draws[:, np.newaxis], num_pulses_per_dwell, axis=1)
        else:
            # Independent draw per (dwell, pulse)
            rcs_draws = rng.chisquare(dof, size=(num_dwells, num_pulses_per_dwell)) * (
                self.sigma / dof
            )

        # Convert RCS power to amplitude scale factor
        return np.sqrt(rcs_draws)
```

- [ ] **Step 4: Update `targets/__init__.py` to export RCS classes**

Replace the contents of `python/spectra/targets/__init__.py` with:

```python
# python/spectra/targets/__init__.py
from spectra.targets.rcs import NonFluctuatingRCS, SwerlingRCS
from spectra.targets.trajectory import ConstantTurnRate, ConstantVelocity, Trajectory

__all__ = [
    "ConstantTurnRate",
    "ConstantVelocity",
    "NonFluctuatingRCS",
    "SwerlingRCS",
    "Trajectory",
]
```

- [ ] **Step 5: Run tests to verify they pass**

```bash
/Users/gditzler/.venvs/base/bin/pytest tests/test_rcs.py -v
```
Expected: 8/8 PASS

- [ ] **Step 6: Commit**

```bash
git add python/spectra/targets/rcs.py python/spectra/targets/__init__.py tests/test_rcs.py
git commit -m "feat(targets): add NonFluctuatingRCS and SwerlingRCS models (cases 0-IV)"
```

---

## Task 3: Radar Clutter (`spectra/impairments/clutter.py`)

**Files:**
- Create: `python/spectra/impairments/clutter.py`
- Modify: `python/spectra/impairments/__init__.py`
- Create: `tests/test_clutter.py`

- [ ] **Step 1: Write the failing tests**

```python
# tests/test_clutter.py
"""Tests for RadarClutter impairment."""
import numpy as np
import pytest


def _make_pulse_matrix(num_pulses=32, num_range_bins=128, rng=None):
    """Create a clean pulse matrix for testing."""
    if rng is None:
        rng = np.random.default_rng(42)
    return np.zeros((num_pulses, num_range_bins), dtype=complex)


def test_clutter_output_shape():
    from spectra.impairments.clutter import RadarClutter
    clutter = RadarClutter(cnr=20.0, doppler_spread=50.0, sample_rate=1e6)
    rng = np.random.default_rng(42)
    X = _make_pulse_matrix(num_pulses=16, num_range_bins=64, rng=rng)
    out = clutter(X, rng)
    assert out.shape == X.shape
    assert out.dtype == complex


def test_clutter_adds_power():
    from spectra.impairments.clutter import RadarClutter
    clutter = RadarClutter(cnr=30.0, doppler_spread=100.0, sample_rate=1e6)
    rng = np.random.default_rng(42)
    X = _make_pulse_matrix()
    out = clutter(X, rng)
    assert np.mean(np.abs(out) ** 2) > 0


def test_clutter_deterministic():
    from spectra.impairments.clutter import RadarClutter
    clutter = RadarClutter(cnr=20.0, doppler_spread=50.0, sample_rate=1e6)
    X = _make_pulse_matrix()
    out1 = clutter(X, np.random.default_rng(99))
    out2 = clutter(X, np.random.default_rng(99))
    assert np.allclose(out1, out2)


def test_clutter_range_extent():
    from spectra.impairments.clutter import RadarClutter
    clutter = RadarClutter(
        cnr=30.0, doppler_spread=50.0, sample_rate=1e6, range_extent=(10, 20)
    )
    rng = np.random.default_rng(42)
    X = _make_pulse_matrix(num_pulses=16, num_range_bins=64)
    out = clutter(X, rng)
    # Outside range_extent should be unchanged (input was zeros)
    assert np.allclose(out[:, :10], 0.0)
    assert np.allclose(out[:, 20:], 0.0)
    # Inside range_extent should have clutter
    assert np.mean(np.abs(out[:, 10:20]) ** 2) > 0


def test_clutter_doppler_spectrum_shape():
    """Clutter should have energy concentrated near doppler_center."""
    from spectra.impairments.clutter import RadarClutter
    clutter = RadarClutter(
        cnr=40.0, doppler_spread=10.0, doppler_center=0.0, sample_rate=1e6
    )
    rng = np.random.default_rng(42)
    X = _make_pulse_matrix(num_pulses=256, num_range_bins=1)
    out = clutter(X, rng)
    # FFT along slow-time dimension — energy should peak near DC
    spec = np.abs(np.fft.fft(out[:, 0])) ** 2
    dc_bin = 0
    dc_power = spec[dc_bin]
    edge_power = np.mean(spec[len(spec) // 4 : 3 * len(spec) // 4])
    assert dc_power > edge_power * 5, "Clutter spectrum should peak near DC"


def test_ground_preset():
    from spectra.impairments.clutter import RadarClutter
    clutter = RadarClutter.ground(sample_rate=1e6, terrain="rural")
    assert clutter.cnr > 0
    assert clutter.doppler_spread > 0


def test_sea_preset():
    from spectra.impairments.clutter import RadarClutter
    clutter = RadarClutter.sea(sample_rate=1e6, sea_state=3)
    assert clutter.cnr > 0


def test_weather_preset():
    from spectra.impairments.clutter import RadarClutter
    clutter = RadarClutter.weather(sample_rate=1e6, rain_rate_mmhr=10)
    assert clutter.doppler_center != 0.0  # Weather has non-zero Doppler center
```

- [ ] **Step 2: Run tests to verify they fail**

```bash
/Users/gditzler/.venvs/base/bin/pytest tests/test_clutter.py -v
```
Expected: `ModuleNotFoundError: No module named 'spectra.impairments.clutter'`

- [ ] **Step 3: Write the implementation**

```python
# python/spectra/impairments/clutter.py
"""Radar clutter model with terrain-typed presets.

:class:`RadarClutter` generates Doppler-colored complex Gaussian clutter
on a 2-D slow-time / fast-time matrix. It is a standalone callable (not a
:class:`~spectra.impairments.base.Transform` subclass) because radar clutter
operates on ``(num_pulses, num_range_bins)`` matrices rather than 1-D IQ.

Usage::

    clutter = RadarClutter.sea(sample_rate=1e6, sea_state=3)
    received = clutter(pulse_matrix, rng)
"""

from __future__ import annotations

from typing import Optional, Tuple

import numpy as np


# ── Preset lookup tables ────────────────────────────────────────────────────

_GROUND_PRESETS = {
    #                  CNR (dB), doppler_spread (Hz)
    "rural":          (20.0,     5.0),
    "urban":          (30.0,     2.0),
    "forest":         (25.0,     8.0),
    "desert":         (15.0,     3.0),
}

_SEA_PRESETS = {
    # sea_state: (CNR dB, doppler_spread Hz)
    1: (10.0, 15.0),
    2: (15.0, 25.0),
    3: (20.0, 40.0),
    4: (25.0, 60.0),
    5: (30.0, 80.0),
    6: (35.0, 100.0),
}


class RadarClutter:
    """Radar clutter generator with Doppler-coloured noise.

    Generates complex Gaussian clutter shaped by a Doppler power spectral
    density and adds it to a slow-time / fast-time pulse matrix.

    Args:
        cnr: Clutter-to-noise ratio in dB (relative to unit thermal noise).
        doppler_spread: Clutter Doppler spectral width in Hz.
        sample_rate: Receiver sample rate in Hz (needed to normalise Doppler).
        doppler_center: Center Doppler frequency in Hz. Default 0 (ground).
        range_extent: ``(start, stop)`` range bin indices where clutter is
            active. Default ``None`` applies clutter to all range bins.
        spectral_shape: ``"gaussian"`` (default) or ``"exponential"``.
    """

    def __init__(
        self,
        cnr: float,
        doppler_spread: float,
        sample_rate: float,
        doppler_center: float = 0.0,
        range_extent: Optional[Tuple[int, int]] = None,
        spectral_shape: str = "gaussian",
    ) -> None:
        self.cnr = cnr
        self.doppler_spread = doppler_spread
        self.sample_rate = sample_rate
        self.doppler_center = doppler_center
        self.range_extent = range_extent
        self.spectral_shape = spectral_shape

    def __call__(
        self, pulse_matrix: np.ndarray, rng: np.random.Generator
    ) -> np.ndarray:
        """Add clutter to a ``(num_pulses, num_range_bins)`` complex matrix."""
        num_pulses, num_range_bins = pulse_matrix.shape
        r_start = 0
        r_stop = num_range_bins
        if self.range_extent is not None:
            r_start, r_stop = self.range_extent

        n_bins = r_stop - r_start

        # Generate white noise in slow-time for each range bin
        white = np.sqrt(0.5) * (
            rng.standard_normal((num_pulses, n_bins))
            + 1j * rng.standard_normal((num_pulses, n_bins))
        )

        # Build Doppler shaping filter in frequency domain
        freqs = np.fft.fftfreq(num_pulses, d=1.0)  # normalised frequency
        # Convert doppler_spread and doppler_center to normalised freq
        # slow-time sample rate = PRF = 1/PRI, but we don't know PRI here.
        # We use sample_rate as the slow-time effective rate for Doppler shaping.
        norm_center = self.doppler_center / self.sample_rate
        norm_spread = self.doppler_spread / self.sample_rate

        if self.spectral_shape == "gaussian":
            if norm_spread > 0:
                psd = np.exp(-0.5 * ((freqs - norm_center) / norm_spread) ** 2)
            else:
                psd = np.ones(num_pulses)
        elif self.spectral_shape == "exponential":
            if norm_spread > 0:
                psd = np.exp(-np.abs(freqs - norm_center) / norm_spread)
            else:
                psd = np.ones(num_pulses)
        else:
            raise ValueError(
                f"spectral_shape must be 'gaussian' or 'exponential', got {self.spectral_shape!r}"
            )

        # Normalise PSD so total power = 1 before scaling
        psd = psd / (np.sum(psd) + 1e-30) * num_pulses

        # Shape noise in frequency domain per range bin
        H = np.sqrt(psd)  # amplitude shaping filter
        white_fft = np.fft.fft(white, axis=0)
        shaped_fft = white_fft * H[:, np.newaxis]
        clutter = np.fft.ifft(shaped_fft, axis=0)

        # Scale to desired CNR (relative to unit noise power)
        cnr_linear = 10.0 ** (self.cnr / 10.0)
        current_power = np.mean(np.abs(clutter) ** 2)
        if current_power > 0:
            clutter = clutter * np.sqrt(cnr_linear / current_power)

        # Add clutter to the specified range extent
        out = pulse_matrix.copy()
        out[:, r_start:r_stop] = out[:, r_start:r_stop] + clutter
        return out

    # ── Presets ──────────────────────────────────────────────────────────────

    @classmethod
    def ground(
        cls, sample_rate: float, terrain: str = "rural", **overrides
    ) -> RadarClutter:
        """Ground clutter preset.

        Args:
            sample_rate: Receiver sample rate in Hz.
            terrain: One of ``"rural"``, ``"urban"``, ``"forest"``, ``"desert"``.
            **overrides: Override any constructor parameter.
        """
        if terrain not in _GROUND_PRESETS:
            raise ValueError(
                f"terrain must be one of {list(_GROUND_PRESETS)}, got {terrain!r}"
            )
        cnr, spread = _GROUND_PRESETS[terrain]
        defaults = dict(
            cnr=cnr, doppler_spread=spread, sample_rate=sample_rate,
            doppler_center=0.0, spectral_shape="gaussian",
        )
        defaults.update(overrides)
        return cls(**defaults)

    @classmethod
    def sea(
        cls, sample_rate: float, sea_state: int = 3, **overrides
    ) -> RadarClutter:
        """Sea clutter preset.

        Args:
            sample_rate: Receiver sample rate in Hz.
            sea_state: Sea state (1–6).
            **overrides: Override any constructor parameter.
        """
        if sea_state not in _SEA_PRESETS:
            raise ValueError(f"sea_state must be 1–6, got {sea_state}")
        cnr, spread = _SEA_PRESETS[sea_state]
        defaults = dict(
            cnr=cnr, doppler_spread=spread, sample_rate=sample_rate,
            doppler_center=0.0, spectral_shape="exponential",
        )
        defaults.update(overrides)
        return cls(**defaults)

    @classmethod
    def weather(
        cls, sample_rate: float, rain_rate_mmhr: float = 10.0, **overrides
    ) -> RadarClutter:
        """Weather clutter preset.

        Args:
            sample_rate: Receiver sample rate in Hz.
            rain_rate_mmhr: Rain rate in mm/hr. Higher rates increase CNR
                and Doppler spread.
            **overrides: Override any constructor parameter.
        """
        # Empirical scaling: CNR ~ 10*log10(rain_rate), spread ~ sqrt(rain_rate)
        cnr = 10.0 * np.log10(max(rain_rate_mmhr, 0.1)) + 10.0
        spread = 20.0 * np.sqrt(rain_rate_mmhr)
        doppler_center = 30.0  # weather moves at ~30 Hz Doppler
        defaults = dict(
            cnr=cnr, doppler_spread=spread, sample_rate=sample_rate,
            doppler_center=doppler_center, spectral_shape="gaussian",
        )
        defaults.update(overrides)
        return cls(**defaults)
```

- [ ] **Step 4: Add export to impairments `__init__.py`**

Add to `python/spectra/impairments/__init__.py`:

```python
from spectra.impairments.clutter import RadarClutter
```

And add `"RadarClutter"` to the `__all__` list.

- [ ] **Step 5: Run tests to verify they pass**

```bash
/Users/gditzler/.venvs/base/bin/pytest tests/test_clutter.py -v
```
Expected: 8/8 PASS

- [ ] **Step 6: Commit**

```bash
git add python/spectra/impairments/clutter.py python/spectra/impairments/__init__.py tests/test_clutter.py
git commit -m "feat(impairments): add RadarClutter with ground/sea/weather presets"
```

---

## Task 4: MTI Processing (`spectra/algorithms/mti.py`)

**Files:**
- Create: `python/spectra/algorithms/mti.py`
- Modify: `python/spectra/algorithms/__init__.py`
- Create: `tests/test_mti.py`

- [ ] **Step 1: Write the failing tests**

```python
# tests/test_mti.py
"""Tests for MTI pulse cancellers and Doppler filter bank."""
import numpy as np
import pytest


def _make_clutter_plus_target(num_pulses=32, num_range_bins=64, target_bin=30,
                               target_doppler_hz=500.0, prf=1000.0):
    """Pulse matrix with zero-Doppler clutter + a moving target."""
    X = np.ones((num_pulses, num_range_bins), dtype=complex) * 5.0  # strong DC clutter
    # Add moving target at target_bin with Doppler phase progression
    for n in range(num_pulses):
        phase = np.exp(1j * 2 * np.pi * target_doppler_hz * n / prf)
        X[n, target_bin] += 10.0 * phase
    return X


def test_single_canceller_shape():
    from spectra.algorithms.mti import single_pulse_canceller
    X = np.zeros((16, 64), dtype=complex)
    out = single_pulse_canceller(X)
    assert out.shape == (15, 64)


def test_single_canceller_removes_dc():
    from spectra.algorithms.mti import single_pulse_canceller
    # Constant signal (DC only) should be nearly zeroed
    X = np.ones((32, 10), dtype=complex) * 100.0
    out = single_pulse_canceller(X)
    assert np.max(np.abs(out)) < 1e-10


def test_single_canceller_passes_moving_target():
    from spectra.algorithms.mti import single_pulse_canceller
    X = _make_clutter_plus_target()
    out = single_pulse_canceller(X)
    # Target bin should still have significant power
    target_power = np.mean(np.abs(out[:, 30]) ** 2)
    clutter_power = np.mean(np.abs(out[:, 0]) ** 2)
    assert target_power > clutter_power * 10


def test_double_canceller_shape():
    from spectra.algorithms.mti import double_pulse_canceller
    X = np.zeros((16, 64), dtype=complex)
    out = double_pulse_canceller(X)
    assert out.shape == (14, 64)


def test_double_canceller_removes_dc():
    from spectra.algorithms.mti import double_pulse_canceller
    X = np.ones((32, 10), dtype=complex) * 100.0
    out = double_pulse_canceller(X)
    assert np.max(np.abs(out)) < 1e-10


def test_doppler_filter_bank_shape():
    from spectra.algorithms.mti import doppler_filter_bank
    X = np.zeros((16, 64), dtype=complex)
    rdm = doppler_filter_bank(X)
    assert rdm.shape == (16, 64)


def test_doppler_filter_bank_zero_padded():
    from spectra.algorithms.mti import doppler_filter_bank
    X = np.zeros((16, 64), dtype=complex)
    rdm = doppler_filter_bank(X, num_doppler_bins=32)
    assert rdm.shape == (32, 64)


def test_doppler_filter_bank_peak_at_target():
    from spectra.algorithms.mti import doppler_filter_bank
    X = _make_clutter_plus_target(num_pulses=64, target_doppler_hz=200.0, prf=1000.0)
    rdm = doppler_filter_bank(X, window="hann")
    # Max in target range bin column should be away from DC
    target_col = rdm[:, 30]
    dc_bin = 0
    peak_bin = np.argmax(target_col)
    assert peak_bin != dc_bin, "Target Doppler peak should not be at DC"


def test_doppler_filter_bank_returns_power():
    from spectra.algorithms.mti import doppler_filter_bank
    X = np.random.default_rng(0).standard_normal((16, 32)) + 0j
    rdm = doppler_filter_bank(X)
    assert np.all(rdm >= 0), "Power should be non-negative"
```

- [ ] **Step 2: Run tests to verify they fail**

```bash
/Users/gditzler/.venvs/base/bin/pytest tests/test_mti.py -v
```
Expected: `ModuleNotFoundError: No module named 'spectra.algorithms.mti'`

- [ ] **Step 3: Write the implementation**

```python
# python/spectra/algorithms/mti.py
"""Moving Target Indication (MTI) algorithms.

Stateless functions for clutter suppression in radar pulse trains:

- :func:`single_pulse_canceller` — first-order high-pass (nulls DC).
- :func:`double_pulse_canceller` — second-order high-pass (deeper DC null).
- :func:`doppler_filter_bank` — FFT-based range-Doppler map.
"""

from __future__ import annotations

from typing import Optional

import numpy as np


def single_pulse_canceller(pulses: np.ndarray) -> np.ndarray:
    """First-order pulse canceller: ``y[n] = x[n+1] - x[n]``.

    Args:
        pulses: Complex pulse matrix, shape ``(num_pulses, num_range_bins)``.

    Returns:
        Cancelled output, shape ``(num_pulses - 1, num_range_bins)``.
    """
    return pulses[1:] - pulses[:-1]


def double_pulse_canceller(pulses: np.ndarray) -> np.ndarray:
    """Second-order pulse canceller: ``y[n] = x[n+2] - 2*x[n+1] + x[n]``.

    Args:
        pulses: Complex pulse matrix, shape ``(num_pulses, num_range_bins)``.

    Returns:
        Cancelled output, shape ``(num_pulses - 2, num_range_bins)``.
    """
    return pulses[2:] - 2 * pulses[1:-1] + pulses[:-2]


def doppler_filter_bank(
    pulses: np.ndarray,
    num_doppler_bins: Optional[int] = None,
    window: str = "hann",
) -> np.ndarray:
    """FFT-based Doppler filter bank producing a range-Doppler map.

    Applies a window along the slow-time (pulse) dimension, then computes
    the FFT to form a range-Doppler power map.

    Args:
        pulses: Complex pulse matrix, shape ``(num_pulses, num_range_bins)``.
        num_doppler_bins: FFT size along the pulse axis. Defaults to
            ``num_pulses``. Values larger than ``num_pulses`` zero-pad.
        window: Window function name (``"hann"``, ``"hamming"``, ``"rect"``).

    Returns:
        Range-Doppler power map (magnitude squared), shape
        ``(num_doppler_bins, num_range_bins)``.
    """
    num_pulses, num_range_bins = pulses.shape
    if num_doppler_bins is None:
        num_doppler_bins = num_pulses

    # Apply window along slow-time axis
    if window == "hann":
        w = np.hanning(num_pulses)
    elif window == "hamming":
        w = np.hamming(num_pulses)
    elif window == "rect":
        w = np.ones(num_pulses)
    else:
        raise ValueError(f"Unsupported window: {window!r}")

    windowed = pulses * w[:, np.newaxis]

    # FFT along pulse dimension (axis=0) with optional zero-padding
    fft_out = np.fft.fft(windowed, n=num_doppler_bins, axis=0)

    return np.abs(fft_out) ** 2
```

- [ ] **Step 4: Add exports to algorithms `__init__.py`**

Add to `python/spectra/algorithms/__init__.py`:

```python
from spectra.algorithms.mti import (
    doppler_filter_bank,
    double_pulse_canceller,
    single_pulse_canceller,
)
```

And add `"doppler_filter_bank"`, `"double_pulse_canceller"`, `"single_pulse_canceller"` to `__all__`.

- [ ] **Step 5: Run tests to verify they pass**

```bash
/Users/gditzler/.venvs/base/bin/pytest tests/test_mti.py -v
```
Expected: 9/9 PASS

- [ ] **Step 6: Commit**

```bash
git add python/spectra/algorithms/mti.py python/spectra/algorithms/__init__.py tests/test_mti.py
git commit -m "feat(algorithms): add MTI pulse cancellers and Doppler filter bank"
```

---

## Task 5: Kalman Filter (`spectra/tracking/kalman.py`)

**Files:**
- Create: `python/spectra/tracking/__init__.py`
- Create: `python/spectra/tracking/kalman.py`
- Create: `tests/test_kalman.py`

- [ ] **Step 1: Write the failing tests**

```python
# tests/test_kalman.py
"""Tests for generic Kalman filter and CV factory."""
import numpy as np
import pytest


def test_kf_initial_state():
    from spectra.tracking.kalman import KalmanFilter
    F = np.eye(2)
    H = np.array([[1.0, 0.0]])
    Q = np.eye(2) * 0.01
    R = np.array([[1.0]])
    kf = KalmanFilter(F, H, Q, R)
    assert kf.state.shape == (2,)
    assert np.allclose(kf.state, 0.0)
    assert kf.covariance.shape == (2, 2)


def test_kf_custom_initial():
    from spectra.tracking.kalman import KalmanFilter
    F = np.eye(2)
    H = np.array([[1.0, 0.0]])
    Q = np.eye(2) * 0.01
    R = np.array([[1.0]])
    x0 = np.array([100.0, 5.0])
    P0 = np.eye(2) * 10.0
    kf = KalmanFilter(F, H, Q, R, x0=x0, P0=P0)
    assert np.allclose(kf.state, x0)
    assert np.allclose(kf.covariance, P0)


def test_kf_predict():
    from spectra.tracking.kalman import KalmanFilter
    dt = 1.0
    F = np.array([[1, dt], [0, 1]])
    H = np.array([[1.0, 0.0]])
    Q = np.eye(2) * 0.01
    R = np.array([[1.0]])
    x0 = np.array([100.0, 10.0])
    kf = KalmanFilter(F, H, Q, R, x0=x0)
    predicted = kf.predict()
    assert predicted[0] == pytest.approx(110.0)  # 100 + 10*1
    assert predicted[1] == pytest.approx(10.0)


def test_kf_update_moves_toward_measurement():
    from spectra.tracking.kalman import KalmanFilter
    F = np.eye(2)
    H = np.array([[1.0, 0.0]])
    Q = np.eye(2) * 0.01
    R = np.array([[0.1]])
    x0 = np.array([100.0, 0.0])
    P0 = np.eye(2) * 100.0  # high uncertainty
    kf = KalmanFilter(F, H, Q, R, x0=x0, P0=P0)
    kf.predict()
    updated = kf.update(np.array([105.0]))
    # Should move toward measurement (105) from prior (100)
    assert updated[0] > 100.0
    assert updated[0] < 106.0


def test_kf_step():
    from spectra.tracking.kalman import KalmanFilter
    F = np.eye(2)
    H = np.array([[1.0, 0.0]])
    Q = np.eye(2) * 0.01
    R = np.array([[1.0]])
    kf = KalmanFilter(F, H, Q, R, x0=np.array([0.0, 0.0]))
    state = kf.step(np.array([10.0]))
    assert state[0] > 0.0  # moved toward 10


def test_kf_run_batch():
    from spectra.tracking.kalman import KalmanFilter
    dt = 1.0
    F = np.array([[1, dt], [0, 1]])
    H = np.array([[1.0, 0.0]])
    Q = np.eye(2) * 0.1
    R = np.array([[1.0]])
    x0 = np.array([0.0, 10.0])
    kf = KalmanFilter(F, H, Q, R, x0=x0)
    # True trajectory: range = 10*t, velocity = 10
    measurements = np.array([[10.0 * t + np.random.default_rng(t).normal(0, 1)]
                              for t in range(1, 21)])
    states = kf.run(measurements)
    assert states.shape == (20, 2)
    # Final range estimate should be near 200
    assert abs(states[-1, 0] - 200.0) < 20.0


def test_cv_kf_factory():
    from spectra.tracking.kalman import ConstantVelocityKF, KalmanFilter
    kf = ConstantVelocityKF(dt=0.5, process_noise_std=1.0, measurement_noise_std=5.0)
    assert isinstance(kf, KalmanFilter)
    assert kf.state.shape == (2,)
    # Predict from [100, 20] with dt=0.5 should give [110, 20]
    kf2 = ConstantVelocityKF(dt=0.5, process_noise_std=1.0, measurement_noise_std=5.0,
                               x0=np.array([100.0, 20.0]))
    pred = kf2.predict()
    assert pred[0] == pytest.approx(110.0)  # 100 + 20*0.5
    assert pred[1] == pytest.approx(20.0)


def test_cv_kf_tracks_linear_target():
    from spectra.tracking.kalman import ConstantVelocityKF
    kf = ConstantVelocityKF(dt=1.0, process_noise_std=0.5, measurement_noise_std=2.0)
    rng = np.random.default_rng(42)
    # True: range = 100 + 20*t
    measurements = np.array([[100.0 + 20.0 * t + rng.normal(0, 2)]
                              for t in range(1, 51)])
    states = kf.run(measurements)
    # Final estimate should be near 100 + 20*50 = 1100
    assert abs(states[-1, 0] - 1100.0) < 30.0
    # Velocity estimate should converge near 20
    assert abs(states[-1, 1] - 20.0) < 5.0
```

- [ ] **Step 2: Run tests to verify they fail**

```bash
/Users/gditzler/.venvs/base/bin/pytest tests/test_kalman.py -v
```
Expected: `ModuleNotFoundError: No module named 'spectra.tracking'`

- [ ] **Step 3: Write the implementation**

```python
# python/spectra/tracking/kalman.py
"""Generic linear Kalman filter and radar convenience factories.

:class:`KalmanFilter` is state-dimension agnostic — it works for any
``(n, m)`` state/measurement pair. :func:`ConstantVelocityKF` returns a
pre-configured instance for range-only radar tracking.
"""

from __future__ import annotations

from typing import Optional

import numpy as np


class KalmanFilter:
    """Generic discrete-time linear Kalman filter.

    Args:
        F: State transition matrix, shape ``(n, n)``.
        H: Measurement matrix, shape ``(m, n)``.
        Q: Process noise covariance, shape ``(n, n)``.
        R: Measurement noise covariance, shape ``(m, m)``.
        x0: Initial state, shape ``(n,)``. Defaults to zeros.
        P0: Initial covariance, shape ``(n, n)``. Defaults to identity.
    """

    def __init__(
        self,
        F: np.ndarray,
        H: np.ndarray,
        Q: np.ndarray,
        R: np.ndarray,
        x0: Optional[np.ndarray] = None,
        P0: Optional[np.ndarray] = None,
    ) -> None:
        self._F = np.asarray(F, dtype=float)
        self._H = np.asarray(H, dtype=float)
        self._Q = np.asarray(Q, dtype=float)
        self._R = np.asarray(R, dtype=float)

        n = self._F.shape[0]
        self._x = np.zeros(n) if x0 is None else np.asarray(x0, dtype=float).copy()
        self._P = np.eye(n) if P0 is None else np.asarray(P0, dtype=float).copy()

    @property
    def state(self) -> np.ndarray:
        """Current state estimate, shape ``(n,)``."""
        return self._x.copy()

    @property
    def covariance(self) -> np.ndarray:
        """Current error covariance, shape ``(n, n)``."""
        return self._P.copy()

    def predict(self) -> np.ndarray:
        """Propagate state and covariance one step forward.

        Returns:
            Predicted state, shape ``(n,)``.
        """
        self._x = self._F @ self._x
        self._P = self._F @ self._P @ self._F.T + self._Q
        return self._x.copy()

    def update(self, z: np.ndarray) -> np.ndarray:
        """Incorporate a measurement.

        Args:
            z: Measurement vector, shape ``(m,)``.

        Returns:
            Updated state, shape ``(n,)``.
        """
        z = np.asarray(z, dtype=float)
        y = z - self._H @ self._x                     # innovation
        S = self._H @ self._P @ self._H.T + self._R   # innovation covariance
        K = self._P @ self._H.T @ np.linalg.inv(S)    # Kalman gain
        self._x = self._x + K @ y
        n = len(self._x)
        self._P = (np.eye(n) - K @ self._H) @ self._P
        return self._x.copy()

    def step(self, z: np.ndarray) -> np.ndarray:
        """Predict then update in one call.

        Args:
            z: Measurement vector, shape ``(m,)``.

        Returns:
            Updated state, shape ``(n,)``.
        """
        self.predict()
        return self.update(z)

    def run(self, measurements: np.ndarray) -> np.ndarray:
        """Batch-process a sequence of measurements.

        Args:
            measurements: Shape ``(T, m)`` array of measurements.

        Returns:
            State history, shape ``(T, n)``.
        """
        measurements = np.asarray(measurements, dtype=float)
        T = measurements.shape[0]
        n = len(self._x)
        states = np.empty((T, n))
        for t in range(T):
            states[t] = self.step(measurements[t])
        return states


def ConstantVelocityKF(
    dt: float,
    process_noise_std: float,
    measurement_noise_std: float,
    x0: Optional[np.ndarray] = None,
    P0: Optional[np.ndarray] = None,
) -> KalmanFilter:
    """Create a constant-velocity Kalman filter for range-only tracking.

    State: ``[range, range_rate]``.  Measurement: ``[range]``.

    Args:
        dt: Time step between measurements in seconds.
        process_noise_std: Acceleration process noise standard deviation.
        measurement_noise_std: Range measurement noise standard deviation.
        x0: Initial state ``(2,)``. Defaults to zeros.
        P0: Initial covariance ``(2, 2)``. Defaults to identity.

    Returns:
        Configured :class:`KalmanFilter` instance.
    """
    F = np.array([[1.0, dt], [0.0, 1.0]])
    H = np.array([[1.0, 0.0]])

    # Discrete white-noise acceleration model
    q = process_noise_std**2
    Q = q * np.array([
        [dt**4 / 4, dt**3 / 2],
        [dt**3 / 2, dt**2],
    ])

    R = np.array([[measurement_noise_std**2]])

    return KalmanFilter(F, H, Q, R, x0=x0, P0=P0)
```

```python
# python/spectra/tracking/__init__.py
from spectra.tracking.kalman import ConstantVelocityKF, KalmanFilter

__all__ = ["ConstantVelocityKF", "KalmanFilter"]
```

- [ ] **Step 4: Run tests to verify they pass**

```bash
/Users/gditzler/.venvs/base/bin/pytest tests/test_kalman.py -v
```
Expected: 8/8 PASS

- [ ] **Step 5: Commit**

```bash
git add python/spectra/tracking/__init__.py python/spectra/tracking/kalman.py tests/test_kalman.py
git commit -m "feat(tracking): add generic KalmanFilter and ConstantVelocityKF factory"
```

---

## Task 6: Radar Pipeline Dataset (`spectra/datasets/radar_pipeline.py`)

**Files:**
- Create: `python/spectra/datasets/radar_pipeline.py`
- Modify: `python/spectra/datasets/__init__.py`
- Create: `tests/test_radar_pipeline.py`

- [ ] **Step 1: Write the failing tests**

```python
# tests/test_radar_pipeline.py
"""Tests for RadarPipelineDataset."""
import numpy as np
import pytest
import torch


def _make_pipeline_ds(**kwargs):
    from spectra.datasets.radar_pipeline import RadarPipelineDataset
    from spectra.targets.trajectory import ConstantVelocity
    from spectra.impairments.clutter import RadarClutter
    from spectra.waveforms import LFM

    defaults = dict(
        waveform_pool=[LFM()],
        trajectory_pool=[
            ConstantVelocity(initial_range=100.0, velocity=0.5, dt=1.0),  # range in bin units
        ],
        swerling_cases=[0],
        clutter_presets=[RadarClutter.ground(sample_rate=1e6, terrain="rural")],
        num_range_bins=256,
        sample_rate=1e6,
        carrier_frequency=10e9,
        pri=1e-3,
        snr_range=(10.0, 20.0),
        num_targets_range=(1, 2),
        sequence_length=5,
        pulses_per_cpi=16,
        apply_mti=True,
        cfar_type="ca",
        num_samples=10,
        seed=42,
    )
    defaults.update(kwargs)
    return RadarPipelineDataset(**defaults)


def test_pipeline_len():
    ds = _make_pipeline_ds(num_samples=15)
    assert len(ds) == 15


def test_pipeline_output_shape():
    ds = _make_pipeline_ds(sequence_length=5, num_range_bins=128)
    data, target = ds[0]
    assert isinstance(data, torch.Tensor)
    assert data.shape == (5, 128)


def test_pipeline_single_frame():
    ds = _make_pipeline_ds(sequence_length=1, num_range_bins=64)
    data, target = ds[0]
    assert data.shape == (1, 64)


def test_pipeline_target_fields():
    from spectra.datasets.radar_pipeline import RadarPipelineTarget
    ds = _make_pipeline_ds(sequence_length=3)
    _, target = ds[0]
    assert isinstance(target, RadarPipelineTarget)
    assert target.true_ranges.shape[0] == 3  # sequence_length
    assert target.true_velocities.shape[0] == 3
    assert target.rcs_amplitudes.shape[0] == 3
    assert len(target.detections) == 3
    assert target.kf_states.shape[0] == 3
    assert target.num_targets >= 1


def test_pipeline_deterministic():
    ds = _make_pipeline_ds()
    d1, t1 = ds[3]
    d2, t2 = ds[3]
    assert torch.allclose(d1, d2)
    assert np.allclose(t1.true_ranges, t2.true_ranges)


def test_pipeline_no_mti():
    ds = _make_pipeline_ds(apply_mti=False)
    data, target = ds[0]
    assert isinstance(data, torch.Tensor)


def test_pipeline_os_cfar():
    ds = _make_pipeline_ds(cfar_type="os")
    data, target = ds[0]
    assert isinstance(data, torch.Tensor)


def test_pipeline_tensor_normalized():
    ds = _make_pipeline_ds()
    data, _ = ds[0]
    assert data.min() >= 0.0
    assert data.max() <= 1.0


def test_pipeline_dataloader():
    from spectra.datasets import collate_fn
    ds = _make_pipeline_ds(num_samples=8)
    loader = torch.utils.data.DataLoader(ds, batch_size=4, collate_fn=collate_fn)
    batch_data, batch_targets = next(iter(loader))
    assert batch_data.shape[0] == 4
    assert len(batch_targets) == 4
```

- [ ] **Step 2: Run tests to verify they fail**

```bash
/Users/gditzler/.venvs/base/bin/pytest tests/test_radar_pipeline.py -v
```
Expected: `ModuleNotFoundError: No module named 'spectra.datasets.radar_pipeline'`

- [ ] **Step 3: Write the implementation**

```python
# python/spectra/datasets/radar_pipeline.py
"""End-to-end radar processing pipeline dataset.

Generates multi-CPI radar scenarios: waveform → target injection → clutter →
matched filter → MTI → CFAR → Kalman tracker, producing training data for
radar ML tasks.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import List, Optional, Tuple, Union

import numpy as np
import torch
from torch.utils.data import Dataset

from spectra.algorithms.mti import doppler_filter_bank, single_pulse_canceller
from spectra.algorithms.radar import ca_cfar, matched_filter, os_cfar
from spectra.impairments.clutter import RadarClutter
from spectra.targets.rcs import NonFluctuatingRCS, SwerlingRCS
from spectra.targets.trajectory import Trajectory
from spectra.tracking.kalman import ConstantVelocityKF
from spectra.waveforms.base import Waveform


@dataclass
class RadarPipelineTarget:
    """Ground-truth and processing results for one pipeline sample.

    Attributes:
        true_ranges: Shape ``(sequence_length, num_targets)``.
        true_velocities: Shape ``(sequence_length, num_targets)``.
        rcs_amplitudes: Shape ``(sequence_length, num_targets)``.
        detections: Per-frame CFAR detection bin indices, length ``sequence_length``.
        kf_states: Shape ``(sequence_length, num_targets, state_dim)``.
        num_targets: Number of targets in this sample.
        waveform_label: Radar waveform type string.
        snr_db: Nominal target SNR in dB (before clutter).
        clutter_preset: Clutter configuration description.
    """

    true_ranges: np.ndarray
    true_velocities: np.ndarray
    rcs_amplitudes: np.ndarray
    detections: List[np.ndarray]
    kf_states: np.ndarray
    num_targets: int
    waveform_label: str
    snr_db: float
    clutter_preset: str


class RadarPipelineDataset(Dataset):
    """On-the-fly end-to-end radar pipeline dataset.

    Each ``__getitem__`` call generates a complete radar scenario:
    pulse transmission → target reflection → clutter + noise → matched filter
    → MTI → CFAR detection → Kalman tracking.

    Note:
        Trajectory ``range_at()`` returns values in range-bin units (not metres).
        Configure your trajectory ``initial_range`` and ``velocity`` accordingly.

    Args:
        waveform_pool: Radar waveforms to draw from.
        trajectory_pool: Target trajectory configurations.
        swerling_cases: RCS cases to sample from (default ``[0, 1, 2, 3, 4]``).
        clutter_presets: Clutter configurations to sample from.
        num_range_bins: Range profile length.
        sample_rate: Receiver sample rate in Hz.
        carrier_frequency: RF carrier frequency in Hz (for Doppler computation).
        pri: Pulse repetition interval in seconds.
        snr_range: ``(min_db, max_db)`` per-target SNR.
        num_targets_range: ``(min, max)`` targets per sample.
        sequence_length: CPIs per sample (1 = single-frame).
        pulses_per_cpi: Pulses per coherent processing interval.
        apply_mti: Apply pulse cancellation before CFAR.
        cfar_type: ``"ca"`` or ``"os"``.
        num_samples: Dataset size.
        seed: Base seed for reproducibility.
    """

    def __init__(
        self,
        waveform_pool: List[Waveform],
        trajectory_pool: List,
        swerling_cases: List[int] = None,
        clutter_presets: List[RadarClutter] = None,
        num_range_bins: int = 256,
        sample_rate: float = 1e6,
        carrier_frequency: float = 10e9,
        pri: float = 1e-3,
        snr_range: Tuple[float, float] = (5.0, 25.0),
        num_targets_range: Tuple[int, int] = (1, 3),
        sequence_length: int = 1,
        pulses_per_cpi: int = 16,
        apply_mti: bool = True,
        cfar_type: str = "ca",
        num_samples: int = 10000,
        seed: int = 0,
    ) -> None:
        self.waveform_pool = waveform_pool
        self.trajectory_pool = trajectory_pool
        self.swerling_cases = swerling_cases if swerling_cases is not None else [0, 1, 2, 3, 4]
        self.clutter_presets = clutter_presets if clutter_presets is not None else []
        self.num_range_bins = num_range_bins
        self.sample_rate = sample_rate
        self.carrier_frequency = carrier_frequency
        self.pri = pri
        self.snr_range = snr_range
        self.num_targets_range = num_targets_range
        self.sequence_length = sequence_length
        self.pulses_per_cpi = pulses_per_cpi
        self.apply_mti = apply_mti
        self.cfar_type = cfar_type
        self.num_samples = num_samples
        self.seed = seed

    def __len__(self) -> int:
        return self.num_samples

    def __getitem__(self, idx: int) -> Tuple[torch.Tensor, RadarPipelineTarget]:
        rng = np.random.default_rng(seed=(self.seed, idx))

        # ── Step 1: Sample configuration ────────────────────────────────────
        wf_idx = int(rng.integers(0, len(self.waveform_pool)))
        waveform = self.waveform_pool[wf_idx]

        n_targets = int(rng.integers(self.num_targets_range[0], self.num_targets_range[1] + 1))

        # Sample per-target trajectories with randomised initial conditions
        trajectories = []
        for _ in range(n_targets):
            traj_template = self.trajectory_pool[int(rng.integers(0, len(self.trajectory_pool)))]
            # Clone with randomised initial range offset
            offset = rng.uniform(-50, 50)
            from spectra.targets.trajectory import ConstantVelocity, ConstantTurnRate
            if isinstance(traj_template, ConstantTurnRate):
                traj = ConstantTurnRate(
                    initial_range=traj_template.initial_range + offset,
                    velocity=traj_template.velocity + rng.uniform(-5, 5),
                    turn_rate=traj_template.turn_rate,
                    dt=traj_template.dt,
                )
            else:
                traj = ConstantVelocity(
                    initial_range=traj_template.initial_range + offset,
                    velocity=traj_template.velocity + rng.uniform(-5, 5),
                    dt=traj_template.dt,
                )
            trajectories.append(traj)

        # Swerling RCS model
        swerling_case = int(rng.choice(self.swerling_cases))
        if swerling_case == 0:
            rcs_model = NonFluctuatingRCS(sigma=1.0)
        else:
            rcs_model = SwerlingRCS(case=swerling_case, sigma=1.0)

        rcs_amps = rcs_model.amplitudes(
            num_dwells=self.sequence_length,
            num_pulses_per_dwell=self.pulses_per_cpi,
            rng=rng,
        )

        # Per-target SNR
        snr_db = float(rng.uniform(self.snr_range[0], self.snr_range[1]))
        snr_linear = 10.0 ** (snr_db / 10.0)

        # Clutter preset
        if self.clutter_presets:
            clutter = self.clutter_presets[int(rng.integers(0, len(self.clutter_presets)))]
            clutter_name = f"cnr_{clutter.cnr:.0f}dB"
        else:
            clutter = None
            clutter_name = "none"

        # Generate transmitted pulse template
        sps = getattr(waveform, "samples_per_symbol", 8)
        num_sym = self.num_range_bins // sps + 1
        template = waveform.generate(
            num_symbols=num_sym, sample_rate=self.sample_rate, seed=int(rng.integers(0, 2**32))
        )
        template = template[: self.num_range_bins]
        if len(template) < self.num_range_bins:
            padded = np.zeros(self.num_range_bins, dtype=np.complex64)
            padded[: len(template)] = template
            template = padded
        template_len = len(template)

        # ── Step 2: Per-CPI loop ────────────────────────────────────────────
        range_profiles = []
        all_detections = []
        true_ranges = np.zeros((self.sequence_length, n_targets))
        true_velocities = np.zeros((self.sequence_length, n_targets))
        all_rcs_amps = np.zeros((self.sequence_length, n_targets))

        wavelength = 3e8 / self.carrier_frequency

        for frame in range(self.sequence_length):
            # (a) Form pulse matrix — each row is a copy of the template
            pulse_matrix = np.zeros(
                (self.pulses_per_cpi, self.num_range_bins), dtype=complex
            )

            # (c) Inject target returns
            for k, traj in enumerate(trajectories):
                true_state = traj.state_at(frame)
                true_range = true_state[0]
                true_vel = true_state[1]
                true_ranges[frame, k] = true_range
                true_velocities[frame, k] = true_vel

                range_bin = int(np.clip(round(true_range), 0, self.num_range_bins - 1))

                # Doppler frequency from velocity
                f_d = 2.0 * true_vel / wavelength if wavelength > 0 else 0.0

                for n in range(self.pulses_per_cpi):
                    amp = rcs_amps[frame, n] * np.sqrt(snr_linear)
                    doppler_phase = np.exp(1j * 2 * np.pi * f_d * n * self.pri)
                    # Place delayed template at target range bin
                    end_bin = min(range_bin + template_len, self.num_range_bins)
                    seg_len = end_bin - range_bin
                    if seg_len > 0:
                        pulse_matrix[n, range_bin:end_bin] += (
                            amp * doppler_phase * template[:seg_len]
                        )
                    all_rcs_amps[frame, k] = float(rcs_amps[frame, 0])

            # (d) Add thermal noise (AWGN)
            noise = np.sqrt(0.5) * (
                rng.standard_normal(pulse_matrix.shape)
                + 1j * rng.standard_normal(pulse_matrix.shape)
            )
            pulse_matrix = pulse_matrix + noise

            # (e) Add clutter
            if clutter is not None:
                pulse_matrix = clutter(pulse_matrix, rng)

            # (f) Matched filter each pulse (mode="same" preserves range-bin alignment)
            h = np.conj(template[::-1])
            mf_matrix = np.zeros_like(pulse_matrix)
            for n in range(self.pulses_per_cpi):
                mf_matrix[n] = np.convolve(pulse_matrix[n], h, mode="same")

            # (g) MTI
            if self.apply_mti and self.pulses_per_cpi > 1:
                mf_matrix = single_pulse_canceller(mf_matrix)

            # (h) Doppler filter bank
            rdm = doppler_filter_bank(mf_matrix, window="hann")

            # (i) Range profile + CFAR
            range_profile = np.max(rdm, axis=0)  # (num_range_bins,)

            if self.cfar_type == "os":
                det_mask = os_cfar(range_profile, guard_cells=2, training_cells=8)
            else:
                det_mask = ca_cfar(range_profile, guard_cells=2, training_cells=8)

            det_bins = np.where(det_mask)[0]
            all_detections.append(det_bins)
            range_profiles.append(range_profile)

        # ── Step 3: Kalman tracking (ground-truth-aided) ────────────────────
        state_dim = 2
        kf_states = np.zeros((self.sequence_length, n_targets, state_dim))

        for k in range(n_targets):
            kf = ConstantVelocityKF(
                dt=self.pri * self.pulses_per_cpi,
                process_noise_std=1.0,
                measurement_noise_std=5.0,
                x0=np.array([true_ranges[0, k], true_velocities[0, k]]),
            )
            for frame in range(self.sequence_length):
                predicted = kf.predict()
                pred_range = predicted[0]
                dets = all_detections[frame]
                if len(dets) > 0:
                    nearest_idx = np.argmin(np.abs(dets - pred_range))
                    nearest_det = float(dets[nearest_idx])
                    gate = 20.0  # gating threshold in range bins
                    if abs(nearest_det - pred_range) < gate:
                        kf.update(np.array([nearest_det]))
                kf_states[frame, k] = kf.state

        # ── Step 4: Build output ────────────────────────────────────────────
        profiles = np.stack(range_profiles)  # (seq_len, num_range_bins)

        # Log-magnitude normalise to [0, 1]
        profiles_db = 10.0 * np.log10(profiles + 1e-30)
        p_min = profiles_db.min()
        p_max = profiles_db.max()
        if p_max > p_min:
            profiles_norm = (profiles_db - p_min) / (p_max - p_min)
        else:
            profiles_norm = np.zeros_like(profiles_db)

        tensor = torch.from_numpy(profiles_norm.astype(np.float32))

        target = RadarPipelineTarget(
            true_ranges=true_ranges,
            true_velocities=true_velocities,
            rcs_amplitudes=all_rcs_amps,
            detections=all_detections,
            kf_states=kf_states,
            num_targets=n_targets,
            waveform_label=waveform.label,
            snr_db=snr_db,
            clutter_preset=clutter_name,
        )
        return tensor, target
```

- [ ] **Step 4: Add exports to datasets `__init__.py`**

Add to `python/spectra/datasets/__init__.py`:

```python
from spectra.datasets.radar_pipeline import RadarPipelineDataset, RadarPipelineTarget
```

And add `"RadarPipelineDataset"`, `"RadarPipelineTarget"` to `__all__`.

- [ ] **Step 5: Run tests to verify they pass**

```bash
/Users/gditzler/.venvs/base/bin/pytest tests/test_radar_pipeline.py -v
```
Expected: 9/9 PASS

- [ ] **Step 6: Commit**

```bash
git add python/spectra/datasets/radar_pipeline.py python/spectra/datasets/__init__.py tests/test_radar_pipeline.py
git commit -m "feat(datasets): add RadarPipelineDataset with end-to-end radar processing"
```

---

## Task 7: Full Verification

- [ ] **Step 1: Run all tests (no regressions)**

```bash
/Users/gditzler/.venvs/base/bin/pytest tests/ -q --tb=short
```
Expected: All existing + new tests pass.

- [ ] **Step 2: Verify imports from top-level packages**

```bash
/Users/gditzler/.venvs/base/bin/python -c "
from spectra.targets import ConstantVelocity, ConstantTurnRate, NonFluctuatingRCS, SwerlingRCS, Trajectory
from spectra.impairments import RadarClutter
from spectra.algorithms import single_pulse_canceller, double_pulse_canceller, doppler_filter_bank
from spectra.tracking import KalmanFilter, ConstantVelocityKF
from spectra.datasets import RadarPipelineDataset, RadarPipelineTarget
print('All imports OK')
"
```
Expected: `All imports OK`

- [ ] **Step 3: Commit any remaining changes**

```bash
git status
# If clean, nothing to do. Otherwise commit fixes.
```
