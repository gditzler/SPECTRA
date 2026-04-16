# Environment & Propagation Modeling Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Add a physics-based `Environment` layer that computes per-signal SNR, path loss, delay, and Doppler from transmitter/receiver geometry and pluggable propagation models, integrating with the existing Composer and impairment pipeline.

**Architecture:** A new `python/spectra/environment/` module with dataclasses for geometry (`Position`, `Emitter`, `ReceiverConfig`, `LinkParams`), a `PropagationModel` ABC with 3 concrete implementations (free-space, log-distance, COST-231 Hata), an `Environment` orchestrator, and a bridge function (`link_params_to_impairments`) that converts computed parameters into existing SPECTRA impairment chains. Pure Python — no Rust FFI.

**Tech Stack:** Python 3.10+, NumPy, dataclasses, PyYAML (optional dep), pytest

---

## File Structure

```
python/spectra/environment/
    __init__.py          # public API exports
    position.py          # Position dataclass, geometry helpers
    propagation.py       # PropagationModel ABC, PathLossResult, 3 concrete models
    presets.py           # propagation_presets dict
    core.py              # Emitter, ReceiverConfig, LinkParams, Environment
    integration.py       # link_params_to_impairments(), _fading_from_suggestion()

tests/
    test_position.py
    test_propagation.py
    test_environment.py  # link budget + Environment.compute()
    test_environment_integration.py  # impairment bridge + end-to-end

python/spectra/__init__.py  # add new exports
```

---

### Task 1: Position Dataclass and Geometry Helpers

**Files:**
- Create: `python/spectra/environment/__init__.py`
- Create: `python/spectra/environment/position.py`
- Create: `tests/test_position.py`

- [ ] **Step 1: Write failing tests for Position**

Create `tests/test_position.py`:

```python
"""Tests for Position dataclass and geometry helpers."""

import math

import pytest

from spectra.environment.position import Position


class TestPositionDistance:
    def test_2d_distance_345_triangle(self):
        a = Position(0.0, 0.0)
        b = Position(3.0, 4.0)
        assert math.isclose(a.distance_to(b), 5.0)

    def test_2d_distance_symmetric(self):
        a = Position(1.0, 2.0)
        b = Position(4.0, 6.0)
        assert math.isclose(a.distance_to(b), b.distance_to(a))

    def test_same_position_zero_distance(self):
        a = Position(5.0, 5.0)
        assert a.distance_to(a) == 0.0

    def test_3d_distance_when_z_provided(self):
        a = Position(0.0, 0.0, z=0.0)
        b = Position(1.0, 2.0, z=2.0)
        assert math.isclose(a.distance_to(b), 3.0)

    def test_2d_ignores_z_when_one_is_none(self):
        a = Position(0.0, 0.0)
        b = Position(3.0, 4.0, z=100.0)
        assert math.isclose(a.distance_to(b), 5.0)


class TestPositionBearing:
    def test_bearing_east(self):
        a = Position(0.0, 0.0)
        b = Position(1.0, 0.0)
        assert math.isclose(a.bearing_to(b), 0.0, abs_tol=1e-10)

    def test_bearing_north(self):
        a = Position(0.0, 0.0)
        b = Position(0.0, 1.0)
        assert math.isclose(a.bearing_to(b), math.pi / 2)

    def test_bearing_west(self):
        a = Position(0.0, 0.0)
        b = Position(-1.0, 0.0)
        assert math.isclose(abs(a.bearing_to(b)), math.pi)


class TestPositionAngle:
    def test_angle_2d(self):
        a = Position(0.0, 0.0)
        b = Position(1.0, 1.0)
        assert math.isclose(a.angle_to(b), math.pi / 4)

    def test_elevation_angle_with_z(self):
        a = Position(0.0, 0.0, z=0.0)
        b = Position(100.0, 0.0, z=100.0)
        assert math.isclose(a.elevation_to(b), math.pi / 4)

    def test_elevation_returns_none_without_z(self):
        a = Position(0.0, 0.0)
        b = Position(100.0, 0.0)
        assert a.elevation_to(b) is None


class TestPositionDefaults:
    def test_z_defaults_to_none(self):
        p = Position(1.0, 2.0)
        assert p.z is None

    def test_z_can_be_set(self):
        p = Position(1.0, 2.0, z=3.0)
        assert p.z == 3.0
```

- [ ] **Step 2: Run tests to verify they fail**

Run: `pytest tests/test_position.py -v`
Expected: FAIL — `ModuleNotFoundError: No module named 'spectra.environment'`

- [ ] **Step 3: Create the environment package and Position implementation**

Create `python/spectra/environment/__init__.py`:

```python
"""Environment and propagation modeling for SPECTRA."""

from spectra.environment.position import Position

__all__ = ["Position"]
```

Create `python/spectra/environment/position.py`:

```python
"""Position dataclass with 2D/3D geometry helpers."""

from __future__ import annotations

import math
from dataclasses import dataclass


@dataclass
class Position:
    """A point in 2D (or optionally 3D) space. All units in meters."""

    x: float
    y: float
    z: float | None = None

    def distance_to(self, other: Position) -> float:
        """Euclidean distance. Uses 3D when both positions have z, else 2D."""
        dx = other.x - self.x
        dy = other.y - self.y
        if self.z is not None and other.z is not None:
            dz = other.z - self.z
            return math.sqrt(dx * dx + dy * dy + dz * dz)
        return math.sqrt(dx * dx + dy * dy)

    def bearing_to(self, other: Position) -> float:
        """Azimuth angle from self to other in radians. 0 = +x, pi/2 = +y."""
        dx = other.x - self.x
        dy = other.y - self.y
        return math.atan2(dy, dx)

    def angle_to(self, other: Position) -> float:
        """Azimuth angle from self to other (alias for bearing_to)."""
        return self.bearing_to(other)

    def elevation_to(self, other: Position) -> float | None:
        """Elevation angle in radians. Returns None if either z is None."""
        if self.z is None or other.z is None:
            return None
        horiz = math.sqrt((other.x - self.x) ** 2 + (other.y - self.y) ** 2)
        dz = other.z - self.z
        return math.atan2(dz, horiz)
```

- [ ] **Step 4: Run tests to verify they pass**

Run: `pytest tests/test_position.py -v`
Expected: All 12 tests PASS

- [ ] **Step 5: Commit**

```bash
git add python/spectra/environment/__init__.py python/spectra/environment/position.py tests/test_position.py
git commit -m "feat(environment): add Position dataclass with 2D/3D geometry helpers"
```

---

### Task 2: PropagationModel ABC and FreeSpacePathLoss

**Files:**
- Create: `python/spectra/environment/propagation.py`
- Create: `tests/test_propagation.py`
- Modify: `python/spectra/environment/__init__.py`

- [ ] **Step 1: Write failing tests for PropagationModel and FreeSpacePathLoss**

Create `tests/test_propagation.py`:

```python
"""Tests for propagation models."""

import math

import pytest

from spectra.environment.propagation import (
    FreeSpacePathLoss,
    PathLossResult,
    PropagationModel,
)

SPEED_OF_LIGHT = 299_792_458.0


class TestPathLossResult:
    def test_defaults(self):
        r = PathLossResult(path_loss_db=100.0)
        assert r.path_loss_db == 100.0
        assert r.shadow_fading_db == 0.0
        assert r.rms_delay_spread_s is None
        assert r.k_factor_db is None
        assert r.angular_spread_deg is None


class TestFreeSpacePathLoss:
    def test_is_propagation_model(self):
        assert isinstance(FreeSpacePathLoss(), PropagationModel)

    def test_1km_2400mhz(self):
        """Free-space PL at 1 km, 2.4 GHz should be ~100 dB."""
        model = FreeSpacePathLoss()
        result = model(distance_m=1000.0, freq_hz=2.4e9)
        expected = (
            20 * math.log10(1000.0)
            + 20 * math.log10(2.4e9)
            + 20 * math.log10(4 * math.pi / SPEED_OF_LIGHT)
        )
        assert math.isclose(result.path_loss_db, expected, rel_tol=1e-6)
        assert 99.0 < result.path_loss_db < 101.0

    def test_no_shadow_fading(self):
        model = FreeSpacePathLoss()
        result = model(distance_m=500.0, freq_hz=1e9)
        assert result.shadow_fading_db == 0.0

    def test_no_fading_metadata(self):
        model = FreeSpacePathLoss()
        result = model(distance_m=500.0, freq_hz=1e9)
        assert result.k_factor_db is None
        assert result.rms_delay_spread_s is None

    def test_inverse_square_law(self):
        """Doubling distance adds ~6 dB of path loss."""
        model = FreeSpacePathLoss()
        r1 = model(distance_m=100.0, freq_hz=1e9)
        r2 = model(distance_m=200.0, freq_hz=1e9)
        assert math.isclose(r2.path_loss_db - r1.path_loss_db, 20 * math.log10(2), rel_tol=1e-6)

    def test_minimum_distance_clamp(self):
        """Distance <= 0 should raise ValueError."""
        model = FreeSpacePathLoss()
        with pytest.raises(ValueError, match="distance_m must be positive"):
            model(distance_m=0.0, freq_hz=1e9)

    def test_very_small_distance(self):
        """Very small but positive distance should work."""
        model = FreeSpacePathLoss()
        result = model(distance_m=0.01, freq_hz=1e9)
        assert result.path_loss_db > 0
```

- [ ] **Step 2: Run tests to verify they fail**

Run: `pytest tests/test_propagation.py -v`
Expected: FAIL — `ModuleNotFoundError: No module named 'spectra.environment.propagation'`

- [ ] **Step 3: Implement PropagationModel ABC and FreeSpacePathLoss**

Create `python/spectra/environment/propagation.py`:

```python
"""Propagation models for path loss computation."""

from __future__ import annotations

import math
from abc import ABC, abstractmethod
from dataclasses import dataclass

SPEED_OF_LIGHT = 299_792_458.0


@dataclass
class PathLossResult:
    """Result of a propagation model computation."""

    path_loss_db: float
    shadow_fading_db: float = 0.0
    rms_delay_spread_s: float | None = None
    k_factor_db: float | None = None
    angular_spread_deg: float | None = None


class PropagationModel(ABC):
    """Abstract base class for propagation models."""

    @abstractmethod
    def __call__(self, distance_m: float, freq_hz: float, **kwargs) -> PathLossResult:
        """Compute path loss for given distance and frequency."""
        ...


class FreeSpacePathLoss(PropagationModel):
    """Friis free-space path loss model."""

    def __call__(self, distance_m: float, freq_hz: float, **kwargs) -> PathLossResult:
        if distance_m <= 0:
            raise ValueError("distance_m must be positive")
        pl_db = (
            20 * math.log10(distance_m)
            + 20 * math.log10(freq_hz)
            + 20 * math.log10(4 * math.pi / SPEED_OF_LIGHT)
        )
        return PathLossResult(path_loss_db=pl_db)
```

Update `python/spectra/environment/__init__.py`:

```python
"""Environment and propagation modeling for SPECTRA."""

from spectra.environment.position import Position
from spectra.environment.propagation import (
    FreeSpacePathLoss,
    PathLossResult,
    PropagationModel,
)

__all__ = [
    "FreeSpacePathLoss",
    "PathLossResult",
    "Position",
    "PropagationModel",
]
```

- [ ] **Step 4: Run tests to verify they pass**

Run: `pytest tests/test_propagation.py -v`
Expected: All 7 tests PASS

- [ ] **Step 5: Commit**

```bash
git add python/spectra/environment/propagation.py python/spectra/environment/__init__.py tests/test_propagation.py
git commit -m "feat(environment): add PropagationModel ABC and FreeSpacePathLoss"
```

---

### Task 3: LogDistancePL Model

**Files:**
- Modify: `python/spectra/environment/propagation.py`
- Modify: `tests/test_propagation.py`

- [ ] **Step 1: Write failing tests for LogDistancePL**

Append to `tests/test_propagation.py`:

```python
import numpy as np

from spectra.environment.propagation import LogDistancePL


class TestLogDistancePL:
    def test_is_propagation_model(self):
        assert isinstance(LogDistancePL(), PropagationModel)

    def test_n2_matches_free_space(self):
        """Log-distance with n=2, sigma=0 should match free-space at same distance."""
        log_model = LogDistancePL(n=2.0, sigma_db=0.0, d0=1.0)
        fs_model = FreeSpacePathLoss()
        freq = 2.4e9
        for d in [100.0, 500.0, 1000.0]:
            log_result = log_model(distance_m=d, freq_hz=freq)
            fs_result = fs_model(distance_m=d, freq_hz=freq)
            assert math.isclose(log_result.path_loss_db, fs_result.path_loss_db, rel_tol=1e-4)

    def test_higher_exponent_more_loss(self):
        """Higher path loss exponent should give more loss."""
        m1 = LogDistancePL(n=2.0, sigma_db=0.0)
        m2 = LogDistancePL(n=3.5, sigma_db=0.0)
        r1 = m1(distance_m=500.0, freq_hz=2.4e9)
        r2 = m2(distance_m=500.0, freq_hz=2.4e9)
        assert r2.path_loss_db > r1.path_loss_db

    def test_zero_sigma_no_shadow_fading(self):
        model = LogDistancePL(n=3.0, sigma_db=0.0)
        result = model(distance_m=500.0, freq_hz=1e9)
        assert result.shadow_fading_db == 0.0

    def test_nonzero_sigma_produces_shadow_fading(self):
        model = LogDistancePL(n=3.0, sigma_db=8.0)
        result = model(distance_m=500.0, freq_hz=1e9, seed=42)
        assert result.shadow_fading_db != 0.0

    def test_shadow_fading_deterministic_with_seed(self):
        model = LogDistancePL(n=3.0, sigma_db=8.0)
        r1 = model(distance_m=500.0, freq_hz=1e9, seed=42)
        r2 = model(distance_m=500.0, freq_hz=1e9, seed=42)
        assert r1.shadow_fading_db == r2.shadow_fading_db

    def test_shadow_fading_varies_without_seed(self):
        """Without seed, successive calls may differ (non-deterministic)."""
        model = LogDistancePL(n=3.0, sigma_db=8.0)
        results = [model(distance_m=500.0, freq_hz=1e9).shadow_fading_db for _ in range(20)]
        assert len(set(results)) > 1

    def test_minimum_distance_clamp(self):
        model = LogDistancePL(n=3.0)
        with pytest.raises(ValueError, match="distance_m must be positive"):
            model(distance_m=0.0, freq_hz=1e9)

    def test_shadow_included_in_path_loss(self):
        """path_loss_db should include the shadow fading sample."""
        model = LogDistancePL(n=3.0, sigma_db=8.0)
        result = model(distance_m=500.0, freq_hz=1e9, seed=42)
        model_no_shadow = LogDistancePL(n=3.0, sigma_db=0.0)
        result_no_shadow = model_no_shadow(distance_m=500.0, freq_hz=1e9)
        expected = result_no_shadow.path_loss_db + result.shadow_fading_db
        assert math.isclose(result.path_loss_db, expected, rel_tol=1e-6)
```

- [ ] **Step 2: Run tests to verify new tests fail**

Run: `pytest tests/test_propagation.py::TestLogDistancePL -v`
Expected: FAIL — `ImportError: cannot import name 'LogDistancePL'`

- [ ] **Step 3: Implement LogDistancePL**

Add to `python/spectra/environment/propagation.py` after `FreeSpacePathLoss`:

```python
class LogDistancePL(PropagationModel):
    """Log-distance path loss model with optional shadow fading.

    PL(dB) = PL(d0) + 10*n*log10(d/d0) + X_sigma
    where PL(d0) is free-space path loss at reference distance d0.
    """

    def __init__(
        self,
        n: float = 3.0,
        sigma_db: float = 0.0,
        d0: float = 1.0,
    ):
        self.n = n
        self.sigma_db = sigma_db
        self.d0 = d0
        self._fspl = FreeSpacePathLoss()

    def __call__(self, distance_m: float, freq_hz: float, **kwargs) -> PathLossResult:
        if distance_m <= 0:
            raise ValueError("distance_m must be positive")
        pl_d0 = self._fspl(self.d0, freq_hz).path_loss_db
        pl_db = pl_d0 + 10 * self.n * math.log10(distance_m / self.d0)

        shadow = 0.0
        if self.sigma_db > 0:
            seed = kwargs.get("seed")
            rng = np.random.default_rng(seed)
            shadow = float(rng.normal(0.0, self.sigma_db))

        return PathLossResult(
            path_loss_db=pl_db + shadow,
            shadow_fading_db=shadow,
        )
```

Add `import numpy as np` to the top of `propagation.py`.

Update `python/spectra/environment/__init__.py` to add `LogDistancePL` to imports and `__all__`.

- [ ] **Step 4: Run tests to verify they pass**

Run: `pytest tests/test_propagation.py -v`
Expected: All 16 tests PASS

- [ ] **Step 5: Commit**

```bash
git add python/spectra/environment/propagation.py python/spectra/environment/__init__.py tests/test_propagation.py
git commit -m "feat(environment): add LogDistancePL propagation model"
```

---

### Task 4: COST231HataPL Model

**Files:**
- Modify: `python/spectra/environment/propagation.py`
- Modify: `tests/test_propagation.py`

- [ ] **Step 1: Write failing tests for COST231HataPL**

Append to `tests/test_propagation.py`:

```python
from spectra.environment.propagation import COST231HataPL


class TestCOST231HataPL:
    def test_is_propagation_model(self):
        assert isinstance(COST231HataPL(), PropagationModel)

    def test_urban_more_loss_than_suburban(self):
        urban = COST231HataPL(environment="urban")
        suburban = COST231HataPL(environment="suburban")
        r_urban = urban(distance_m=1000.0, freq_hz=1800e6)
        r_suburban = suburban(distance_m=1000.0, freq_hz=1800e6)
        assert r_urban.path_loss_db > r_suburban.path_loss_db

    def test_suburban_more_loss_than_rural(self):
        suburban = COST231HataPL(environment="suburban")
        rural = COST231HataPL(environment="rural")
        r_sub = suburban(distance_m=1000.0, freq_hz=1800e6)
        r_rural = rural(distance_m=1000.0, freq_hz=1800e6)
        assert r_sub.path_loss_db > r_rural.path_loss_db

    def test_farther_distance_more_loss(self):
        model = COST231HataPL()
        r1 = model(distance_m=1000.0, freq_hz=1800e6)
        r2 = model(distance_m=5000.0, freq_hz=1800e6)
        assert r2.path_loss_db > r1.path_loss_db

    def test_higher_frequency_more_loss(self):
        model = COST231HataPL()
        r1 = model(distance_m=1000.0, freq_hz=1500e6)
        r2 = model(distance_m=1000.0, freq_hz=2000e6)
        assert r2.path_loss_db > r1.path_loss_db

    def test_reasonable_range_1km_1800mhz(self):
        """Urban PL at 1 km, 1800 MHz should be roughly 120-150 dB."""
        model = COST231HataPL(h_bs_m=30, h_ms_m=1.5, environment="urban")
        result = model(distance_m=1000.0, freq_hz=1800e6)
        assert 110 < result.path_loss_db < 160

    def test_invalid_environment(self):
        with pytest.raises(ValueError, match="environment must be"):
            COST231HataPL(environment="space")

    def test_minimum_distance_clamp(self):
        model = COST231HataPL()
        with pytest.raises(ValueError, match="distance_m must be positive"):
            model(distance_m=0.0, freq_hz=1800e6)

    def test_no_fading_metadata(self):
        model = COST231HataPL()
        result = model(distance_m=1000.0, freq_hz=1800e6)
        assert result.shadow_fading_db == 0.0
```

- [ ] **Step 2: Run tests to verify new tests fail**

Run: `pytest tests/test_propagation.py::TestCOST231HataPL -v`
Expected: FAIL — `ImportError: cannot import name 'COST231HataPL'`

- [ ] **Step 3: Implement COST231HataPL**

Add to `python/spectra/environment/propagation.py` after `LogDistancePL`:

```python
_VALID_ENVIRONMENTS = {"urban", "suburban", "rural"}


class COST231HataPL(PropagationModel):
    """COST-231 Hata path loss model for 1500-2000 MHz.

    Valid ranges: fc 1500-2000 MHz, h_bs 30-200 m, h_ms 1-10 m, d 1-20 km.
    """

    def __init__(
        self,
        h_bs_m: float = 30.0,
        h_ms_m: float = 1.5,
        environment: str = "urban",
    ):
        if environment not in _VALID_ENVIRONMENTS:
            raise ValueError(f"environment must be one of {_VALID_ENVIRONMENTS}, got '{environment}'")
        self.h_bs_m = h_bs_m
        self.h_ms_m = h_ms_m
        self.environment = environment

    def __call__(self, distance_m: float, freq_hz: float, **kwargs) -> PathLossResult:
        if distance_m <= 0:
            raise ValueError("distance_m must be positive")

        fc_mhz = freq_hz / 1e6
        d_km = distance_m / 1000.0

        # Mobile station antenna height correction factor (small/medium city)
        a_hms = (1.1 * math.log10(fc_mhz) - 0.7) * self.h_ms_m - (
            1.56 * math.log10(fc_mhz) - 0.8
        )

        # Metropolitan correction factor
        c_m = 3.0 if self.environment == "urban" else 0.0

        # Base COST-231 Hata formula
        pl_db = (
            46.3
            + 33.9 * math.log10(fc_mhz)
            - 13.82 * math.log10(self.h_bs_m)
            - a_hms
            + (44.9 - 6.55 * math.log10(self.h_bs_m)) * math.log10(d_km)
            + c_m
        )

        # Suburban/rural correction
        if self.environment == "suburban":
            pl_db -= 2 * (math.log10(fc_mhz / 28)) ** 2 + 5.4
        elif self.environment == "rural":
            pl_db -= 4.78 * (math.log10(fc_mhz)) ** 2 + 18.33 * math.log10(fc_mhz) - 40.94

        return PathLossResult(path_loss_db=pl_db)
```

Update `python/spectra/environment/__init__.py` to add `COST231HataPL` to imports and `__all__`.

- [ ] **Step 4: Run tests to verify they pass**

Run: `pytest tests/test_propagation.py -v`
Expected: All 25 tests PASS

- [ ] **Step 5: Commit**

```bash
git add python/spectra/environment/propagation.py python/spectra/environment/__init__.py tests/test_propagation.py
git commit -m "feat(environment): add COST231HataPL propagation model"
```

---

### Task 5: Presets

**Files:**
- Create: `python/spectra/environment/presets.py`
- Create: `tests/test_presets.py`
- Modify: `python/spectra/environment/__init__.py`

- [ ] **Step 1: Write failing tests for presets**

Create `tests/test_presets.py`:

```python
"""Tests for propagation model presets."""

from spectra.environment.presets import propagation_presets
from spectra.environment.propagation import PropagationModel


class TestPresets:
    def test_all_presets_are_propagation_models(self):
        for name, model in propagation_presets.items():
            assert isinstance(model, PropagationModel), f"Preset '{name}' is not a PropagationModel"

    def test_expected_presets_exist(self):
        expected = {"free_space", "urban_macro", "suburban", "indoor_office", "cost231_urban"}
        assert set(propagation_presets.keys()) == expected

    def test_all_presets_callable(self):
        for name, model in propagation_presets.items():
            result = model(distance_m=100.0, freq_hz=1800e6)
            assert result.path_loss_db > 0, f"Preset '{name}' returned non-positive path loss"
```

- [ ] **Step 2: Run tests to verify they fail**

Run: `pytest tests/test_presets.py -v`
Expected: FAIL — `ModuleNotFoundError: No module named 'spectra.environment.presets'`

- [ ] **Step 3: Implement presets**

Create `python/spectra/environment/presets.py`:

```python
"""Pre-configured propagation model instances for common scenarios."""

from spectra.environment.propagation import (
    COST231HataPL,
    FreeSpacePathLoss,
    LogDistancePL,
    PropagationModel,
)

propagation_presets: dict[str, PropagationModel] = {
    "free_space": FreeSpacePathLoss(),
    "urban_macro": LogDistancePL(n=3.5, sigma_db=8.0),
    "suburban": LogDistancePL(n=3.0, sigma_db=6.0),
    "indoor_office": LogDistancePL(n=2.0, sigma_db=4.0),
    "cost231_urban": COST231HataPL(h_bs_m=30, h_ms_m=1.5, environment="urban"),
}
```

Update `python/spectra/environment/__init__.py` to add `propagation_presets` to imports and `__all__`.

- [ ] **Step 4: Run tests to verify they pass**

Run: `pytest tests/test_presets.py -v`
Expected: All 3 tests PASS

- [ ] **Step 5: Commit**

```bash
git add python/spectra/environment/presets.py python/spectra/environment/__init__.py tests/test_presets.py
git commit -m "feat(environment): add propagation model presets"
```

---

### Task 6: Emitter, ReceiverConfig, LinkParams, and Environment

**Files:**
- Create: `python/spectra/environment/core.py`
- Create: `tests/test_environment.py`
- Modify: `python/spectra/environment/__init__.py`

- [ ] **Step 1: Write failing tests for core dataclasses and Environment**

Create `tests/test_environment.py`:

```python
"""Tests for Environment, Emitter, ReceiverConfig, and LinkParams."""

import math

import numpy as np
import pytest

from spectra.environment.core import Emitter, Environment, LinkParams, ReceiverConfig
from spectra.environment.position import Position
from spectra.environment.propagation import FreeSpacePathLoss, LogDistancePL
from spectra.waveforms import QPSK

SPEED_OF_LIGHT = 299_792_458.0
BOLTZMANN_DBM_HZ = -174.0  # 10*log10(k_B * 290) in dBm/Hz


@pytest.fixture
def simple_env():
    """Single QPSK emitter at 1 km, free-space."""
    return Environment(
        propagation=FreeSpacePathLoss(),
        emitters=[
            Emitter(
                waveform=QPSK(samples_per_symbol=8),
                position=Position(1000.0, 0.0),
                power_dbm=30.0,
                freq_hz=2.4e9,
            ),
        ],
        receiver=ReceiverConfig(
            position=Position(0.0, 0.0),
            noise_figure_db=6.0,
            bandwidth_hz=1e6,
        ),
    )


class TestEmitter:
    def test_defaults(self):
        e = Emitter(
            waveform=QPSK(samples_per_symbol=8),
            position=Position(0, 0),
            power_dbm=30.0,
            freq_hz=2.4e9,
        )
        assert e.velocity_mps is None
        assert e.antenna_gain_dbi == 0.0

    def test_with_velocity(self):
        e = Emitter(
            waveform=QPSK(samples_per_symbol=8),
            position=Position(0, 0),
            power_dbm=30.0,
            freq_hz=2.4e9,
            velocity_mps=(30.0, 0.0),
        )
        assert e.velocity_mps == (30.0, 0.0)


class TestReceiverConfig:
    def test_defaults(self):
        r = ReceiverConfig(position=Position(0, 0))
        assert r.noise_figure_db == 6.0
        assert r.bandwidth_hz == 1e6
        assert r.antenna_gain_dbi == 0.0
        assert r.temperature_k == 290.0


class TestLinkParams:
    def test_fields(self):
        lp = LinkParams(
            emitter_index=0,
            snr_db=15.0,
            path_loss_db=100.0,
            received_power_dbm=-70.0,
            delay_s=3.3e-6,
            doppler_hz=0.0,
            distance_m=1000.0,
            fading_suggestion=None,
        )
        assert lp.snr_db == 15.0
        assert lp.fading_suggestion is None

    def test_mutable_for_override(self):
        lp = LinkParams(
            emitter_index=0,
            snr_db=15.0,
            path_loss_db=100.0,
            received_power_dbm=-70.0,
            delay_s=3.3e-6,
            doppler_hz=0.0,
            distance_m=1000.0,
            fading_suggestion=None,
        )
        lp.snr_db = 20.0
        assert lp.snr_db == 20.0


class TestEnvironmentCompute:
    def test_returns_list_of_link_params(self, simple_env):
        result = simple_env.compute()
        assert isinstance(result, list)
        assert len(result) == 1
        assert isinstance(result[0], LinkParams)

    def test_distance_computed(self, simple_env):
        result = simple_env.compute()
        assert math.isclose(result[0].distance_m, 1000.0)

    def test_delay_from_distance(self, simple_env):
        result = simple_env.compute()
        expected_delay = 1000.0 / SPEED_OF_LIGHT
        assert math.isclose(result[0].delay_s, expected_delay, rel_tol=1e-6)

    def test_snr_link_budget(self, simple_env):
        result = simple_env.compute()
        fspl = FreeSpacePathLoss()
        pl = fspl(1000.0, 2.4e9).path_loss_db
        rx_power = 30.0 + 0.0 + 0.0 - pl
        noise_floor = BOLTZMANN_DBM_HZ + 10 * math.log10(1e6) + 6.0
        expected_snr = rx_power - noise_floor
        assert math.isclose(result[0].snr_db, expected_snr, rel_tol=1e-4)

    def test_no_doppler_without_velocity(self, simple_env):
        result = simple_env.compute()
        assert result[0].doppler_hz == 0.0

    def test_doppler_with_velocity(self):
        env = Environment(
            propagation=FreeSpacePathLoss(),
            emitters=[
                Emitter(
                    waveform=QPSK(samples_per_symbol=8),
                    position=Position(1000.0, 0.0),
                    power_dbm=30.0,
                    freq_hz=2.4e9,
                    velocity_mps=(-30.0, 0.0),
                ),
            ],
            receiver=ReceiverConfig(position=Position(0.0, 0.0)),
        )
        result = env.compute()
        expected = (30.0 / SPEED_OF_LIGHT) * 2.4e9
        assert math.isclose(result[0].doppler_hz, expected, rel_tol=1e-4)

    def test_doppler_perpendicular_velocity_zero(self):
        """Velocity perpendicular to LOS should produce ~zero Doppler."""
        env = Environment(
            propagation=FreeSpacePathLoss(),
            emitters=[
                Emitter(
                    waveform=QPSK(samples_per_symbol=8),
                    position=Position(1000.0, 0.0),
                    power_dbm=30.0,
                    freq_hz=2.4e9,
                    velocity_mps=(0.0, 30.0),
                ),
            ],
            receiver=ReceiverConfig(position=Position(0.0, 0.0)),
        )
        result = env.compute()
        assert abs(result[0].doppler_hz) < 0.1

    def test_multiple_emitters(self):
        env = Environment(
            propagation=FreeSpacePathLoss(),
            emitters=[
                Emitter(
                    waveform=QPSK(samples_per_symbol=8),
                    position=Position(100.0, 0.0),
                    power_dbm=30.0,
                    freq_hz=2.4e9,
                ),
                Emitter(
                    waveform=QPSK(samples_per_symbol=8),
                    position=Position(500.0, 0.0),
                    power_dbm=30.0,
                    freq_hz=2.4e9,
                ),
            ],
            receiver=ReceiverConfig(position=Position(0.0, 0.0)),
        )
        result = env.compute()
        assert len(result) == 2
        assert result[0].emitter_index == 0
        assert result[1].emitter_index == 1
        assert result[0].snr_db > result[1].snr_db

    def test_antenna_gain_increases_snr(self):
        env_no_gain = Environment(
            propagation=FreeSpacePathLoss(),
            emitters=[
                Emitter(
                    waveform=QPSK(samples_per_symbol=8),
                    position=Position(1000.0, 0.0),
                    power_dbm=30.0,
                    freq_hz=2.4e9,
                    antenna_gain_dbi=0.0,
                ),
            ],
            receiver=ReceiverConfig(position=Position(0.0, 0.0), antenna_gain_dbi=0.0),
        )
        env_with_gain = Environment(
            propagation=FreeSpacePathLoss(),
            emitters=[
                Emitter(
                    waveform=QPSK(samples_per_symbol=8),
                    position=Position(1000.0, 0.0),
                    power_dbm=30.0,
                    freq_hz=2.4e9,
                    antenna_gain_dbi=10.0,
                ),
            ],
            receiver=ReceiverConfig(position=Position(0.0, 0.0), antenna_gain_dbi=5.0),
        )
        r1 = env_no_gain.compute()
        r2 = env_with_gain.compute()
        assert math.isclose(r2[0].snr_db - r1[0].snr_db, 15.0, rel_tol=1e-4)

    def test_deterministic_with_seed(self):
        env = Environment(
            propagation=LogDistancePL(n=3.5, sigma_db=8.0),
            emitters=[
                Emitter(
                    waveform=QPSK(samples_per_symbol=8),
                    position=Position(500.0, 0.0),
                    power_dbm=30.0,
                    freq_hz=2.4e9,
                ),
            ],
            receiver=ReceiverConfig(position=Position(0.0, 0.0)),
        )
        r1 = env.compute(seed=42)
        r2 = env.compute(seed=42)
        assert r1[0].snr_db == r2[0].snr_db

    def test_emitter_index_preserved(self):
        env = Environment(
            propagation=FreeSpacePathLoss(),
            emitters=[
                Emitter(
                    waveform=QPSK(samples_per_symbol=8),
                    position=Position(100.0, 0.0),
                    power_dbm=20.0,
                    freq_hz=1e9,
                ),
                Emitter(
                    waveform=QPSK(samples_per_symbol=8),
                    position=Position(200.0, 0.0),
                    power_dbm=40.0,
                    freq_hz=2e9,
                ),
            ],
            receiver=ReceiverConfig(position=Position(0.0, 0.0)),
        )
        result = env.compute()
        assert result[0].emitter_index == 0
        assert result[1].emitter_index == 1
```

- [ ] **Step 2: Run tests to verify they fail**

Run: `pytest tests/test_environment.py -v`
Expected: FAIL — `ModuleNotFoundError: No module named 'spectra.environment.core'`

- [ ] **Step 3: Implement core dataclasses and Environment**

Create `python/spectra/environment/core.py`:

```python
"""Core Environment, Emitter, ReceiverConfig, and LinkParams classes."""

from __future__ import annotations

import math
from dataclasses import dataclass

import numpy as np

from spectra.environment.position import Position
from spectra.environment.propagation import PropagationModel
from spectra.waveforms.base import Waveform

SPEED_OF_LIGHT = 299_792_458.0
BOLTZMANN_K = 1.380649e-23  # J/K


@dataclass
class Emitter:
    """A transmitting source with waveform, position, and RF parameters."""

    waveform: Waveform
    position: Position
    power_dbm: float
    freq_hz: float
    velocity_mps: tuple[float, float] | None = None
    antenna_gain_dbi: float = 0.0


@dataclass
class ReceiverConfig:
    """Receiver parameters for link budget computation."""

    position: Position
    noise_figure_db: float = 6.0
    bandwidth_hz: float = 1e6
    antenna_gain_dbi: float = 0.0
    temperature_k: float = 290.0


@dataclass
class LinkParams:
    """Derived link parameters for a single emitter."""

    emitter_index: int
    snr_db: float
    path_loss_db: float
    received_power_dbm: float
    delay_s: float
    doppler_hz: float
    distance_m: float
    fading_suggestion: str | None


class Environment:
    """Computes per-emitter link parameters from geometry and propagation."""

    def __init__(
        self,
        propagation: PropagationModel,
        emitters: list[Emitter],
        receiver: ReceiverConfig,
    ):
        self.propagation = propagation
        self.emitters = emitters
        self.receiver = receiver

    def compute(self, seed: int | None = None) -> list[LinkParams]:
        """Compute link parameters for each emitter."""
        results = []
        for i, emitter in enumerate(self.emitters):
            distance = emitter.position.distance_to(self.receiver.position)

            # Propagation model — derive per-emitter seed from master seed
            kwargs = {}
            if seed is not None:
                kwargs["seed"] = seed + i
            pl_result = self.propagation(distance, emitter.freq_hz, **kwargs)

            # Link budget
            rx_power = (
                emitter.power_dbm
                + emitter.antenna_gain_dbi
                + self.receiver.antenna_gain_dbi
                - pl_result.path_loss_db
            )
            noise_power = (
                10 * math.log10(BOLTZMANN_K * self.receiver.temperature_k)
                + 30  # convert to dBm
                + 10 * math.log10(self.receiver.bandwidth_hz)
                + self.receiver.noise_figure_db
            )
            snr = rx_power - noise_power

            # Propagation delay
            delay = distance / SPEED_OF_LIGHT

            # Doppler
            doppler = 0.0
            if emitter.velocity_mps is not None:
                bearing = self.receiver.position.bearing_to(emitter.position)
                vx, vy = emitter.velocity_mps
                v_radial = -(vx * math.cos(bearing) + vy * math.sin(bearing))
                doppler = (v_radial / SPEED_OF_LIGHT) * emitter.freq_hz

            # Fading suggestion from propagation model metadata
            fading = None
            if pl_result.k_factor_db is not None:
                fading = f"rician_k{int(pl_result.k_factor_db)}"
            elif pl_result.rms_delay_spread_s is not None:
                fading = "rayleigh"

            results.append(
                LinkParams(
                    emitter_index=i,
                    snr_db=snr,
                    path_loss_db=pl_result.path_loss_db,
                    received_power_dbm=rx_power,
                    delay_s=delay,
                    doppler_hz=doppler,
                    distance_m=distance,
                    fading_suggestion=fading,
                )
            )
        return results
```

Update `python/spectra/environment/__init__.py`:

```python
"""Environment and propagation modeling for SPECTRA."""

from spectra.environment.core import Emitter, Environment, LinkParams, ReceiverConfig
from spectra.environment.position import Position
from spectra.environment.presets import propagation_presets
from spectra.environment.propagation import (
    COST231HataPL,
    FreeSpacePathLoss,
    LogDistancePL,
    PathLossResult,
    PropagationModel,
)

__all__ = [
    "COST231HataPL",
    "Emitter",
    "Environment",
    "FreeSpacePathLoss",
    "LinkParams",
    "LogDistancePL",
    "PathLossResult",
    "Position",
    "PropagationModel",
    "ReceiverConfig",
    "propagation_presets",
]
```

- [ ] **Step 4: Run tests to verify they pass**

Run: `pytest tests/test_environment.py -v`
Expected: All 13 tests PASS

- [ ] **Step 5: Commit**

```bash
git add python/spectra/environment/core.py python/spectra/environment/__init__.py tests/test_environment.py
git commit -m "feat(environment): add Environment, Emitter, ReceiverConfig, LinkParams"
```

---

### Task 7: Integration Bridge — link_params_to_impairments

**Files:**
- Create: `python/spectra/environment/integration.py`
- Create: `tests/test_environment_integration.py`
- Modify: `python/spectra/environment/__init__.py`

- [ ] **Step 1: Write failing tests for impairment bridge**

Create `tests/test_environment_integration.py`:

```python
"""Tests for environment-to-impairment integration."""

import math

import numpy as np
import pytest

from spectra.environment.core import Emitter, Environment, LinkParams, ReceiverConfig
from spectra.environment.integration import link_params_to_impairments
from spectra.environment.position import Position
from spectra.environment.propagation import FreeSpacePathLoss
from spectra.impairments import AWGN, Compose, DopplerShift, RayleighFading, RicianFading
from spectra.impairments.base import Transform
from spectra.scene import SignalDescription
from spectra.waveforms import QPSK


@pytest.fixture
def signal_description():
    return SignalDescription(
        t_start=0.0,
        t_stop=0.001,
        f_low=-5e3,
        f_high=5e3,
        label="QPSK",
        snr=20.0,
    )


class TestLinkParamsToImpairments:
    def test_returns_list_of_transforms(self):
        lp = LinkParams(
            emitter_index=0,
            snr_db=15.0,
            path_loss_db=100.0,
            received_power_dbm=-70.0,
            delay_s=3.3e-6,
            doppler_hz=0.0,
            distance_m=1000.0,
            fading_suggestion=None,
        )
        result = link_params_to_impairments(lp)
        assert isinstance(result, list)
        assert all(isinstance(t, Transform) for t in result)

    def test_awgn_always_present(self):
        lp = LinkParams(
            emitter_index=0,
            snr_db=15.0,
            path_loss_db=100.0,
            received_power_dbm=-70.0,
            delay_s=3.3e-6,
            doppler_hz=0.0,
            distance_m=1000.0,
            fading_suggestion=None,
        )
        result = link_params_to_impairments(lp)
        awgn_list = [t for t in result if isinstance(t, AWGN)]
        assert len(awgn_list) == 1

    def test_no_doppler_when_zero(self):
        lp = LinkParams(
            emitter_index=0,
            snr_db=15.0,
            path_loss_db=100.0,
            received_power_dbm=-70.0,
            delay_s=3.3e-6,
            doppler_hz=0.0,
            distance_m=1000.0,
            fading_suggestion=None,
        )
        result = link_params_to_impairments(lp)
        doppler_list = [t for t in result if isinstance(t, DopplerShift)]
        assert len(doppler_list) == 0

    def test_doppler_included_when_nonzero(self):
        lp = LinkParams(
            emitter_index=0,
            snr_db=15.0,
            path_loss_db=100.0,
            received_power_dbm=-70.0,
            delay_s=3.3e-6,
            doppler_hz=240.0,
            distance_m=1000.0,
            fading_suggestion=None,
        )
        result = link_params_to_impairments(lp)
        doppler_list = [t for t in result if isinstance(t, DopplerShift)]
        assert len(doppler_list) == 1

    def test_rician_fading_from_suggestion(self):
        lp = LinkParams(
            emitter_index=0,
            snr_db=15.0,
            path_loss_db=100.0,
            received_power_dbm=-70.0,
            delay_s=3.3e-6,
            doppler_hz=0.0,
            distance_m=1000.0,
            fading_suggestion="rician_k6",
        )
        result = link_params_to_impairments(lp)
        rician_list = [t for t in result if isinstance(t, RicianFading)]
        assert len(rician_list) == 1

    def test_rayleigh_fading_from_suggestion(self):
        lp = LinkParams(
            emitter_index=0,
            snr_db=15.0,
            path_loss_db=100.0,
            received_power_dbm=-70.0,
            delay_s=3.3e-6,
            doppler_hz=0.0,
            distance_m=1000.0,
            fading_suggestion="rayleigh",
        )
        result = link_params_to_impairments(lp)
        rayleigh_list = [t for t in result if isinstance(t, RayleighFading)]
        assert len(rayleigh_list) == 1

    def test_no_fading_when_none(self):
        lp = LinkParams(
            emitter_index=0,
            snr_db=15.0,
            path_loss_db=100.0,
            received_power_dbm=-70.0,
            delay_s=3.3e-6,
            doppler_hz=0.0,
            distance_m=1000.0,
            fading_suggestion=None,
        )
        result = link_params_to_impairments(lp)
        fading_list = [t for t in result if isinstance(t, (RayleighFading, RicianFading))]
        assert len(fading_list) == 0

    def test_impairment_order_doppler_fading_awgn(self):
        lp = LinkParams(
            emitter_index=0,
            snr_db=15.0,
            path_loss_db=100.0,
            received_power_dbm=-70.0,
            delay_s=3.3e-6,
            doppler_hz=240.0,
            distance_m=1000.0,
            fading_suggestion="rayleigh",
        )
        result = link_params_to_impairments(lp)
        assert len(result) == 3
        assert isinstance(result[0], DopplerShift)
        assert isinstance(result[1], RayleighFading)
        assert isinstance(result[2], AWGN)


class TestEndToEnd:
    def test_environment_to_impairments_apply(self, signal_description, assert_valid_iq):
        """Full pipeline: Environment -> compute -> impairments -> apply to waveform."""
        env = Environment(
            propagation=FreeSpacePathLoss(),
            emitters=[
                Emitter(
                    waveform=QPSK(samples_per_symbol=8),
                    position=Position(500.0, 0.0),
                    power_dbm=30.0,
                    freq_hz=2.4e9,
                ),
            ],
            receiver=ReceiverConfig(
                position=Position(0.0, 0.0),
                noise_figure_db=6.0,
                bandwidth_hz=1e6,
            ),
        )
        params = env.compute()[0]
        impairments = link_params_to_impairments(params)

        iq = env.emitters[0].waveform.generate(
            num_symbols=128, sample_rate=1e6, seed=42
        )
        desc = signal_description
        for t in impairments:
            iq, desc = t(iq, desc, sample_rate=1e6)

        assert_valid_iq(iq)

    def test_compose_wrapping(self, signal_description, assert_valid_iq):
        """Impairments can be wrapped in Compose."""
        lp = LinkParams(
            emitter_index=0,
            snr_db=20.0,
            path_loss_db=80.0,
            received_power_dbm=-50.0,
            delay_s=1e-6,
            doppler_hz=0.0,
            distance_m=300.0,
            fading_suggestion=None,
        )
        chain = Compose(link_params_to_impairments(lp))
        waveform = QPSK(samples_per_symbol=8)
        iq = waveform.generate(num_symbols=128, sample_rate=1e6, seed=42)
        iq, desc = chain(iq, signal_description, sample_rate=1e6)
        assert_valid_iq(iq)

    def test_override_snr_before_conversion(self, signal_description, assert_valid_iq):
        """Users can override LinkParams fields before converting to impairments."""
        env = Environment(
            propagation=FreeSpacePathLoss(),
            emitters=[
                Emitter(
                    waveform=QPSK(samples_per_symbol=8),
                    position=Position(500.0, 0.0),
                    power_dbm=30.0,
                    freq_hz=2.4e9,
                ),
            ],
            receiver=ReceiverConfig(position=Position(0.0, 0.0)),
        )
        params = env.compute()[0]
        params.snr_db = 5.0  # override to low SNR
        impairments = link_params_to_impairments(params)

        iq = env.emitters[0].waveform.generate(
            num_symbols=128, sample_rate=1e6, seed=42
        )
        for t in impairments:
            iq, desc = t(iq, signal_description, sample_rate=1e6)
        assert_valid_iq(iq)
```

- [ ] **Step 2: Run tests to verify they fail**

Run: `pytest tests/test_environment_integration.py -v`
Expected: FAIL — `ModuleNotFoundError: No module named 'spectra.environment.integration'`

- [ ] **Step 3: Implement integration bridge**

Create `python/spectra/environment/integration.py`:

```python
"""Bridge between Environment link parameters and SPECTRA impairments."""

from __future__ import annotations

import re

from spectra.environment.core import LinkParams
from spectra.impairments import AWGN, DopplerShift, RayleighFading, RicianFading
from spectra.impairments.base import Transform


def _fading_from_suggestion(suggestion: str) -> Transform:
    """Map a fading suggestion string to a configured impairment instance."""
    if suggestion == "rayleigh":
        return RayleighFading()
    match = re.match(r"rician_k(\d+)", suggestion)
    if match:
        k = float(match.group(1))
        return RicianFading(k_factor=k)
    return RayleighFading()


def link_params_to_impairments(params: LinkParams) -> list[Transform]:
    """Convert derived link parameters to an ordered impairment chain.

    Order: Doppler (if nonzero) -> Fading (if suggested) -> AWGN (always last).
    """
    impairments: list[Transform] = []

    if abs(params.doppler_hz) > 0.01:
        impairments.append(DopplerShift(fd_hz=params.doppler_hz))

    if params.fading_suggestion is not None:
        impairments.append(_fading_from_suggestion(params.fading_suggestion))

    impairments.append(AWGN(snr_db=params.snr_db))

    return impairments
```

Update `python/spectra/environment/__init__.py` to add `link_params_to_impairments` to imports and `__all__`.

- [ ] **Step 4: Run tests to verify they pass**

Run: `pytest tests/test_environment_integration.py -v`
Expected: All 11 tests PASS

- [ ] **Step 5: Run full test suite to check for regressions**

Run: `pytest tests/ -v --timeout=120`
Expected: All existing tests still pass, plus all new tests pass.

- [ ] **Step 6: Commit**

```bash
git add python/spectra/environment/integration.py python/spectra/environment/__init__.py tests/test_environment_integration.py
git commit -m "feat(environment): add link_params_to_impairments integration bridge"
```

---

### Task 8: Top-Level Exports

**Files:**
- Modify: `python/spectra/__init__.py`

- [ ] **Step 1: Add environment exports to top-level __init__.py**

Add the following import block to `python/spectra/__init__.py` after the `# Scene composition` section:

```python
# Environment
from spectra.environment import (
    COST231HataPL,
    Emitter,
    Environment,
    FreeSpacePathLoss,
    LinkParams,
    LogDistancePL,
    PathLossResult,
    Position,
    PropagationModel,
    ReceiverConfig,
    link_params_to_impairments,
    propagation_presets,
)
```

Add the following entries to `__all__` after the `# Scene` section:

```python
    # Environment
    "COST231HataPL",
    "Emitter",
    "Environment",
    "FreeSpacePathLoss",
    "LinkParams",
    "LogDistancePL",
    "PathLossResult",
    "Position",
    "PropagationModel",
    "ReceiverConfig",
    "link_params_to_impairments",
    "propagation_presets",
```

- [ ] **Step 2: Verify imports work**

Run: `python -c "from spectra import Environment, Position, Emitter, ReceiverConfig, FreeSpacePathLoss, LogDistancePL, COST231HataPL, LinkParams, link_params_to_impairments, propagation_presets; print('All imports OK')"`
Expected: `All imports OK`

- [ ] **Step 3: Run full test suite**

Run: `pytest tests/ -v --timeout=120`
Expected: All tests pass.

- [ ] **Step 4: Commit**

```bash
git add python/spectra/__init__.py
git commit -m "feat(environment): add environment module exports to top-level __init__"
```

---

### Task 9: YAML Serialization

**Files:**
- Modify: `python/spectra/environment/core.py`
- Add YAML tests to: `tests/test_environment.py`

- [ ] **Step 1: Write failing tests for YAML round-trip**

Append to `tests/test_environment.py`:

```python
import tempfile
import os

yaml = pytest.importorskip("yaml")


class TestEnvironmentYAML:
    def test_to_yaml_creates_file(self, simple_env, tmp_path):
        path = str(tmp_path / "env.yaml")
        simple_env.to_yaml(path)
        assert os.path.exists(path)

    def test_round_trip(self, tmp_path):
        env = Environment(
            propagation=LogDistancePL(n=3.5, sigma_db=8.0),
            emitters=[
                Emitter(
                    waveform=QPSK(samples_per_symbol=8),
                    position=Position(500.0, 200.0),
                    power_dbm=30.0,
                    freq_hz=2.4e9,
                ),
            ],
            receiver=ReceiverConfig(
                position=Position(0.0, 0.0),
                noise_figure_db=6.0,
                bandwidth_hz=1e6,
            ),
        )
        path = str(tmp_path / "env.yaml")
        env.to_yaml(path)
        loaded = Environment.from_yaml(path)

        orig = env.compute(seed=42)
        reloaded = loaded.compute(seed=42)
        assert len(orig) == len(reloaded)
        assert math.isclose(orig[0].snr_db, reloaded[0].snr_db, rel_tol=1e-6)
        assert math.isclose(orig[0].distance_m, reloaded[0].distance_m, rel_tol=1e-6)

    def test_round_trip_free_space(self, tmp_path):
        env = Environment(
            propagation=FreeSpacePathLoss(),
            emitters=[
                Emitter(
                    waveform=QPSK(samples_per_symbol=8),
                    position=Position(1000.0, 0.0),
                    power_dbm=20.0,
                    freq_hz=1e9,
                ),
            ],
            receiver=ReceiverConfig(position=Position(0.0, 0.0)),
        )
        path = str(tmp_path / "env.yaml")
        env.to_yaml(path)
        loaded = Environment.from_yaml(path)
        orig = env.compute()
        reloaded = loaded.compute()
        assert math.isclose(orig[0].snr_db, reloaded[0].snr_db, rel_tol=1e-6)

    def test_round_trip_cost231(self, tmp_path):
        from spectra.environment.propagation import COST231HataPL

        env = Environment(
            propagation=COST231HataPL(h_bs_m=50, h_ms_m=2.0, environment="suburban"),
            emitters=[
                Emitter(
                    waveform=QPSK(samples_per_symbol=8),
                    position=Position(2000.0, 0.0),
                    power_dbm=40.0,
                    freq_hz=1800e6,
                ),
            ],
            receiver=ReceiverConfig(position=Position(0.0, 0.0)),
        )
        path = str(tmp_path / "env.yaml")
        env.to_yaml(path)
        loaded = Environment.from_yaml(path)
        orig = env.compute()
        reloaded = loaded.compute()
        assert math.isclose(orig[0].snr_db, reloaded[0].snr_db, rel_tol=1e-6)
```

- [ ] **Step 2: Run tests to verify they fail**

Run: `pytest tests/test_environment.py::TestEnvironmentYAML -v`
Expected: FAIL — `AttributeError: type object 'Environment' has no attribute 'to_yaml'`

- [ ] **Step 3: Implement YAML serialization on Environment**

Add to `python/spectra/environment/core.py` — add the following imports at the top:

```python
from spectra.environment.propagation import (
    COST231HataPL,
    FreeSpacePathLoss,
    LogDistancePL,
    PropagationModel,
)
```

Add a module-level propagation registry dict:

```python
_PROPAGATION_REGISTRY: dict[str, type[PropagationModel]] = {
    "free_space": FreeSpacePathLoss,
    "log_distance": LogDistancePL,
    "cost231_hata": COST231HataPL,
}
```

Add a module-level waveform registry helper:

```python
def _waveform_to_dict(waveform: Waveform) -> dict:
    """Serialize a waveform to a type/params dict."""
    cls_name = type(waveform).__name__
    params = {}
    if hasattr(waveform, "__dict__"):
        for k, v in waveform.__dict__.items():
            if not k.startswith("_"):
                params[k] = v
    return {"type": cls_name, "params": params} if params else {"type": cls_name}


def _waveform_from_dict(d: dict) -> Waveform:
    """Deserialize a waveform from a type/params dict."""
    import spectra.waveforms as wmod

    cls = getattr(wmod, d["type"])
    params = d.get("params", {})
    return cls(**params)
```

Add `to_yaml` and `from_yaml` methods to the `Environment` class:

```python
    def to_yaml(self, path: str) -> None:
        """Serialize this Environment to a YAML file."""
        import yaml

        prop = self.propagation
        prop_dict: dict = {}
        if isinstance(prop, FreeSpacePathLoss):
            prop_dict = {"type": "free_space"}
        elif isinstance(prop, LogDistancePL):
            prop_dict = {"type": "log_distance", "n": prop.n, "sigma_db": prop.sigma_db, "d0": prop.d0}
        elif isinstance(prop, COST231HataPL):
            prop_dict = {
                "type": "cost231_hata",
                "h_bs_m": prop.h_bs_m,
                "h_ms_m": prop.h_ms_m,
                "environment": prop.environment,
            }

        emitters_list = []
        for e in self.emitters:
            entry: dict = {
                "waveform": _waveform_to_dict(e.waveform),
                "position": [e.position.x, e.position.y] + ([e.position.z] if e.position.z is not None else []),
                "power_dbm": e.power_dbm,
                "freq_hz": e.freq_hz,
            }
            if e.velocity_mps is not None:
                entry["velocity_mps"] = list(e.velocity_mps)
            if e.antenna_gain_dbi != 0.0:
                entry["antenna_gain_dbi"] = e.antenna_gain_dbi
            emitters_list.append(entry)

        rx = self.receiver
        rx_dict: dict = {
            "position": [rx.position.x, rx.position.y] + ([rx.position.z] if rx.position.z is not None else []),
            "noise_figure_db": rx.noise_figure_db,
            "bandwidth_hz": rx.bandwidth_hz,
        }
        if rx.antenna_gain_dbi != 0.0:
            rx_dict["antenna_gain_dbi"] = rx.antenna_gain_dbi
        if rx.temperature_k != 290.0:
            rx_dict["temperature_k"] = rx.temperature_k

        data = {
            "environment": {
                "propagation": prop_dict,
                "receiver": rx_dict,
                "emitters": emitters_list,
            }
        }
        with open(path, "w") as f:
            yaml.dump(data, f, default_flow_style=False, sort_keys=False)

    @classmethod
    def from_yaml(cls, path: str) -> "Environment":
        """Deserialize an Environment from a YAML file."""
        import yaml

        with open(path) as f:
            data = yaml.safe_load(f)

        env_data = data["environment"]

        # Propagation model
        prop_data = env_data["propagation"]
        prop_type = prop_data["type"]
        prop_cls = _PROPAGATION_REGISTRY[prop_type]
        prop_params = {k: v for k, v in prop_data.items() if k != "type"}
        propagation = prop_cls(**prop_params)

        # Receiver
        rx_data = env_data["receiver"]
        rx_pos_list = rx_data["position"]
        rx_pos = Position(rx_pos_list[0], rx_pos_list[1], rx_pos_list[2] if len(rx_pos_list) > 2 else None)
        receiver = ReceiverConfig(
            position=rx_pos,
            noise_figure_db=rx_data.get("noise_figure_db", 6.0),
            bandwidth_hz=rx_data.get("bandwidth_hz", 1e6),
            antenna_gain_dbi=rx_data.get("antenna_gain_dbi", 0.0),
            temperature_k=rx_data.get("temperature_k", 290.0),
        )

        # Emitters
        emitters = []
        for e_data in env_data["emitters"]:
            pos_list = e_data["position"]
            pos = Position(pos_list[0], pos_list[1], pos_list[2] if len(pos_list) > 2 else None)
            vel = tuple(e_data["velocity_mps"]) if "velocity_mps" in e_data else None
            emitters.append(
                Emitter(
                    waveform=_waveform_from_dict(e_data["waveform"]),
                    position=pos,
                    power_dbm=e_data["power_dbm"],
                    freq_hz=e_data["freq_hz"],
                    velocity_mps=vel,
                    antenna_gain_dbi=e_data.get("antenna_gain_dbi", 0.0),
                )
            )

        return cls(propagation=propagation, emitters=emitters, receiver=receiver)
```

- [ ] **Step 4: Run tests to verify they pass**

Run: `pytest tests/test_environment.py -v`
Expected: All tests PASS (including new YAML tests)

- [ ] **Step 5: Run full test suite**

Run: `pytest tests/ -v --timeout=120`
Expected: All tests pass.

- [ ] **Step 6: Commit**

```bash
git add python/spectra/environment/core.py tests/test_environment.py
git commit -m "feat(environment): add YAML serialization for Environment"
```

---

### Task 10: Final Verification and Cleanup

**Files:**
- All files in `python/spectra/environment/`
- All new test files

- [ ] **Step 1: Run the full test suite**

Run: `pytest tests/ -v --timeout=120`
Expected: All tests pass with no regressions.

- [ ] **Step 2: Run linting**

Run: `ruff check python/spectra/environment/ tests/test_position.py tests/test_propagation.py tests/test_presets.py tests/test_environment.py tests/test_environment_integration.py`
Expected: No errors. If there are lint issues, fix them.

- [ ] **Step 3: Run formatting check**

Run: `ruff format --check python/spectra/environment/ tests/test_position.py tests/test_propagation.py tests/test_presets.py tests/test_environment.py tests/test_environment_integration.py`
Expected: All files formatted. If not, run `ruff format` to fix.

- [ ] **Step 4: Verify import from top-level**

Run: `python -c "import spectra; env = spectra.Environment(propagation=spectra.FreeSpacePathLoss(), emitters=[spectra.Emitter(waveform=spectra.QPSK(samples_per_symbol=8), position=spectra.Position(1000, 0), power_dbm=30, freq_hz=2.4e9)], receiver=spectra.ReceiverConfig(position=spectra.Position(0, 0))); print(env.compute())"`
Expected: Prints a list with one `LinkParams` object showing computed values.

- [ ] **Step 5: Commit any lint/format fixes**

```bash
git add -u
git commit -m "style: fix lint and formatting in environment module"
```
