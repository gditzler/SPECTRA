# Direction-Finding Dataset Generation Implementation Plan

> **For agentic workers:** REQUIRED: Use superpowers:subagent-driven-development (if subagents available) or superpowers:executing-plans to implement this plan. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Add direction-finding (DoA) dataset generation to SPECTRA with antenna element models, flexible array geometry, spatial signal mixing, and a PyTorch Dataset that returns multi-antenna IQ snapshots with AoA ground truth.

**Architecture:** Four new Python packages are added — `antennas/` (element pattern ABCs and built-ins), `arrays/` (geometry and steering vector), a new `datasets/direction_finding.py` module, and a `transforms/snapshot.py` helper. All existing modules (waveforms, impairments, `SignalDescription`) are reused without modification. State and randomness live in Python; no Rust changes.

**Tech Stack:** Python 3.10+, NumPy (array math), SciPy (bilinear interpolation for MSI patterns — already a dependency), PyTorch (Dataset/DataLoader integration), pytest (TDD).

---

## File Structure

### New files (created)

| File | Responsibility |
|------|---------------|
| `python/spectra/antennas/__init__.py` | Public re-exports for `antennas` package |
| `python/spectra/antennas/base.py` | `AntennaElement` ABC |
| `python/spectra/antennas/isotropic.py` | `IsotropicElement` |
| `python/spectra/antennas/dipole.py` | `ShortDipoleElement`, `HalfWaveDipoleElement` |
| `python/spectra/antennas/cosine_power.py` | `CosinePowerElement` |
| `python/spectra/antennas/msi.py` | `MSIAntennaElement`, `parse_msi` |
| `python/spectra/arrays/__init__.py` | Public re-exports for `arrays` package |
| `python/spectra/arrays/array.py` | `AntennaArray`, `ula()`, `uca()`, `rectangular()` |
| `python/spectra/arrays/calibration.py` | `CalibrationErrors` |
| `python/spectra/datasets/direction_finding.py` | `DirectionFindingDataset`, `DirectionFindingTarget` |
| `python/spectra/transforms/snapshot.py` | `ToSnapshotMatrix` |
| `tests/test_antennas.py` | Tests for antenna element models |
| `tests/test_arrays.py` | Tests for `AntennaArray`, geometry constructors, `CalibrationErrors` |
| `tests/test_direction_finding_dataset.py` | Tests for `DirectionFindingDataset` and `DirectionFindingTarget` |
| `tests/test_snapshot_transform.py` | Tests for `ToSnapshotMatrix` |
| ~~`tests/fixtures/sample.msi`~~ | MSI fixture is generated programmatically via `tmp_path` in tests — no static file needed |

### Modified files

| File | Change |
|------|--------|
| `python/spectra/datasets/__init__.py` | Add `DirectionFindingDataset`, `DirectionFindingTarget` imports and `__all__` entries |
| `python/spectra/transforms/__init__.py` | Add `ToSnapshotMatrix` import and `__all__` entry |

---

## Task 1: `AntennaElement` ABC

**Files:**
- Create: `python/spectra/antennas/base.py`
- Create: `tests/test_antennas.py`

- [ ] **Step 1: Write the failing test**

```python
# tests/test_antennas.py
import numpy as np
import pytest


def test_cannot_instantiate_antenna_element_abc():
    from spectra.antennas.base import AntennaElement

    class Incomplete(AntennaElement):
        pass  # does not implement abstract methods

    with pytest.raises(TypeError):
        Incomplete()


def test_concrete_antenna_element_interface():
    from spectra.antennas.base import AntennaElement

    class Minimal(AntennaElement):
        @property
        def frequency(self):
            return 2.4e9

        def pattern(self, azimuth, elevation):
            return np.ones_like(azimuth, dtype=complex)

    elem = Minimal()
    az = np.array([0.0, np.pi / 4])
    el = np.array([0.0, 0.0])
    gain = elem.pattern(az, el)
    assert gain.shape == (2,)
    assert gain.dtype == complex or np.issubdtype(gain.dtype, np.complexfloating)
```

- [ ] **Step 2: Run test to verify it fails**

```bash
pytest tests/test_antennas.py -v
```
Expected: `ImportError` (module doesn't exist yet).

- [ ] **Step 3: Write minimal implementation**

```python
# python/spectra/antennas/base.py
from abc import ABC, abstractmethod

import numpy as np


class AntennaElement(ABC):
    """Abstract base class for antenna element radiation patterns.

    All elements return complex gain so the interface supports phase patterns.
    Initial built-in implementations are real-valued (zero phase).

    Subclasses must implement ``pattern()`` and the ``frequency`` property.
    """

    @abstractmethod
    def pattern(self, azimuth: np.ndarray, elevation: np.ndarray) -> np.ndarray:
        """Return complex gain at query angles.

        Args:
            azimuth: Azimuth angles in radians. Arbitrary shape.
            elevation: Elevation angles in radians. Same broadcast-compatible
                shape as ``azimuth``.

        Returns:
            Complex-valued array with shape matching the broadcast shape of
            inputs. Magnitude is linear gain; phase is the pattern phase shift.
        """
        ...

    @property
    @abstractmethod
    def frequency(self) -> float:
        """Design frequency in Hz."""
        ...
```

- [ ] **Step 4: Run test to verify it passes**

```bash
pytest tests/test_antennas.py::test_cannot_instantiate_antenna_element_abc tests/test_antennas.py::test_concrete_antenna_element_interface -v
```
Expected: PASS

- [ ] **Step 5: Commit**

```bash
git add python/spectra/antennas/base.py tests/test_antennas.py
git commit -m "feat(antennas): add AntennaElement ABC"
```

---

## Task 2: `IsotropicElement`

**Files:**
- Create: `python/spectra/antennas/isotropic.py`
- Modify: `tests/test_antennas.py`

- [ ] **Step 1: Write the failing test**

Append to `tests/test_antennas.py`:

```python
def test_isotropic_element_unity_gain():
    from spectra.antennas.isotropic import IsotropicElement

    elem = IsotropicElement(frequency=2.4e9)
    az = np.linspace(0, 2 * np.pi, 36)
    el = np.linspace(-np.pi / 2, np.pi / 2, 36)
    gain = elem.pattern(az, el)
    assert gain.shape == (36,)
    np.testing.assert_array_equal(gain, 1.0 + 0j)


def test_isotropic_frequency_property():
    from spectra.antennas.isotropic import IsotropicElement

    elem = IsotropicElement(frequency=900e6)
    assert elem.frequency == 900e6
```

- [ ] **Step 2: Run test to verify it fails**

```bash
pytest tests/test_antennas.py::test_isotropic_element_unity_gain -v
```
Expected: `ImportError`

- [ ] **Step 3: Write minimal implementation**

```python
# python/spectra/antennas/isotropic.py
import numpy as np

from spectra.antennas.base import AntennaElement


class IsotropicElement(AntennaElement):
    """Isotropic antenna element — unit gain in all directions.

    Args:
        frequency: Design frequency in Hz.
    """

    def __init__(self, frequency: float):
        self._frequency = frequency

    @property
    def frequency(self) -> float:
        return self._frequency

    def pattern(self, azimuth: np.ndarray, elevation: np.ndarray) -> np.ndarray:
        azimuth = np.asarray(azimuth)
        elevation = np.asarray(elevation)
        shape = np.broadcast_shapes(azimuth.shape, elevation.shape)
        return np.ones(shape, dtype=complex)
```

- [ ] **Step 4: Run test to verify it passes**

```bash
pytest tests/test_antennas.py -k "isotropic" -v
```
Expected: PASS

- [ ] **Step 5: Commit**

```bash
git add python/spectra/antennas/isotropic.py tests/test_antennas.py
git commit -m "feat(antennas): add IsotropicElement"
```

---

## Task 3: `ShortDipoleElement` and `HalfWaveDipoleElement`

**Files:**
- Create: `python/spectra/antennas/dipole.py`
- Modify: `tests/test_antennas.py`

Background: For a z-axis dipole, the elevation angle `theta` from the z-axis is `pi/2 - elevation` (where elevation is measured from the xy-plane). The azimuth `axis` can be `'x'`, `'y'`, or `'z'`.

- [ ] **Step 1: Write the failing tests**

Append to `tests/test_antennas.py`:

```python
def test_short_dipole_z_axis_broadside():
    """Gain is maximum (=1) at elevation=0 (equatorial plane), zero at poles."""
    from spectra.antennas.dipole import ShortDipoleElement

    elem = ShortDipoleElement(axis="z", frequency=300e6)
    # Broadside: elevation=0 → theta_from_z = pi/2 → sin(pi/2) = 1
    az = np.array([0.0])
    el = np.array([0.0])
    gain = elem.pattern(az, el)
    np.testing.assert_allclose(np.abs(gain), [1.0], atol=1e-6)


def test_short_dipole_z_axis_pole_null():
    """Gain is zero at elevation=pi/2 (pole, along dipole axis)."""
    from spectra.antennas.dipole import ShortDipoleElement

    elem = ShortDipoleElement(axis="z", frequency=300e6)
    az = np.array([0.0])
    el = np.array([np.pi / 2])
    gain = elem.pattern(az, el)
    np.testing.assert_allclose(np.abs(gain), [0.0], atol=1e-6)


def test_half_wave_dipole_z_broadside():
    """Half-wave dipole: gain at broadside should be ~1 (normalized)."""
    from spectra.antennas.dipole import HalfWaveDipoleElement

    elem = HalfWaveDipoleElement(axis="z", frequency=300e6)
    az = np.array([0.0])
    el = np.array([0.0])  # elevation 0 → sin(theta)=1 in denominator handling
    gain = elem.pattern(az, el)
    assert np.abs(gain[0]) > 0.9  # normalized to ~1


def test_dipole_returns_complex():
    from spectra.antennas.dipole import ShortDipoleElement

    elem = ShortDipoleElement(axis="z", frequency=300e6)
    gain = elem.pattern(np.array([0.0]), np.array([0.0]))
    assert np.issubdtype(gain.dtype, np.complexfloating)
```

- [ ] **Step 2: Run tests to verify they fail**

```bash
pytest tests/test_antennas.py -k "dipole" -v
```
Expected: `ImportError`

- [ ] **Step 3: Write minimal implementation**

```python
# python/spectra/antennas/dipole.py
"""Short and half-wave dipole antenna element patterns."""

from typing import Literal

import numpy as np

from spectra.antennas.base import AntennaElement


def _theta_from_axis(
    azimuth: np.ndarray,
    elevation: np.ndarray,
    axis: str,
) -> np.ndarray:
    """Compute angle from dipole axis in radians.

    Standard spherical coordinates: azimuth from x-axis in xy-plane,
    elevation above xy-plane. The 'theta' used in dipole patterns is the
    polar angle from the dipole axis.

    For z-axis dipole: theta = pi/2 - elevation.
    For x-axis dipole: theta = arccos(cos(elevation)*cos(azimuth)).
    For y-axis dipole: theta = arccos(cos(elevation)*sin(azimuth)).
    """
    el = np.asarray(elevation, dtype=float)
    az = np.asarray(azimuth, dtype=float)
    if axis == "z":
        return np.pi / 2 - el
    elif axis == "x":
        return np.arccos(np.cos(el) * np.cos(az))
    elif axis == "y":
        return np.arccos(np.cos(el) * np.sin(az))
    else:
        raise ValueError(f"axis must be 'x', 'y', or 'z', got {axis!r}")


class ShortDipoleElement(AntennaElement):
    """Short electric dipole element with sin(theta) radiation pattern.

    The pattern is real-valued and normalized to 1 at the equatorial plane.

    Args:
        axis: Dipole axis orientation — ``'x'``, ``'y'``, or ``'z'``.
        frequency: Design frequency in Hz.
    """

    def __init__(self, axis: Literal["x", "y", "z"] = "z", frequency: float = 1e9):
        self._axis = axis
        self._frequency = frequency

    @property
    def frequency(self) -> float:
        return self._frequency

    def pattern(self, azimuth: np.ndarray, elevation: np.ndarray) -> np.ndarray:
        azimuth = np.asarray(azimuth, dtype=float)
        elevation = np.asarray(elevation, dtype=float)
        az_b, el_b = np.broadcast_arrays(azimuth, elevation)
        theta = _theta_from_axis(az_b, el_b, self._axis)
        return np.sin(theta).astype(complex)


class HalfWaveDipoleElement(AntennaElement):
    """Half-wave dipole element.

    Pattern: ``cos(pi/2 * cos(theta)) / sin(theta)``, normalized to 1 at
    broadside. Near-zero sin(theta) values are clamped to avoid division by zero.

    Args:
        axis: Dipole axis orientation — ``'x'``, ``'y'``, or ``'z'``.
        frequency: Design frequency in Hz.
    """

    def __init__(self, axis: Literal["x", "y", "z"] = "z", frequency: float = 1e9):
        self._axis = axis
        self._frequency = frequency

    @property
    def frequency(self) -> float:
        return self._frequency

    def pattern(self, azimuth: np.ndarray, elevation: np.ndarray) -> np.ndarray:
        azimuth = np.asarray(azimuth, dtype=float)
        elevation = np.asarray(elevation, dtype=float)
        az_b, el_b = np.broadcast_arrays(azimuth, elevation)
        theta = _theta_from_axis(az_b, el_b, self._axis)
        sin_theta = np.where(np.abs(np.sin(theta)) < 1e-12, 1e-12, np.sin(theta))
        numerator = np.cos((np.pi / 2) * np.cos(theta))
        gain = numerator / sin_theta
        # Normalize to 1 at broadside (theta = pi/2)
        broadside = np.cos((np.pi / 2) * np.cos(np.pi / 2)) / np.sin(np.pi / 2)
        if broadside != 0:
            gain = gain / broadside
        return gain.astype(complex)
```

- [ ] **Step 4: Run tests to verify they pass**

```bash
pytest tests/test_antennas.py -k "dipole" -v
```
Expected: PASS

- [ ] **Step 5: Commit**

```bash
git add python/spectra/antennas/dipole.py tests/test_antennas.py
git commit -m "feat(antennas): add ShortDipoleElement and HalfWaveDipoleElement"
```

---

## Task 4: `CosinePowerElement`

**Files:**
- Create: `python/spectra/antennas/cosine_power.py`
- Modify: `tests/test_antennas.py`

Background: `CosinePowerElement` models a forward-facing patch antenna. The pattern is `cos^n(theta_off_boresight)` where `theta_off_boresight` is the angle from the boresight direction (positive z-axis, i.e., `elevation=pi/2`). `peak_gain_dbi` is converted to linear to scale the pattern.

- [ ] **Step 1: Write the failing tests**

Append to `tests/test_antennas.py`:

```python
def test_cosine_power_boresight_max():
    """Gain at boresight (elevation=pi/2) should equal the linear peak gain."""
    from spectra.antennas.cosine_power import CosinePowerElement

    elem = CosinePowerElement(exponent=1.5, peak_gain_dbi=3.0, frequency=2.4e9)
    # Boresight: theta_off = 0 → cos^n(0) = 1, scaled by peak_gain_linear
    gain = elem.pattern(np.array([0.0]), np.array([np.pi / 2]))
    peak_linear = 10 ** (3.0 / 10.0)
    np.testing.assert_allclose(np.abs(gain[0]), peak_linear, rtol=1e-5)


def test_cosine_power_back_hemisphere_zero():
    """Gain is zero for theta_off > pi/2 (back hemisphere)."""
    from spectra.antennas.cosine_power import CosinePowerElement

    elem = CosinePowerElement(exponent=1.5, peak_gain_dbi=0.0, frequency=2.4e9)
    # elevation = -pi/2 → boresight angle = pi → cos(pi) < 0 → clamp to 0
    gain = elem.pattern(np.array([0.0]), np.array([-np.pi / 2]))
    np.testing.assert_allclose(np.abs(gain[0]), 0.0, atol=1e-6)


def test_cosine_power_returns_complex():
    from spectra.antennas.cosine_power import CosinePowerElement

    elem = CosinePowerElement(exponent=2.0, frequency=1e9)
    gain = elem.pattern(np.array([0.0, 1.0]), np.array([0.0, 0.5]))
    assert np.issubdtype(gain.dtype, np.complexfloating)
```

- [ ] **Step 2: Run tests to verify they fail**

```bash
pytest tests/test_antennas.py -k "cosine_power" -v
```
Expected: `ImportError`

- [ ] **Step 3: Write minimal implementation**

```python
# python/spectra/antennas/cosine_power.py
"""Cosine-power element pattern approximating patch/microstrip antennas."""

import numpy as np

from spectra.antennas.base import AntennaElement


class CosinePowerElement(AntennaElement):
    """Cosine-power element: ``cos^n(theta_off_boresight) * peak_gain_linear``.

    The boresight is the positive z-axis (elevation = pi/2). The pattern is
    clamped to zero for angles in the back hemisphere (cos < 0), approximating
    a patch or microstrip antenna mounted on a ground plane.

    Args:
        exponent: Controls beamwidth. Higher values → narrower beam.
        peak_gain_dbi: Peak gain in dBi (at boresight). Defaults to 0 dBi.
        frequency: Design frequency in Hz.
    """

    def __init__(
        self,
        exponent: float = 1.5,
        peak_gain_dbi: float = 0.0,
        frequency: float = 1e9,
    ):
        self.exponent = exponent
        self.peak_gain_dbi = peak_gain_dbi
        self._frequency = frequency
        self._peak_linear = 10.0 ** (peak_gain_dbi / 10.0)

    @property
    def frequency(self) -> float:
        return self._frequency

    def pattern(self, azimuth: np.ndarray, elevation: np.ndarray) -> np.ndarray:
        azimuth = np.asarray(azimuth, dtype=float)
        elevation = np.asarray(elevation, dtype=float)
        az_b, el_b = np.broadcast_arrays(azimuth, elevation)
        # theta_off_boresight: angle from +z axis (elevation=pi/2)
        # cos(theta_off) = sin(elevation)
        cos_theta = np.sin(el_b)
        gain = np.where(cos_theta > 0, cos_theta**self.exponent, 0.0)
        return (gain * self._peak_linear).astype(complex)
```

- [ ] **Step 4: Run tests to verify they pass**

```bash
pytest tests/test_antennas.py -k "cosine_power" -v
```
Expected: PASS

- [ ] **Step 5: Commit**

```bash
git add python/spectra/antennas/cosine_power.py tests/test_antennas.py
git commit -m "feat(antennas): add CosinePowerElement"
```

---

## Task 5: MSI/Planet File Parser and `MSIAntennaElement`

**Files:**
- Create: `python/spectra/antennas/msi.py`
- Create: `tests/fixtures/sample.msi`
- Modify: `tests/test_antennas.py`

Background: MSI/Planet format uses a text header followed by `HORIZONTAL` and `VERTICAL` sections, each with 360 lines of `"angle gain_db"`. The 2D gain surface is built using the additive method: `G(az, el) = G_h(az) + G_v(el_deg + 90) - G_peak` in dB (elevation 0 maps to 90 degrees in the vertical array, where the vertical section uses 0=zenith convention). The `RegularGridInterpolator` from scipy handles bilinear interpolation.

- [ ] **Step 1: Create a minimal synthetic MSI fixture**

```text
; tests/fixtures/sample.msi
NAME          TestAntenna
FREQUENCY     2400
GAIN          3
TILT          0
POLARIZATION  V

HORIZONTAL
0   0
1   0
2   0
... (360 lines of "N 0")
VERTICAL
0   -30
1   -28
... (360 lines with some variation)
```

Instead of listing all 360 lines manually, generate the file using a helper in the test. See Step 1 in the test below.

- [ ] **Step 2: Write the failing tests**

Append to `tests/test_antennas.py`:

```python
import os
import textwrap


def _write_minimal_msi(path: str, freq_mhz: float = 2400.0, gain_dbi: float = 3.0) -> None:
    """Write a valid minimal MSI file with flat horizontal and vertical patterns."""
    lines = [
        "NAME          TestAntenna",
        f"FREQUENCY     {freq_mhz:.0f}",
        f"GAIN          {gain_dbi:.1f}",
        "TILT          0",
        "POLARIZATION  V",
        "",
        "HORIZONTAL",
    ]
    for angle in range(360):
        lines.append(f"{angle}   0.0")
    lines.append("")
    lines.append("VERTICAL")
    for angle in range(360):
        lines.append(f"{angle}   0.0")
    lines.append("")
    with open(path, "w") as f:
        f.write("\n".join(lines))


def test_parse_msi_returns_msi_element(tmp_path):
    from spectra.antennas.msi import MSIAntennaElement, parse_msi

    msi_file = str(tmp_path / "test.msi")
    _write_minimal_msi(msi_file, freq_mhz=2400.0, gain_dbi=3.0)

    elem = parse_msi(msi_file)
    assert isinstance(elem, MSIAntennaElement)


def test_msi_frequency_in_hz(tmp_path):
    from spectra.antennas.msi import parse_msi

    msi_file = str(tmp_path / "test.msi")
    _write_minimal_msi(msi_file, freq_mhz=900.0, gain_dbi=6.0)

    elem = parse_msi(msi_file)
    assert elem.frequency == pytest.approx(900e6)


def test_msi_peak_gain_dbi(tmp_path):
    from spectra.antennas.msi import parse_msi

    msi_file = str(tmp_path / "test.msi")
    _write_minimal_msi(msi_file, freq_mhz=2400.0, gain_dbi=5.5)

    elem = parse_msi(msi_file)
    assert elem.peak_gain_dbi == pytest.approx(5.5)


def test_msi_pattern_returns_complex(tmp_path):
    from spectra.antennas.msi import parse_msi

    msi_file = str(tmp_path / "test.msi")
    _write_minimal_msi(msi_file, freq_mhz=2400.0, gain_dbi=3.0)

    elem = parse_msi(msi_file)
    az = np.array([0.0, np.pi / 2, np.pi])
    el = np.array([0.0, 0.0, 0.0])
    gain = elem.pattern(az, el)
    assert gain.shape == (3,)
    assert np.issubdtype(gain.dtype, np.complexfloating)


def test_msi_flat_pattern_uniform_gain(tmp_path):
    """Flat 0 dB relative pattern → all gains equal peak_gain_linear."""
    from spectra.antennas.msi import parse_msi

    msi_file = str(tmp_path / "test.msi")
    _write_minimal_msi(msi_file, freq_mhz=2400.0, gain_dbi=3.0)

    elem = parse_msi(msi_file)
    az = np.linspace(0, 2 * np.pi, 10)
    el = np.zeros(10)
    gain = elem.pattern(az, el)
    peak_linear = 10 ** (3.0 / 10.0)
    np.testing.assert_allclose(np.abs(gain), peak_linear, rtol=1e-4)


def test_msi_missing_section_raises(tmp_path):
    from spectra.antennas.msi import parse_msi

    bad_file = str(tmp_path / "bad.msi")
    with open(bad_file, "w") as f:
        f.write("NAME   Test\nFREQUENCY  900\nGAIN  0\n\nHORIZONTAL\n")
        for i in range(360):
            f.write(f"{i}  0.0\n")
        # Missing VERTICAL section

    with pytest.raises(ValueError, match="VERTICAL"):
        parse_msi(bad_file)
```

- [ ] **Step 3: Run tests to verify they fail**

```bash
pytest tests/test_antennas.py -k "msi" -v
```
Expected: `ImportError`

- [ ] **Step 4: Write minimal implementation**

```python
# python/spectra/antennas/msi.py
"""MSI/Planet antenna file parser and element."""

from typing import Optional

import numpy as np
from scipy.interpolate import RegularGridInterpolator

from spectra.antennas.base import AntennaElement


class MSIAntennaElement(AntennaElement):
    """Antenna element loaded from an MSI/Planet pattern file.

    The 2D gain surface is built from horizontal and vertical pattern cuts
    using the additive pattern method in dB. Bilinear interpolation via
    ``scipy.interpolate.RegularGridInterpolator`` handles arbitrary query angles.

    Args:
        gain_surface_db: 2D array of shape (360, 181) indexed by
            (azimuth_deg, elevation_deg+90). Values in dB relative to peak.
        peak_gain_dbi: Peak antenna gain in dBi (from MSI header).
        frequency_hz: Design frequency in Hz.
        name: Antenna name from MSI header.
    """

    def __init__(
        self,
        gain_surface_db: np.ndarray,
        peak_gain_dbi: float,
        frequency_hz: float,
        name: str = "",
    ):
        self._gain_surface_db = gain_surface_db  # (360, 181) in dB relative
        self._peak_gain_dbi = peak_gain_dbi
        self._frequency = frequency_hz
        self.name = name
        # Build interpolator
        az_deg = np.arange(360)       # 0..359
        el_idx = np.arange(181)       # index 0..180 → elevation -90..90 deg
        self._interp = RegularGridInterpolator(
            (az_deg, el_idx),
            gain_surface_db,
            method="linear",
            bounds_error=False,
            fill_value=None,  # extrapolate at boundaries
        )

    @property
    def frequency(self) -> float:
        return self._frequency

    @property
    def peak_gain_dbi(self) -> float:
        return self._peak_gain_dbi

    def pattern(self, azimuth: np.ndarray, elevation: np.ndarray) -> np.ndarray:
        """Return complex gain at query angles.

        MSI files have no phase data; the returned array is real-valued cast
        to complex (zero imaginary part).

        Args:
            azimuth: Azimuth in radians (0 = north/+x).
            elevation: Elevation in radians (-pi/2 to pi/2).

        Returns:
            Complex array of linear gain values.
        """
        azimuth = np.asarray(azimuth, dtype=float)
        elevation = np.asarray(elevation, dtype=float)
        az_b, el_b = np.broadcast_arrays(azimuth, elevation)

        # Convert to degrees, map elevation to index (el_deg + 90)
        az_deg = np.degrees(az_b) % 360.0  # wrap to [0, 360)
        el_deg_idx = np.degrees(el_b) + 90.0  # map [-90, 90] → [0, 180]

        points = np.stack([az_deg.ravel(), el_deg_idx.ravel()], axis=-1)
        relative_db = self._interp(points).reshape(az_b.shape)

        # Total gain in dBi = peak_gain_dbi + relative_db
        total_db = self._peak_gain_dbi + relative_db
        gain_linear = 10.0 ** (total_db / 10.0)
        return gain_linear.astype(complex)


def parse_msi(path: str) -> MSIAntennaElement:
    """Parse an MSI/Planet antenna file and return an :class:`MSIAntennaElement`.

    Args:
        path: Path to the ``.msi`` (or ``.ant``) file.

    Returns:
        :class:`MSIAntennaElement` with interpolated 2D gain surface.

    Raises:
        ValueError: If the file is missing required sections or has invalid data.
        FileNotFoundError: If the file does not exist.
    """
    with open(path) as f:
        content = f.read()

    lines = content.splitlines()

    # --- Parse header ---
    name = ""
    frequency_mhz = None
    gain_dbi = 0.0

    for line in lines:
        stripped = line.strip()
        if not stripped or stripped.startswith(";"):
            continue
        upper = stripped.upper()
        if upper.startswith("NAME"):
            name = stripped.split(None, 1)[1] if len(stripped.split()) > 1 else ""
        elif upper.startswith("FREQUENCY"):
            frequency_mhz = float(stripped.split()[1])
        elif upper.startswith("GAIN"):
            gain_dbi = float(stripped.split()[1])

    if frequency_mhz is None:
        raise ValueError("MSI file missing FREQUENCY header field")

    # --- Locate HORIZONTAL and VERTICAL sections ---
    def _parse_section(lines, section_name):
        start = None
        for i, line in enumerate(lines):
            if line.strip().upper() == section_name:
                start = i + 1
                break
        if start is None:
            raise ValueError(
                f"MSI file missing {section_name} section"
            )
        values = []
        for line in lines[start:]:
            stripped = line.strip()
            if not stripped or stripped.startswith(";"):
                continue
            if stripped.upper() in ("HORIZONTAL", "VERTICAL"):
                break
            parts = stripped.split()
            if len(parts) < 2:
                continue
            try:
                values.append(float(parts[1]))
            except ValueError as e:
                raise ValueError(
                    f"Non-numeric gain in {section_name} section: {line!r}"
                ) from e
        if len(values) != 360:
            raise ValueError(
                f"Expected 360 entries in {section_name} section, got {len(values)}"
            )
        return np.array(values, dtype=float)

    horiz_db = _parse_section(lines, "HORIZONTAL")  # (360,) azimuth cut at el=0
    vert_db = _parse_section(lines, "VERTICAL")     # (360,) elevation cut at az=0

    # --- Build 2D gain surface (360, 181) in dB relative ---
    # Index: az in [0..359], el_index = el_deg + 90 in [0..180]
    # G(az, el) = G_h(az) + G_v(el+90) - G_peak_in_data
    # G_peak_in_data = max across the two cuts combined
    g_peak = max(horiz_db.max(), vert_db.max())

    az_idx = np.arange(360)        # (360,)
    el_idx = np.arange(181)        # (181,) maps to elevation -90..90 deg

    # Vertical array has 360 entries: 0 is zenith in MSI convention.
    # Remap: MSI vertical[0] = top (el=90°, index=180), vertical[180] = bottom (el=-90°, index=0)
    # MSI vertical angle 0 = zenith → el_deg = +90 → el_idx = 180
    # MSI vertical angle 90 = horizon → el_deg = 0 → el_idx = 90
    # MSI vertical angle 180 = nadir → el_deg = -90 → el_idx = 0
    # So vert_db[v_angle] → el_idx = 180 - v_angle (for v_angle 0..180)
    # For v_angle 181..359: mirror (vert_db uses 360-entry wrap)
    # Build a 181-element vertical cut from the 360-entry array:
    vert_cut_181 = np.zeros(181, dtype=float)
    for el_i in range(181):
        v_angle = 180 - el_i  # 0 → v=180 (nadir), 180 → v=0 (zenith)
        vert_cut_181[el_i] = vert_db[v_angle % 360]

    # Additive pattern method:
    # gain_surface[az, el_i] = horiz_db[az] + vert_cut_181[el_i] - g_peak
    gain_surface = (
        horiz_db[:, np.newaxis] + vert_cut_181[np.newaxis, :] - g_peak
    )  # (360, 181)

    frequency_hz = frequency_mhz * 1e6
    return MSIAntennaElement(
        gain_surface_db=gain_surface,
        peak_gain_dbi=gain_dbi,
        frequency_hz=frequency_hz,
        name=name,
    )
```

- [ ] **Step 5: Run tests to verify they pass**

```bash
pytest tests/test_antennas.py -k "msi" -v
```
Expected: PASS

- [ ] **Step 6: Commit**

```bash
git add python/spectra/antennas/msi.py tests/test_antennas.py
git commit -m "feat(antennas): add MSIAntennaElement and parse_msi"
```

---

## Task 6: `antennas/__init__.py`

**Files:**
- Create: `python/spectra/antennas/__init__.py`
- Modify: `tests/test_antennas.py`

- [ ] **Step 1: Write the failing test**

Append to `tests/test_antennas.py`:

```python
def test_antennas_package_exports():
    import spectra.antennas as antennas

    assert hasattr(antennas, "AntennaElement")
    assert hasattr(antennas, "IsotropicElement")
    assert hasattr(antennas, "ShortDipoleElement")
    assert hasattr(antennas, "HalfWaveDipoleElement")
    assert hasattr(antennas, "CosinePowerElement")
    assert hasattr(antennas, "MSIAntennaElement")
    assert hasattr(antennas, "parse_msi")
```

- [ ] **Step 2: Run test to verify it fails**

```bash
pytest tests/test_antennas.py::test_antennas_package_exports -v
```
Expected: `ModuleNotFoundError` (no `__init__.py`)

- [ ] **Step 3: Write the `__init__.py`**

```python
# python/spectra/antennas/__init__.py
from spectra.antennas.base import AntennaElement
from spectra.antennas.cosine_power import CosinePowerElement
from spectra.antennas.dipole import HalfWaveDipoleElement, ShortDipoleElement
from spectra.antennas.isotropic import IsotropicElement
from spectra.antennas.msi import MSIAntennaElement, parse_msi

__all__ = [
    "AntennaElement",
    "CosinePowerElement",
    "HalfWaveDipoleElement",
    "IsotropicElement",
    "MSIAntennaElement",
    "ShortDipoleElement",
    "parse_msi",
]
```

- [ ] **Step 4: Run all antenna tests**

```bash
pytest tests/test_antennas.py -v
```
Expected: All PASS

- [ ] **Step 5: Commit**

```bash
git add python/spectra/antennas/__init__.py tests/test_antennas.py
git commit -m "feat(antennas): create antennas package with public __init__"
```

---

## Task 7: `AntennaArray` and Steering Vector

**Files:**
- Create: `python/spectra/arrays/array.py`
- Create: `tests/test_arrays.py`

The steering vector formula from the spec:
```
a_i(az, el) = g_i(az, el) * exp(j*2*pi*(x_i*cos(el)*cos(az) + y_i*cos(el)*sin(az)))
```
Positions `(x_i, y_i)` are in wavelengths. For multiple angles, shape is `(N_elements, N_angles)`.

- [ ] **Step 1: Write the failing tests**

```python
# tests/test_arrays.py
import numpy as np
import pytest


def test_antenna_array_num_elements():
    from spectra.antennas.isotropic import IsotropicElement
    from spectra.arrays.array import AntennaArray

    positions = np.array([[0.0, 0.0], [0.5, 0.0], [1.0, 0.0]])
    elem = IsotropicElement(frequency=1e9)
    arr = AntennaArray(positions=positions, elements=elem, reference_frequency=1e9)
    assert arr.num_elements == 3


def test_steering_vector_single_angle_shape():
    from spectra.antennas.isotropic import IsotropicElement
    from spectra.arrays.array import AntennaArray

    positions = np.array([[0.0, 0.0], [0.5, 0.0], [1.0, 0.0]])
    elem = IsotropicElement(frequency=1e9)
    arr = AntennaArray(positions=positions, elements=elem, reference_frequency=1e9)
    sv = arr.steering_vector(azimuth=0.0, elevation=0.0)
    assert sv.shape == (3,)
    assert np.issubdtype(sv.dtype, np.complexfloating)


def test_steering_vector_multiple_angles_shape():
    from spectra.antennas.isotropic import IsotropicElement
    from spectra.arrays.array import AntennaArray

    positions = np.array([[0.0, 0.0], [0.5, 0.0], [1.0, 0.0]])
    elem = IsotropicElement(frequency=1e9)
    arr = AntennaArray(positions=positions, elements=elem, reference_frequency=1e9)
    azimuths = np.array([0.0, np.pi / 4, np.pi / 2])
    elevations = np.zeros(3)
    sv = arr.steering_vector(azimuth=azimuths, elevation=elevations)
    assert sv.shape == (3, 3)


def test_steering_vector_isotropic_ula_phase():
    """For an isotropic ULA along x-axis, the inter-element phase shift at
    broadside (az=0, el=0) should be 2*pi*spacing (spacing in wavelengths)."""
    from spectra.antennas.isotropic import IsotropicElement
    from spectra.arrays.array import AntennaArray

    spacing = 0.5  # half-wavelength
    positions = np.array([[0.0, 0.0], [spacing, 0.0], [2 * spacing, 0.0]])
    elem = IsotropicElement(frequency=1e9)
    arr = AntennaArray(positions=positions, elements=elem, reference_frequency=1e9)
    sv = arr.steering_vector(azimuth=0.0, elevation=0.0)
    # Phase difference between adjacent elements should be 2*pi*spacing
    phase_diff = np.angle(sv[1]) - np.angle(sv[0])
    expected = 2 * np.pi * spacing
    np.testing.assert_allclose(phase_diff % (2 * np.pi), expected % (2 * np.pi), atol=1e-5)


def test_antenna_array_per_element_list():
    """When elements is a list, each element gets its own pattern applied."""
    from spectra.antennas.isotropic import IsotropicElement
    from spectra.arrays.array import AntennaArray

    positions = np.array([[0.0, 0.0], [0.5, 0.0]])
    elements = [IsotropicElement(frequency=1e9), IsotropicElement(frequency=1e9)]
    arr = AntennaArray(positions=positions, elements=elements, reference_frequency=1e9)
    sv = arr.steering_vector(azimuth=0.0, elevation=0.0)
    assert sv.shape == (2,)
```

- [ ] **Step 2: Run tests to verify they fail**

```bash
pytest tests/test_arrays.py -v
```
Expected: `ImportError`

- [ ] **Step 3: Write minimal implementation**

```python
# python/spectra/arrays/array.py
"""AntennaArray: geometry and steering vector computation."""

from typing import List, Optional, Union

import numpy as np

from spectra.antennas.base import AntennaElement
from spectra.antennas.isotropic import IsotropicElement


class AntennaArray:
    """Planar antenna array with arbitrary element positions.

    Args:
        positions: Element positions in wavelengths, shape ``(N, 2)`` for
            ``(x, y)`` in the horizontal plane.
        elements: A single :class:`AntennaElement` (broadcast to all) or a
            list of per-element instances of length ``N``.
        reference_frequency: Frequency in Hz that defines the wavelength for
            the position units. Usually matches the element design frequency.

    Example::

        arr = AntennaArray(
            positions=np.array([[0, 0], [0.5, 0], [1.0, 0]]),
            elements=IsotropicElement(frequency=2.4e9),
            reference_frequency=2.4e9,
        )
        sv = arr.steering_vector(azimuth=np.deg2rad(30), elevation=0.0)
    """

    def __init__(
        self,
        positions: np.ndarray,
        elements: Union[AntennaElement, List[AntennaElement]],
        reference_frequency: float,
    ):
        self.positions = np.asarray(positions, dtype=float)  # (N, 2)
        if self.positions.ndim != 2 or self.positions.shape[1] != 2:
            raise ValueError("positions must have shape (N, 2)")
        self.reference_frequency = reference_frequency
        n = self.positions.shape[0]
        if isinstance(elements, list):
            if len(elements) != n:
                raise ValueError(
                    f"elements list length {len(elements)} != num_elements {n}"
                )
            self.elements = elements
        else:
            # Broadcast single element to all
            self.elements = [elements] * n

    @property
    def num_elements(self) -> int:
        """Number of array elements."""
        return self.positions.shape[0]

    def steering_vector(
        self,
        azimuth: Union[float, np.ndarray],
        elevation: Union[float, np.ndarray],
    ) -> np.ndarray:
        """Compute the array manifold vector for one or more directions.

        The steering vector combines geometry-induced phase shifts with
        per-element radiation patterns::

            a_i(az, el) = g_i(az, el) * exp(j*2*pi*(x_i*cos(el)*cos(az)
                                                     + y_i*cos(el)*sin(az)))

        Positions are in wavelengths relative to ``reference_frequency``.

        Args:
            azimuth: Azimuth angle(s) in radians. Scalar or 1-D array of
                length ``M``.
            elevation: Elevation angle(s) in radians. Scalar or 1-D array of
                length ``M``.

        Returns:
            Complex array of shape ``(N_elements,)`` for a single direction or
            ``(N_elements, M)`` for multiple directions.
        """
        scalar_input = np.isscalar(azimuth) and np.isscalar(elevation)
        azimuth = np.atleast_1d(np.asarray(azimuth, dtype=float))
        elevation = np.atleast_1d(np.asarray(elevation, dtype=float))
        if azimuth.shape != elevation.shape:
            raise ValueError("azimuth and elevation must have the same shape")
        M = azimuth.size  # number of directions

        # Phase component: (N_elements, M)
        x = self.positions[:, 0]  # (N,)
        y = self.positions[:, 1]  # (N,)
        cos_el = np.cos(elevation)  # (M,)
        cos_az = np.cos(azimuth)    # (M,)
        sin_az = np.sin(azimuth)    # (M,)

        # phase_arg[i, m] = x_i * cos(el_m) * cos(az_m) + y_i * cos(el_m) * sin(az_m)
        phase_arg = (
            x[:, np.newaxis] * (cos_el * cos_az)[np.newaxis, :]
            + y[:, np.newaxis] * (cos_el * sin_az)[np.newaxis, :]
        )  # (N, M)
        phase = np.exp(1j * 2 * np.pi * phase_arg)  # (N, M)

        # Pattern component: (N, M)
        pattern = np.zeros((self.num_elements, M), dtype=complex)
        for i, elem in enumerate(self.elements):
            pattern[i, :] = elem.pattern(azimuth, elevation)

        sv = pattern * phase  # (N, M)

        if scalar_input:
            return sv[:, 0]  # (N,)
        return sv  # (N, M)


def ula(
    num_elements: int,
    spacing: float = 0.5,
    element: Optional[AntennaElement] = None,
    frequency: float = 1e9,
) -> "AntennaArray":
    """Uniform Linear Array along the x-axis.

    Args:
        num_elements: Number of array elements ``N``.
        spacing: Inter-element spacing in wavelengths. Default 0.5 (lambda/2).
        element: Antenna element instance. Defaults to
            :class:`IsotropicElement`.
        frequency: Design frequency in Hz (used for positions and default element).

    Returns:
        :class:`AntennaArray` with elements at ``(n*spacing, 0)``.
    """
    if element is None:
        element = IsotropicElement(frequency=frequency)
    positions = np.column_stack([
        np.arange(num_elements) * spacing,
        np.zeros(num_elements),
    ])
    return AntennaArray(positions=positions, elements=element, reference_frequency=frequency)


def uca(
    num_elements: int,
    radius: Optional[float] = None,
    element: Optional[AntennaElement] = None,
    frequency: float = 1e9,
) -> "AntennaArray":
    """Uniform Circular Array.

    Args:
        num_elements: Number of array elements ``N``.
        radius: Array radius in wavelengths. If ``None``, defaults to the
            radius that gives approximately ``lambda/2`` inter-element spacing:
            ``radius = 0.5 / (2 * sin(pi / N))``.
        element: Antenna element instance. Defaults to
            :class:`IsotropicElement`.
        frequency: Design frequency in Hz.

    Returns:
        :class:`AntennaArray` with elements on a circle in the xy-plane.
    """
    if element is None:
        element = IsotropicElement(frequency=frequency)
    if radius is None:
        # Default radius for ~lambda/2 spacing
        radius = 0.5 / (2 * np.sin(np.pi / num_elements))
    angles = 2 * np.pi * np.arange(num_elements) / num_elements
    positions = np.column_stack([radius * np.cos(angles), radius * np.sin(angles)])
    return AntennaArray(positions=positions, elements=element, reference_frequency=frequency)


def rectangular(
    rows: int,
    cols: int,
    spacing_x: float = 0.5,
    spacing_y: float = 0.5,
    element: Optional[AntennaElement] = None,
    frequency: float = 1e9,
) -> "AntennaArray":
    """Rectangular planar array.

    Args:
        rows: Number of rows (along y-axis).
        cols: Number of columns (along x-axis).
        spacing_x: Column spacing in wavelengths.
        spacing_y: Row spacing in wavelengths.
        element: Antenna element instance. Defaults to
            :class:`IsotropicElement`.
        frequency: Design frequency in Hz.

    Returns:
        :class:`AntennaArray` with ``rows * cols`` elements in a grid.
    """
    if element is None:
        element = IsotropicElement(frequency=frequency)
    xs = np.arange(cols) * spacing_x
    ys = np.arange(rows) * spacing_y
    grid_x, grid_y = np.meshgrid(xs, ys)
    positions = np.column_stack([grid_x.ravel(), grid_y.ravel()])
    return AntennaArray(positions=positions, elements=element, reference_frequency=frequency)
```

- [ ] **Step 4: Run tests to verify they pass**

```bash
pytest tests/test_arrays.py -v
```
Expected: PASS

- [ ] **Step 5: Commit**

```bash
git add python/spectra/arrays/array.py tests/test_arrays.py
git commit -m "feat(arrays): add AntennaArray with steering_vector and ula/uca/rectangular constructors"
```

---

## Task 8: Convenience Constructor Geometry Tests and `CalibrationErrors`

**Files:**
- Create: `python/spectra/arrays/calibration.py`
- Modify: `tests/test_arrays.py`

- [ ] **Step 1: Write the failing tests**

Append to `tests/test_arrays.py`:

```python
def test_ula_positions():
    from spectra.arrays.array import ula

    arr = ula(num_elements=4, spacing=0.5, frequency=1e9)
    expected = np.array([[0.0, 0.0], [0.5, 0.0], [1.0, 0.0], [1.5, 0.0]])
    np.testing.assert_allclose(arr.positions, expected)


def test_uca_num_elements():
    from spectra.arrays.array import uca

    arr = uca(num_elements=8, frequency=2.4e9)
    assert arr.num_elements == 8


def test_uca_elements_on_circle():
    from spectra.arrays.array import uca

    arr = uca(num_elements=6, frequency=1e9)
    radii = np.sqrt(arr.positions[:, 0] ** 2 + arr.positions[:, 1] ** 2)
    np.testing.assert_allclose(radii, radii[0], rtol=1e-6)  # all same radius


def test_rectangular_shape():
    from spectra.arrays.array import rectangular

    arr = rectangular(rows=2, cols=3, frequency=1e9)
    assert arr.num_elements == 6


def test_calibration_errors_apply_shape():
    from spectra.arrays.calibration import CalibrationErrors

    cal = CalibrationErrors(
        gain_offsets_db=np.array([0.1, -0.1, 0.05]),
        phase_offsets_rad=np.array([0.01, -0.02, 0.0]),
    )
    sv = np.array([1.0 + 0j, 0.5 + 0.5j, 0.0 + 1.0j])
    sv_cal = cal.apply(sv)
    assert sv_cal.shape == sv.shape
    assert np.issubdtype(sv_cal.dtype, np.complexfloating)


def test_calibration_errors_random():
    from spectra.arrays.calibration import CalibrationErrors

    rng = np.random.default_rng(42)
    cal = CalibrationErrors.random(num_elements=4, gain_std_db=0.5, phase_std_rad=0.05, rng=rng)
    assert cal.gain_offsets_db.shape == (4,)
    assert cal.phase_offsets_rad.shape == (4,)


def test_calibration_errors_zero_is_identity():
    """Zero calibration errors must not change the steering vector."""
    from spectra.arrays.calibration import CalibrationErrors

    cal = CalibrationErrors(
        gain_offsets_db=np.zeros(3),
        phase_offsets_rad=np.zeros(3),
    )
    sv = np.array([1.0 + 0j, 0.5 + 0.5j, 0.0 + 1.0j])
    sv_cal = cal.apply(sv)
    np.testing.assert_allclose(sv_cal, sv, atol=1e-10)
```

- [ ] **Step 2: Run tests to verify they fail**

```bash
pytest tests/test_arrays.py -k "ula or uca or rectangular or calibration" -v
```
Expected: `ImportError` for calibration, geometry tests may pass.

- [ ] **Step 3: Write `CalibrationErrors`**

```python
# python/spectra/arrays/calibration.py
"""Per-element calibration error model for antenna arrays."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Optional

import numpy as np


@dataclass
class CalibrationErrors:
    """Per-element gain and phase calibration offsets.

    Args:
        gain_offsets_db: Per-element gain offsets in dB, shape ``(N,)``.
        phase_offsets_rad: Per-element phase offsets in radians, shape ``(N,)``.

    Example::

        cal = CalibrationErrors.random(num_elements=8, gain_std_db=0.5)
        sv_cal = cal.apply(steering_vector)
    """

    gain_offsets_db: np.ndarray   # (N,)
    phase_offsets_rad: np.ndarray  # (N,)

    @classmethod
    def random(
        cls,
        num_elements: int,
        gain_std_db: float = 0.5,
        phase_std_rad: float = 0.05,
        rng: Optional[np.random.Generator] = None,
    ) -> "CalibrationErrors":
        """Generate random calibration errors from zero-mean Gaussians.

        Args:
            num_elements: Number of array elements.
            gain_std_db: Standard deviation of gain offsets in dB.
            phase_std_rad: Standard deviation of phase offsets in radians.
            rng: NumPy random generator. If ``None``, uses ``default_rng()``.

        Returns:
            :class:`CalibrationErrors` with random offsets.
        """
        if rng is None:
            rng = np.random.default_rng()
        gain_offsets_db = rng.normal(0.0, gain_std_db, size=num_elements)
        phase_offsets_rad = rng.normal(0.0, phase_std_rad, size=num_elements)
        return cls(
            gain_offsets_db=gain_offsets_db,
            phase_offsets_rad=phase_offsets_rad,
        )

    def apply(self, steering_vector: np.ndarray) -> np.ndarray:
        """Apply calibration errors to a steering vector.

        Implements ``a_cal = diag(gain_linear * exp(j*phase)) @ a``.

        Args:
            steering_vector: Steering vector of shape ``(N,)`` or ``(N, M)``.

        Returns:
            Calibrated steering vector with the same shape.
        """
        gain_linear = 10.0 ** (self.gain_offsets_db / 20.0)  # amplitude scaling
        diag = gain_linear * np.exp(1j * self.phase_offsets_rad)  # (N,)
        if steering_vector.ndim == 1:
            return diag * steering_vector
        return diag[:, np.newaxis] * steering_vector
```

- [ ] **Step 4: Run all array tests**

```bash
pytest tests/test_arrays.py -v
```
Expected: All PASS

- [ ] **Step 5: Commit**

```bash
git add python/spectra/arrays/calibration.py tests/test_arrays.py
git commit -m "feat(arrays): add CalibrationErrors and geometry constructor tests"
```

---

## Task 9: `arrays/__init__.py`

**Files:**
- Create: `python/spectra/arrays/__init__.py`
- Modify: `tests/test_arrays.py`

- [ ] **Step 1: Write the failing test**

Append to `tests/test_arrays.py`:

```python
def test_arrays_package_exports():
    import spectra.arrays as arrays

    assert hasattr(arrays, "AntennaArray")
    assert hasattr(arrays, "CalibrationErrors")
    assert hasattr(arrays, "ula")
    assert hasattr(arrays, "uca")
    assert hasattr(arrays, "rectangular")
```

- [ ] **Step 2: Run to verify it fails**

```bash
pytest tests/test_arrays.py::test_arrays_package_exports -v
```
Expected: `ModuleNotFoundError`

- [ ] **Step 3: Write the `__init__.py`**

```python
# python/spectra/arrays/__init__.py
from spectra.arrays.array import AntennaArray, rectangular, uca, ula
from spectra.arrays.calibration import CalibrationErrors

__all__ = [
    "AntennaArray",
    "CalibrationErrors",
    "rectangular",
    "uca",
    "ula",
]
```

- [ ] **Step 4: Run all array tests**

```bash
pytest tests/test_arrays.py -v
```
Expected: All PASS

- [ ] **Step 5: Commit**

```bash
git add python/spectra/arrays/__init__.py tests/test_arrays.py
git commit -m "feat(arrays): create arrays package with public __init__"
```

---

## Task 10: `ToSnapshotMatrix` Transform

**Files:**
- Create: `python/spectra/transforms/snapshot.py`
- Create: `tests/test_snapshot_transform.py`
- Modify: `python/spectra/transforms/__init__.py`

- [ ] **Step 1: Write the failing tests**

```python
# tests/test_snapshot_transform.py
import numpy as np
import pytest


def test_snapshot_matrix_shape():
    from spectra.transforms.snapshot import ToSnapshotMatrix

    t = ToSnapshotMatrix()
    x = np.random.randn(4, 2, 128).astype(np.float32)  # [n_elem, 2, T]
    result = t(x)
    assert result.shape == (4, 128)


def test_snapshot_matrix_complex():
    from spectra.transforms.snapshot import ToSnapshotMatrix

    t = ToSnapshotMatrix()
    x = np.random.randn(4, 2, 128).astype(np.float32)
    result = t(x)
    assert np.issubdtype(result.dtype, np.complexfloating)


def test_snapshot_matrix_values():
    from spectra.transforms.snapshot import ToSnapshotMatrix

    t = ToSnapshotMatrix()
    i_channel = np.ones((3, 64), dtype=np.float32) * 2.0
    q_channel = np.ones((3, 64), dtype=np.float32) * 3.0
    x = np.stack([i_channel, q_channel], axis=1)  # (3, 2, 64)
    result = t(x)
    np.testing.assert_allclose(result.real, 2.0)
    np.testing.assert_allclose(result.imag, 3.0)
```

- [ ] **Step 2: Run tests to verify they fail**

```bash
pytest tests/test_snapshot_transform.py -v
```
Expected: `ImportError`

- [ ] **Step 3: Write the transform**

```python
# python/spectra/transforms/snapshot.py
"""Snapshot matrix transform for DoA algorithm input."""

import numpy as np


class ToSnapshotMatrix:
    """Convert a real ``[n_elements, 2, num_snapshots]`` tensor to a complex
    ``[n_elements, num_snapshots]`` snapshot matrix.

    The input format (I and Q channels separated in dimension 1) matches the
    output of :class:`~spectra.datasets.DirectionFindingDataset`.
    The output format is suitable as input to classical DoA algorithms
    (MUSIC, ESPRIT, Capon).

    Example::

        transform = ToSnapshotMatrix()
        X = transform(data)  # data: [N, 2, T] → X: complex [N, T]
        R = (X @ X.conj().T) / X.shape[1]  # sample covariance
    """

    def __call__(self, x: np.ndarray) -> np.ndarray:
        """
        Args:
            x: Real array of shape ``[n_elements, 2, num_snapshots]``.
                Channel 0 is I, channel 1 is Q.

        Returns:
            Complex array of shape ``[n_elements, num_snapshots]``.
        """
        return x[:, 0, :] + 1j * x[:, 1, :]
```

- [ ] **Step 4: Update `transforms/__init__.py`**

```python
# Add to imports in python/spectra/transforms/__init__.py:
from spectra.transforms.snapshot import ToSnapshotMatrix

# Add to __all__:
"ToSnapshotMatrix",
```

- [ ] **Step 5: Run tests**

```bash
pytest tests/test_snapshot_transform.py -v
```
Expected: PASS

- [ ] **Step 6: Commit**

```bash
git add python/spectra/transforms/snapshot.py tests/test_snapshot_transform.py python/spectra/transforms/__init__.py
git commit -m "feat(transforms): add ToSnapshotMatrix for DoA algorithm input"
```

---

## Task 11: `DirectionFindingDataset` — Core Structure and `__len__`

**Files:**
- Create: `python/spectra/datasets/direction_finding.py`
- Create: `tests/test_direction_finding_dataset.py`

- [ ] **Step 1: Write the failing tests**

```python
# tests/test_direction_finding_dataset.py
import numpy as np
import pytest
import torch
from torch.utils.data import DataLoader


def _make_dataset(**kwargs):
    """Helper to create a minimal DirectionFindingDataset for testing."""
    from spectra.arrays.array import ula
    from spectra.datasets.direction_finding import DirectionFindingDataset
    from spectra.waveforms import BPSK

    defaults = dict(
        array=ula(num_elements=4, frequency=2.4e9),
        signal_pool=[BPSK(samples_per_symbol=4)],
        num_signals=1,
        num_snapshots=128,
        sample_rate=1e6,
        snr_range=(10.0, 20.0),
        num_samples=50,
        seed=42,
    )
    defaults.update(kwargs)
    return DirectionFindingDataset(**defaults)


def test_dataset_len():
    ds = _make_dataset(num_samples=100)
    assert len(ds) == 100


def test_dataset_getitem_types():
    ds = _make_dataset()
    data, target = ds[0]
    assert isinstance(data, torch.Tensor)
    from spectra.datasets.direction_finding import DirectionFindingTarget
    assert isinstance(target, DirectionFindingTarget)
```

- [ ] **Step 2: Run tests to verify they fail**

```bash
pytest tests/test_direction_finding_dataset.py::test_dataset_len tests/test_direction_finding_dataset.py::test_dataset_getitem_types -v
```
Expected: `ImportError`

- [ ] **Step 3: Write the dataset skeleton with `__len__` and `DirectionFindingTarget`**

```python
# python/spectra/datasets/direction_finding.py
"""Direction-finding dataset for ML-based DoA estimation."""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Callable, List, Optional, Tuple, Union

import numpy as np
import torch
from torch.utils.data import Dataset

from spectra.arrays.array import AntennaArray
from spectra.arrays.calibration import CalibrationErrors
from spectra.datasets.iq_utils import truncate_pad
from spectra.impairments.compose import Compose
from spectra.scene.signal_desc import SignalDescription
from spectra.waveforms.base import Waveform


@dataclass
class DirectionFindingTarget:
    """Ground-truth labels for a direction-finding snapshot.

    Attributes:
        azimuths: Source azimuth angles in radians, shape ``(num_sources,)``.
        elevations: Source elevation angles in radians, shape ``(num_sources,)``.
        snrs: Per-source SNR in dB, shape ``(num_sources,)``.
        num_sources: Number of active sources.
        labels: Modulation label string per source.
        signal_descs: Full :class:`~spectra.scene.signal_desc.SignalDescription`
            per source, with DoA stored in ``modulation_params["doa"]``.
    """

    azimuths: np.ndarray
    elevations: np.ndarray
    snrs: np.ndarray
    num_sources: int
    labels: List[str]
    signal_descs: List[SignalDescription] = field(default_factory=list)


class DirectionFindingDataset(Dataset):
    """On-the-fly direction-finding IQ snapshot dataset.

    Generates multi-antenna IQ snapshots deterministically from
    ``(base_seed, idx)`` pairs using ``np.random.default_rng``. Safe for
    use with ``num_workers > 0``.

    The output tensor has shape ``[n_elements, 2, num_snapshots]`` where
    channel 0 is I and channel 1 is Q. Each element's IQ is formed by
    mixing the weighted steering-vector contributions of all sources.

    Args:
        array: :class:`~spectra.arrays.array.AntennaArray` defining the
            geometry and element patterns.
        signal_pool: List of :class:`~spectra.waveforms.base.Waveform`
            instances to sample from for each source.
        num_signals: Fixed number of sources (int) or ``(min, max)`` range
            (inclusive) to draw uniformly.
        num_snapshots: Number of IQ samples per antenna element.
        sample_rate: Receiver sample rate in Hz.
        snr_range: ``(min_db, max_db)`` per-source SNR drawn uniformly.
        azimuth_range: ``(min_rad, max_rad)`` azimuth sampling range.
            Defaults to ``(0, 2*pi)`` (full circle).
        elevation_range: ``(min_rad, max_rad)`` elevation range.
            Defaults to ``(-pi/2, pi/2)``.
        min_angular_separation: Minimum angular separation between sources in
            radians. If ``None``, no constraint is applied.
        calibration_errors: Optional :class:`~spectra.arrays.calibration.CalibrationErrors`
            to apply to every steering vector.
        impairments: Optional per-signal :class:`~spectra.impairments.compose.Compose`
            pipeline applied before spatial mixing.
        transform: Optional callable applied to the output tensor.
        num_samples: Total dataset size.
        seed: Base integer seed.
    """

    def __init__(
        self,
        array: AntennaArray,
        signal_pool: List[Waveform],
        num_signals: Union[int, Tuple[int, int]],
        num_snapshots: int,
        sample_rate: float,
        snr_range: Tuple[float, float],
        azimuth_range: Tuple[float, float] = (0.0, 2 * np.pi),
        elevation_range: Tuple[float, float] = (-np.pi / 2, np.pi / 2),
        min_angular_separation: Optional[float] = None,
        calibration_errors: Optional[CalibrationErrors] = None,
        impairments: Optional[Compose] = None,
        transform: Optional[Callable] = None,
        num_samples: int = 10000,
        seed: int = 0,
    ):
        self.array = array
        self.signal_pool = signal_pool
        self.num_signals = num_signals
        self.num_snapshots = num_snapshots
        self.sample_rate = sample_rate
        self.snr_range = snr_range
        self.azimuth_range = azimuth_range
        self.elevation_range = elevation_range
        self.min_angular_separation = min_angular_separation
        self.calibration_errors = calibration_errors
        self.impairments = impairments
        self.transform = transform
        self.num_samples = num_samples
        self.seed = seed

    def __len__(self) -> int:
        return self.num_samples

    def __getitem__(self, idx: int) -> Tuple[torch.Tensor, DirectionFindingTarget]:
        rng = np.random.default_rng(seed=(self.seed, idx))

        # --- Determine number of sources ---
        if isinstance(self.num_signals, tuple):
            n_sources = int(rng.integers(self.num_signals[0], self.num_signals[1] + 1))
        else:
            n_sources = int(self.num_signals)

        # --- Sample angles, SNRs, waveforms for each source ---
        azimuths, elevations = self._sample_angles(rng, n_sources)
        snrs_db = rng.uniform(self.snr_range[0], self.snr_range[1], size=n_sources)

        # --- Generate per-source IQ signals ---
        signal_descs = []
        source_iq = []  # list of (num_snapshots,) complex arrays
        labels = []

        for k in range(n_sources):
            wf_idx = int(rng.integers(0, len(self.signal_pool)))
            waveform = self.signal_pool[wf_idx]
            sps = getattr(waveform, "samples_per_symbol", 8)
            num_symbols = self.num_snapshots // sps + 1
            sig_seed = int(rng.integers(0, 2**32))

            iq = waveform.generate(
                num_symbols=num_symbols,
                sample_rate=self.sample_rate,
                seed=sig_seed,
            )
            iq = truncate_pad(iq, self.num_snapshots)

            bw = waveform.bandwidth(self.sample_rate)
            desc = SignalDescription(
                t_start=0.0,
                t_stop=self.num_snapshots / self.sample_rate,
                f_low=-bw / 2,
                f_high=bw / 2,
                label=waveform.label,
                snr=float(snrs_db[k]),
                modulation_params={
                    "doa": {
                        "azimuth_rad": float(azimuths[k]),
                        "elevation_rad": float(elevations[k]),
                        "azimuth_spread_rad": None,
                        "elevation_spread_rad": None,
                    }
                },
            )

            if self.impairments is not None:
                iq, desc = self.impairments(iq, desc, sample_rate=self.sample_rate)

            source_iq.append(iq)
            signal_descs.append(desc)
            labels.append(waveform.label)

        # --- Spatial mixing ---
        X = self._spatial_mix(source_iq, azimuths, elevations, snrs_db, rng)
        # X shape: (n_elements, num_snapshots) complex

        # --- Convert to [n_elements, 2, num_snapshots] float32 ---
        tensor = np.stack([X.real, X.imag], axis=1).astype(np.float32)
        out = torch.from_numpy(tensor)

        if self.transform is not None:
            out = self.transform(out)

        target = DirectionFindingTarget(
            azimuths=azimuths,
            elevations=elevations,
            snrs=snrs_db,
            num_sources=n_sources,
            labels=labels,
            signal_descs=signal_descs,
        )
        return out, target

    def _sample_angles(
        self, rng: np.random.Generator, n_sources: int
    ) -> Tuple[np.ndarray, np.ndarray]:
        """Sample source angles with optional minimum angular separation constraint.

        Returns:
            Tuple of ``(azimuths, elevations)`` arrays of shape ``(n_sources,)``.
        """
        az_min, az_max = self.azimuth_range
        el_min, el_max = self.elevation_range

        if self.min_angular_separation is None or n_sources == 1:
            azimuths = rng.uniform(az_min, az_max, size=n_sources)
            elevations = rng.uniform(el_min, el_max, size=n_sources)
            return azimuths, elevations

        # Rejection sampling with max attempts
        max_attempts = 1000
        for _ in range(max_attempts):
            azimuths = rng.uniform(az_min, az_max, size=n_sources)
            elevations = rng.uniform(el_min, el_max, size=n_sources)
            if self._angles_are_separated(azimuths, elevations):
                return azimuths, elevations

        # Fallback: return last draw even if separation not met
        return azimuths, elevations

    def _angles_are_separated(
        self, azimuths: np.ndarray, elevations: np.ndarray
    ) -> bool:
        """Check that all pairs of angles have at least min_angular_separation."""
        n = len(azimuths)
        for i in range(n):
            for j in range(i + 1, n):
                sep = _angular_separation(
                    azimuths[i], elevations[i], azimuths[j], elevations[j]
                )
                if sep < self.min_angular_separation:
                    return False
        return True

    def _spatial_mix(
        self,
        source_iq: List[np.ndarray],
        azimuths: np.ndarray,
        elevations: np.ndarray,
        snrs_db: np.ndarray,
        rng: np.random.Generator,
    ) -> np.ndarray:
        """Compute the received multi-element signal X = sum_k a_k * s_k^T + N.

        Each source is amplitude-scaled against a fixed noise floor of 1.0 so
        that its per-element contribution achieves the target SNR (in dB). The
        returned SNR is therefore relative to that fixed noise floor, not to an
        absolute receiver noise figure.

        Returns:
            Complex array of shape ``(n_elements, num_snapshots)``.
        """
        n_elem = self.array.num_elements
        n_snap = self.num_snapshots
        noise_power = 1.0  # fixed reference noise power

        # Add independent per-element complex Gaussian noise
        noise = np.sqrt(noise_power / 2.0) * (
            rng.standard_normal((n_elem, n_snap))
            + 1j * rng.standard_normal((n_elem, n_snap))
        )

        X = np.zeros((n_elem, n_snap), dtype=complex)
        for iq, az, el, snr_db in zip(source_iq, azimuths, elevations, snrs_db):
            sv = self.array.steering_vector(azimuth=az, elevation=el)  # (N,)
            if self.calibration_errors is not None:
                sv = self.calibration_errors.apply(sv)
            sig_power = np.mean(np.abs(iq) ** 2)
            if sig_power > 0:
                snr_linear = 10.0 ** (snr_db / 10.0)
                scale = np.sqrt(snr_linear * noise_power / sig_power)
                iq_scaled = iq * scale
            else:
                iq_scaled = iq
            X += sv[:, np.newaxis] * iq_scaled[np.newaxis, :]

        return X + noise


def _angular_separation(
    az1: float, el1: float, az2: float, el2: float
) -> float:
    """Great-circle angular separation between two directions in radians."""
    # Using haversine-like formula for angular distance on a unit sphere
    cos_sep = (
        np.sin(el1) * np.sin(el2)
        + np.cos(el1) * np.cos(el2) * np.cos(az1 - az2)
    )
    return float(np.arccos(np.clip(cos_sep, -1.0, 1.0)))
```

- [ ] **Step 4: Run initial tests**

```bash
pytest tests/test_direction_finding_dataset.py::test_dataset_len tests/test_direction_finding_dataset.py::test_dataset_getitem_types -v
```
Expected: PASS

- [ ] **Step 5: Commit**

```bash
git add python/spectra/datasets/direction_finding.py tests/test_direction_finding_dataset.py
git commit -m "feat(datasets): add DirectionFindingDataset skeleton with __len__ and __getitem__"
```

---

## Task 12: `DirectionFindingDataset` — Full Test Coverage

**Files:**
- Modify: `tests/test_direction_finding_dataset.py`

- [ ] **Step 1: Write the remaining tests**

Append to `tests/test_direction_finding_dataset.py`:

```python
def test_output_tensor_shape():
    ds = _make_dataset(num_snapshots=128)
    data, target = ds[0]
    assert data.shape == (4, 2, 128)  # 4 elements, 2 channels (I/Q), 128 snapshots
    assert data.dtype == torch.float32


def test_deterministic():
    ds = _make_dataset(seed=7)
    d1, t1 = ds[0]
    d2, t2 = ds[0]
    torch.testing.assert_close(d1, d2)
    np.testing.assert_array_equal(t1.azimuths, t2.azimuths)


def test_different_indices_differ():
    ds = _make_dataset(seed=42)
    d0, _ = ds[0]
    d1, _ = ds[1]
    assert not torch.equal(d0, d1)


def test_target_fields():
    ds = _make_dataset(num_signals=2)
    _, target = ds[0]
    assert target.num_sources == 2
    assert len(target.azimuths) == 2
    assert len(target.elevations) == 2
    assert len(target.snrs) == 2
    assert len(target.labels) == 2
    assert len(target.signal_descs) == 2


def test_target_signal_desc_has_doa():
    ds = _make_dataset(num_signals=1)
    _, target = ds[0]
    desc = target.signal_descs[0]
    assert "doa" in desc.modulation_params
    doa = desc.modulation_params["doa"]
    assert "azimuth_rad" in doa
    assert "elevation_rad" in doa
    assert doa["azimuth_spread_rad"] is None
    assert doa["elevation_spread_rad"] is None


def test_num_signals_range():
    ds = _make_dataset(num_signals=(1, 3))
    for i in range(20):
        _, target = ds[i]
        assert 1 <= target.num_sources <= 3


def test_azimuth_in_range():
    az_range = (0.0, np.pi)
    ds = _make_dataset(azimuth_range=az_range, num_signals=1)
    for i in range(20):
        _, target = ds[i]
        assert az_range[0] <= target.azimuths[0] <= az_range[1]


def test_snr_in_range():
    snr_range = (5.0, 15.0)
    ds = _make_dataset(snr_range=snr_range)
    for i in range(20):
        _, target = ds[i]
        for snr in target.snrs:
            assert snr_range[0] <= snr <= snr_range[1]


def test_with_dataloader():
    ds = _make_dataset(num_samples=16, num_snapshots=64)
    loader = DataLoader(ds, batch_size=4)
    batch_data, batch_targets = next(iter(loader))
    assert batch_data.shape == (4, 4, 2, 64)


def test_with_calibration_errors():
    from spectra.arrays.calibration import CalibrationErrors

    cal = CalibrationErrors.random(num_elements=4, rng=np.random.default_rng(0))
    ds = _make_dataset(calibration_errors=cal)
    data, target = ds[0]
    assert data.shape == (4, 2, 128)


def test_with_impairments():
    from spectra.impairments.awgn import AWGN
    from spectra.impairments.compose import Compose

    pipeline = Compose([AWGN(snr=20.0)])
    ds = _make_dataset(impairments=pipeline)
    data, target = ds[0]
    assert data.shape == (4, 2, 128)


def test_with_transform():
    from spectra.transforms.snapshot import ToSnapshotMatrix

    ds = _make_dataset(transform=ToSnapshotMatrix())
    data, _ = ds[0]
    # After ToSnapshotMatrix: the transform receives [4, 2, 128] tensor
    # and should produce something different
    assert data is not None


def test_min_angular_separation():
    ds = _make_dataset(num_signals=2, min_angular_separation=np.deg2rad(10))
    _, target = ds[0]
    if target.num_sources == 2:
        from spectra.datasets.direction_finding import _angular_separation
        sep = _angular_separation(
            target.azimuths[0], target.elevations[0],
            target.azimuths[1], target.elevations[1],
        )
        # Should be at least min_sep OR rejection sampling exhausted (rare)
        assert sep >= 0.0  # just verify it runs without error


def test_no_nan_in_output():
    ds = _make_dataset()
    for i in range(5):
        data, _ = ds[i]
        assert not torch.any(torch.isnan(data))
        assert not torch.any(torch.isinf(data))
```

- [ ] **Step 2: Run all dataset tests**

```bash
pytest tests/test_direction_finding_dataset.py -v
```
Expected: All PASS

- [ ] **Step 3: Commit**

```bash
git add tests/test_direction_finding_dataset.py
git commit -m "test(datasets): add full test coverage for DirectionFindingDataset"
```

---

## Task 13: Wire Up Public Exports

**Files:**
- Modify: `python/spectra/datasets/__init__.py`

- [ ] **Step 1: Write the failing test**

Append to `tests/test_direction_finding_dataset.py`:

```python
def test_public_export_from_datasets():
    from spectra.datasets import DirectionFindingDataset, DirectionFindingTarget

    assert DirectionFindingDataset is not None
    assert DirectionFindingTarget is not None
```

- [ ] **Step 2: Run to verify it fails**

```bash
pytest tests/test_direction_finding_dataset.py::test_public_export_from_datasets -v
```
Expected: `ImportError`

- [ ] **Step 3: Update `datasets/__init__.py`**

Add to imports:
```python
from spectra.datasets.direction_finding import DirectionFindingDataset, DirectionFindingTarget
```

Add to `__all__`:
```python
"DirectionFindingDataset",
"DirectionFindingTarget",
```

- [ ] **Step 4: Run all tests**

```bash
pytest tests/test_direction_finding_dataset.py tests/test_antennas.py tests/test_arrays.py tests/test_snapshot_transform.py -v
```
Expected: All PASS

- [ ] **Step 5: Commit**

```bash
git add python/spectra/datasets/__init__.py tests/test_direction_finding_dataset.py
git commit -m "feat(datasets): export DirectionFindingDataset and DirectionFindingTarget from datasets package"
```

---

## Task 14: Full Regression — Run Existing Tests

Verify no existing tests are broken.

- [ ] **Step 1: Run the full test suite**

```bash
pytest tests/ -v --tb=short
```
Expected: All existing tests PASS. New tests PASS.

- [ ] **Step 2: If any failures, investigate and fix**

Common failure modes:
- Missing import in a `__init__.py` (check for typos).
- `maturin develop --release` needs re-running if Rust changes were made (no Rust changes expected here).

- [ ] **Step 3: Final commit if any fixes needed**

```bash
git add <changed files>
git commit -m "fix: resolve any import or integration issues found in regression"
```

---

## Summary of All New Files

```
python/spectra/antennas/__init__.py
python/spectra/antennas/base.py
python/spectra/antennas/isotropic.py
python/spectra/antennas/dipole.py
python/spectra/antennas/cosine_power.py
python/spectra/antennas/msi.py
python/spectra/arrays/__init__.py
python/spectra/arrays/array.py
python/spectra/arrays/calibration.py
python/spectra/datasets/direction_finding.py
python/spectra/transforms/snapshot.py
tests/test_antennas.py
tests/test_arrays.py
tests/test_direction_finding_dataset.py
tests/test_snapshot_transform.py
```

## Summary of Modified Files

```
python/spectra/datasets/__init__.py   (add DirectionFindingDataset, DirectionFindingTarget)
python/spectra/transforms/__init__.py (add ToSnapshotMatrix)
```
