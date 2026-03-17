"""Short and half-wave dipole antenna element patterns."""

from typing import Literal
import numpy as np
from spectra.antennas.base import AntennaElement


def _theta_from_axis(azimuth: np.ndarray, elevation: np.ndarray, axis: str) -> np.ndarray:
    """Compute angle from dipole axis in radians.

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
        axis: Dipole axis orientation — 'x', 'y', or 'z'.
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

    Pattern: cos(pi/2 * cos(theta)) / sin(theta), normalized to 1 at
    broadside. Near-zero sin(theta) values are clamped to avoid division by zero.

    Args:
        axis: Dipole axis orientation — 'x', 'y', or 'z'.
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
