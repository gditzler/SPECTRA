"""Cosine-power element pattern approximating patch/microstrip antennas."""

import numpy as np

from spectra.antennas.base import AntennaElement


class CosinePowerElement(AntennaElement):
    """Cosine-power element: cos^n(theta_off_boresight) * peak_gain_linear.

    The boresight is the positive z-axis (elevation = pi/2). The pattern is
    clamped to zero for angles in the back hemisphere (cos < 0).

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
        self._peak_linear = 10.0 ** (peak_gain_dbi / 20.0)

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
        # Clamp cos_theta to non-negative to avoid power-of-negative warnings;
        # np.where evaluates both branches eagerly on the full array.
        cos_theta_clipped = np.maximum(cos_theta, 0.0)
        gain = np.where(cos_theta > 0, cos_theta_clipped**self.exponent, 0.0)
        return (gain * self._peak_linear).astype(complex)
