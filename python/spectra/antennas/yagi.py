"""Yagi-Uda directional antenna element pattern."""

import numpy as np
from spectra.antennas.base import AntennaElement


class YagiElement(AntennaElement):
    """Yagi-Uda directional element with boresight along the +x axis.

    Simplified far-field pattern::

        g(az, el) = max(0, cos(el))^2 * max(0, cos(az))^n

    where ``n = n_elements`` controls the azimuth beamwidth (more elements →
    narrower beam, higher directivity). Gain is normalised to 1.0 at boresight
    (az=0, el=0). Back hemisphere (cos(az) < 0) is set to zero.

    Args:
        n_elements: Number of Yagi elements (3–10 typical). Controls
            azimuth directivity exponent. Default 3.
        frequency: Design frequency in Hz.

    Example::

        arr = ula(num_elements=8, element=YagiElement(n_elements=5), frequency=1e9)
        sv = arr.steering_vector(azimuth=np.deg2rad(30), elevation=0.0)
    """

    def __init__(self, n_elements: int = 3, frequency: float = 1e9):
        if n_elements < 1:
            raise ValueError(f"n_elements must be >= 1, got {n_elements}")
        self._n_elements = n_elements
        self._frequency = frequency

    @property
    def frequency(self) -> float:
        return self._frequency

    def pattern(self, azimuth: np.ndarray, elevation: np.ndarray) -> np.ndarray:
        """Compute element gain for given azimuth(s) and elevation(s).

        Args:
            azimuth: Azimuth angle(s) in radians.
            elevation: Elevation angle(s) in radians.

        Returns:
            Complex gain array matching the broadcast shape of inputs.
        """
        az = np.asarray(azimuth, dtype=float)
        el = np.asarray(elevation, dtype=float)
        az_b, el_b = np.broadcast_arrays(az, el)

        # Elevation roll-off: cos²(el) — zero at ±90° elevation
        el_gain = np.maximum(0.0, np.cos(el_b)) ** 2

        # Azimuth directivity: cos^n(az) — zero at ±90° and back hemisphere
        cos_az = np.cos(az_b)
        az_gain = np.where(cos_az > 0.0, cos_az ** self._n_elements, 0.0)

        gain = el_gain * az_gain
        return gain.astype(complex)
