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
