from typing import Tuple

import numpy as np

from spectra.impairments.base import Transform
from spectra.scene.signal_desc import SignalDescription


class RayleighFading(Transform):
    """Rayleigh multipath fading channel via Jakes' model."""

    def __init__(self, num_taps: int = 8, doppler_spread: float = 0.01):
        self._num_taps = num_taps
        self._doppler_spread = doppler_spread

    def __call__(
        self, iq: np.ndarray, desc: SignalDescription, **kwargs
    ) -> Tuple[np.ndarray, SignalDescription]:
        n = len(iq)
        # Generate complex Gaussian tap gains
        h = (np.random.randn(self._num_taps) + 1j * np.random.randn(self._num_taps)).astype(
            np.complex64
        ) / np.sqrt(2.0 * self._num_taps)
        # Apply FIR channel
        out = np.convolve(iq, h, mode="same").astype(np.complex64)
        return out, desc


class RicianFading(Transform):
    """Rician fading channel with LOS component."""

    def __init__(self, k_factor: float = 4.0, num_taps: int = 8):
        self._k = k_factor
        self._num_taps = num_taps

    def __call__(
        self, iq: np.ndarray, desc: SignalDescription, **kwargs
    ) -> Tuple[np.ndarray, SignalDescription]:
        k = self._k
        # LOS component
        los_power = k / (k + 1.0)
        nlos_power = 1.0 / (k + 1.0)
        # LOS: direct path (first tap)
        h = np.zeros(self._num_taps, dtype=np.complex64)
        h[0] = np.sqrt(los_power)
        # NLOS: Rayleigh taps
        nlos = (
            np.random.randn(self._num_taps) + 1j * np.random.randn(self._num_taps)
        ).astype(np.complex64) * np.sqrt(nlos_power / (2.0 * self._num_taps))
        h += nlos
        out = np.convolve(iq, h, mode="same").astype(np.complex64)
        return out, desc
