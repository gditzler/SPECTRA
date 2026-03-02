from typing import Tuple

import numpy as np

from spectra.impairments.base import Transform
from spectra.scene.signal_desc import SignalDescription


class PhaseNoise(Transform):
    """LO phase noise via Wiener process (cumulative Gaussian)."""

    def __init__(self, noise_power_db: float = -30.0):
        self._noise_power_db = noise_power_db

    def __call__(
        self, iq: np.ndarray, desc: SignalDescription, **kwargs
    ) -> Tuple[np.ndarray, SignalDescription]:
        noise_power = 10.0 ** (self._noise_power_db / 10.0)
        phase_increments = np.random.randn(len(iq)).astype(np.float32) * np.sqrt(noise_power)
        phase = np.cumsum(phase_increments)
        return (iq * np.exp(1j * phase)).astype(np.complex64), desc
