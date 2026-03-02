from typing import Tuple

import numpy as np

from spectra.impairments.base import Transform
from spectra.scene.signal_desc import SignalDescription


class FrequencyDrift(Transform):
    """Linear frequency drift as quadratic phase."""

    def __init__(self, max_drift: float = 100.0):
        self._max_drift = max_drift

    def __call__(
        self, iq: np.ndarray, desc: SignalDescription, **kwargs
    ) -> Tuple[np.ndarray, SignalDescription]:
        sample_rate = kwargs.get("sample_rate")
        if sample_rate is None:
            raise ValueError("FrequencyDrift requires sample_rate kwarg")
        drift = np.random.uniform(-self._max_drift, self._max_drift)
        n = len(iq)
        t = np.arange(n) / sample_rate
        duration = n / sample_rate
        # Quadratic phase: drift linearly from 0 to drift Hz
        phase = 2.0 * np.pi * 0.5 * drift * t**2 / duration
        return (iq * np.exp(1j * phase)).astype(np.complex64), desc
