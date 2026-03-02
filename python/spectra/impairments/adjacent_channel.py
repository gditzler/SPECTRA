from typing import Tuple

import numpy as np

from spectra.impairments.base import Transform
from spectra.scene.signal_desc import SignalDescription


class AdjacentChannelInterference(Transform):
    """Add adjacent channel interference signal."""

    def __init__(self, power_db: float = -20.0, offset: float = 0.0):
        self._power_db = power_db
        self._offset = offset

    def __call__(
        self, iq: np.ndarray, desc: SignalDescription, **kwargs
    ) -> Tuple[np.ndarray, SignalDescription]:
        sample_rate = kwargs.get("sample_rate")
        if sample_rate is None:
            raise ValueError("AdjacentChannelInterference requires sample_rate kwarg")
        n = len(iq)
        # Generate random interference
        interference = (
            np.random.randn(n) + 1j * np.random.randn(n)
        ).astype(np.complex64) / np.sqrt(2.0)
        # Frequency shift to offset
        t = np.arange(n) / sample_rate
        shift = np.exp(1j * 2.0 * np.pi * self._offset * t).astype(np.complex64)
        interference *= shift
        # Scale to desired power relative to signal
        signal_power = np.mean(np.abs(iq) ** 2)
        target_power = signal_power * 10.0 ** (self._power_db / 10.0)
        int_power = np.mean(np.abs(interference) ** 2)
        if int_power > 0:
            interference *= np.sqrt(target_power / int_power)
        return (iq + interference).astype(np.complex64), desc
