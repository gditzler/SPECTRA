from typing import Tuple

import numpy as np

from spectra.impairments.base import Transform
from spectra.scene.signal_desc import SignalDescription


class PassbandRipple(Transform):
    """Apply passband filter distortion (sinusoidal gain ripple)."""

    def __init__(self, max_ripple_db: float = 1.0, num_ripples: int = 5):
        self._max_ripple_db = max_ripple_db
        self._num_ripples = num_ripples

    def __call__(
        self, iq: np.ndarray, desc: SignalDescription, **kwargs
    ) -> Tuple[np.ndarray, SignalDescription]:
        n = len(iq)
        ripple_db = np.random.uniform(0, self._max_ripple_db)
        freq = np.random.uniform(1, self._num_ripples)
        t = np.linspace(0, 1, n)
        gain_db = ripple_db * np.sin(2.0 * np.pi * freq * t)
        gain_linear = 10.0 ** (gain_db / 20.0)
        return (iq * gain_linear).astype(np.complex64), desc
