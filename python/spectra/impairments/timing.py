from typing import Optional, Tuple

import numpy as np

from spectra.impairments.base import Transform
from spectra.scene.signal_desc import SignalDescription


class FractionalDelay(Transform):
    """Sub-sample timing offset via windowed-sinc FIR interpolation."""

    def __init__(self, delay: float = 0.5, max_delay: Optional[float] = None):
        self._delay = delay
        self._max_delay = max_delay

    def __call__(
        self, iq: np.ndarray, desc: SignalDescription, **kwargs
    ) -> Tuple[np.ndarray, SignalDescription]:
        if self._max_delay is not None:
            delay = np.random.uniform(0, self._max_delay)
        else:
            delay = self._delay

        n_taps = 21
        half = n_taps // 2
        n = np.arange(n_taps) - half
        # Windowed sinc interpolation filter
        sinc_vals = np.sinc(n - delay)
        window = np.hamming(n_taps)
        h = (sinc_vals * window).astype(np.float64)
        h /= h.sum()

        out = np.convolve(iq, h, mode="same").astype(np.complex64)
        return out, desc


class SamplingJitter(Transform):
    """Per-sample random timing variation via linear interpolation."""

    def __init__(self, std_samples: float = 0.01):
        self._std = std_samples

    def __call__(
        self, iq: np.ndarray, desc: SignalDescription, **kwargs
    ) -> Tuple[np.ndarray, SignalDescription]:
        n = len(iq)
        # Jittered sample positions
        jitter = np.random.randn(n) * self._std
        positions = np.arange(n, dtype=float) + jitter
        # Clamp to valid range
        positions = np.clip(positions, 0, n - 1)
        # Linear interpolation
        indices = np.floor(positions).astype(int)
        frac = positions - indices
        # Prevent out-of-bounds
        indices = np.clip(indices, 0, n - 2)
        out = iq[indices] * (1 - frac) + iq[indices + 1] * frac
        return out.astype(np.complex64), desc
