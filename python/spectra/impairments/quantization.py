from typing import Tuple

import numpy as np

from spectra.impairments.base import Transform
from spectra.scene.signal_desc import SignalDescription


class Quantization(Transform):
    """ADC quantization on I/Q channels."""

    def __init__(self, num_bits: int = 8):
        self._num_bits = num_bits

    def __call__(
        self, iq: np.ndarray, desc: SignalDescription, **kwargs
    ) -> Tuple[np.ndarray, SignalDescription]:
        num_levels = 2**self._num_bits
        # Scale to [-1, 1] based on peak
        peak = max(np.max(np.abs(iq.real)), np.max(np.abs(iq.imag)))
        if peak == 0:
            return iq.copy(), desc
        i_norm = iq.real / peak
        q_norm = iq.imag / peak
        # Quantize
        i_q = np.round(i_norm * (num_levels / 2 - 1)) / (num_levels / 2 - 1)
        q_q = np.round(q_norm * (num_levels / 2 - 1)) / (num_levels / 2 - 1)
        return ((i_q + 1j * q_q) * peak).astype(np.complex64), desc
