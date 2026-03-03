from typing import Optional, Tuple

import numpy as np

from spectra.impairments.base import Transform
from spectra.scene.signal_desc import SignalDescription


class SampleRateOffset(Transform):
    def __init__(
        self,
        ppm: Optional[float] = None,
        max_ppm: Optional[float] = None,
    ):
        if ppm is None and max_ppm is None:
            raise ValueError("Must provide either ppm or max_ppm")
        self.ppm = ppm
        self.max_ppm = max_ppm

    def __call__(
        self, iq: np.ndarray, desc: SignalDescription, **kwargs
    ) -> Tuple[np.ndarray, SignalDescription]:
        if self.max_ppm is not None:
            ppm_val = np.random.uniform(-self.max_ppm, self.max_ppm)
        else:
            ppm_val = self.ppm

        n = len(iq)
        ratio = 1.0 + ppm_val * 1e-6
        # New sample indices mapped back to original time grid
        new_indices = np.arange(n) * ratio
        # Clip to valid range for interpolation
        new_indices = np.clip(new_indices, 0, n - 1)
        # Linear interpolation of real and imaginary parts
        old_indices = np.arange(n, dtype=np.float64)
        re = np.interp(new_indices, old_indices, iq.real)
        im = np.interp(new_indices, old_indices, iq.imag)
        return (re + 1j * im).astype(np.complex64), desc
