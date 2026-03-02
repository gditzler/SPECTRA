from typing import Tuple

import numpy as np

from spectra.impairments.base import Transform
from spectra.scene.signal_desc import SignalDescription


class SpectralInversion(Transform):
    """Spectral inversion (complex conjugate) — flips frequency bounds."""

    def __call__(
        self, iq: np.ndarray, desc: SignalDescription, **kwargs
    ) -> Tuple[np.ndarray, SignalDescription]:
        from dataclasses import replace

        new_desc = replace(desc, f_low=-desc.f_high, f_high=-desc.f_low)
        return iq.conj().astype(np.complex64), new_desc
