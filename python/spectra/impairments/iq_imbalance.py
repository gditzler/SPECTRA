from typing import Tuple

import numpy as np

from spectra.impairments.base import Transform
from spectra.scene.signal_desc import SignalDescription


class IQImbalance(Transform):
    """Apply I/Q gain and phase mismatch."""

    def __init__(
        self,
        amplitude_imbalance_db: float = 1.0,
        phase_imbalance_deg: float = 5.0,
    ):
        self._amp_db = amplitude_imbalance_db
        self._phase_deg = phase_imbalance_deg

    def __call__(
        self, iq: np.ndarray, desc: SignalDescription, **kwargs
    ) -> Tuple[np.ndarray, SignalDescription]:
        g = 10.0 ** (self._amp_db / 20.0)
        phi = np.deg2rad(self._phase_deg)
        i_out = iq.real * g * np.cos(phi) - iq.imag * g * np.sin(phi)
        q_out = iq.imag
        return (i_out + 1j * q_out).astype(np.complex64), desc
