from typing import Tuple

import numpy as np

from spectra.impairments.base import Transform
from spectra.scene.signal_desc import SignalDescription


class IntermodulationProducts(Transform):
    """AM/AM + AM/PM nonlinearity (3rd-order intermodulation)."""

    def __init__(self, iip3_db: float = 30.0):
        self._iip3_db = iip3_db

    def __call__(
        self, iq: np.ndarray, desc: SignalDescription, **kwargs
    ) -> Tuple[np.ndarray, SignalDescription]:
        iip3_linear = 10.0 ** (self._iip3_db / 20.0)
        # Third-order nonlinearity: y = x + (2/3) * |x|^2 * x / iip3^2
        out = iq + (2.0 / 3.0) * np.abs(iq) ** 2 * iq / (iip3_linear**2)
        return out.astype(np.complex64), desc
