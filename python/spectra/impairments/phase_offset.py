from typing import Optional, Tuple

import numpy as np

from spectra.impairments._param_utils import resolve_param, validate_fixed_or_random
from spectra.impairments.base import Transform
from spectra.scene.signal_desc import SignalDescription


class PhaseOffset(Transform):
    def __init__(
        self,
        offset: Optional[float] = None,
        max_offset: Optional[float] = None,
    ):
        validate_fixed_or_random(offset, max_offset, "offset")
        self.offset = offset
        self.max_offset = max_offset

    def __call__(
        self, iq: np.ndarray, desc: SignalDescription, **kwargs
    ) -> Tuple[np.ndarray, SignalDescription]:
        theta = resolve_param(self.offset, self.max_offset)

        rotated = (iq * np.complex64(np.exp(1j * theta))).astype(np.complex64)
        return rotated, desc
