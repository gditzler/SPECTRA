from typing import Optional, Tuple

import numpy as np

from spectra.impairments.base import Transform
from spectra.scene.signal_desc import SignalDescription


class PhaseOffset(Transform):
    def __init__(
        self,
        offset: Optional[float] = None,
        max_offset: Optional[float] = None,
    ):
        if offset is None and max_offset is None:
            raise ValueError("Must provide either offset or max_offset")
        self.offset = offset
        self.max_offset = max_offset

    def __call__(
        self, iq: np.ndarray, desc: SignalDescription, **kwargs
    ) -> Tuple[np.ndarray, SignalDescription]:
        if self.max_offset is not None:
            theta = np.random.uniform(-self.max_offset, self.max_offset)
        else:
            theta = self.offset

        rotated = (iq * np.complex64(np.exp(1j * theta))).astype(np.complex64)
        return rotated, desc
