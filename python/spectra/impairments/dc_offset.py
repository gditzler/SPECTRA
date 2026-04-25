from typing import Optional, Tuple

import numpy as np

from spectra.impairments._param_utils import validate_fixed_or_random
from spectra.impairments.base import Transform
from spectra.scene.signal_desc import SignalDescription


class DCOffset(Transform):
    def __init__(
        self,
        offset: Optional[complex] = None,
        max_offset: Optional[float] = None,
    ):
        # validate_fixed_or_random takes Optional[float]; ``offset`` is ``complex``
        # here but only its ``None``-ness matters for that check.
        validate_fixed_or_random(
            None if offset is None else 0.0, max_offset, "offset"
        )
        self.offset = offset
        self.max_offset = max_offset

    def __call__(
        self, iq: np.ndarray, desc: SignalDescription, **kwargs
    ) -> Tuple[np.ndarray, SignalDescription]:
        if self.max_offset is not None:
            dc = np.random.uniform(-self.max_offset, self.max_offset) + 1j * np.random.uniform(
                -self.max_offset, self.max_offset
            )
        else:
            dc = self.offset

        return (iq + np.complex64(dc)).astype(np.complex64), desc
