from typing import Optional, Tuple

import numpy as np

from spectra.impairments._param_utils import resolve_param, validate_fixed_or_random
from spectra.impairments.base import Transform
from spectra.scene.signal_desc import SignalDescription


class FrequencyOffset(Transform):
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
        sample_rate = kwargs.get("sample_rate")
        if sample_rate is None:
            raise ValueError("FrequencyOffset requires sample_rate kwarg")

        fo = resolve_param(self.offset, self.max_offset)

        t = np.arange(len(iq)) / sample_rate
        shift = np.exp(1j * 2.0 * np.pi * fo * t).astype(np.complex64)
        shifted_iq = iq * shift

        from dataclasses import replace

        new_desc = replace(desc, f_low=desc.f_low + fo, f_high=desc.f_high + fo)
        return shifted_iq, new_desc
