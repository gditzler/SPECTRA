from typing import Optional, Tuple

import numpy as np

from spectra.impairments.base import Transform
from spectra.scene.signal_desc import SignalDescription


class IQImbalance(Transform):
    def __init__(
        self,
        gain_imbalance: Optional[float] = None,
        phase_imbalance: Optional[float] = None,
        gain_imbalance_range: Optional[Tuple[float, float]] = None,
        phase_imbalance_range: Optional[Tuple[float, float]] = None,
    ):
        has_gain = gain_imbalance is not None or gain_imbalance_range is not None
        has_phase = phase_imbalance is not None or phase_imbalance_range is not None
        if not has_gain:
            raise ValueError(
                "Must provide either gain_imbalance or gain_imbalance_range"
            )
        if not has_phase:
            raise ValueError(
                "Must provide either phase_imbalance or phase_imbalance_range"
            )
        self.gain_imbalance = gain_imbalance
        self.phase_imbalance = phase_imbalance
        self.gain_imbalance_range = gain_imbalance_range
        self.phase_imbalance_range = phase_imbalance_range

    def __call__(
        self, iq: np.ndarray, desc: SignalDescription, **kwargs
    ) -> Tuple[np.ndarray, SignalDescription]:
        if self.gain_imbalance_range is not None:
            g = np.random.uniform(*self.gain_imbalance_range)
        else:
            g = self.gain_imbalance

        if self.phase_imbalance_range is not None:
            p = np.random.uniform(*self.phase_imbalance_range)
        else:
            p = self.phase_imbalance

        i_out = iq.real
        q_out = g * (np.sin(p) * iq.real + np.cos(p) * iq.imag)
        result = (i_out + 1j * q_out).astype(np.complex64)
        return result, desc
