from typing import Optional

import numpy as np

from spectra._rust import apply_rrc_filter, generate_qam_symbols
from spectra.waveforms.base import Waveform


class _QAMBase(Waveform):
    """Base class for square QAM waveforms."""

    _order: int = 16

    def __init__(
        self,
        samples_per_symbol: int = 8,
        rolloff: float = 0.35,
        filter_span: int = 10,
    ):
        self.samples_per_symbol = samples_per_symbol
        self.rolloff = rolloff
        self.filter_span = filter_span

    def generate(
        self,
        num_symbols: int,
        sample_rate: float,
        seed: Optional[int] = None,
    ) -> np.ndarray:
        s = seed if seed is not None else np.random.randint(0, 2**32)
        symbols = generate_qam_symbols(num_symbols, self._order, seed=s)
        filtered = apply_rrc_filter(
            symbols, self.rolloff, self.filter_span, self.samples_per_symbol
        )
        return filtered

    def bandwidth(self, sample_rate: float) -> float:
        symbol_rate = sample_rate / self.samples_per_symbol
        return symbol_rate * (1.0 + self.rolloff)


class QAM16(_QAMBase):
    _order = 16

    @property
    def label(self) -> str:
        return "16QAM"


class QAM64(_QAMBase):
    _order = 64

    @property
    def label(self) -> str:
        return "64QAM"


class QAM256(_QAMBase):
    _order = 256

    @property
    def label(self) -> str:
        return "256QAM"
