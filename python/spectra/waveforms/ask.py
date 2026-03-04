from typing import Optional

import numpy as np

from spectra._rust import apply_rrc_filter_with_taps, generate_ask_symbols
from spectra.utils.rrc_cache import cached_rrc_taps
from spectra.waveforms.base import Waveform


class _ASKBase(Waveform):
    """Base class for M-ary ASK waveforms."""

    _order: int = 2

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
        symbols = generate_ask_symbols(num_symbols, self._order, seed=s)
        taps = cached_rrc_taps(self.rolloff, self.filter_span, self.samples_per_symbol)
        filtered = apply_rrc_filter_with_taps(symbols, taps, self.samples_per_symbol)
        return filtered

    def bandwidth(self, sample_rate: float) -> float:
        symbol_rate = sample_rate / self.samples_per_symbol
        return symbol_rate * (1.0 + self.rolloff)


class OOK(_ASKBase):
    _order = 2

    @property
    def label(self) -> str:
        return "OOK"


class ASK4(_ASKBase):
    _order = 4

    @property
    def label(self) -> str:
        return "4ASK"


class ASK8(_ASKBase):
    _order = 8

    @property
    def label(self) -> str:
        return "8ASK"


class ASK16(_ASKBase):
    _order = 16

    @property
    def label(self) -> str:
        return "16ASK"


class ASK32(_ASKBase):
    _order = 32

    @property
    def label(self) -> str:
        return "32ASK"


class ASK64(_ASKBase):
    _order = 64

    @property
    def label(self) -> str:
        return "64ASK"
