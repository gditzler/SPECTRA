"""Shared base class for RRC-filtered waveforms (PSK, QAM, ASK)."""
from abc import abstractmethod
from typing import Optional

import numpy as np

from spectra._rust import apply_rrc_filter_with_taps
from spectra.utils.rrc_cache import cached_rrc_taps
from spectra.waveforms.base import Waveform


class _RRCWaveformBase(Waveform):
    """Base for waveforms that generate symbols then apply RRC pulse shaping.

    Subclasses must define ``label`` and implement ``_generate_symbols()``.
    """

    def __init__(
        self,
        rolloff: float = 0.35,
        filter_span: int = 10,
        samples_per_symbol: int = 8,
    ):
        self.rolloff = rolloff
        self.filter_span = filter_span
        self.samples_per_symbol = samples_per_symbol

    def bandwidth(self, sample_rate: float) -> float:
        symbol_rate = sample_rate / self.samples_per_symbol
        return symbol_rate * (1.0 + self.rolloff)

    @abstractmethod
    def _generate_symbols(self, num_symbols: int, seed: int) -> np.ndarray:
        """Return complex64 symbol array of length num_symbols."""
        ...

    def generate(
        self, num_symbols: int, sample_rate: float, seed: Optional[int] = None
    ) -> np.ndarray:
        s = seed if seed is not None else np.random.randint(0, 2**32)
        symbols = self._generate_symbols(num_symbols, s)
        taps = cached_rrc_taps(self.rolloff, self.filter_span, self.samples_per_symbol)
        filtered = apply_rrc_filter_with_taps(symbols, taps, self.samples_per_symbol)
        return filtered
