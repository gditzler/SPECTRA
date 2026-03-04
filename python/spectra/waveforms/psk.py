from typing import Optional

import numpy as np

from spectra._rust import (
    apply_rrc_filter_with_taps,
    generate_8psk_symbols,
    generate_bpsk_symbols,
    generate_psk_symbols,
    generate_qpsk_symbols,
)
from spectra.utils.rrc_cache import cached_rrc_taps
from spectra.waveforms.base import Waveform


class QPSK(Waveform):
    def __init__(
        self,
        samples_per_symbol: int = 8,
        pulse_shape: str = "rrc",
        rolloff: float = 0.35,
        filter_span: int = 10,
    ):
        self.samples_per_symbol = samples_per_symbol
        self.pulse_shape = pulse_shape
        self.rolloff = rolloff
        self.filter_span = filter_span

    def generate(
        self,
        num_symbols: int,
        sample_rate: float,
        seed: Optional[int] = None,
    ) -> np.ndarray:
        s = seed if seed is not None else np.random.randint(0, 2**32)
        symbols = generate_qpsk_symbols(num_symbols, seed=s)
        taps = cached_rrc_taps(self.rolloff, self.filter_span, self.samples_per_symbol)
        filtered = apply_rrc_filter_with_taps(symbols, taps, self.samples_per_symbol)
        return filtered

    def bandwidth(self, sample_rate: float) -> float:
        symbol_rate = sample_rate / self.samples_per_symbol
        return symbol_rate * (1.0 + self.rolloff)

    @property
    def label(self) -> str:
        return "QPSK"


class BPSK(Waveform):
    def __init__(
        self,
        samples_per_symbol: int = 8,
        pulse_shape: str = "rrc",
        rolloff: float = 0.35,
        filter_span: int = 10,
    ):
        self.samples_per_symbol = samples_per_symbol
        self.pulse_shape = pulse_shape
        self.rolloff = rolloff
        self.filter_span = filter_span

    def generate(
        self,
        num_symbols: int,
        sample_rate: float,
        seed: Optional[int] = None,
    ) -> np.ndarray:
        s = seed if seed is not None else np.random.randint(0, 2**32)
        symbols = generate_bpsk_symbols(num_symbols, seed=s)
        taps = cached_rrc_taps(self.rolloff, self.filter_span, self.samples_per_symbol)
        filtered = apply_rrc_filter_with_taps(symbols, taps, self.samples_per_symbol)
        return filtered

    def bandwidth(self, sample_rate: float) -> float:
        symbol_rate = sample_rate / self.samples_per_symbol
        return symbol_rate * (1.0 + self.rolloff)

    @property
    def label(self) -> str:
        return "BPSK"


class PSK8(Waveform):
    def __init__(
        self,
        samples_per_symbol: int = 8,
        pulse_shape: str = "rrc",
        rolloff: float = 0.35,
        filter_span: int = 10,
    ):
        self.samples_per_symbol = samples_per_symbol
        self.pulse_shape = pulse_shape
        self.rolloff = rolloff
        self.filter_span = filter_span

    def generate(
        self,
        num_symbols: int,
        sample_rate: float,
        seed: Optional[int] = None,
    ) -> np.ndarray:
        s = seed if seed is not None else np.random.randint(0, 2**32)
        symbols = generate_8psk_symbols(num_symbols, seed=s)
        taps = cached_rrc_taps(self.rolloff, self.filter_span, self.samples_per_symbol)
        filtered = apply_rrc_filter_with_taps(symbols, taps, self.samples_per_symbol)
        return filtered

    def bandwidth(self, sample_rate: float) -> float:
        symbol_rate = sample_rate / self.samples_per_symbol
        return symbol_rate * (1.0 + self.rolloff)

    @property
    def label(self) -> str:
        return "8PSK"


class _PSKBase(Waveform):
    """Base class for higher-order M-PSK waveforms using generic generator."""

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
        symbols = generate_psk_symbols(num_symbols, self._order, seed=s)
        taps = cached_rrc_taps(self.rolloff, self.filter_span, self.samples_per_symbol)
        filtered = apply_rrc_filter_with_taps(symbols, taps, self.samples_per_symbol)
        return filtered

    def bandwidth(self, sample_rate: float) -> float:
        symbol_rate = sample_rate / self.samples_per_symbol
        return symbol_rate * (1.0 + self.rolloff)


class PSK16(_PSKBase):
    _order = 16

    @property
    def label(self) -> str:
        return "16PSK"


class PSK32(_PSKBase):
    _order = 32

    @property
    def label(self) -> str:
        return "32PSK"


class PSK64(_PSKBase):
    _order = 64

    @property
    def label(self) -> str:
        return "64PSK"
