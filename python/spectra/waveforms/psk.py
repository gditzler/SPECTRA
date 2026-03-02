from typing import Optional

import numpy as np

from spectra._rust import (
    apply_rrc_filter,
    generate_8psk_symbols,
    generate_bpsk_symbols,
    generate_qpsk_symbols,
)
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
        filtered = apply_rrc_filter(
            symbols, self.rolloff, self.filter_span, self.samples_per_symbol
        )
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
        filtered = apply_rrc_filter(
            symbols, self.rolloff, self.filter_span, self.samples_per_symbol
        )
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
        filtered = apply_rrc_filter(
            symbols, self.rolloff, self.filter_span, self.samples_per_symbol
        )
        return filtered

    def bandwidth(self, sample_rate: float) -> float:
        symbol_rate = sample_rate / self.samples_per_symbol
        return symbol_rate * (1.0 + self.rolloff)

    @property
    def label(self) -> str:
        return "8PSK"
