from typing import Optional

import numpy as np

from spectra._rust import (
    generate_frank_code,
    generate_p1_code,
    generate_p2_code,
    generate_p3_code,
    generate_p4_code,
)
from spectra.waveforms.base import Waveform


class FrankCode(Waveform):
    """Frank polyphase radar code. Code length = code_order^2 chips."""

    def __init__(self, code_order: int = 4, samples_per_chip: int = 8):
        self._code_order = code_order
        self._samples_per_chip = samples_per_chip
        self.samples_per_symbol = code_order * code_order * samples_per_chip

    def generate(
        self,
        num_symbols: int,
        sample_rate: float,
        seed: Optional[int] = None,
    ) -> np.ndarray:
        chips = generate_frank_code(self._code_order)
        one_code = np.repeat(chips, self._samples_per_chip)
        return np.tile(one_code, num_symbols)

    def bandwidth(self, sample_rate: float) -> float:
        return sample_rate / self._samples_per_chip

    @property
    def label(self) -> str:
        return "Frank"


class P1Code(Waveform):
    """P1 polyphase radar code. Code length = code_order^2 chips."""

    def __init__(self, code_order: int = 4, samples_per_chip: int = 8):
        self._code_order = code_order
        self._samples_per_chip = samples_per_chip
        self.samples_per_symbol = code_order * code_order * samples_per_chip

    def generate(
        self,
        num_symbols: int,
        sample_rate: float,
        seed: Optional[int] = None,
    ) -> np.ndarray:
        chips = generate_p1_code(self._code_order)
        one_code = np.repeat(chips, self._samples_per_chip)
        return np.tile(one_code, num_symbols)

    def bandwidth(self, sample_rate: float) -> float:
        return sample_rate / self._samples_per_chip

    @property
    def label(self) -> str:
        return "P1"


class P2Code(Waveform):
    """P2 polyphase radar code. Code length = code_order^2 chips.
    Requires even code_order.
    """

    def __init__(self, code_order: int = 4, samples_per_chip: int = 8):
        if code_order % 2 != 0:
            raise ValueError("P2 code requires even code_order")
        self._code_order = code_order
        self._samples_per_chip = samples_per_chip
        self.samples_per_symbol = code_order * code_order * samples_per_chip

    def generate(
        self,
        num_symbols: int,
        sample_rate: float,
        seed: Optional[int] = None,
    ) -> np.ndarray:
        chips = generate_p2_code(self._code_order)
        one_code = np.repeat(chips, self._samples_per_chip)
        return np.tile(one_code, num_symbols)

    def bandwidth(self, sample_rate: float) -> float:
        return sample_rate / self._samples_per_chip

    @property
    def label(self) -> str:
        return "P2"


class P3Code(Waveform):
    """P3 polyphase radar code. Arbitrary code length."""

    def __init__(self, code_length: int = 16, samples_per_chip: int = 8):
        self._code_length = code_length
        self._samples_per_chip = samples_per_chip
        self.samples_per_symbol = code_length * samples_per_chip

    def generate(
        self,
        num_symbols: int,
        sample_rate: float,
        seed: Optional[int] = None,
    ) -> np.ndarray:
        chips = generate_p3_code(self._code_length)
        one_code = np.repeat(chips, self._samples_per_chip)
        return np.tile(one_code, num_symbols)

    def bandwidth(self, sample_rate: float) -> float:
        return sample_rate / self._samples_per_chip

    @property
    def label(self) -> str:
        return "P3"


class P4Code(Waveform):
    """P4 polyphase radar code. Arbitrary code length."""

    def __init__(self, code_length: int = 16, samples_per_chip: int = 8):
        self._code_length = code_length
        self._samples_per_chip = samples_per_chip
        self.samples_per_symbol = code_length * samples_per_chip

    def generate(
        self,
        num_symbols: int,
        sample_rate: float,
        seed: Optional[int] = None,
    ) -> np.ndarray:
        chips = generate_p4_code(self._code_length)
        one_code = np.repeat(chips, self._samples_per_chip)
        return np.tile(one_code, num_symbols)

    def bandwidth(self, sample_rate: float) -> float:
        return sample_rate / self._samples_per_chip

    @property
    def label(self) -> str:
        return "P4"
