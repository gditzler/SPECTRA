from abc import abstractmethod
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


class _PolyphaseCodeBase(Waveform):
    """Shared base for polyphase radar code waveforms."""

    def __init__(self, samples_per_chip: Optional[int] = None, chip_rate: Optional[float] = None):
        if chip_rate is not None and samples_per_chip is not None:
            raise ValueError("chip_rate and samples_per_chip are mutually exclusive")
        self._chip_rate = chip_rate
        self._samples_per_chip = 8 if samples_per_chip is None else samples_per_chip

    def _resolved_spc(self, sample_rate: float) -> int:
        if self._chip_rate is None:
            return self._samples_per_chip
        from spectra.waveforms.physical import resolve_symbol_rate

        spc, _, _ = resolve_symbol_rate(sample_rate, self._chip_rate)
        return spc

    def bandwidth(self, sample_rate: float) -> float:
        if self._chip_rate is not None:
            return self._chip_rate
        return sample_rate / self._samples_per_chip

    def num_symbols_for(self, num_samples: int, sample_rate: float) -> int:
        code_len = len(self._get_chips())
        return max(1, int(num_samples // (code_len * self._resolved_spc(sample_rate))))

    @abstractmethod
    def _get_chips(self) -> np.ndarray:
        """Return complex64 chip array for one code period."""
        ...

    def generate(
        self,
        num_symbols: int,
        sample_rate: float,
        seed: Optional[int] = None,
    ) -> np.ndarray:
        chips = self._get_chips()
        one_code = np.repeat(chips, self._resolved_spc(sample_rate))
        return np.tile(one_code, num_symbols)


class FrankCode(_PolyphaseCodeBase):
    """Frank polyphase radar code. Code length = code_order^2 chips."""

    def __init__(
        self,
        code_order: int = 4,
        samples_per_chip: Optional[int] = None,
        chip_rate: Optional[float] = None,
    ):
        super().__init__(samples_per_chip, chip_rate)
        self._code_order = code_order
        self.samples_per_symbol = code_order * code_order * self._samples_per_chip

    def _get_chips(self):
        return generate_frank_code(self._code_order)

    @property
    def label(self) -> str:
        return "Frank"


class P1Code(_PolyphaseCodeBase):
    """P1 polyphase radar code. Code length = code_order^2 chips."""

    def __init__(
        self,
        code_order: int = 4,
        samples_per_chip: Optional[int] = None,
        chip_rate: Optional[float] = None,
    ):
        super().__init__(samples_per_chip, chip_rate)
        self._code_order = code_order
        self.samples_per_symbol = code_order * code_order * self._samples_per_chip

    def _get_chips(self):
        return generate_p1_code(self._code_order)

    @property
    def label(self) -> str:
        return "P1"


class P2Code(_PolyphaseCodeBase):
    """P2 polyphase radar code. Code length = code_order^2 chips.
    Requires even code_order.
    """

    def __init__(
        self,
        code_order: int = 4,
        samples_per_chip: Optional[int] = None,
        chip_rate: Optional[float] = None,
    ):
        if code_order % 2 != 0:
            raise ValueError("P2 code requires even code_order")
        super().__init__(samples_per_chip, chip_rate)
        self._code_order = code_order
        self.samples_per_symbol = code_order * code_order * self._samples_per_chip

    def _get_chips(self):
        return generate_p2_code(self._code_order)

    @property
    def label(self) -> str:
        return "P2"


class P3Code(_PolyphaseCodeBase):
    """P3 polyphase radar code. Arbitrary code length."""

    def __init__(
        self,
        code_length: int = 16,
        samples_per_chip: Optional[int] = None,
        chip_rate: Optional[float] = None,
    ):
        super().__init__(samples_per_chip, chip_rate)
        self._code_length = code_length
        self.samples_per_symbol = code_length * self._samples_per_chip

    def _get_chips(self):
        return generate_p3_code(self._code_length)

    @property
    def label(self) -> str:
        return "P3"


class P4Code(_PolyphaseCodeBase):
    """P4 polyphase radar code. Arbitrary code length."""

    def __init__(
        self,
        code_length: int = 16,
        samples_per_chip: Optional[int] = None,
        chip_rate: Optional[float] = None,
    ):
        super().__init__(samples_per_chip, chip_rate)
        self._code_length = code_length
        self.samples_per_symbol = code_length * self._samples_per_chip

    def _get_chips(self):
        return generate_p4_code(self._code_length)

    @property
    def label(self) -> str:
        return "P4"
