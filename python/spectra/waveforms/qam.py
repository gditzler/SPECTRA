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


class QAM1024(_QAMBase):
    _order = 1024

    @property
    def label(self) -> str:
        return "1024QAM"


class _CrossQAMBase(Waveform):
    """Base class for non-square cross QAM constellations (32, 128, 512)."""

    _order: int = 32

    def __init__(
        self,
        samples_per_symbol: int = 8,
        rolloff: float = 0.35,
        filter_span: int = 10,
    ):
        self.samples_per_symbol = samples_per_symbol
        self.rolloff = rolloff
        self.filter_span = filter_span

    @staticmethod
    def _cross_constellation(order: int) -> np.ndarray:
        """Build a cross QAM constellation for non-square orders.

        Uses the next larger square grid and removes corner points
        to reach the desired order.
        """
        side = int(np.ceil(np.sqrt(order)))
        if side % 2 != 0:
            side += 1
        # Build full square grid
        points = []
        for i in range(side):
            for j in range(side):
                re = 2.0 * i - (side - 1)
                im = 2.0 * j - (side - 1)
                points.append(complex(re, im))
        points = np.array(points)
        # Sort by magnitude to keep inner points
        mags = np.abs(points)
        indices = np.argsort(mags)
        constellation = points[indices[:order]]
        # Normalize to unit average power
        avg_power = np.mean(np.abs(constellation) ** 2)
        if avg_power > 0:
            constellation /= np.sqrt(avg_power)
        return constellation.astype(np.complex64)

    def generate(
        self,
        num_symbols: int,
        sample_rate: float,
        seed: Optional[int] = None,
    ) -> np.ndarray:
        rng = np.random.default_rng(seed)
        constellation = self._cross_constellation(self._order)
        indices = rng.integers(0, len(constellation), size=num_symbols)
        symbols = constellation[indices]
        filtered = apply_rrc_filter(
            symbols, self.rolloff, self.filter_span, self.samples_per_symbol
        )
        return filtered

    def bandwidth(self, sample_rate: float) -> float:
        symbol_rate = sample_rate / self.samples_per_symbol
        return symbol_rate * (1.0 + self.rolloff)


class QAM32(_CrossQAMBase):
    _order = 32

    @property
    def label(self) -> str:
        return "32QAM"


class QAM128(_CrossQAMBase):
    _order = 128

    @property
    def label(self) -> str:
        return "128QAM"


class QAM512(_CrossQAMBase):
    _order = 512

    @property
    def label(self) -> str:
        return "512QAM"
