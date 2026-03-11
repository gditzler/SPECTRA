from typing import Optional

import numpy as np

from spectra.waveforms.base import Waveform


class ZadoffChu(Waveform):
    """Zadoff-Chu (CAZAC) sequence used in LTE/5G synchronization."""

    def __init__(
        self,
        length: int = 63,
        root: int = 25,
        samples_per_chip: int = 8,
    ):
        if root < 1 or root >= length:
            raise ValueError(f"Root index must be in [1, {length - 1}], got {root}")
        self._length = length
        self._root = root
        self._samples_per_chip = samples_per_chip
        self.samples_per_symbol = length * samples_per_chip

    def _generate_sequence(self) -> np.ndarray:
        n = np.arange(self._length)
        u = self._root
        N = self._length
        if N % 2 == 1:
            phase = -np.pi * u * n * (n + 1) / N
        else:
            phase = -np.pi * u * n * n / N
        return np.exp(1j * phase).astype(np.complex64)

    def generate(
        self,
        num_symbols: int,
        sample_rate: float,
        seed: Optional[int] = None,
    ) -> np.ndarray:
        seq = self._generate_sequence()
        chips_up = np.repeat(seq, self._samples_per_chip)
        return np.tile(chips_up, num_symbols)

    def bandwidth(self, sample_rate: float) -> float:
        return sample_rate / self._samples_per_chip

    @property
    def label(self) -> str:
        return "ZadoffChu"
