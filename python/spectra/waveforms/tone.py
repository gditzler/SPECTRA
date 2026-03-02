from typing import Optional

import numpy as np

from spectra._rust import generate_tone
from spectra.waveforms.base import Waveform


class Tone(Waveform):
    """Single-frequency tone (complex sinusoid)."""

    def __init__(self, frequency: float = 0.0):
        self._frequency = frequency
        self.samples_per_symbol = 1

    def generate(
        self,
        num_symbols: int,
        sample_rate: float,
        seed: Optional[int] = None,
    ) -> np.ndarray:
        duration = num_symbols / sample_rate
        return generate_tone(self._frequency, duration, sample_rate)

    def bandwidth(self, sample_rate: float) -> float:
        return 0.0

    @property
    def label(self) -> str:
        return "Tone"
