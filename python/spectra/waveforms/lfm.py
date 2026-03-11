from typing import Optional

import numpy as np

from spectra._rust import generate_chirp
from spectra.waveforms.base import Waveform


class LFM(Waveform):
    """Linear Frequency Modulation (chirp) radar waveform.

    Each 'symbol' is one complete chirp pulse sweeping across the
    configured bandwidth.
    """

    def __init__(
        self,
        bandwidth_fraction: float = 0.5,
        samples_per_pulse: int = 256,
    ):
        self._bandwidth_fraction = bandwidth_fraction
        self._samples_per_pulse = samples_per_pulse
        self.samples_per_symbol = samples_per_pulse

    def generate(
        self,
        num_symbols: int,
        sample_rate: float,
        seed: Optional[int] = None,
    ) -> np.ndarray:
        bw = sample_rate * self._bandwidth_fraction
        f0 = -bw / 2.0
        f1 = bw / 2.0
        duration = self._samples_per_pulse / sample_rate

        pulses = [generate_chirp(duration, sample_rate, f0, f1) for _ in range(num_symbols)]
        return np.concatenate(pulses) if pulses else np.array([], dtype=np.complex64)

    def bandwidth(self, sample_rate: float) -> float:
        return sample_rate * self._bandwidth_fraction

    @property
    def label(self) -> str:
        return "LFM"
