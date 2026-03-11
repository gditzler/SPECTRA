from typing import Optional

import numpy as np

from spectra._rust import generate_costas_sequence
from spectra.waveforms.base import Waveform


class CostasCode(Waveform):
    """Costas frequency-hopping radar waveform (Welch construction)."""

    def __init__(self, prime: int = 7, samples_per_hop: int = 64):
        self._prime = prime
        self._samples_per_hop = samples_per_hop
        self._n_hops = prime - 1  # Welch gives order p-1
        self.samples_per_symbol = self._n_hops * samples_per_hop
        # Cache the sequence (deterministic for a given prime)
        self._sequence = generate_costas_sequence(prime)

    def generate(
        self,
        num_symbols: int,
        sample_rate: float,
        seed: Optional[int] = None,
    ) -> np.ndarray:
        delta_f = sample_rate / self._samples_per_hop
        t_local = np.arange(self._samples_per_hop) / sample_rate

        # Center frequencies around DC
        center = (self._prime + 1) / 2.0  # midpoint of 1..p-1
        one_code = np.zeros(self.samples_per_symbol, dtype=np.complex64)

        for k, freq_idx in enumerate(self._sequence):
            f_k = (freq_idx - center) * delta_f
            start = k * self._samples_per_hop
            phase = 2.0 * np.pi * f_k * t_local
            one_code[start : start + self._samples_per_hop] = np.exp(1j * phase).astype(
                np.complex64
            )

        return np.tile(one_code, num_symbols)

    def bandwidth(self, sample_rate: float) -> float:
        delta_f = sample_rate / self._samples_per_hop
        return self._n_hops * delta_f

    @property
    def label(self) -> str:
        return "Costas"
