from typing import Optional

import numpy as np

from spectra.waveforms.base import Waveform


class ChirpSS(Waveform):
    """LoRa-like Chirp Spread Spectrum waveform.

    Each symbol is a cyclic-shifted base chirp. The spreading factor
    determines the number of chips per symbol (2^SF).
    """

    def __init__(
        self,
        spreading_factor: int = 7,
        bandwidth_fraction: float = 0.5,
    ):
        self._sf = spreading_factor
        self._chips_per_symbol = 2**spreading_factor
        self._bw_fraction = bandwidth_fraction
        self.samples_per_symbol = self._chips_per_symbol

    def generate(
        self,
        num_symbols: int,
        sample_rate: float,
        seed: Optional[int] = None,
    ) -> np.ndarray:
        rng = np.random.default_rng(seed)
        n_chips = self._chips_per_symbol
        bw = sample_rate * self._bw_fraction

        # Base upchirp: frequency sweeps from -bw/2 to +bw/2 over one symbol
        t = np.arange(n_chips) / sample_rate
        T_sym = n_chips / sample_rate
        chirp_rate = bw / T_sym
        base_phase = 2.0 * np.pi * (-bw / 2 * t + 0.5 * chirp_rate * t**2)
        base_chirp = np.exp(1j * base_phase)

        segments = []
        for _ in range(num_symbols):
            # Random symbol value determines cyclic shift
            symbol_val = rng.integers(0, n_chips)
            shifted = np.roll(base_chirp, symbol_val)
            segments.append(shifted)

        return np.concatenate(segments).astype(np.complex64)

    def bandwidth(self, sample_rate: float) -> float:
        return sample_rate * self._bw_fraction

    @property
    def label(self) -> str:
        return "ChirpSS"
