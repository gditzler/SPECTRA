from typing import Optional

import numpy as np

from spectra.waveforms.base import Waveform


class OFDM(Waveform):
    """OFDM waveform with configurable subcarriers and cyclic prefix.

    Uses QPSK subcarrier modulation. Active subcarriers are placed
    symmetrically around DC in the frequency domain.
    """

    def __init__(
        self,
        num_subcarriers: int = 64,
        fft_size: int = 256,
        cp_length: int = 16,
    ):
        self._num_subcarriers = num_subcarriers
        self._fft_size = fft_size
        self._cp_length = cp_length
        self.samples_per_symbol = fft_size + cp_length

    def generate(
        self,
        num_symbols: int,
        sample_rate: float,
        seed: Optional[int] = None,
    ) -> np.ndarray:
        rng = np.random.default_rng(seed)
        segments = []

        for _ in range(num_symbols):
            # Random QPSK symbols for active subcarriers
            angles = rng.choice(
                [np.pi / 4, 3 * np.pi / 4, -3 * np.pi / 4, -np.pi / 4],
                size=self._num_subcarriers,
            )
            data = np.exp(1j * angles)

            # Place subcarriers symmetrically around DC in FFT bins
            fft_bins = np.zeros(self._fft_size, dtype=np.complex128)
            half = self._num_subcarriers // 2
            # Positive frequencies: bins 1..half
            fft_bins[1 : half + 1] = data[:half]
            # Negative frequencies: bins fft_size-half..fft_size-1
            fft_bins[self._fft_size - half :] = data[half:]

            # IFFT to time domain
            time_signal = np.fft.ifft(fft_bins)

            # Add cyclic prefix
            if self._cp_length > 0:
                cp = time_signal[-self._cp_length :]
                ofdm_symbol = np.concatenate([cp, time_signal])
            else:
                ofdm_symbol = time_signal

            segments.append(ofdm_symbol)

        iq = np.concatenate(segments).astype(np.complex64)
        return iq

    def bandwidth(self, sample_rate: float) -> float:
        return self._num_subcarriers * sample_rate / self._fft_size

    @property
    def label(self) -> str:
        return "OFDM"
