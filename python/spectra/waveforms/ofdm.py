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


class OFDM72(OFDM):
    def __init__(self, cp_length: int = 16):
        super().__init__(num_subcarriers=72, fft_size=128, cp_length=cp_length)

    @property
    def label(self) -> str:
        return "OFDM-72"


class OFDM128(OFDM):
    def __init__(self, cp_length: int = 16):
        super().__init__(num_subcarriers=128, fft_size=256, cp_length=cp_length)

    @property
    def label(self) -> str:
        return "OFDM-128"


class OFDM180(OFDM):
    def __init__(self, cp_length: int = 32):
        super().__init__(num_subcarriers=180, fft_size=256, cp_length=cp_length)

    @property
    def label(self) -> str:
        return "OFDM-180"


class OFDM256(OFDM):
    def __init__(self, cp_length: int = 32):
        super().__init__(num_subcarriers=256, fft_size=512, cp_length=cp_length)

    @property
    def label(self) -> str:
        return "OFDM-256"


class OFDM300(OFDM):
    def __init__(self, cp_length: int = 32):
        super().__init__(num_subcarriers=300, fft_size=512, cp_length=cp_length)

    @property
    def label(self) -> str:
        return "OFDM-300"


class OFDM512(OFDM):
    def __init__(self, cp_length: int = 64):
        super().__init__(num_subcarriers=512, fft_size=1024, cp_length=cp_length)

    @property
    def label(self) -> str:
        return "OFDM-512"


class OFDM600(OFDM):
    def __init__(self, cp_length: int = 64):
        super().__init__(num_subcarriers=600, fft_size=1024, cp_length=cp_length)

    @property
    def label(self) -> str:
        return "OFDM-600"


class OFDM900(OFDM):
    def __init__(self, cp_length: int = 128):
        super().__init__(num_subcarriers=900, fft_size=1024, cp_length=cp_length)

    @property
    def label(self) -> str:
        return "OFDM-900"


class OFDM1200(OFDM):
    def __init__(self, cp_length: int = 128):
        super().__init__(num_subcarriers=1200, fft_size=2048, cp_length=cp_length)

    @property
    def label(self) -> str:
        return "OFDM-1200"


class OFDM2048(OFDM):
    def __init__(self, cp_length: int = 256):
        super().__init__(num_subcarriers=2048, fft_size=4096, cp_length=cp_length)

    @property
    def label(self) -> str:
        return "OFDM-2048"
