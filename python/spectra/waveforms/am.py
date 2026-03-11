from typing import Optional

import numpy as np

from spectra._rust import convolve_complex, lowpass_taps
from spectra.waveforms.base import Waveform


class _AMBase(Waveform):
    """Base class for AM waveforms."""

    _mode: str = "dsb-sc"

    def __init__(
        self,
        audio_bw_fraction: float = 0.1,
        num_taps: int = 101,
    ):
        self._audio_bw_fraction = audio_bw_fraction
        self._num_taps = num_taps
        self.samples_per_symbol = 1

    def _generate_audio(
        self, num_samples: int, sample_rate: float, rng: np.random.Generator
    ) -> np.ndarray:
        """Generate lowpass-filtered noise as audio baseband signal."""
        cutoff = 2.0 * self._audio_bw_fraction
        taps = np.array(lowpass_taps(self._num_taps, cutoff), dtype=np.float32)
        noise = rng.standard_normal(num_samples).astype(np.float32)
        noise_complex = (noise + 0j).astype(np.complex64)
        filtered = np.array(convolve_complex(noise_complex, taps))
        # Trim to original length (center crop)
        start = (len(filtered) - num_samples) // 2
        audio = filtered[start : start + num_samples].real.astype(np.float32)
        # Normalize to [-1, 1]
        peak = np.max(np.abs(audio))
        if peak > 0:
            audio = audio / peak
        return audio

    def generate(
        self,
        num_symbols: int,
        sample_rate: float,
        seed: Optional[int] = None,
    ) -> np.ndarray:
        rng = np.random.default_rng(seed)
        num_samples = num_symbols
        audio = self._generate_audio(num_samples, sample_rate, rng)

        if self._mode == "dsb-sc":
            iq = audio.astype(np.complex64)
        elif self._mode == "dsb":
            iq = (1.0 + audio).astype(np.complex64)
        elif self._mode in ("lsb", "usb"):
            # FFT-based Hilbert transform for single sideband
            analytic = self._hilbert(audio)
            if self._mode == "lsb":
                iq = analytic.conj().astype(np.complex64)
            else:
                iq = analytic.astype(np.complex64)
        else:
            raise ValueError(f"Unknown AM mode: {self._mode}")

        # Normalize to unit power
        power = np.mean(np.abs(iq) ** 2)
        if power > 0:
            iq = iq / np.sqrt(power)
        return iq.astype(np.complex64)

    @staticmethod
    def _hilbert(x: np.ndarray) -> np.ndarray:
        """FFT-based analytic signal (no scipy dependency)."""
        n = len(x)
        X = np.fft.fft(x)
        h = np.zeros(n)
        if n > 0:
            h[0] = 1
            if n % 2 == 0:
                h[n // 2] = 1
                h[1 : n // 2] = 2
            else:
                h[1 : (n + 1) // 2] = 2
        return np.fft.ifft(X * h)

    def bandwidth(self, sample_rate: float) -> float:
        audio_bw = sample_rate * self._audio_bw_fraction
        if self._mode in ("dsb-sc", "dsb"):
            return 2.0 * audio_bw
        else:
            return audio_bw


class AMDSB_SC(_AMBase):
    _mode = "dsb-sc"

    @property
    def label(self) -> str:
        return "AM-DSB-SC"


class AMDSB(_AMBase):
    _mode = "dsb"

    @property
    def label(self) -> str:
        return "AM-DSB"


class AMLSB(_AMBase):
    _mode = "lsb"

    @property
    def label(self) -> str:
        return "AM-LSB"


class AMUSB(_AMBase):
    _mode = "usb"

    @property
    def label(self) -> str:
        return "AM-USB"
