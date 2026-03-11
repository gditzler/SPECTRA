from typing import Optional

import numpy as np

from spectra.waveforms.base import Waveform


class Noise(Waveform):
    """Band-limited complex Gaussian noise waveform (negative/null class)."""

    def __init__(
        self,
        bandwidth_fraction: float = 0.5,
        samples_per_symbol: int = 8,
    ):
        self._bandwidth_fraction = bandwidth_fraction
        self.samples_per_symbol = samples_per_symbol

    def generate(
        self,
        num_symbols: int,
        sample_rate: float,
        seed: Optional[int] = None,
    ) -> np.ndarray:
        rng = np.random.default_rng(seed if seed is not None else np.random.randint(0, 2**32))
        n_samples = num_symbols * self.samples_per_symbol

        # Complex white Gaussian noise
        noise = (rng.standard_normal(n_samples) + 1j * rng.standard_normal(n_samples)).astype(
            np.complex64
        ) / np.sqrt(2)

        # Band-limit via FFT
        noise_fft = np.fft.fft(noise)
        freqs = np.fft.fftfreq(n_samples, d=1.0 / sample_rate)
        cutoff = self._bandwidth_fraction * sample_rate / 2.0
        noise_fft[np.abs(freqs) > cutoff] = 0.0
        filtered = np.fft.ifft(noise_fft).astype(np.complex64)

        # Normalize to unit average power
        power = np.mean(np.abs(filtered) ** 2)
        if power > 0:
            filtered = filtered / np.sqrt(power)

        return filtered

    def bandwidth(self, sample_rate: float) -> float:
        return self._bandwidth_fraction * sample_rate

    @property
    def label(self) -> str:
        return "Noise"
