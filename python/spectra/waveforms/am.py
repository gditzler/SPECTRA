from typing import Optional

import numpy as np

from spectra.waveforms.base import Waveform


class AM(Waveform):
    """DSB-AM (Double Sideband Amplitude Modulation) waveform."""

    def __init__(
        self,
        mod_index: float = 0.5,
        message_bandwidth: float = 5e3,
        samples_per_symbol: int = 8,
    ):
        self._mod_index = mod_index
        self._message_bandwidth = message_bandwidth
        self.samples_per_symbol = samples_per_symbol

    def generate(
        self,
        num_symbols: int,
        sample_rate: float,
        seed: Optional[int] = None,
    ) -> np.ndarray:
        rng = np.random.default_rng(
            seed if seed is not None else np.random.randint(0, 2**32)
        )
        n_samples = num_symbols * self.samples_per_symbol

        # Generate band-limited message: white noise filtered to message_bandwidth
        noise = rng.standard_normal(n_samples).astype(np.float32)
        # Low-pass filter via FFT
        freqs = np.fft.rfftfreq(n_samples, d=1.0 / sample_rate)
        noise_fft = np.fft.rfft(noise)
        noise_fft[freqs > self._message_bandwidth] = 0.0
        message = np.fft.irfft(noise_fft, n=n_samples)

        # Normalize message to [-1, 1]
        peak = np.max(np.abs(message)) + 1e-10
        message = message / peak

        # AM: (1 + m * message(t)), real-valued baseband
        envelope = (1.0 + self._mod_index * message).astype(np.float32)
        return (envelope + 0j).astype(np.complex64)

    def bandwidth(self, sample_rate: float) -> float:
        return 2.0 * self._message_bandwidth

    @property
    def label(self) -> str:
        return "AM"
