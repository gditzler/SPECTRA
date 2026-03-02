from typing import Tuple

import numpy as np

from spectra.impairments.base import Transform
from spectra.scene.signal_desc import SignalDescription


class ColoredNoise(Transform):
    """Add colored noise (pink or red/brown) at specified SNR."""

    def __init__(self, snr: float = 10.0, color: str = "pink"):
        self._snr = snr
        self._color = color

    def __call__(
        self, iq: np.ndarray, desc: SignalDescription, **kwargs
    ) -> Tuple[np.ndarray, SignalDescription]:
        n = len(iq)
        # Generate white noise in frequency domain
        freqs = np.fft.rfftfreq(n)
        freqs[0] = 1.0  # avoid division by zero
        if self._color == "pink":
            # 1/f spectrum
            spectrum = 1.0 / np.sqrt(freqs)
        elif self._color == "red":
            # 1/f^2 spectrum
            spectrum = 1.0 / freqs
        else:
            spectrum = np.ones_like(freqs)
        # Generate noise
        white = np.random.randn(len(freqs)) + 1j * np.random.randn(len(freqs))
        colored_freq = white * spectrum
        colored_time = np.fft.irfft(colored_freq, n=n)
        noise = (colored_time + 1j * np.fft.irfft(
            (np.random.randn(len(freqs)) + 1j * np.random.randn(len(freqs))) * spectrum, n=n
        )).astype(np.complex64)
        # Scale to desired SNR
        signal_power = np.mean(np.abs(iq) ** 2)
        snr_linear = 10.0 ** (self._snr / 10.0)
        noise_power = signal_power / snr_linear
        current_noise_power = np.mean(np.abs(noise) ** 2)
        if current_noise_power > 0:
            noise *= np.sqrt(noise_power / current_noise_power)
        return (iq + noise).astype(np.complex64), desc
