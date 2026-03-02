from typing import Optional

import numpy as np

from spectra._rust import lowpass_taps, convolve_complex
from spectra.waveforms.base import Waveform


class FM(Waveform):
    """Frequency Modulation waveform.

    Bandwidth via Carson's rule: 2 * (deviation + audio_bw).
    """

    def __init__(
        self,
        deviation_fraction: float = 0.1,
        audio_bw_fraction: float = 0.05,
        num_taps: int = 101,
    ):
        self._deviation_fraction = deviation_fraction
        self._audio_bw_fraction = audio_bw_fraction
        self._num_taps = num_taps
        self.samples_per_symbol = 1

    def generate(
        self,
        num_symbols: int,
        sample_rate: float,
        seed: Optional[int] = None,
    ) -> np.ndarray:
        rng = np.random.default_rng(seed)
        num_samples = num_symbols

        # Generate lowpass-filtered noise as audio
        cutoff = 2.0 * self._audio_bw_fraction
        taps = np.array(lowpass_taps(self._num_taps, cutoff), dtype=np.float32)
        noise = rng.standard_normal(num_samples).astype(np.float32)
        noise_complex = (noise + 0j).astype(np.complex64)
        filtered = np.array(convolve_complex(noise_complex, taps))
        start = (len(filtered) - num_samples) // 2
        audio = filtered[start : start + num_samples].real.astype(np.float32)
        peak = np.max(np.abs(audio))
        if peak > 0:
            audio = audio / peak

        # FM: integrate audio into phase
        deviation = sample_rate * self._deviation_fraction
        phase = 2.0 * np.pi * deviation * np.cumsum(audio) / sample_rate
        iq = np.exp(1j * phase).astype(np.complex64)
        return iq

    def bandwidth(self, sample_rate: float) -> float:
        deviation = sample_rate * self._deviation_fraction
        audio_bw = sample_rate * self._audio_bw_fraction
        return 2.0 * (deviation + audio_bw)

    @property
    def label(self) -> str:
        return "FM"
