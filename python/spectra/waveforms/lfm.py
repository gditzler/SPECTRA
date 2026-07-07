from typing import Optional

import numpy as np

from spectra._rust import generate_chirp
from spectra.waveforms.base import Waveform


class LFM(Waveform):
    """Linear Frequency Modulation (chirp) radar waveform.

    Each 'symbol' is one complete chirp pulse sweeping across the
    configured bandwidth.
    """

    _VALID_DIRECTIONS = ("up", "down", "random")

    def __init__(
        self,
        bandwidth_fraction: Optional[float] = None,
        samples_per_pulse: Optional[int] = None,
        sweep_bandwidth: Optional[float] = None,
        pulse_duration: Optional[float] = None,
        direction: str = "up",
    ):
        if sweep_bandwidth is not None and bandwidth_fraction is not None:
            raise ValueError("sweep_bandwidth and bandwidth_fraction are mutually exclusive")
        if pulse_duration is not None and samples_per_pulse is not None:
            raise ValueError("pulse_duration and samples_per_pulse are mutually exclusive")
        if direction not in self._VALID_DIRECTIONS:
            raise ValueError(
                f"direction must be one of {self._VALID_DIRECTIONS}, got {direction!r}"
            )
        self._direction = direction
        self._sweep_bandwidth = sweep_bandwidth
        self._pulse_duration = pulse_duration
        self._bandwidth_fraction = 0.5 if bandwidth_fraction is None else bandwidth_fraction
        self._samples_per_pulse = 256 if samples_per_pulse is None else samples_per_pulse
        self.samples_per_symbol = self._samples_per_pulse

    def generate(
        self,
        num_symbols: int,
        sample_rate: float,
        seed: Optional[int] = None,
    ) -> np.ndarray:
        bw = self.bandwidth(sample_rate)
        if bw > sample_rate:
            raise ValueError(
                f"LFM sweep bandwidth {bw:g} Hz exceeds sample_rate {sample_rate:g} Hz"
            )
        n = (
            round(self._pulse_duration * sample_rate)
            if self._pulse_duration is not None
            else self._samples_per_pulse
        )
        if self._direction == "random":
            s = seed if seed is not None else np.random.randint(0, 2**32)
            rng = np.random.default_rng(s)
            direction = "down" if rng.random() < 0.5 else "up"
        else:
            direction = self._direction

        if direction == "down":
            f0, f1 = bw / 2.0, -bw / 2.0
        else:  # "up"
            f0, f1 = -bw / 2.0, bw / 2.0
        duration = n / sample_rate

        pulses = [generate_chirp(duration, sample_rate, f0, f1) for _ in range(num_symbols)]
        return np.concatenate(pulses) if pulses else np.array([], dtype=np.complex64)

    def bandwidth(self, sample_rate: float) -> float:
        if self._sweep_bandwidth is not None:
            return self._sweep_bandwidth
        return sample_rate * self._bandwidth_fraction

    def num_symbols_for(self, num_samples: int, sample_rate: float) -> int:
        n = (
            round(self._pulse_duration * sample_rate)
            if self._pulse_duration is not None
            else self._samples_per_pulse
        )
        return max(1, int(num_samples // n))

    @property
    def label(self) -> str:
        return "LFM"
