from dataclasses import dataclass, field
from typing import Dict, Optional


@dataclass
class SignalDescription:
    """Ground-truth record for a single signal in a scene or impairment pipeline.

    Frequencies are in Hz relative to DC (baseband). A signal centered at
    100 kHz with 50 kHz bandwidth has ``f_low=75_000``, ``f_high=125_000``.
    Times are in seconds from the start of the capture.

    Attributes:
        t_start: Start time of the signal in seconds.
        t_stop: End time of the signal in seconds.
        f_low: Lower edge of the signal's occupied bandwidth in Hz.
        f_high: Upper edge of the signal's occupied bandwidth in Hz.
        label: Modulation/waveform class string (e.g., ``"QPSK"``).
        snr: Signal-to-noise ratio in dB at the point this description was created.
        modulation_params: Reserved dict for waveform-specific metadata.
        mode: Optional operating-mode label for emitter-internal timelines
            (e.g., ``"search"``, ``"track"``, ``"comms"``). ``None`` for
            single-mode signals.
    """

    t_start: float
    t_stop: float
    f_low: float
    f_high: float
    label: str
    snr: float
    modulation_params: Dict = field(default_factory=dict)
    mode: Optional[str] = None

    @property
    def f_center(self) -> float:
        return (self.f_low + self.f_high) / 2.0

    @property
    def bandwidth(self) -> float:
        return self.f_high - self.f_low

    @property
    def duration(self) -> float:
        return self.t_stop - self.t_start
