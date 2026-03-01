from dataclasses import dataclass, field
from typing import Dict


@dataclass
class SignalDescription:
    t_start: float
    t_stop: float
    f_low: float
    f_high: float
    label: str
    snr: float
    modulation_params: Dict = field(default_factory=dict)

    @property
    def f_center(self) -> float:
        return (self.f_low + self.f_high) / 2.0

    @property
    def bandwidth(self) -> float:
        return self.f_high - self.f_low

    @property
    def duration(self) -> float:
        return self.t_stop - self.t_start
