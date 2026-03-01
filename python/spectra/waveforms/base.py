from abc import ABC, abstractmethod
from typing import Optional

import numpy as np


class Waveform(ABC):
    """Abstract base class for all waveform generators."""

    @abstractmethod
    def generate(
        self,
        num_symbols: int,
        sample_rate: float,
        seed: Optional[int] = None,
    ) -> np.ndarray:
        """Generate complex baseband IQ samples."""
        ...

    @abstractmethod
    def bandwidth(self, sample_rate: float) -> float:
        """Signal bandwidth in Hz given a sample rate."""
        ...

    @property
    @abstractmethod
    def label(self) -> str:
        """Classification label string."""
        ...
