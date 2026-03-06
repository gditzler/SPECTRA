from abc import ABC, abstractmethod
from typing import Optional

import numpy as np


class Waveform(ABC):
    """Abstract base class for all waveform generators.

    Waveforms generate complex baseband IQ samples on demand. They are
    stateless — all randomness is controlled via the ``seed`` argument to
    ``generate()``, making them safe for use across DataLoader workers.

    Subclasses must implement ``generate()``, ``bandwidth()``, and the
    ``label`` property.
    """

    @abstractmethod
    def generate(
        self,
        num_symbols: int,
        sample_rate: float,
        seed: Optional[int] = None,
    ) -> np.ndarray:
        """Generate complex baseband IQ samples.

        Args:
            num_symbols: Number of symbols to generate (actual sample count
                depends on ``samples_per_symbol`` for pulse-shaped waveforms).
            sample_rate: Receiver sample rate in Hz.
            seed: Optional integer seed for reproducibility. Pass the same
                seed to get identical output across calls.

        Returns:
            Complex64 NumPy array of IQ samples.
        """
        ...

    @abstractmethod
    def bandwidth(self, sample_rate: float) -> float:
        """Signal bandwidth in Hz.

        Args:
            sample_rate: Receiver sample rate in Hz.

        Returns:
            Positive float bandwidth in Hz. Must be <= ``sample_rate``.
        """
        ...

    @property
    @abstractmethod
    def label(self) -> str:
        """Classification label string (e.g., ``"QPSK"``, ``"16QAM"``)."""
        ...
