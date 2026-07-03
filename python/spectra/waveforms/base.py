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

    def center_offset(self, sample_rate: float) -> float:
        """Offset of the occupied band's center from baseband 0 Hz.

        Most waveforms are symmetric about DC and use the default of 0.
        Waveforms whose occupancy is asymmetric (e.g. OFDM with asymmetric
        guard bands) override this so scene composers can place ground-truth
        boxes on the band actually occupied:
        ``[center_offset - bandwidth/2, center_offset + bandwidth/2]``.

        Args:
            sample_rate: Receiver sample rate in Hz.

        Returns:
            Signed offset in Hz.
        """
        return 0.0

    def num_symbols_for(self, num_samples: int, sample_rate: float) -> int:
        """Number of symbols needed to fill ``num_samples`` at ``sample_rate``.

        Default reproduces the legacy Composer heuristic
        (``num_samples // samples_per_symbol``, falling back to 8).
        Waveforms with physical-unit parameters override this.
        """
        sps = getattr(self, "samples_per_symbol", 8)
        return int(num_samples // sps)

    @property
    @abstractmethod
    def label(self) -> str:
        """Classification label string (e.g., ``"QPSK"``, ``"16QAM"``)."""
        ...
