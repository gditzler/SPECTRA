from abc import ABC, abstractmethod
from dataclasses import dataclass, field
from typing import Dict, List, Optional, Tuple

import numpy as np


@dataclass
class SignalMetadata:
    """Metadata extracted from an RF recording file."""

    sample_rate: Optional[float] = None
    center_frequency: Optional[float] = None
    datatype: Optional[str] = None
    num_samples: Optional[int] = None
    annotations: List[Dict] = field(default_factory=list)
    extra: Dict = field(default_factory=dict)


class FileReader(ABC):
    """Abstract base for RF file format readers."""

    @abstractmethod
    def read(self, path: str) -> Tuple[np.ndarray, SignalMetadata]:
        """Read IQ samples and metadata.

        Returns:
            Tuple of (iq_samples, metadata) where iq_samples is a 1-D
            ``np.complex64`` array.
        """
        ...

    @staticmethod
    @abstractmethod
    def extensions() -> Tuple[str, ...]:
        """File extensions this reader handles (e.g. ``('.npy',)``)."""
        ...
