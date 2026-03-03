from abc import ABC, abstractmethod
from typing import Optional, Tuple

import numpy as np

from spectra.utils.file_handlers.base_reader import SignalMetadata


class FileWriter(ABC):
    """Abstract base for RF file format writers."""

    @abstractmethod
    def write(self, iq: np.ndarray, metadata: Optional[SignalMetadata] = None) -> None:
        """Write IQ samples and optional metadata to disk.

        Args:
            iq: 1-D ``complex64`` array.
            metadata: Optional signal metadata.
        """
        ...

    @staticmethod
    @abstractmethod
    def extensions() -> Tuple[str, ...]:
        """File extensions this writer produces (e.g. ``('.npy',)``)."""
        ...
