from abc import ABC, abstractmethod
import numpy as np


class AntennaElement(ABC):
    """Abstract base class for antenna element radiation patterns.

    All elements return complex gain so the interface supports phase patterns.
    Initial built-in implementations are real-valued (zero phase).

    Subclasses must implement ``pattern()`` and the ``frequency`` property.
    """

    @abstractmethod
    def pattern(self, azimuth: np.ndarray, elevation: np.ndarray) -> np.ndarray:
        """Return complex gain at query angles.

        Args:
            azimuth: Azimuth angles in radians. Arbitrary shape.
            elevation: Elevation angles in radians. Same broadcast-compatible
                shape as ``azimuth``.

        Returns:
            Complex-valued array with shape matching the broadcast shape of
            inputs. Magnitude is linear gain; phase is the pattern phase shift.
        """
        ...

    @property
    @abstractmethod
    def frequency(self) -> float:
        """Design frequency in Hz."""
        ...
