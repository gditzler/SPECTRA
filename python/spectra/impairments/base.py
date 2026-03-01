from abc import ABC, abstractmethod
from typing import Tuple

import numpy as np

from spectra.scene.signal_desc import SignalDescription


class Transform(ABC):
    @abstractmethod
    def __call__(
        self, iq: np.ndarray, desc: SignalDescription, **kwargs
    ) -> Tuple[np.ndarray, SignalDescription]:
        ...
