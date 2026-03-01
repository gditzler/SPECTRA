from typing import List, Tuple

import numpy as np

from spectra.impairments.base import Transform
from spectra.scene.signal_desc import SignalDescription


class Compose(Transform):
    def __init__(self, transforms: List[Transform]):
        self.transforms = transforms

    def __call__(
        self, iq: np.ndarray, desc: SignalDescription, **kwargs
    ) -> Tuple[np.ndarray, SignalDescription]:
        for t in self.transforms:
            iq, desc = t(iq, desc, **kwargs)
        return iq, desc
