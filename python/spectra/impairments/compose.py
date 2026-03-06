from typing import List, Tuple

import numpy as np

from spectra.impairments.base import Transform
from spectra.scene.signal_desc import SignalDescription


class Compose(Transform):
    """Sequential impairment pipeline, analogous to ``torchvision.transforms.Compose``.

    Applies each :class:`Transform` in order, threading ``(iq, desc)`` through
    the chain. The ``**kwargs`` (typically ``sample_rate``) are forwarded to every
    transform.

    Args:
        transforms: Ordered list of :class:`Transform` instances to apply.

    Example::

        pipeline = Compose([
            AWGN(snr_range=(-5.0, 20.0)),
            FrequencyOffset(max_offset=5000.0),
            PhaseNoise(level=-80.0),
        ])
        iq_impaired, desc = pipeline(iq, desc, sample_rate=1e6)
    """

    def __init__(self, transforms: List[Transform]):
        self.transforms = transforms

    def __call__(
        self, iq: np.ndarray, desc: SignalDescription, **kwargs
    ) -> Tuple[np.ndarray, SignalDescription]:
        for t in self.transforms:
            iq, desc = t(iq, desc, **kwargs)
        return iq, desc
