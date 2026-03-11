from abc import ABC, abstractmethod
from typing import Tuple

import numpy as np

from spectra.scene.signal_desc import SignalDescription


class Transform(ABC):
    """Abstract base class for all channel impairments.

    A Transform is a callable that applies a channel effect to a complex baseband
    IQ signal. Transforms receive the signal alongside its ``SignalDescription``
    metadata and may update the metadata to reflect the effect (e.g.,
    ``FrequencyOffset`` shifts ``f_low`` and ``f_high``).

    All keyword arguments (typically ``sample_rate``) are forwarded from
    ``Compose`` to every transform in the chain.

    Example:
        Minimal custom impairment::

            class ScaleAmplitude(Transform):
                def __init__(self, factor: float):
                    self.factor = factor

                def __call__(self, iq, desc, **kwargs):
                    return iq * self.factor, desc
    """

    @abstractmethod
    def __call__(
        self, iq: np.ndarray, desc: SignalDescription, **kwargs
    ) -> Tuple[np.ndarray, SignalDescription]: ...
