from typing import Optional, Tuple

import numpy as np

from spectra.impairments.base import Transform
from spectra.scene.signal_desc import SignalDescription


class AWGN(Transform):
    """Additive White Gaussian Noise.

    Computes noise power from measured signal power and target SNR, then adds
    complex circular Gaussian noise. Exactly one of ``snr`` or ``snr_range``
    must be provided.

    Args:
        snr: Fixed signal-to-noise ratio in dB.
        snr_range: ``(min_db, max_db)`` tuple; actual SNR is drawn uniformly
            per call. Useful for dataset diversity.

    Raises:
        ValueError: If both or neither of ``snr`` and ``snr_range`` are given.

    Example::

        awgn = AWGN(snr_range=(-5.0, 30.0))
        iq_noisy, desc = awgn(iq, desc, sample_rate=1e6)
    """

    def __init__(
        self,
        snr: Optional[float] = None,
        snr_range: Optional[Tuple[float, float]] = None,
    ):
        if snr is None and snr_range is None:
            raise ValueError("Must provide either snr or snr_range")
        self.snr = snr
        self.snr_range = snr_range

    def __call__(
        self, iq: np.ndarray, desc: SignalDescription, **kwargs
    ) -> Tuple[np.ndarray, SignalDescription]:
        if self.snr_range is not None:
            snr_db = float(np.random.uniform(*self.snr_range))
        else:
            assert self.snr is not None  # enforced in __init__
            snr_db = float(self.snr)

        signal_power = np.mean(np.abs(iq) ** 2)
        snr_linear = 10.0 ** (snr_db / 10.0)
        noise_power = signal_power / snr_linear

        noise = np.sqrt(noise_power / 2.0) * (
            np.random.randn(len(iq)) + 1j * np.random.randn(len(iq))
        ).astype(np.complex64)

        return iq + noise, desc
