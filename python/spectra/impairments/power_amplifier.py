from typing import Tuple

import numpy as np

from spectra.impairments.base import Transform
from spectra.scene.signal_desc import SignalDescription


class RappPA(Transform):
    """Rapp power amplifier model (AM/AM only).

    Smooth clipping model standard for OFDM/satellite:
    y = x / (1 + |x/sat|^(2p))^(1/2p)
    """

    def __init__(self, smoothness: float = 3.0, saturation: float = 1.0):
        self._p = smoothness
        self._sat = saturation

    def __call__(
        self, iq: np.ndarray, desc: SignalDescription, **kwargs
    ) -> Tuple[np.ndarray, SignalDescription]:
        amplitude = np.abs(iq)
        phase = np.angle(iq)
        normalized = amplitude / self._sat
        gain = 1.0 / (1.0 + normalized ** (2 * self._p)) ** (1.0 / (2 * self._p))
        out_amp = amplitude * gain
        out = (out_amp * np.exp(1j * phase)).astype(np.complex64)
        return out, desc


class SalehPA(Transform):
    """Saleh TWT power amplifier model (AM/AM + AM/PM).

    A(r) = alpha_a * r / (1 + beta_a * r^2)
    phi(r) = alpha_p * r^2 / (1 + beta_p * r^2)
    """

    def __init__(
        self,
        alpha_a: float = 2.0,
        beta_a: float = 1.0,
        alpha_p: float = 1.0,
        beta_p: float = 1.0,
    ):
        self._alpha_a = alpha_a
        self._beta_a = beta_a
        self._alpha_p = alpha_p
        self._beta_p = beta_p

    def __call__(
        self, iq: np.ndarray, desc: SignalDescription, **kwargs
    ) -> Tuple[np.ndarray, SignalDescription]:
        r = np.abs(iq)
        phase = np.angle(iq)

        # AM/AM
        out_amp = self._alpha_a * r / (1.0 + self._beta_a * r**2)
        # AM/PM
        phase_shift = self._alpha_p * r**2 / (1.0 + self._beta_p * r**2)

        out = (out_amp * np.exp(1j * (phase + phase_shift))).astype(np.complex64)
        return out, desc
