from dataclasses import replace
from typing import Optional, Tuple

import numpy as np

from spectra.impairments.base import Transform
from spectra.scene.signal_desc import SignalDescription

_C = 3e8  # speed of light (m/s)


class DopplerShift(Transform):
    """
    Doppler frequency shift from relative motion between transmitter and receiver.

    Parameterization options (exactly one group must be provided):
      - ``fd_hz``: fixed Doppler shift in Hz
      - ``max_fd_hz``: random Doppler shift drawn from Uniform(-max, +max) each call
      - ``speed_mps`` + ``carrier_hz``: physical parameters; angle_deg defaults to 0
        (head-on approach). Computes fd = speed * cos(angle) / c * carrier_hz.

    Profiles:
      - ``"constant"`` (default): constant radial velocity throughout the signal.
        Phase ramp = 2*pi*fd*t. SignalDescription f_low/f_high are shifted by fd.
      - ``"linear"``: velocity reverses linearly (flyby scenario: approaching then
        receding). fd varies from +fd to -fd. Net phase ~ 0; SignalDescription
        is unchanged.

    Parameters
    ----------
    fd_hz : float, optional
        Fixed Doppler shift in Hz. Positive = approaching.
    max_fd_hz : float, optional
        Maximum Doppler shift magnitude in Hz; actual fd drawn from Uniform(-max, max).
    speed_mps : float, optional
        Relative speed in m/s. Requires ``carrier_hz``.
    carrier_hz : float, optional
        Carrier/center frequency in Hz. Required with ``speed_mps``.
    angle_deg : float, optional
        Angle between velocity vector and line-of-sight in degrees (default 0 = head-on).
    profile : {"constant", "linear"}, optional
        Velocity profile over the signal duration. Default "constant".
    """

    _VALID_PROFILES = ("constant", "linear")

    def __init__(
        self,
        fd_hz: Optional[float] = None,
        max_fd_hz: Optional[float] = None,
        speed_mps: Optional[float] = None,
        carrier_hz: Optional[float] = None,
        angle_deg: float = 0.0,
        profile: str = "constant",
    ):
        if profile not in self._VALID_PROFILES:
            raise ValueError(f"profile must be one of {self._VALID_PROFILES}, got {profile!r}")

        has_direct = fd_hz is not None or max_fd_hz is not None
        has_physical = speed_mps is not None

        if not has_direct and not has_physical:
            raise ValueError("Provide fd_hz, max_fd_hz, or (speed_mps + carrier_hz)")
        if has_physical and carrier_hz is None:
            raise ValueError("speed_mps requires carrier_hz")

        self._fd_hz = fd_hz
        self._max_fd_hz = max_fd_hz
        self._speed_mps = speed_mps
        self._carrier_hz = carrier_hz
        self._angle_deg = angle_deg
        self._profile = profile

    def _resolve_fd(self) -> float:
        """Compute the Doppler shift in Hz for this call."""
        if self._speed_mps is not None:
            return self._speed_mps * np.cos(np.radians(self._angle_deg)) / _C * self._carrier_hz
        if self._max_fd_hz is not None:
            return float(np.random.uniform(-self._max_fd_hz, self._max_fd_hz))
        return float(self._fd_hz)

    def __call__(
        self, iq: np.ndarray, desc: SignalDescription, **kwargs
    ) -> Tuple[np.ndarray, SignalDescription]:
        sample_rate = kwargs.get("sample_rate")
        if sample_rate is None:
            raise ValueError("DopplerShift requires sample_rate kwarg")

        n = len(iq)
        fd = self._resolve_fd()
        t = np.arange(n) / sample_rate

        if self._profile == "constant":
            phase = 2.0 * np.pi * fd * t
            new_desc = replace(desc, f_low=desc.f_low + fd, f_high=desc.f_high + fd)
        else:  # "linear" flyby: fd goes from +fd to -fd
            fd_t = np.linspace(fd, -fd, n)
            phase = 2.0 * np.pi * np.cumsum(fd_t) / sample_rate
            new_desc = desc  # net shift is zero

        out = (iq * np.exp(1j * phase).astype(np.complex64)).astype(np.complex64)
        return out, new_desc
