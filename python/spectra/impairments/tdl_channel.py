from typing import Dict, List, Optional, Tuple

import numpy as np

from spectra.impairments.base import Transform
from spectra.scene.signal_desc import SignalDescription

PROFILES: Dict[str, Dict] = {
    "TDL-A": {
        "delays_ns": [0, 30, 70, 90, 110, 190, 410],
        "powers_db": [0.0, -1.0, -2.0, -3.0, -8.0, -17.2, -20.8],
    },
    "TDL-B": {
        "delays_ns": [0, 10, 20, 30, 50, 65, 120],
        "powers_db": [0.0, -2.2, -4.0, -3.2, -9.8, -1.2, -3.4],
    },
    "TDL-C": {
        "delays_ns": [0, 65, 70, 190, 195, 200, 240, 325, 520, 1045, 1510, 2595],
        "powers_db": [0.0, -2.2, -0.6, -0.6, -4.4, -7.6, -5.2, -6.4, -3.2, -5.2, -7.4, -2.8],
    },
    "TDL-D": {
        "delays_ns": [0, 0, 10, 15, 20, 25, 50, 65, 75, 85, 170, 190, 275],
        "powers_db": [
            -0.2,
            -13.5,
            -18.8,
            -21.0,
            -22.8,
            -17.9,
            -20.1,
            -21.9,
            -22.9,
            -27.8,
            -23.6,
            -24.8,
            -30.0,
        ],
        "k_factor_db": 13.3,
    },
    "TDL-E": {
        "delays_ns": [0, 0, 50, 55, 60, 100, 170, 195, 220, 350, 520, 610, 705, 750, 780],
        "powers_db": [
            -0.03,
            -22.03,
            -15.8,
            -18.1,
            -19.8,
            -22.9,
            -22.4,
            -18.6,
            -22.8,
            -22.6,
            -27.4,
            -20.2,
            -24.1,
            -30.0,
            -27.7,
        ],
        "k_factor_db": 22.0,
    },
    "PedestrianA": {
        "delays_ns": [0, 110, 190, 410],
        "powers_db": [0.0, -9.7, -19.2, -22.8],
    },
    "PedestrianB": {
        "delays_ns": [0, 200, 800, 1200, 2300, 3700],
        "powers_db": [0.0, -0.9, -4.9, -8.0, -7.8, -23.9],
    },
    "VehicularA": {
        "delays_ns": [0, 310, 710, 1090, 1730, 2510],
        "powers_db": [0.0, -1.0, -9.0, -10.0, -15.0, -20.0],
    },
    "VehicularB": {
        "delays_ns": [0, 300, 8900, 12900, 17100, 20000],
        "powers_db": [-2.5, 0.0, -12.8, -10.0, -25.2, -16.0],
    },
}


class TDLChannel(Transform):
    """3GPP TDL multipath fading channel.

    Implements standardized delay/power profiles from 3GPP TS 38.901
    and ITU channel models.
    """

    PROFILES = PROFILES

    def __init__(self, profile: str = "TDL-A", doppler_hz: float = 5.0):
        if profile not in self.PROFILES:
            raise ValueError(
                f"Unknown profile '{profile}'. Choose from: {list(self.PROFILES.keys())}"
            )
        self._profile_name = profile
        self._profile = self.PROFILES[profile]
        self._doppler_hz = doppler_hz

    @classmethod
    def custom(
        cls,
        delays_ns: List[float],
        powers_db: List[float],
        doppler_hz: float = 5.0,
        k_factor_db: Optional[float] = None,
    ) -> "TDLChannel":
        instance = cls.__new__(cls)
        instance._profile_name = "custom"
        instance._profile = {"delays_ns": delays_ns, "powers_db": powers_db}
        if k_factor_db is not None:
            instance._profile["k_factor_db"] = k_factor_db
        instance._doppler_hz = doppler_hz
        return instance

    def __call__(
        self, iq: np.ndarray, desc: SignalDescription, **kwargs
    ) -> Tuple[np.ndarray, SignalDescription]:
        sample_rate = kwargs.get("sample_rate", 1e6)
        delays_ns = np.array(self._profile["delays_ns"], dtype=float)
        powers_db = np.array(self._profile["powers_db"], dtype=float)

        # Convert delays to samples
        delays_samples = (delays_ns * 1e-9 * sample_rate).astype(int)

        # Convert powers to linear and normalize
        powers_linear = 10.0 ** (powers_db / 10.0)
        powers_linear /= powers_linear.sum()

        # Build impulse response
        max_delay = max(delays_samples.max(), 1)
        h = np.zeros(max_delay + 1, dtype=np.complex128)

        k_factor_db = self._profile.get("k_factor_db", None)

        for i, (delay, power) in enumerate(zip(delays_samples, powers_linear)):
            tap = (np.random.randn() + 1j * np.random.randn()) / np.sqrt(2.0)

            # For LOS profiles, first tap has deterministic component
            if k_factor_db is not None and i == 0:
                k_lin = 10.0 ** (k_factor_db / 10.0)
                los_power = k_lin / (k_lin + 1.0)
                nlos_power = 1.0 / (k_lin + 1.0)
                tap = np.sqrt(los_power) + tap * np.sqrt(nlos_power)

            h[delay] += tap * np.sqrt(power)

        out = np.convolve(iq, h, mode="same").astype(np.complex64)
        return out, desc
