"""ITU-R P.1411 site-general short-range outdoor terrestrial model."""

from __future__ import annotations

import math
from typing import Literal

import numpy as np

from spectra.environment.propagation._base import (
    LOSMode,
    PathLossResult,
    PropagationModel,
    _check_distance_range,
    _check_freq_range,
    _resolve_los,
)

# Site-general coefficients (alpha, beta, gamma, sigma) per ITU-R P.1411-12
# Table 4. Keyed by (environment, is_los).
_P1411_COEFFS: dict[tuple[str, bool], tuple[float, float, float, float]] = {
    ("urban_high_rise", True): (2.29, 28.6, 1.96, 3.48),
    ("urban_high_rise", False): (4.39, -6.27, 2.30, 6.89),
    ("urban_low_rise_suburban", True): (2.12, 29.2, 2.11, 5.06),
    ("urban_low_rise_suburban", False): (4.00, 10.2, 2.36, 7.60),
    ("residential", True): (2.29, 28.6, 1.96, 3.48),
    ("residential", False): (4.39, -6.27, 2.30, 6.89),
}

_VALID_P1411_ENVS = {"urban_high_rise", "urban_low_rise_suburban", "residential"}


class ITU_R_P1411(PropagationModel):
    """ITU-R P.1411-12 site-general short-range outdoor model.

    Valid envelope: 300 MHz - 100 GHz, 50 m - 3 km.

    Parameters
    ----------
    environment
        Site-general environment category.
    los_mode
        "stochastic" (default), "force_los", or "force_nlos".
    strict_range
        Raise ValueError outside validity envelope (default True).
    """

    MODEL_NAME = "ITU_R_P1411"
    FREQ_RANGE_HZ = (300e6, 100e9)
    DISTANCE_RANGE_M = (50.0, 3000.0)

    def __init__(
        self,
        environment: Literal["urban_high_rise", "urban_low_rise_suburban", "residential"],
        los_mode: LOSMode = "stochastic",
        strict_range: bool = True,
    ):
        if environment not in _VALID_P1411_ENVS:
            raise ValueError(f"environment must be one of {_VALID_P1411_ENVS}, got '{environment}'")
        self.environment = environment
        self.los_mode = los_mode
        self.strict_range = strict_range

    def _los_probability(self, d_2d_m: float) -> float:
        """Approximate LOS probability per ITU-R P.1411 section 4.3.

        Uses a site-general exponential decay with environment-specific
        characteristic distance. These are simplified fits - the full
        section 4.3 formulation depends on building clutter density.
        """
        if self.environment == "urban_high_rise":
            char_d = 60.0
        elif self.environment == "urban_low_rise_suburban":
            char_d = 150.0
        else:  # residential
            char_d = 300.0
        return float(math.exp(-d_2d_m / char_d))

    def __call__(self, distance_m: float, freq_hz: float, **kwargs) -> PathLossResult:
        if distance_m <= 0:
            raise ValueError("distance_m must be positive")
        _check_freq_range(freq_hz, *self.FREQ_RANGE_HZ, self.MODEL_NAME, strict=self.strict_range)
        _check_distance_range(
            distance_m,
            *self.DISTANCE_RANGE_M,
            self.MODEL_NAME,
            strict=self.strict_range,
        )

        seed = kwargs.get("seed")
        rng = np.random.default_rng(seed)

        is_los = _resolve_los(self.los_mode, self._los_probability(distance_m), rng)
        alpha, beta, gamma, sigma = _P1411_COEFFS[(self.environment, is_los)]

        f_ghz = freq_hz / 1e9
        pl_mean = alpha * 10.0 * math.log10(distance_m) + beta + gamma * 10.0 * math.log10(f_ghz)
        sf = float(rng.normal(0.0, sigma))

        return PathLossResult(
            path_loss_db=pl_mean + sf,
            shadow_fading_db=sf,
        )
