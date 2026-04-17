"""Empirical path-loss models: log-distance, Hata-family."""

from __future__ import annotations

import math
from typing import Literal

import numpy as np

from spectra.environment.propagation._base import (
    PathLossResult,
    PropagationModel,
    _check_distance_range,
    _check_freq_range,
)
from spectra.environment.propagation.free_space import FreeSpacePathLoss


class LogDistancePL(PropagationModel):
    """Log-distance path loss model with optional shadow fading.

    PL(dB) = PL(d0) + 10*n*log10(d/d0) + X_sigma
    where PL(d0) is free-space path loss at reference distance d0.
    """

    def __init__(
        self,
        n: float = 3.0,
        sigma_db: float = 0.0,
        d0: float = 1.0,
    ):
        self.n = n
        self.sigma_db = sigma_db
        self.d0 = d0
        self._fspl = FreeSpacePathLoss()

    def __call__(self, distance_m: float, freq_hz: float, **kwargs) -> PathLossResult:
        if distance_m <= 0:
            raise ValueError("distance_m must be positive")
        pl_d0 = self._fspl(self.d0, freq_hz).path_loss_db
        pl_db = pl_d0 + 10 * self.n * math.log10(distance_m / self.d0)

        shadow = 0.0
        if self.sigma_db > 0:
            seed = kwargs.get("seed")
            rng = np.random.default_rng(seed)
            shadow = float(rng.normal(0.0, self.sigma_db))

        return PathLossResult(
            path_loss_db=pl_db + shadow,
            shadow_fading_db=shadow,
        )


_VALID_ENVIRONMENTS_COST231 = {"urban", "suburban", "rural"}


class COST231HataPL(PropagationModel):
    """COST-231 Hata path loss model for 1500-2000 MHz.

    Valid ranges: fc 1500-2000 MHz, h_bs 30-200 m, h_ms 1-10 m, d 1-20 km.
    """

    def __init__(
        self,
        h_bs_m: float = 30.0,
        h_ms_m: float = 1.5,
        environment: str = "urban",
    ):
        if environment not in _VALID_ENVIRONMENTS_COST231:
            raise ValueError(
                f"environment must be one of {_VALID_ENVIRONMENTS_COST231}, got '{environment}'"
            )
        self.h_bs_m = h_bs_m
        self.h_ms_m = h_ms_m
        self.environment = environment

    def __call__(self, distance_m: float, freq_hz: float, **kwargs) -> PathLossResult:
        if distance_m <= 0:
            raise ValueError("distance_m must be positive")

        fc_mhz = freq_hz / 1e6
        d_km = distance_m / 1000.0

        a_hms = (1.1 * math.log10(fc_mhz) - 0.7) * self.h_ms_m - (1.56 * math.log10(fc_mhz) - 0.8)

        c_m = 3.0 if self.environment == "urban" else 0.0

        pl_db = (
            46.3
            + 33.9 * math.log10(fc_mhz)
            - 13.82 * math.log10(self.h_bs_m)
            - a_hms
            + (44.9 - 6.55 * math.log10(self.h_bs_m)) * math.log10(d_km)
            + c_m
        )

        if self.environment == "suburban":
            pl_db -= 2 * (math.log10(fc_mhz / 28)) ** 2 + 5.4
        elif self.environment == "rural":
            pl_db -= 4.78 * (math.log10(fc_mhz)) ** 2 + 18.33 * math.log10(fc_mhz) - 40.94

        return PathLossResult(path_loss_db=pl_db)


_VALID_ENVIRONMENTS_HATA = {
    "urban_large",
    "urban_small_medium",
    "suburban",
    "rural",
}


class OkumuraHataPL(PropagationModel):
    """Okumura-Hata path loss model (Hata, 1980), valid 150-1500 MHz.

    Parameters
    ----------
    h_bs_m
        Base station antenna height above ground (m). Nominal range 30-200 m.
    h_ms_m
        Mobile station antenna height above ground (m). Nominal range 1-10 m.
    environment
        "urban_large" uses the large-city mobile antenna correction;
        "urban_small_medium", "suburban", and "rural" use the small/medium
        city form with appropriate environmental offsets.
    sigma_db
        Lognormal shadow fading std dev (dB). 0.0 disables.
    strict_range
        If True, raise ValueError for out-of-range freq/distance. If False, warn.
    """

    def __init__(
        self,
        h_bs_m: float,
        h_ms_m: float,
        environment: Literal["urban_large", "urban_small_medium", "suburban", "rural"],
        sigma_db: float = 0.0,
        strict_range: bool = True,
    ):
        if environment not in _VALID_ENVIRONMENTS_HATA:
            raise ValueError(
                f"environment must be one of {_VALID_ENVIRONMENTS_HATA}, "
                f"got '{environment}'. For 1500-2000 MHz use COST231HataPL."
            )
        self.h_bs_m = h_bs_m
        self.h_ms_m = h_ms_m
        self.environment = environment
        self.sigma_db = sigma_db
        self.strict_range = strict_range

    def __call__(self, distance_m: float, freq_hz: float, **kwargs) -> PathLossResult:
        if distance_m <= 0:
            raise ValueError("distance_m must be positive")

        # Validity envelope
        _check_freq_range(freq_hz, 150e6, 1500e6, "OkumuraHataPL", strict=self.strict_range)
        _check_distance_range(
            distance_m, 1000.0, 20000.0, "OkumuraHataPL", strict=self.strict_range
        )

        fc_mhz = freq_hz / 1e6
        d_km = distance_m / 1000.0

        # Mobile antenna correction a(h_ms)
        if self.environment == "urban_large":
            if fc_mhz >= 400:
                a_hms = 3.2 * (math.log10(11.75 * self.h_ms_m)) ** 2 - 4.97
            else:
                a_hms = 8.29 * (math.log10(1.54 * self.h_ms_m)) ** 2 - 1.1
        else:
            a_hms = (1.1 * math.log10(fc_mhz) - 0.7) * self.h_ms_m - (
                1.56 * math.log10(fc_mhz) - 0.8
            )

        # Basic urban PL (Hata urban small-medium city form)
        pl_urban = (
            69.55
            + 26.16 * math.log10(fc_mhz)
            - 13.82 * math.log10(self.h_bs_m)
            - a_hms
            + (44.9 - 6.55 * math.log10(self.h_bs_m)) * math.log10(d_km)
        )

        if self.environment in ("urban_small_medium", "urban_large"):
            pl_db = pl_urban
        elif self.environment == "suburban":
            pl_db = pl_urban - 2 * (math.log10(fc_mhz / 28)) ** 2 - 5.4
        else:  # rural
            pl_db = pl_urban - 4.78 * (math.log10(fc_mhz)) ** 2 + 18.33 * math.log10(fc_mhz) - 40.94

        # Shadow fading
        shadow = 0.0
        if self.sigma_db > 0:
            seed = kwargs.get("seed")
            rng = np.random.default_rng(seed)
            shadow = float(rng.normal(0.0, self.sigma_db))

        return PathLossResult(
            path_loss_db=pl_db + shadow,
            shadow_fading_db=shadow,
        )
