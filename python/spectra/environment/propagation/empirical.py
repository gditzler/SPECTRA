"""Empirical path-loss models: log-distance, Hata-family."""

from __future__ import annotations

import math

import numpy as np

from spectra.environment.propagation._base import PathLossResult, PropagationModel
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

        a_hms = (1.1 * math.log10(fc_mhz) - 0.7) * self.h_ms_m - (
            1.56 * math.log10(fc_mhz) - 0.8
        )

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
            pl_db -= (
                4.78 * (math.log10(fc_mhz)) ** 2
                + 18.33 * math.log10(fc_mhz)
                - 40.94
            )

        return PathLossResult(path_loss_db=pl_db)
