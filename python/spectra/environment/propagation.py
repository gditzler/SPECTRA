"""Propagation models for path loss computation."""

from __future__ import annotations

import math
from abc import ABC, abstractmethod
from dataclasses import dataclass

import numpy as np

SPEED_OF_LIGHT = 299_792_458.0


@dataclass
class PathLossResult:
    """Result of a propagation model computation."""

    path_loss_db: float
    shadow_fading_db: float = 0.0
    rms_delay_spread_s: float | None = None
    k_factor_db: float | None = None
    angular_spread_deg: float | None = None


class PropagationModel(ABC):
    """Abstract base class for propagation models."""

    @abstractmethod
    def __call__(self, distance_m: float, freq_hz: float, **kwargs) -> PathLossResult:
        """Compute path loss for given distance and frequency."""
        ...


class FreeSpacePathLoss(PropagationModel):
    """Friis free-space path loss model."""

    def __call__(self, distance_m: float, freq_hz: float, **kwargs) -> PathLossResult:
        if distance_m <= 0:
            raise ValueError("distance_m must be positive")
        pl_db = (
            20 * math.log10(distance_m)
            + 20 * math.log10(freq_hz)
            + 20 * math.log10(4 * math.pi / SPEED_OF_LIGHT)
        )
        return PathLossResult(path_loss_db=pl_db)


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


_VALID_ENVIRONMENTS = {"urban", "suburban", "rural"}


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
        if environment not in _VALID_ENVIRONMENTS:
            raise ValueError(
                f"environment must be one of {_VALID_ENVIRONMENTS}, got '{environment}'"
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
