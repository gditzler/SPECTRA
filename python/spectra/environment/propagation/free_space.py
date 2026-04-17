"""Free-space propagation models (Friis, ITU-R P.525)."""

from __future__ import annotations

import math

from spectra.environment.propagation._base import (
    SPEED_OF_LIGHT,
    PathLossResult,
    PropagationModel,
)
from spectra.environment.propagation.atmospheric import gaseous_attenuation_db


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


class ITU_R_P525(PropagationModel):
    """ITU-R P.525-4 free-space path loss, with optional P.676 gaseous absorption.

    Without gaseous attenuation this is numerically identical to `FreeSpacePathLoss`;
    setting `include_gaseous=True` adds one-way oxygen + water vapor attenuation
    along a horizontal terrestrial path per ITU-R P.676-13 Annex 2.

    Parameters
    ----------
    include_gaseous
        If True, add P.676 gaseous attenuation.
    temperature_k
        Atmospheric temperature for P.676 (default = 288.15 K).
    pressure_hpa
        Dry-air pressure for P.676 (default = 1013.25 hPa).
    water_vapor_density_g_m3
        Surface water vapor density for P.676 (default = 7.5 g/m^3).
    """

    def __init__(
        self,
        *,
        include_gaseous: bool = False,
        temperature_k: float = 288.15,
        pressure_hpa: float = 1013.25,
        water_vapor_density_g_m3: float = 7.5,
    ):
        self.include_gaseous = include_gaseous
        self.temperature_k = temperature_k
        self.pressure_hpa = pressure_hpa
        self.water_vapor_density_g_m3 = water_vapor_density_g_m3
        self._fspl = FreeSpacePathLoss()

    def __call__(self, distance_m: float, freq_hz: float, **kwargs) -> PathLossResult:
        if distance_m <= 0:
            raise ValueError("distance_m must be positive")
        fspl_db = self._fspl(distance_m, freq_hz).path_loss_db
        extra_db = 0.0
        if self.include_gaseous:
            extra_db = gaseous_attenuation_db(
                distance_m,
                freq_hz,
                temperature_k=self.temperature_k,
                pressure_hpa=self.pressure_hpa,
                water_vapor_density_g_m3=self.water_vapor_density_g_m3,
            )
        return PathLossResult(path_loss_db=fspl_db + extra_db)
