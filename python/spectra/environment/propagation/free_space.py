"""Free-space propagation models (Friis, ITU-R P.525)."""

from __future__ import annotations

import math

from spectra.environment.propagation._base import (
    SPEED_OF_LIGHT,
    PathLossResult,
    PropagationModel,
)


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
