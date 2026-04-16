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
