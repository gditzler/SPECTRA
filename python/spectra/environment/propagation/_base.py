"""Shared base classes and helpers for propagation models."""

from __future__ import annotations

import warnings
from abc import ABC, abstractmethod
from dataclasses import dataclass
from typing import Literal

import numpy as np

SPEED_OF_LIGHT = 299_792_458.0

LOSMode = Literal["stochastic", "force_los", "force_nlos"]


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


def _resolve_los(los_mode: LOSMode, p_los: float, rng: np.random.Generator) -> bool:
    """Return True if this evaluation is LOS.

    Parameters
    ----------
    los_mode
        "stochastic" -> sample Bernoulli(p_los); "force_los" -> always True;
        "force_nlos" -> always False.
    p_los
        LOS probability in [0, 1]. Ignored unless los_mode == "stochastic".
    rng
        RNG used only in stochastic mode.
    """
    if los_mode == "force_los":
        return True
    if los_mode == "force_nlos":
        return False
    if los_mode == "stochastic":
        return bool(rng.random() < p_los)
    raise ValueError(
        f"los_mode must be one of 'stochastic', 'force_los', 'force_nlos'; got {los_mode!r}"
    )


def _check_freq_range(
    freq_hz: float,
    lo_hz: float,
    hi_hz: float,
    model_name: str,
    strict: bool = True,
) -> None:
    """Raise ValueError (strict) or emit UserWarning for out-of-range frequency."""
    if lo_hz <= freq_hz <= hi_hz:
        return
    msg = (
        f"{model_name}: freq_hz={freq_hz:.3g} outside validity envelope "
        f"[{lo_hz:.3g}, {hi_hz:.3g}] Hz."
    )
    if strict:
        raise ValueError(msg)
    warnings.warn(msg, UserWarning, stacklevel=2)


def _check_distance_range(
    distance_m: float,
    lo_m: float,
    hi_m: float,
    model_name: str,
    strict: bool = True,
) -> None:
    """Raise ValueError (strict) or emit UserWarning for out-of-range distance."""
    if lo_m <= distance_m <= hi_m:
        return
    msg = (
        f"{model_name}: distance_m={distance_m:.3g} outside validity envelope "
        f"[{lo_m:.3g}, {hi_m:.3g}] m."
    )
    if strict:
        raise ValueError(msg)
    warnings.warn(msg, UserWarning, stacklevel=2)
