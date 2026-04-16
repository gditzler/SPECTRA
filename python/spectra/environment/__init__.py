"""Environment and propagation modeling for SPECTRA."""

from spectra.environment.core import Emitter, Environment, LinkParams, ReceiverConfig
from spectra.environment.integration import link_params_to_impairments
from spectra.environment.position import Position
from spectra.environment.presets import propagation_presets
from spectra.environment.propagation import (
    COST231HataPL,
    FreeSpacePathLoss,
    LogDistancePL,
    PathLossResult,
    PropagationModel,
)

__all__ = [
    "COST231HataPL",
    "Emitter",
    "Environment",
    "FreeSpacePathLoss",
    "LinkParams",
    "LogDistancePL",
    "PathLossResult",
    "Position",
    "PropagationModel",
    "ReceiverConfig",
    "link_params_to_impairments",
    "propagation_presets",
]
