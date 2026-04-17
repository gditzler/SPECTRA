"""Environment and propagation modeling for SPECTRA."""

from spectra.environment.core import Emitter, Environment, LinkParams, ReceiverConfig
from spectra.environment.integration import link_params_to_impairments
from spectra.environment.position import Position
from spectra.environment.presets import propagation_presets
from spectra.environment.propagation import (
    ITU_R_P525,
    ITU_R_P1411,
    COST231HataPL,
    FreeSpacePathLoss,
    GPP38901InH,
    GPP38901RMa,
    GPP38901UMa,
    GPP38901UMi,
    LogDistancePL,
    OkumuraHataPL,
    PathLossResult,
    PropagationModel,
)

__all__ = [
    "COST231HataPL",
    "Emitter",
    "Environment",
    "FreeSpacePathLoss",
    "GPP38901InH",
    "GPP38901RMa",
    "GPP38901UMa",
    "GPP38901UMi",
    "ITU_R_P525",
    "ITU_R_P1411",
    "LinkParams",
    "LogDistancePL",
    "OkumuraHataPL",
    "PathLossResult",
    "Position",
    "PropagationModel",
    "ReceiverConfig",
    "link_params_to_impairments",
    "propagation_presets",
]
