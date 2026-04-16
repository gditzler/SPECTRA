"""Pre-configured propagation model instances for common scenarios."""

from spectra.environment.propagation import (
    COST231HataPL,
    FreeSpacePathLoss,
    LogDistancePL,
    PropagationModel,
)

propagation_presets: dict[str, PropagationModel] = {
    "free_space": FreeSpacePathLoss(),
    "urban_macro": LogDistancePL(n=3.5, sigma_db=8.0),
    "suburban": LogDistancePL(n=3.0, sigma_db=6.0),
    "indoor_office": LogDistancePL(n=2.0, sigma_db=4.0),
    "cost231_urban": COST231HataPL(h_bs_m=30, h_ms_m=1.5, environment="urban"),
}
