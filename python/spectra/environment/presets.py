"""Pre-configured propagation model instances for common scenarios."""

from spectra.environment.propagation import (
    ITU_R_P1411,
    COST231HataPL,
    FreeSpacePathLoss,
    GPP38901InH,
    GPP38901RMa,
    GPP38901UMa,
    GPP38901UMi,
    LogDistancePL,
    OkumuraHataPL,
    PropagationModel,
)

propagation_presets: dict[str, PropagationModel] = {
    # Existing presets
    "free_space": FreeSpacePathLoss(),
    "urban_macro": LogDistancePL(n=3.5, sigma_db=8.0),
    "suburban": LogDistancePL(n=3.0, sigma_db=6.0),
    "indoor_office": LogDistancePL(n=2.0, sigma_db=4.0),
    "cost231_urban": COST231HataPL(h_bs_m=30, h_ms_m=1.5, environment="urban"),
    # New 38.901 presets
    "urban_macro_5g": GPP38901UMa(h_bs_m=25.0, h_ut_m=1.5),
    "urban_micro_mmwave": GPP38901UMi(h_bs_m=10.0, h_ut_m=1.5),
    "rural_macro_5g": GPP38901RMa(h_bs_m=35.0, h_ut_m=1.5),
    "indoor_office_5g": GPP38901InH(h_bs_m=3.0, h_ut_m=1.0, variant="mixed_office"),
    # New Hata-family preset
    "urban_hata_4g": OkumuraHataPL(
        h_bs_m=50.0, h_ms_m=1.5, environment="urban_small_medium"
    ),
    # New ITU-R preset
    "short_range_urban": ITU_R_P1411(environment="urban_high_rise"),
}
