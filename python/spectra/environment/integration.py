"""Bridge between Environment link parameters and SPECTRA impairments."""

from __future__ import annotations

import re

from spectra.environment.core import LinkParams
from spectra.impairments import (
    AWGN,
    DopplerShift,
    RayleighFading,
    RicianFading,
    TDLChannel,
)
from spectra.impairments.base import Transform

# Reference profile RMS delay spreads (in seconds).
# These are the nominal delay spreads associated with each TDL profile's
# default (unscaled) delays. Computed from PROFILES in tdl_channel.py.
# TDL-A through TDL-E are normalized to 1.0 when nominal_rms is used as the
# divisor — we keep tabulated nominal values for scaling.
_TDL_NOMINAL_RMS_S = {
    "TDL-A": 5.70e-8,  # 57 ns nominal
    "TDL-B": 4.20e-8,  # 42 ns nominal
    "TDL-C": 3.80e-7,  # 380 ns nominal
    "TDL-D": 3.20e-8,  # 32 ns nominal (LOS, K=13.3 dB)
    "TDL-E": 3.00e-8,  # 30 ns nominal (LOS, K=22.0 dB)
}


def _scale_tdl_profile(
    base_profile: str, target_rms_s: float, k_factor_db: float | None
) -> TDLChannel:
    """Return a TDLChannel with delays scaled to `target_rms_s`."""
    base_rms = _TDL_NOMINAL_RMS_S[base_profile]
    scale = target_rms_s / base_rms
    delays = [d * scale for d in TDLChannel.PROFILES[base_profile]["delays_ns"]]
    powers = list(TDLChannel.PROFILES[base_profile]["powers_db"])
    return TDLChannel.custom(
        delays_ns=delays,
        powers_db=powers,
        doppler_hz=5.0,
        k_factor_db=k_factor_db,
    )


def _fading_from_suggestion(suggestion: str) -> Transform:
    """Map a fading suggestion string to a configured impairment instance."""
    if suggestion == "rayleigh":
        return RayleighFading()
    match = re.match(r"rician_k(\d+)", suggestion)
    if match:
        k = float(match.group(1))
        return RicianFading(k_factor=k)
    return RayleighFading()


def link_params_to_impairments(params: LinkParams) -> list[Transform]:
    """Convert derived link parameters to an ordered impairment chain.

    Selection order for the fading stage (first match wins):
      1. `rms_delay_spread_s` populated (38.901-style): emit TDLChannel
         scaled to that delay spread. If `k_factor_db` is also populated,
         use TDL-D base (LOS); otherwise use TDL-B base (NLOS).
      2. `k_factor_db` populated without delay spread: emit RicianFading.
      3. `fading_suggestion` string present (legacy path): map as before.
      4. Otherwise: no fading stage.

    Order: Doppler (if nonzero) -> Fading -> AWGN (always last).
    """
    impairments: list[Transform] = []

    if abs(params.doppler_hz) > 0.01:
        impairments.append(DopplerShift(fd_hz=params.doppler_hz))

    fading: Transform | None = None
    if params.rms_delay_spread_s is not None:
        base = "TDL-D" if params.k_factor_db is not None else "TDL-B"
        fading = _scale_tdl_profile(
            base, params.rms_delay_spread_s, params.k_factor_db
        )
    elif params.k_factor_db is not None:
        # Convert dB to linear
        k_lin = 10.0 ** (params.k_factor_db / 10.0)
        fading = RicianFading(k_factor=k_lin)
    elif params.fading_suggestion is not None:
        fading = _fading_from_suggestion(params.fading_suggestion)

    if fading is not None:
        impairments.append(fading)

    impairments.append(AWGN(snr=params.snr_db))

    return impairments
