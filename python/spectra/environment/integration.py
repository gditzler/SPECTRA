"""Bridge between Environment link parameters and SPECTRA impairments."""

from __future__ import annotations

import re

import numpy as np

from spectra.environment.core import LinkParams
from spectra.impairments import (
    AWGN,
    DopplerShift,
    RayleighFading,
    RicianFading,
    TDLChannel,
)
from spectra.impairments.base import Transform


def _compute_profile_rms_s(delays_ns, powers_db):
    """Compute the normalized RMS delay spread (seconds) of a TDL profile."""
    d = np.asarray(delays_ns, dtype=float) * 1e-9
    p = 10.0 ** (np.asarray(powers_db, dtype=float) / 10.0)
    p /= p.sum()
    mean_d = float((d * p).sum())
    return float(np.sqrt(((d - mean_d) ** 2 * p).sum()))


# Runtime-computed nominal RMS delay spread for each TDL profile.
# Replaces the previously hand-tabulated values, which differed from the
# actual simplified PROFILES by up to ~3x.
_TDL_NOMINAL_RMS_S: dict[str, float] = {
    name: _compute_profile_rms_s(prof["delays_ns"], prof["powers_db"])
    for name, prof in TDLChannel.PROFILES.items()
    if name.startswith("TDL-")
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
        fading = _scale_tdl_profile(base, params.rms_delay_spread_s, params.k_factor_db)
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
