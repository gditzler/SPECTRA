"""Bridge between Environment link parameters and SPECTRA impairments."""

from __future__ import annotations

import re

from spectra.environment.core import LinkParams
from spectra.impairments import AWGN, DopplerShift, RayleighFading, RicianFading
from spectra.impairments.base import Transform


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

    Order: Doppler (if nonzero) -> Fading (if suggested) -> AWGN (always last).
    """
    impairments: list[Transform] = []

    if abs(params.doppler_hz) > 0.01:
        impairments.append(DopplerShift(fd_hz=params.doppler_hz))

    if params.fading_suggestion is not None:
        impairments.append(_fading_from_suggestion(params.fading_suggestion))

    impairments.append(AWGN(snr=params.snr_db))

    return impairments
