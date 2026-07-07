"""Physical-unit -> sample-domain resolution for waveform parameters.

Spec: docs/superpowers/specs/2026-07-02-waveform-realism-design.md.
Rule: generate at the nearest integer samples-per-symbol when the implied
symbol-rate error is <= RATE_TOLERANCE; otherwise generate at
ceil(sample_rate / symbol_rate) sps and rational-resample to the exact rate.
"""

import math
from fractions import Fraction
from typing import Tuple

import numpy as np

from spectra.utils.dsp import multistage_resampler

RATE_TOLERANCE = 0.01


def resolve_symbol_rate(
    sample_rate: float, symbol_rate: float, tol: float = RATE_TOLERANCE
) -> Tuple[int, int, int]:
    """Map a physical symbol rate to ``(sps, up, down)``.

    Generate at integer ``sps``; when ``up == down == 1`` no resampling is
    needed, otherwise resample the generated signal by ``up/down`` so the
    symbol rate at ``sample_rate`` matches ``symbol_rate``.
    """
    if symbol_rate <= 0:
        raise ValueError(f"symbol_rate must be positive, got {symbol_rate}")
    exact = sample_rate / symbol_rate
    if exact < 2.0:
        raise ValueError(
            f"symbol_rate {symbol_rate:g} Hz needs >= 2 samples/symbol "
            f"at sample_rate {sample_rate:g} Hz"
        )
    nearest = round(exact)
    if nearest >= 2 and abs(nearest - exact) / exact <= tol:
        return nearest, 1, 1
    sps = max(2, math.ceil(exact))
    # Generated rate is symbol_rate * sps >= sample_rate; resample down by
    # the rational approximation of sample_rate / generated_rate.
    frac = Fraction(exact / sps).limit_denominator(1000)
    return sps, frac.numerator, frac.denominator


def resample_to_rate(iq: np.ndarray, up: int, down: int) -> np.ndarray:
    """Rational-resample ``iq`` by ``up/down``; identity when ``up == down == 1``."""
    if up == down:
        return iq
    return multistage_resampler(iq, up, down).astype(np.complex64)
