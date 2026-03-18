# python/spectra/algorithms/mti.py
"""Moving Target Indication (MTI) algorithms.

Stateless functions for clutter suppression in radar pulse trains:

- :func:`single_pulse_canceller` — first-order high-pass (nulls DC).
- :func:`double_pulse_canceller` — second-order high-pass (deeper DC null).
- :func:`doppler_filter_bank` — FFT-based range-Doppler map.
"""

from __future__ import annotations

from typing import Optional

import numpy as np


def single_pulse_canceller(pulses: np.ndarray) -> np.ndarray:
    """First-order pulse canceller: ``y[n] = x[n+1] - x[n]``.

    Args:
        pulses: Complex pulse matrix, shape ``(num_pulses, num_range_bins)``.

    Returns:
        Cancelled output, shape ``(num_pulses - 1, num_range_bins)``.
    """
    return pulses[1:] - pulses[:-1]


def double_pulse_canceller(pulses: np.ndarray) -> np.ndarray:
    """Second-order pulse canceller: ``y[n] = x[n+2] - 2*x[n+1] + x[n]``.

    Args:
        pulses: Complex pulse matrix, shape ``(num_pulses, num_range_bins)``.

    Returns:
        Cancelled output, shape ``(num_pulses - 2, num_range_bins)``.
    """
    return pulses[2:] - 2 * pulses[1:-1] + pulses[:-2]


def doppler_filter_bank(
    pulses: np.ndarray,
    num_doppler_bins: Optional[int] = None,
    window: str = "hann",
) -> np.ndarray:
    """FFT-based Doppler filter bank producing a range-Doppler map.

    Args:
        pulses: Complex pulse matrix, shape ``(num_pulses, num_range_bins)``.
        num_doppler_bins: FFT size. Defaults to ``num_pulses``.
        window: Window function (``"hann"``, ``"hamming"``, ``"rect"``).

    Returns:
        Range-Doppler power map (magnitude squared), shape
        ``(num_doppler_bins, num_range_bins)``.
    """
    num_pulses, num_range_bins = pulses.shape
    if num_doppler_bins is None:
        num_doppler_bins = num_pulses

    if window == "hann":
        w = np.hanning(num_pulses)
    elif window == "hamming":
        w = np.hamming(num_pulses)
    elif window == "rect":
        w = np.ones(num_pulses)
    else:
        raise ValueError(f"Unsupported window: {window!r}")

    windowed = pulses * w[:, np.newaxis]
    fft_out = np.fft.fft(windowed, n=num_doppler_bins, axis=0)
    return np.abs(fft_out) ** 2
