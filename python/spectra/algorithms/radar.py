# python/spectra/algorithms/radar.py
"""Radar signal processing: matched filter and CFAR detectors.

These functions operate on 1-D NumPy arrays and are independent of any
specific waveform.  Use them with the output of
:class:`~spectra.datasets.radar.RadarDataset` or with raw captures.
"""

from typing import Optional

import numpy as np


def matched_filter(received: np.ndarray, template: np.ndarray) -> np.ndarray:
    """Compute the matched filter output via correlation.

    The matched filter maximises output SNR for a known signal shape in white
    Gaussian noise.  Implemented as convolution with the time-reversed
    conjugate of the template::

        y[n] = sum_k template*[k] * received[n+k]

    Args:
        received: Received signal, 1-D complex or real, length M.
        template: Reference waveform / pulse replica, 1-D complex or real, length L.

    Returns:
        Matched filter output, length ``M + L - 1``.  Peak location corresponds
        to ``delay + L - 1`` where ``delay`` is the target's range delay in samples.
    """
    h = np.conj(template[::-1])
    return np.convolve(received, h, mode="full")


def ca_cfar(
    power: np.ndarray,
    guard_cells: int,
    training_cells: int,
    pfa: float = 1e-6,
) -> np.ndarray:
    """Cell-Averaging CFAR detector.

    For each cell under test (CUT), estimates the noise power from adjacent
    training cells and compares the CUT to an adaptive threshold designed to
    achieve probability of false alarm ``pfa``.

    Args:
        power: 1-D power profile (e.g. ``|matched_filter(received, template)|**2``).
        guard_cells: Number of guard cells on each side of the CUT.  Guard cells
            are excluded from the noise estimate to avoid target self-masking.
        training_cells: Number of training cells on each side of the guard region.
        pfa: Target probability of false alarm. Default 1e-6.

    Returns:
        Boolean detection mask, same length as ``power``.  ``True`` indicates
        a detection.
    """
    N = len(power)
    n_train = 2 * training_cells
    # Threshold factor for CA-CFAR: alpha = N_train * (P_fa^{-1/N_train} - 1)
    threshold_factor = n_train * (pfa ** (-1.0 / n_train) - 1.0)
    detections = np.zeros(N, dtype=bool)

    for i in range(N):
        left_end   = max(0, i - guard_cells)
        left_start = max(0, left_end - training_cells)
        right_start = min(N, i + guard_cells + 1)
        right_end   = min(N, right_start + training_cells)

        training = np.concatenate([power[left_start:left_end], power[right_start:right_end]])
        if len(training) == 0:
            continue
        threshold = threshold_factor * np.mean(training)
        detections[i] = power[i] > threshold

    return detections


def os_cfar(
    power: np.ndarray,
    guard_cells: int,
    training_cells: int,
    k_rank: Optional[int] = None,
    pfa: float = 1e-6,
) -> np.ndarray:
    """Ordered-Statistics CFAR detector.

    More robust than CA-CFAR in clutter edges and multi-target scenarios.
    Uses the k-th ranked (sorted ascending) training cell as the noise
    reference instead of the mean.

    Args:
        power: 1-D power profile.
        guard_cells: Guard cells on each side of the CUT.
        training_cells: Training cells on each side.
        k_rank: Rank of the order statistic to use (1-indexed).  Defaults to
            ``round(0.75 * 2 * training_cells)``, which gives good performance
            for typical clutter conditions.
        pfa: Target probability of false alarm. Default 1e-6.

    Returns:
        Boolean detection mask, same length as ``power``.
    """
    N = len(power)
    n_train = 2 * training_cells
    if k_rank is None:
        k_rank = max(1, round(0.75 * n_train))
    # OS-CFAR threshold factor (approximate)
    # alpha ≈ k * (pfa^{-1/(n-k)} - 1) using order-statistic result
    n_minus_k = max(1, n_train - k_rank)
    alpha = k_rank * (pfa ** (-1.0 / n_minus_k) - 1.0)

    detections = np.zeros(N, dtype=bool)
    for i in range(N):
        left_end   = max(0, i - guard_cells)
        left_start = max(0, left_end - training_cells)
        right_start = min(N, i + guard_cells + 1)
        right_end   = min(N, right_start + training_cells)

        training = np.concatenate([power[left_start:left_end], power[right_start:right_end]])
        if len(training) == 0:
            continue
        training_sorted = np.sort(training)
        k_idx = min(k_rank - 1, len(training_sorted) - 1)
        threshold = alpha * training_sorted[k_idx]
        detections[i] = power[i] > threshold

    return detections
