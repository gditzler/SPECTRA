# python/spectra/algorithms/beamforming.py
"""Beamforming algorithms: delay-and-sum, MVDR, and LCMV.

All functions accept a complex snapshot matrix ``X`` of shape
``(N_elements, T)`` and return a beamformed time-series of shape ``(T,)``,
except where noted.
"""

from typing import List, Tuple

import numpy as np

from spectra.arrays.array import AntennaArray


def delay_and_sum(
    X: np.ndarray,
    array: AntennaArray,
    target_az: float,
    elevation: float = 0.0,
) -> np.ndarray:
    """Conventional delay-and-sum (phase-shift) beamformer.

    Applies conjugate steering-vector weights normalised by the number of
    elements.  No covariance estimate is needed, so this works even for very
    short snapshots.

    Args:
        X: Complex snapshot matrix, shape ``(N_elements, T)``.
        array: :class:`~spectra.arrays.array.AntennaArray` geometry.
        target_az: Target azimuth in radians.
        elevation: Target elevation in radians (default 0).

    Returns:
        Beamformed complex signal, shape ``(T,)``.
    """
    a = array.steering_vector(azimuth=target_az, elevation=elevation)  # (N,)
    w = a.conj() / array.num_elements
    return w @ X  # (T,)


def _mvdr_weights(
    X: np.ndarray,
    array: AntennaArray,
    target_az: float,
    elevation: float = 0.0,
    diagonal_loading: float = 1e-6,
) -> np.ndarray:
    """Compute MVDR weight vector (internal helper).

    Returns:
        Weight vector, shape ``(N_elements,)``.
    """
    N, T = X.shape
    R = (X @ X.conj().T) / T
    R_reg = R + diagonal_loading * np.eye(N)
    R_inv = np.linalg.inv(R_reg)
    a = array.steering_vector(azimuth=target_az, elevation=elevation)
    R_inv_a = R_inv @ a
    denom = float(np.real(a.conj() @ R_inv_a))
    return R_inv_a / (denom + 1e-30)


def mvdr(
    X: np.ndarray,
    array: AntennaArray,
    target_az: float,
    elevation: float = 0.0,
    diagonal_loading: float = 1e-6,
) -> np.ndarray:
    """Minimum Variance Distortionless Response (MVDR / Capon) beamformer.

    Minimises output power subject to a unit-gain constraint at
    ``target_az``.  Suppresses interference and noise more effectively than
    delay-and-sum when the interference direction differs from the target.

    Args:
        X: Complex snapshot matrix, shape ``(N_elements, T)``.
        array: :class:`~spectra.arrays.array.AntennaArray` geometry.
        target_az: Target azimuth in radians.
        elevation: Target elevation in radians (default 0).
        diagonal_loading: Regularisation for covariance inversion (1e-6).

    Returns:
        Beamformed complex signal, shape ``(T,)``.
    """
    w = _mvdr_weights(X, array, target_az, elevation, diagonal_loading)
    return w.conj() @ X  # (T,)


def lcmv(
    X: np.ndarray,
    array: AntennaArray,
    constraints: List[Tuple[float, float]],
    responses: List[complex],
    diagonal_loading: float = 1e-6,
    return_weights: bool = False,
) -> np.ndarray:
    """Linearly-Constrained Minimum Variance (LCMV) beamformer.

    Generalises MVDR to multiple simultaneous linear constraints.  A common
    use-case is steering a unit-gain beam toward the desired source while
    placing nulls (response = 0) at known interference directions.

    Args:
        X: Complex snapshot matrix, shape ``(N_elements, T)``.
        array: :class:`~spectra.arrays.array.AntennaArray` geometry.
        constraints: List of ``(azimuth_rad, elevation_rad)`` constraint
            directions.  Must have at least one entry.
        responses: Desired complex response at each constraint direction.
            Length must match ``constraints``.  Use ``1+0j`` for unit gain
            and ``0+0j`` for a null.
        diagonal_loading: Regularisation for covariance inversion (1e-6).
        return_weights: If ``True``, return weight vector instead of
            beamformed signal. Useful for pattern analysis or null-depth
            verification.

    Returns:
        Beamformed complex signal shape ``(T,)`` unless ``return_weights=True``,
        in which case returns weight vector shape ``(N_elements,)``.

    Raises:
        ValueError: If ``constraints`` and ``responses`` have different lengths.
    """
    if len(constraints) != len(responses):
        raise ValueError(
            f"constraints ({len(constraints)}) and responses ({len(responses)}) "
            "must have the same length"
        )

    N, T = X.shape
    R = (X @ X.conj().T) / T
    R_reg = R + diagonal_loading * np.eye(N)
    R_inv = np.linalg.inv(R_reg)

    # Constraint matrix C: columns are steering vectors (N, K)
    C = np.column_stack([
        array.steering_vector(azimuth=az, elevation=el)
        for az, el in constraints
    ])
    g = np.asarray(responses, dtype=complex)  # (K,)

    # LCMV solution: w = R^{-1} C (C^H R^{-1} C)^{-1} g
    R_inv_C = R_inv @ C               # (N, K)
    M = C.conj().T @ R_inv_C          # (K, K)
    w = R_inv_C @ np.linalg.solve(M, g)  # (N,)

    if return_weights:
        return w
    return w.conj() @ X              # (T,)


def compute_beam_pattern(
    weights: np.ndarray,
    array: AntennaArray,
    scan_angles: np.ndarray,
    elevation: float = 0.0,
) -> np.ndarray:
    """Evaluate the normalised beam pattern for a fixed weight vector.

    Args:
        weights: Complex weight vector, shape ``(N_elements,)``.
        array: :class:`~spectra.arrays.array.AntennaArray` geometry.
        scan_angles: 1-D array of azimuth angles in radians.
        elevation: Fixed elevation angle in radians.

    Returns:
        Normalised power pattern in ``[0, 1]``, shape ``(len(scan_angles),)``.
    """
    responses = np.array([
        float(np.abs(weights.conj() @ array.steering_vector(azimuth=az, elevation=elevation)))
        for az in scan_angles
    ])
    peak = responses.max()
    if peak > 0:
        responses = responses / peak
    return responses
