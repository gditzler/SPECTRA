"""MIMO antenna array utilities."""

import numpy as np


def steering_vector(n_antennas: int, angle_rad: float, d_lambda: float = 0.5) -> np.ndarray:
    """Uniform Linear Array (ULA) steering vector.

    a(theta) = exp(j * 2 * pi * d * k * sin(theta))

    Args:
        n_antennas: Number of antenna elements.
        angle_rad: Angle of arrival/departure in radians.
        d_lambda: Element spacing in wavelengths. Default 0.5 (half-wavelength).

    Returns:
        Complex array of shape (n_antennas,).
    """
    k = np.arange(n_antennas)
    return np.exp(1j * 2 * np.pi * d_lambda * k * np.sin(angle_rad))


def exponential_correlation(n: int, rho: float) -> np.ndarray:
    """Exponential correlation matrix.

    R[i, j] = rho^|i-j|

    Args:
        n: Matrix dimension (number of antennas).
        rho: Correlation coefficient between adjacent elements (0 <= rho < 1).

    Returns:
        Real symmetric positive semi-definite matrix of shape (n, n).
    """
    indices = np.arange(n)
    return rho ** np.abs(indices[:, None] - indices[None, :])


def kronecker_correlation(R_tx: np.ndarray, R_rx: np.ndarray) -> np.ndarray:
    """Full spatial correlation matrix via Kronecker product.

    R = R_rx (kron) R_tx

    Args:
        R_tx: TX correlation matrix of shape (n_tx, n_tx).
        R_rx: RX correlation matrix of shape (n_rx, n_rx).

    Returns:
        Full correlation matrix of shape (n_rx*n_tx, n_rx*n_tx).
    """
    return np.kron(R_rx, R_tx)
