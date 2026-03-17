# python/spectra/algorithms/doa.py
"""Direction-of-arrival estimation algorithms: MUSIC and ESPRIT.

Both algorithms operate on a complex snapshot matrix ``X`` of shape
``(N_elements, T)`` — the output of
:class:`~spectra.transforms.snapshot.ToSnapshotMatrix`.

MUSIC works with any :class:`~spectra.arrays.array.AntennaArray`.
ESPRIT requires a ULA (shift-invariant array) and the ``spacing`` parameter
must match the array's inter-element spacing in wavelengths.
"""

import numpy as np

from spectra.arrays.array import AntennaArray


def music(
    X: np.ndarray,
    num_sources: int,
    array: AntennaArray,
    scan_angles: np.ndarray,
    elevation: float = 0.0,
) -> np.ndarray:
    """MUSIC pseudospectrum over a 1-D azimuth scan.

    Args:
        X: Complex snapshot matrix, shape ``(N_elements, T)``.
        num_sources: Number of signal sources (rank of signal subspace).
        array: :class:`~spectra.arrays.array.AntennaArray` used for steering
            vector computation.
        scan_angles: 1-D array of candidate azimuth angles in radians to
            evaluate.
        elevation: Fixed elevation angle in radians for the scan (default 0).

    Returns:
        Pseudospectrum values at each scan angle, shape ``(len(scan_angles),)``.
        Peaks correspond to estimated source azimuths.
    """
    N, T = X.shape
    R = (X @ X.conj().T) / T
    _, U = np.linalg.eigh(R)          # eigenvalues ascending, U columns = eigenvectors
    En = U[:, : N - num_sources]       # noise subspace: (N, N-K)

    spectrum = np.empty(len(scan_angles))
    for i, az in enumerate(scan_angles):
        a = array.steering_vector(azimuth=az, elevation=elevation)   # (N,)
        proj = a.conj() @ En                                          # (N-K,)
        denom = float(np.real(proj @ proj.conj()))
        spectrum[i] = 1.0 / (denom + 1e-30)

    return spectrum


def esprit(
    X: np.ndarray,
    num_sources: int,
    spacing: float = 0.5,
) -> np.ndarray:
    """ESPRIT DoA estimates for a Uniform Linear Array.

    Implements the standard ESPRIT algorithm assuming a ULA along the x-axis.
    In SPECTRA's coordinate convention azimuth is measured from the x-axis, so
    the inter-element phase shift at elevation 0 is ``2π · spacing · cos(az)``.

    Args:
        X: Complex snapshot matrix, shape ``(N_elements, T)``.
        num_sources: Number of signal sources.
        spacing: ULA inter-element spacing in **wavelengths** (must match the
            array used to generate ``X``). Default is 0.5 (λ/2).

    Returns:
        Estimated azimuth angles in radians, shape ``(num_sources,)``, sorted
        ascending. Due to the cosine ambiguity of a ULA, returned values are
        in ``[0, π]``.
    """
    N, T = X.shape
    R = (X @ X.conj().T) / T
    _, U = np.linalg.eigh(R)           # ascending eigenvalues
    Es = U[:, N - num_sources:]         # signal subspace: (N, K)

    Es1 = Es[:-1, :]                    # lower subarray: (N-1, K)
    Es2 = Es[1:, :]                     # upper subarray: (N-1, K)

    # Solve Es2 ≈ Es1 @ Phi via least squares
    Phi, _, _, _ = np.linalg.lstsq(Es1, Es2, rcond=None)

    lam = np.linalg.eigvals(Phi)        # (K,) complex eigenvalues

    # phase = 2π · spacing · cos(az)  →  cos(az) = angle(λ) / (2π · spacing)
    cos_az = np.angle(lam) / (2.0 * np.pi * spacing)
    cos_az = np.clip(cos_az.real, -1.0, 1.0)
    azimuths = np.arccos(cos_az)        # in [0, π]

    return np.sort(azimuths)


def capon(
    X: np.ndarray,
    array: AntennaArray,
    scan_angles: np.ndarray,
    elevation: float = 0.0,
    diagonal_loading: float = 1e-6,
) -> np.ndarray:
    """Capon (MVDR) minimum-variance pseudospectrum over a 1-D azimuth scan.

    The Capon beamformer minimises output power subject to a distortionless
    response constraint at each scan angle.  It generally achieves finer
    spatial resolution than MUSIC at moderate SNR and does not require knowing
    the number of sources.

    Args:
        X: Complex snapshot matrix, shape ``(N_elements, T)``.
        array: :class:`~spectra.arrays.array.AntennaArray` for steering vectors.
        scan_angles: 1-D array of candidate azimuth angles in radians.
        elevation: Fixed elevation angle in radians (default 0).
        diagonal_loading: Regularisation added to diagonal of R before
            inversion (prevents ill-conditioning). Default 1e-6.

    Returns:
        Pseudospectrum values, shape ``(len(scan_angles),)``.
    """
    N, T = X.shape
    R = (X @ X.conj().T) / T
    R_reg = R + diagonal_loading * np.eye(N)
    R_inv = np.linalg.inv(R_reg)

    spectrum = np.empty(len(scan_angles))
    for i, az in enumerate(scan_angles):
        a = array.steering_vector(azimuth=az, elevation=elevation)  # (N,)
        denom = float(np.real(a.conj() @ R_inv @ a))
        spectrum[i] = 1.0 / (denom + 1e-30)

    return spectrum


def root_music(
    X: np.ndarray,
    num_sources: int,
    spacing: float = 0.5,
) -> np.ndarray:
    """Root-MUSIC DoA estimates for a Uniform Linear Array.

    Avoids spatial scanning by forming the MUSIC polynomial and finding its
    roots.  The signal roots are the ``num_sources`` roots of the noise-subspace
    polynomial that lie closest to the unit circle.

    Works only with ULAs (shift-invariant arrays along the x-axis).

    Args:
        X: Complex snapshot matrix, shape ``(N_elements, T)``.
        num_sources: Number of signal sources.
        spacing: ULA inter-element spacing in wavelengths. Default 0.5.

    Returns:
        Estimated azimuth angles in radians, shape ``(num_sources,)``,
        sorted ascending, values in ``[0, π]``.
    """
    N, T = X.shape
    R = (X @ X.conj().T) / T
    _, U = np.linalg.eigh(R)          # ascending eigenvalues
    En = U[:, : N - num_sources]       # noise subspace (N, N-K)
    C = En @ En.conj().T               # noise projection (N, N)

    # Form polynomial coefficients: c[k] = trace of k-th diagonal of C
    # Polynomial degree 2*(N-1), coefficients indexed -(N-1) … (N-1)
    coeffs = np.zeros(2 * N - 1, dtype=complex)
    for k in range(-(N - 1), N):
        coeffs[k + N - 1] = np.sum(np.diag(C, k))

    # np.roots expects highest degree first; flip to put z^{2(N-1)} first
    roots = np.roots(coeffs[::-1])

    # For each signal, the Hermitian-symmetric polynomial has a conjugate-reciprocal
    # root pair (z, 1/z*) straddling the unit circle.  Restrict to roots inside the
    # unit circle to pick one root per pair, then take the K closest to |z|=1.
    inside = np.abs(roots) < 1.0
    candidates = roots[inside]
    if len(candidates) < num_sources:
        candidates = roots  # fallback: use all roots
    order = np.argsort(np.abs(np.abs(candidates) - 1.0))
    signal_roots = candidates[order][:num_sources]

    # cos(az) = angle(z) / (2π·d)
    cos_az = np.angle(signal_roots) / (2.0 * np.pi * spacing)
    cos_az = np.clip(cos_az.real, -1.0, 1.0)
    return np.sort(np.arccos(cos_az))


def find_peaks_doa(
    spectrum: np.ndarray,
    scan_angles: np.ndarray,
    num_peaks: int,
) -> np.ndarray:
    """Return the ``num_peaks`` highest local-maximum angles from a DoA spectrum.

    Args:
        spectrum: 1-D pseudospectrum values (e.g. MUSIC output).
        scan_angles: Corresponding angle values in radians.
        num_peaks: Number of peaks to return.

    Returns:
        Angle values at the ``num_peaks`` largest peaks, sorted ascending.
    """
    # Find local maxima: strictly greater than both neighbours
    is_peak = np.zeros(len(spectrum), dtype=bool)
    is_peak[1:-1] = (spectrum[1:-1] > spectrum[:-2]) & (spectrum[1:-1] > spectrum[2:])
    # Also check endpoints
    is_peak[0] = spectrum[0] > spectrum[1]
    is_peak[-1] = spectrum[-1] > spectrum[-2]

    peak_indices = np.where(is_peak)[0]
    if len(peak_indices) == 0:
        return scan_angles[np.argsort(spectrum)[-num_peaks:]]

    # Sort peak indices by spectrum value descending, take top num_peaks
    top_indices = peak_indices[np.argsort(spectrum[peak_indices])[::-1][:num_peaks]]
    return np.sort(scan_angles[top_indices])
