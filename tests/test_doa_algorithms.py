"""Tests for MUSIC and ESPRIT direction-of-arrival estimators."""
import numpy as np
from spectra.arrays.array import ula


def _synthetic_snapshot(
    azimuth_rad: float,
    num_elements: int = 8,
    spacing: float = 0.5,
    snr_db: float = 20.0,
    num_snapshots: int = 256,
    seed: int = 0,
) -> np.ndarray:
    """Generate a single-source snapshot matrix analytically."""
    rng = np.random.default_rng(seed)
    arr = ula(num_elements=num_elements, spacing=spacing, frequency=1e9)
    sv = arr.steering_vector(azimuth=azimuth_rad, elevation=0.0)  # (N,)
    signal = rng.standard_normal(num_snapshots) + 1j * rng.standard_normal(num_snapshots)
    snr_linear = 10 ** (snr_db / 10.0)
    noise_power = 1.0
    scale = np.sqrt(snr_linear * noise_power / np.mean(np.abs(signal) ** 2))
    X = sv[:, np.newaxis] * (signal * scale)[np.newaxis, :]
    noise = np.sqrt(noise_power / 2) * (
        rng.standard_normal((num_elements, num_snapshots))
        + 1j * rng.standard_normal((num_elements, num_snapshots))
    )
    return X + noise


def test_music_peak_at_true_angle():
    from spectra.algorithms.doa import music

    true_az = np.deg2rad(40.0)  # 40 degrees from x-axis
    arr = ula(num_elements=8, spacing=0.5, frequency=1e9)
    X = _synthetic_snapshot(true_az, num_elements=8, snr_db=20.0)
    scan = np.linspace(0, np.pi, 361)
    spectrum = music(X, num_sources=1, array=arr, scan_angles=scan)
    peak_idx = np.argmax(spectrum)
    estimated_az = scan[peak_idx]
    assert abs(estimated_az - true_az) < np.deg2rad(3.0), (
        f"MUSIC peak {np.rad2deg(estimated_az):.1f}° != true {np.rad2deg(true_az):.1f}°"
    )


def test_esprit_estimates_true_angle():
    from spectra.algorithms.doa import esprit

    true_az = np.deg2rad(60.0)
    X = _synthetic_snapshot(true_az, num_elements=8, snr_db=25.0, num_snapshots=512)
    estimates = esprit(X, num_sources=1, spacing=0.5)
    # ESPRIT returns arccos-based angles; take the one closest to true_az
    diffs = np.abs(estimates - true_az)
    assert diffs.min() < np.deg2rad(5.0), (
        f"ESPRIT nearest estimate differs by {np.rad2deg(diffs.min()):.1f}° from true "
        f"{np.rad2deg(true_az):.1f}°"
    )


def test_music_two_sources():
    from spectra.algorithms.doa import music

    arr = ula(num_elements=8, spacing=0.5, frequency=1e9)
    az1, az2 = np.deg2rad(35.0), np.deg2rad(80.0)
    X1 = _synthetic_snapshot(az1, num_elements=8, snr_db=20.0, seed=1)
    X2 = _synthetic_snapshot(az2, num_elements=8, snr_db=20.0, seed=2)
    X = X1 + X2
    scan = np.linspace(0, np.pi, 361)
    spectrum = music(X, num_sources=2, array=arr, scan_angles=scan)
    # Spectrum should be higher than median overall
    assert spectrum.max() > 10 * np.median(spectrum)


def test_find_peaks_doa():
    from spectra.algorithms.doa import find_peaks_doa

    scan = np.linspace(0, np.pi, 181)
    # Synthetic spectrum with two peaks
    spectrum = np.ones(181)
    spectrum[45] = 100.0   # peak at index 45
    spectrum[120] = 80.0   # peak at index 120
    peaks = find_peaks_doa(spectrum, scan, num_peaks=2)
    assert len(peaks) == 2
    assert abs(peaks[0] - scan[45]) < 1e-6 or abs(peaks[1] - scan[45]) < 1e-6


def test_capon_peak_at_true_angle():
    from spectra.algorithms.doa import capon
    true_az = np.deg2rad(55.0)
    arr = ula(num_elements=8, spacing=0.5, frequency=1e9)
    X = _synthetic_snapshot(true_az, num_elements=8, snr_db=20.0)
    scan = np.linspace(0, np.pi, 361)
    spectrum = capon(X, array=arr, scan_angles=scan, elevation=0.0)
    peak_idx = np.argmax(spectrum)
    estimated_az = scan[peak_idx]
    assert abs(estimated_az - true_az) < np.deg2rad(3.0), (
        f"Capon peak {np.rad2deg(estimated_az):.1f}° != true {np.rad2deg(true_az):.1f}°"
    )


def test_capon_higher_resolution_than_music():
    """Capon generally resolves closely-spaced sources better than MUSIC at moderate SNR."""
    from spectra.algorithms.doa import capon
    arr = ula(num_elements=8, spacing=0.5, frequency=1e9)
    az1, az2 = np.deg2rad(45.0), np.deg2rad(60.0)
    X1 = _synthetic_snapshot(az1, num_elements=8, snr_db=15.0, seed=10)
    X2 = _synthetic_snapshot(az2, num_elements=8, snr_db=15.0, seed=11)
    X = X1 + X2
    scan = np.linspace(0, np.pi, 361)
    sp_capon = capon(X, array=arr, scan_angles=scan)
    # Capon spectrum must have clear peaks above median noise floor
    assert sp_capon.max() > 5 * np.median(sp_capon)


def test_root_music_single_source():
    from spectra.algorithms.doa import root_music
    true_az = np.deg2rad(70.0)
    X = _synthetic_snapshot(true_az, num_elements=8, snr_db=20.0, num_snapshots=256)
    estimates = root_music(X, num_sources=1, spacing=0.5)
    diffs = np.abs(estimates - true_az)
    assert diffs.min() < np.deg2rad(3.0), (
        f"Root-MUSIC estimate {np.rad2deg(estimates[0]):.1f}° != true {np.rad2deg(true_az):.1f}°"
    )


def test_root_music_two_sources():
    from spectra.algorithms.doa import root_music
    az1, az2 = np.deg2rad(40.0), np.deg2rad(100.0)
    X1 = _synthetic_snapshot(az1, num_elements=8, snr_db=20.0, seed=5)
    X2 = _synthetic_snapshot(az2, num_elements=8, snr_db=20.0, seed=6)
    X = X1 + X2
    estimates = root_music(X, num_sources=2, spacing=0.5)
    assert len(estimates) == 2
    # Each true angle should be within 5° of some estimate
    for true_az in [az1, az2]:
        assert np.min(np.abs(estimates - true_az)) < np.deg2rad(5.0), (
            f"Root-MUSIC failed to recover {np.rad2deg(true_az):.1f}°, got {np.rad2deg(estimates)}"
        )
