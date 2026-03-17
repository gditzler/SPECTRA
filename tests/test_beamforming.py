# tests/test_beamforming.py
"""Tests for delay-and-sum, MVDR, and LCMV beamformers."""
import numpy as np
import pytest
from spectra.arrays.array import ula


def _snapshot(azimuth_rad: float, num_elements: int = 8, snr_db: float = 20.0,
              num_snapshots: int = 512, seed: int = 0) -> np.ndarray:
    """Single-source snapshot matrix (N_elem, T)."""
    rng = np.random.default_rng(seed)
    arr = ula(num_elements=num_elements, spacing=0.5, frequency=1e9)
    sv = arr.steering_vector(azimuth=azimuth_rad, elevation=0.0)
    signal = rng.standard_normal(num_snapshots) + 1j * rng.standard_normal(num_snapshots)
    snr_linear = 10 ** (snr_db / 10.0)
    scale = np.sqrt(snr_linear / np.mean(np.abs(signal) ** 2))
    noise = np.sqrt(0.5) * (rng.standard_normal((num_elements, num_snapshots))
                            + 1j * rng.standard_normal((num_elements, num_snapshots)))
    return sv[:, np.newaxis] * (signal * scale)[np.newaxis, :] + noise


def test_das_output_shape():
    from spectra.algorithms.beamforming import delay_and_sum
    arr = ula(num_elements=8, spacing=0.5, frequency=1e9)
    X = _snapshot(np.deg2rad(45.0))
    out = delay_and_sum(X, arr, target_az=np.deg2rad(45.0))
    assert out.shape == (512,)
    assert out.dtype == complex


def test_das_gain_at_target():
    """DAS output power should be higher when steered to true source than away from it."""
    from spectra.algorithms.beamforming import delay_and_sum
    arr = ula(num_elements=8, spacing=0.5, frequency=1e9)
    X = _snapshot(np.deg2rad(60.0), snr_db=15.0)
    out_on  = delay_and_sum(X, arr, target_az=np.deg2rad(60.0))
    out_off = delay_and_sum(X, arr, target_az=np.deg2rad(120.0))
    assert np.mean(np.abs(out_on)**2) > np.mean(np.abs(out_off)**2)


def test_mvdr_output_shape():
    from spectra.algorithms.beamforming import mvdr
    arr = ula(num_elements=8, spacing=0.5, frequency=1e9)
    X = _snapshot(np.deg2rad(45.0))
    out = mvdr(X, arr, target_az=np.deg2rad(45.0))
    assert out.shape == (512,)
    assert out.dtype == complex


def test_mvdr_distortionless_constraint():
    """MVDR weights must satisfy |w^H a| ≈ 1 at the target direction."""
    from spectra.algorithms.beamforming import _mvdr_weights
    arr = ula(num_elements=8, spacing=0.5, frequency=1e9)
    X = _snapshot(np.deg2rad(45.0))
    w = _mvdr_weights(X, arr, target_az=np.deg2rad(45.0))
    a = arr.steering_vector(azimuth=np.deg2rad(45.0), elevation=0.0)
    response = float(np.abs(w.conj() @ a))
    assert abs(response - 1.0) < 0.05, f"MVDR distortionless constraint violated: |w^H a| = {response:.4f}"


def test_lcmv_output_shape():
    from spectra.algorithms.beamforming import lcmv
    arr = ula(num_elements=8, spacing=0.5, frequency=1e9)
    X = _snapshot(np.deg2rad(45.0))
    out = lcmv(X, arr, constraints=[(np.deg2rad(45.0), 0.0)], responses=[1.0+0j])
    assert out.shape == (512,)


def test_lcmv_null_steering():
    """LCMV with desired=1 at target and null at interference should suppress interference."""
    from spectra.algorithms.beamforming import lcmv
    arr = ula(num_elements=8, spacing=0.5, frequency=1e9)
    az_sig = np.deg2rad(50.0)
    az_int = np.deg2rad(100.0)
    X_sig = _snapshot(az_sig, snr_db=10.0, seed=1)
    X_int = _snapshot(az_int, snr_db=20.0, seed=2)  # strong interferer
    X = X_sig + X_int
    constraints = [(az_sig, 0.0), (az_int, 0.0)]
    responses = [1.0 + 0j, 0.0 + 0j]
    w_lcmv = lcmv(X, arr, constraints=constraints, responses=responses, return_weights=True)
    a_int = arr.steering_vector(azimuth=az_int, elevation=0.0)
    null_depth = float(np.abs(w_lcmv.conj() @ a_int))
    assert null_depth < 0.1, f"LCMV null depth {null_depth:.4f} should be < 0.1"


def test_compute_beam_pattern():
    from spectra.algorithms.beamforming import compute_beam_pattern
    arr = ula(num_elements=8, spacing=0.5, frequency=1e9)
    scan = np.linspace(0, np.pi, 181)
    weights = np.ones(8, dtype=complex) / 8
    pattern = compute_beam_pattern(weights, arr, scan)
    assert pattern.shape == (181,)
    assert pattern.dtype == float
    assert 0.0 <= pattern.max() <= 1.0 + 1e-6
