"""Tests for MTI pulse cancellers and Doppler filter bank."""
import numpy as np
import pytest


def _make_clutter_plus_target(num_pulses=32, num_range_bins=64, target_bin=30,
                               target_doppler_hz=500.0, prf=1000.0):
    """Pulse matrix with zero-Doppler clutter + a moving target."""
    X = np.ones((num_pulses, num_range_bins), dtype=complex) * 5.0
    for n in range(num_pulses):
        phase = np.exp(1j * 2 * np.pi * target_doppler_hz * n / prf)
        X[n, target_bin] += 10.0 * phase
    return X


def test_single_canceller_shape():
    from spectra.algorithms.mti import single_pulse_canceller
    X = np.zeros((16, 64), dtype=complex)
    out = single_pulse_canceller(X)
    assert out.shape == (15, 64)


def test_single_canceller_removes_dc():
    from spectra.algorithms.mti import single_pulse_canceller
    X = np.ones((32, 10), dtype=complex) * 100.0
    out = single_pulse_canceller(X)
    assert np.max(np.abs(out)) < 1e-10


def test_single_canceller_passes_moving_target():
    from spectra.algorithms.mti import single_pulse_canceller
    X = _make_clutter_plus_target()
    out = single_pulse_canceller(X)
    target_power = np.mean(np.abs(out[:, 30]) ** 2)
    clutter_power = np.mean(np.abs(out[:, 0]) ** 2)
    assert target_power > clutter_power * 10


def test_double_canceller_shape():
    from spectra.algorithms.mti import double_pulse_canceller
    X = np.zeros((16, 64), dtype=complex)
    out = double_pulse_canceller(X)
    assert out.shape == (14, 64)


def test_double_canceller_removes_dc():
    from spectra.algorithms.mti import double_pulse_canceller
    X = np.ones((32, 10), dtype=complex) * 100.0
    out = double_pulse_canceller(X)
    assert np.max(np.abs(out)) < 1e-10


def test_doppler_filter_bank_shape():
    from spectra.algorithms.mti import doppler_filter_bank
    X = np.zeros((16, 64), dtype=complex)
    rdm = doppler_filter_bank(X)
    assert rdm.shape == (16, 64)


def test_doppler_filter_bank_zero_padded():
    from spectra.algorithms.mti import doppler_filter_bank
    X = np.zeros((16, 64), dtype=complex)
    rdm = doppler_filter_bank(X, num_doppler_bins=32)
    assert rdm.shape == (32, 64)


def test_doppler_filter_bank_peak_at_target():
    from spectra.algorithms.mti import doppler_filter_bank
    X = _make_clutter_plus_target(num_pulses=64, target_doppler_hz=200.0, prf=1000.0)
    rdm = doppler_filter_bank(X, window="hann")
    target_col = rdm[:, 30]
    dc_bin = 0
    peak_bin = np.argmax(target_col)
    assert peak_bin != dc_bin, "Target Doppler peak should not be at DC"


def test_doppler_filter_bank_returns_power():
    from spectra.algorithms.mti import doppler_filter_bank
    X = np.random.default_rng(0).standard_normal((16, 32)) + 0j
    rdm = doppler_filter_bank(X)
    assert np.all(rdm >= 0), "Power should be non-negative"
