"""Tests for Swerling RCS fluctuation models."""
import numpy as np
import pytest


def test_non_fluctuating_shape():
    from spectra.targets.rcs import NonFluctuatingRCS
    rcs = NonFluctuatingRCS(sigma=1.0)
    rng = np.random.default_rng(42)
    amps = rcs.amplitudes(num_dwells=5, num_pulses_per_dwell=10, rng=rng)
    assert amps.shape == (5, 10)


def test_non_fluctuating_constant():
    from spectra.targets.rcs import NonFluctuatingRCS
    rcs = NonFluctuatingRCS(sigma=4.0)
    rng = np.random.default_rng(0)
    amps = rcs.amplitudes(num_dwells=3, num_pulses_per_dwell=8, rng=rng)
    assert np.allclose(amps, 2.0)


def test_swerling_i_constant_within_dwell():
    from spectra.targets.rcs import SwerlingRCS
    rcs = SwerlingRCS(case=1, sigma=1.0)
    rng = np.random.default_rng(42)
    amps = rcs.amplitudes(num_dwells=10, num_pulses_per_dwell=20, rng=rng)
    assert amps.shape == (10, 20)
    for row in range(10):
        assert np.allclose(amps[row, :], amps[row, 0])


def test_swerling_ii_varies_pulse_to_pulse():
    from spectra.targets.rcs import SwerlingRCS
    rcs = SwerlingRCS(case=2, sigma=1.0)
    rng = np.random.default_rng(42)
    amps = rcs.amplitudes(num_dwells=1, num_pulses_per_dwell=100, rng=rng)
    assert amps.shape == (1, 100)
    assert not np.allclose(amps[0, :], amps[0, 0])


def test_swerling_iii_constant_within_dwell():
    from spectra.targets.rcs import SwerlingRCS
    rcs = SwerlingRCS(case=3, sigma=1.0)
    rng = np.random.default_rng(42)
    amps = rcs.amplitudes(num_dwells=5, num_pulses_per_dwell=16, rng=rng)
    for row in range(5):
        assert np.allclose(amps[row, :], amps[row, 0])


def test_swerling_iv_varies_pulse_to_pulse():
    from spectra.targets.rcs import SwerlingRCS
    rcs = SwerlingRCS(case=4, sigma=1.0)
    rng = np.random.default_rng(42)
    amps = rcs.amplitudes(num_dwells=1, num_pulses_per_dwell=100, rng=rng)
    assert not np.allclose(amps[0, :], amps[0, 0])


def test_swerling_mean_scales_with_sigma():
    from spectra.targets.rcs import SwerlingRCS
    sigma = 5.0
    rcs = SwerlingRCS(case=2, sigma=sigma)
    rng = np.random.default_rng(42)
    amps = rcs.amplitudes(num_dwells=1, num_pulses_per_dwell=10000, rng=rng)
    mean_power = np.mean(amps**2)
    assert abs(mean_power - sigma) / sigma < 0.1, f"Mean power {mean_power:.2f} != {sigma}"


def test_swerling_invalid_case():
    from spectra.targets.rcs import SwerlingRCS
    with pytest.raises(ValueError, match="case"):
        SwerlingRCS(case=5, sigma=1.0)
