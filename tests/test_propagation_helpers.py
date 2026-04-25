"""Tests for shared propagation helpers."""

import warnings

import numpy as np
import pytest
from spectra.environment.propagation._base import (
    _check_distance_range,
    _check_freq_range,
    _resolve_los,
)


class TestResolveLOS:
    def test_force_los_returns_true(self):
        rng = np.random.default_rng(0)
        assert _resolve_los("force_los", 0.1, rng) is True

    def test_force_nlos_returns_false(self):
        rng = np.random.default_rng(0)
        assert _resolve_los("force_nlos", 0.9, rng) is False

    def test_stochastic_p1_returns_true(self):
        rng = np.random.default_rng(0)
        assert _resolve_los("stochastic", 1.0, rng) is True

    def test_stochastic_p0_returns_false(self):
        rng = np.random.default_rng(0)
        assert _resolve_los("stochastic", 0.0, rng) is False

    def test_stochastic_reproducible_with_seed(self):
        r1 = _resolve_los("stochastic", 0.5, np.random.default_rng(42))
        r2 = _resolve_los("stochastic", 0.5, np.random.default_rng(42))
        assert r1 == r2

    def test_invalid_mode_raises(self):
        rng = np.random.default_rng(0)
        with pytest.raises(ValueError, match="los_mode"):
            _resolve_los("bogus", 0.5, rng)  # ty: ignore[invalid-argument-type] # intentional invalid Literal


class TestCheckFreqRange:
    def test_in_range_passes(self):
        _check_freq_range(1e9, 500e6, 2e9, "Model")

    def test_below_range_raises(self):
        with pytest.raises(ValueError, match="Model.*freq"):
            _check_freq_range(100e6, 500e6, 2e9, "Model")

    def test_above_range_raises(self):
        with pytest.raises(ValueError, match="Model.*freq"):
            _check_freq_range(5e9, 500e6, 2e9, "Model")

    def test_non_strict_below_warns(self):
        with pytest.warns(UserWarning, match="Model"):
            _check_freq_range(100e6, 500e6, 2e9, "Model", strict=False)

    def test_non_strict_in_range_no_warning(self):
        with warnings.catch_warnings():
            warnings.simplefilter("error")
            _check_freq_range(1e9, 500e6, 2e9, "Model", strict=False)


class TestCheckDistanceRange:
    def test_in_range_passes(self):
        _check_distance_range(500.0, 10.0, 5000.0, "Model")

    def test_below_range_raises(self):
        with pytest.raises(ValueError, match="Model.*distance"):
            _check_distance_range(1.0, 10.0, 5000.0, "Model")

    def test_above_range_raises(self):
        with pytest.raises(ValueError, match="Model.*distance"):
            _check_distance_range(1e5, 10.0, 5000.0, "Model")

    def test_non_strict_below_warns(self):
        with pytest.warns(UserWarning):
            _check_distance_range(1.0, 10.0, 5000.0, "Model", strict=False)
