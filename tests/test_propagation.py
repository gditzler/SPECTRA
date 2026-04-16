"""Tests for propagation models."""

import math

import pytest

from spectra.environment.propagation import (
    FreeSpacePathLoss,
    PathLossResult,
    PropagationModel,
)

SPEED_OF_LIGHT = 299_792_458.0


class TestPathLossResult:
    def test_defaults(self):
        r = PathLossResult(path_loss_db=100.0)
        assert r.path_loss_db == 100.0
        assert r.shadow_fading_db == 0.0
        assert r.rms_delay_spread_s is None
        assert r.k_factor_db is None
        assert r.angular_spread_deg is None


class TestFreeSpacePathLoss:
    def test_is_propagation_model(self):
        assert isinstance(FreeSpacePathLoss(), PropagationModel)

    def test_1km_2400mhz(self):
        """Free-space PL at 1 km, 2.4 GHz should be ~100 dB."""
        model = FreeSpacePathLoss()
        result = model(distance_m=1000.0, freq_hz=2.4e9)
        expected = (
            20 * math.log10(1000.0)
            + 20 * math.log10(2.4e9)
            + 20 * math.log10(4 * math.pi / SPEED_OF_LIGHT)
        )
        assert math.isclose(result.path_loss_db, expected, rel_tol=1e-6)
        assert 99.0 < result.path_loss_db < 101.0

    def test_no_shadow_fading(self):
        model = FreeSpacePathLoss()
        result = model(distance_m=500.0, freq_hz=1e9)
        assert result.shadow_fading_db == 0.0

    def test_no_fading_metadata(self):
        model = FreeSpacePathLoss()
        result = model(distance_m=500.0, freq_hz=1e9)
        assert result.k_factor_db is None
        assert result.rms_delay_spread_s is None

    def test_inverse_square_law(self):
        """Doubling distance adds ~6 dB of path loss."""
        model = FreeSpacePathLoss()
        r1 = model(distance_m=100.0, freq_hz=1e9)
        r2 = model(distance_m=200.0, freq_hz=1e9)
        assert math.isclose(r2.path_loss_db - r1.path_loss_db, 20 * math.log10(2), rel_tol=1e-6)

    def test_minimum_distance_clamp(self):
        """Distance <= 0 should raise ValueError."""
        model = FreeSpacePathLoss()
        with pytest.raises(ValueError, match="distance_m must be positive"):
            model(distance_m=0.0, freq_hz=1e9)

    def test_very_small_distance(self):
        """Very small but positive distance should not raise an exception."""
        model = FreeSpacePathLoss()
        result = model(distance_m=0.01, freq_hz=1e9)
        assert isinstance(result, PathLossResult)
