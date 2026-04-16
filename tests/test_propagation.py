"""Tests for propagation models."""

import math

import numpy as np
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


from spectra.environment.propagation import LogDistancePL


class TestLogDistancePL:
    def test_is_propagation_model(self):
        assert isinstance(LogDistancePL(), PropagationModel)

    def test_n2_matches_free_space(self):
        log_model = LogDistancePL(n=2.0, sigma_db=0.0, d0=1.0)
        fs_model = FreeSpacePathLoss()
        freq = 2.4e9
        for d in [100.0, 500.0, 1000.0]:
            log_result = log_model(distance_m=d, freq_hz=freq)
            fs_result = fs_model(distance_m=d, freq_hz=freq)
            assert math.isclose(log_result.path_loss_db, fs_result.path_loss_db, rel_tol=1e-4)

    def test_higher_exponent_more_loss(self):
        m1 = LogDistancePL(n=2.0, sigma_db=0.0)
        m2 = LogDistancePL(n=3.5, sigma_db=0.0)
        r1 = m1(distance_m=500.0, freq_hz=2.4e9)
        r2 = m2(distance_m=500.0, freq_hz=2.4e9)
        assert r2.path_loss_db > r1.path_loss_db

    def test_zero_sigma_no_shadow_fading(self):
        model = LogDistancePL(n=3.0, sigma_db=0.0)
        result = model(distance_m=500.0, freq_hz=1e9)
        assert result.shadow_fading_db == 0.0

    def test_nonzero_sigma_produces_shadow_fading(self):
        model = LogDistancePL(n=3.0, sigma_db=8.0)
        result = model(distance_m=500.0, freq_hz=1e9, seed=42)
        assert result.shadow_fading_db != 0.0

    def test_shadow_fading_deterministic_with_seed(self):
        model = LogDistancePL(n=3.0, sigma_db=8.0)
        r1 = model(distance_m=500.0, freq_hz=1e9, seed=42)
        r2 = model(distance_m=500.0, freq_hz=1e9, seed=42)
        assert r1.shadow_fading_db == r2.shadow_fading_db

    def test_shadow_fading_varies_without_seed(self):
        model = LogDistancePL(n=3.0, sigma_db=8.0)
        results = [model(distance_m=500.0, freq_hz=1e9).shadow_fading_db for _ in range(20)]
        assert len(set(results)) > 1

    def test_minimum_distance_clamp(self):
        model = LogDistancePL(n=3.0)
        with pytest.raises(ValueError, match="distance_m must be positive"):
            model(distance_m=0.0, freq_hz=1e9)

    def test_shadow_included_in_path_loss(self):
        model = LogDistancePL(n=3.0, sigma_db=8.0)
        result = model(distance_m=500.0, freq_hz=1e9, seed=42)
        model_no_shadow = LogDistancePL(n=3.0, sigma_db=0.0)
        result_no_shadow = model_no_shadow(distance_m=500.0, freq_hz=1e9)
        expected = result_no_shadow.path_loss_db + result.shadow_fading_db
        assert math.isclose(result.path_loss_db, expected, rel_tol=1e-6)


from spectra.environment.propagation import COST231HataPL


class TestCOST231HataPL:
    def test_is_propagation_model(self):
        assert isinstance(COST231HataPL(), PropagationModel)

    def test_urban_more_loss_than_suburban(self):
        urban = COST231HataPL(environment="urban")
        suburban = COST231HataPL(environment="suburban")
        r_urban = urban(distance_m=1000.0, freq_hz=1800e6)
        r_suburban = suburban(distance_m=1000.0, freq_hz=1800e6)
        assert r_urban.path_loss_db > r_suburban.path_loss_db

    def test_suburban_more_loss_than_rural(self):
        suburban = COST231HataPL(environment="suburban")
        rural = COST231HataPL(environment="rural")
        r_sub = suburban(distance_m=1000.0, freq_hz=1800e6)
        r_rural = rural(distance_m=1000.0, freq_hz=1800e6)
        assert r_sub.path_loss_db > r_rural.path_loss_db

    def test_farther_distance_more_loss(self):
        model = COST231HataPL()
        r1 = model(distance_m=1000.0, freq_hz=1800e6)
        r2 = model(distance_m=5000.0, freq_hz=1800e6)
        assert r2.path_loss_db > r1.path_loss_db

    def test_higher_frequency_more_loss(self):
        model = COST231HataPL()
        r1 = model(distance_m=1000.0, freq_hz=1500e6)
        r2 = model(distance_m=1000.0, freq_hz=2000e6)
        assert r2.path_loss_db > r1.path_loss_db

    def test_reasonable_range_1km_1800mhz(self):
        model = COST231HataPL(h_bs_m=30, h_ms_m=1.5, environment="urban")
        result = model(distance_m=1000.0, freq_hz=1800e6)
        assert 110 < result.path_loss_db < 160

    def test_invalid_environment(self):
        with pytest.raises(ValueError, match="environment must be"):
            COST231HataPL(environment="space")

    def test_minimum_distance_clamp(self):
        model = COST231HataPL()
        with pytest.raises(ValueError, match="distance_m must be positive"):
            model(distance_m=0.0, freq_hz=1800e6)

    def test_no_fading_metadata(self):
        model = COST231HataPL()
        result = model(distance_m=1000.0, freq_hz=1800e6)
        assert result.shadow_fading_db == 0.0
