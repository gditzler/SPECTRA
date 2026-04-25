"""Tests for propagation models."""

import math
from typing import Literal

import pytest
from spectra.environment.propagation import (
    ITU_R_P525,
    ITU_R_P1411,
    COST231HataPL,
    FreeSpacePathLoss,
    GPP38901InH,
    GPP38901RMa,
    GPP38901UMa,
    GPP38901UMi,
    LogDistancePL,
    OkumuraHataPL,
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


class TestITU_R_P525:
    def test_is_propagation_model(self):
        assert isinstance(ITU_R_P525(), PropagationModel)

    def test_matches_free_space_without_gaseous(self):
        p525 = ITU_R_P525()
        fspl = FreeSpacePathLoss()
        for d, f in [(1000.0, 2.4e9), (10.0, 28e9), (5000.0, 900e6)]:
            assert math.isclose(p525(d, f).path_loss_db, fspl(d, f).path_loss_db, rel_tol=1e-9)

    def test_gaseous_adds_attenuation_at_mmwave(self):
        p525_clean = ITU_R_P525(include_gaseous=False)
        p525_absorb = ITU_R_P525(include_gaseous=True)
        # At 60 GHz the oxygen complex must add measurable loss
        clean = p525_clean(1000.0, 60e9).path_loss_db
        absorb = p525_absorb(1000.0, 60e9).path_loss_db
        assert absorb > clean + 1.0  # at least 1 dB extra

    def test_gaseous_negligible_at_low_freq(self):
        p525_clean = ITU_R_P525(include_gaseous=False)
        p525_absorb = ITU_R_P525(include_gaseous=True)
        clean = p525_clean(1000.0, 2.4e9).path_loss_db
        absorb = p525_absorb(1000.0, 2.4e9).path_loss_db
        assert absorb - clean < 0.1

    def test_no_fading_metadata(self):
        result = ITU_R_P525()(1000.0, 2.4e9)
        assert result.shadow_fading_db == 0.0
        assert result.k_factor_db is None
        assert result.rms_delay_spread_s is None

    def test_minimum_distance_clamp(self):
        with pytest.raises(ValueError, match="distance_m must be positive"):
            ITU_R_P525()(0.0, 2.4e9)


class TestOkumuraHataPL:
    def test_is_propagation_model(self):
        assert isinstance(
            OkumuraHataPL(h_bs_m=50.0, h_ms_m=1.5, environment="urban_small_medium"),
            PropagationModel,
        )

    def test_reasonable_range_1km_900mhz(self):
        # Standard test case: Tokyo-like, 900 MHz, 1 km, h_bs=50m, h_ms=1.5m
        # Canonical Hata for these inputs yields ~123 dB.
        m = OkumuraHataPL(h_bs_m=50.0, h_ms_m=1.5, environment="urban_small_medium")
        result = m(distance_m=1000.0, freq_hz=900e6)
        assert 120.0 < result.path_loss_db < 135.0

    def test_urban_large_higher_than_small(self):
        large = OkumuraHataPL(h_bs_m=50.0, h_ms_m=1.5, environment="urban_large")
        small = OkumuraHataPL(h_bs_m=50.0, h_ms_m=1.5, environment="urban_small_medium")
        # Large-city correction differs from small-city
        assert large(1000.0, 900e6).path_loss_db != small(1000.0, 900e6).path_loss_db

    def test_urban_more_loss_than_suburban(self):
        urban = OkumuraHataPL(h_bs_m=50.0, h_ms_m=1.5, environment="urban_small_medium")
        suburban = OkumuraHataPL(h_bs_m=50.0, h_ms_m=1.5, environment="suburban")
        assert urban(1000.0, 900e6).path_loss_db > suburban(1000.0, 900e6).path_loss_db

    def test_suburban_more_loss_than_rural(self):
        suburban = OkumuraHataPL(h_bs_m=50.0, h_ms_m=1.5, environment="suburban")
        rural = OkumuraHataPL(h_bs_m=50.0, h_ms_m=1.5, environment="rural")
        assert suburban(1000.0, 900e6).path_loss_db > rural(1000.0, 900e6).path_loss_db

    def test_farther_distance_more_loss(self):
        m = OkumuraHataPL(h_bs_m=50.0, h_ms_m=1.5, environment="urban_small_medium")
        assert m(5000.0, 900e6).path_loss_db > m(1000.0, 900e6).path_loss_db

    def test_higher_frequency_more_loss(self):
        m = OkumuraHataPL(h_bs_m=50.0, h_ms_m=1.5, environment="urban_small_medium")
        assert m(1000.0, 1400e6).path_loss_db > m(1000.0, 300e6).path_loss_db

    def test_above_1500mhz_raises_with_hint(self):
        m = OkumuraHataPL(h_bs_m=50.0, h_ms_m=1.5, environment="urban_small_medium")
        with pytest.raises(ValueError, match="freq"):
            m(1000.0, 1800e6)

    def test_invalid_environment_raises(self):
        with pytest.raises(ValueError, match="environment"):
            OkumuraHataPL(h_bs_m=50.0, h_ms_m=1.5, environment="urban")  # ty: ignore[invalid-argument-type] # intentional invalid Literal

    def test_shadow_fading_deterministic_with_seed(self):
        m = OkumuraHataPL(h_bs_m=50.0, h_ms_m=1.5, environment="urban_small_medium", sigma_db=8.0)
        r1 = m(1000.0, 900e6, seed=42)
        r2 = m(1000.0, 900e6, seed=42)
        assert r1.shadow_fading_db == r2.shadow_fading_db
        assert r1.shadow_fading_db != 0.0

    def test_zero_sigma_no_shadow(self):
        m = OkumuraHataPL(h_bs_m=50.0, h_ms_m=1.5, environment="urban_small_medium", sigma_db=0.0)
        assert m(1000.0, 900e6).shadow_fading_db == 0.0

    def test_non_strict_range_warns(self):
        m = OkumuraHataPL(
            h_bs_m=50.0,
            h_ms_m=1.5,
            environment="urban_small_medium",
            strict_range=False,
        )
        with pytest.warns(UserWarning):
            m(1000.0, 1800e6)

    def test_multipath_fields_none(self):
        m = OkumuraHataPL(h_bs_m=50.0, h_ms_m=1.5, environment="urban_small_medium")
        result = m(1000.0, 900e6)
        assert result.rms_delay_spread_s is None
        assert result.k_factor_db is None
        assert result.angular_spread_deg is None


class TestGPP38901UMa:
    def test_is_propagation_model(self):
        assert isinstance(GPP38901UMa(h_bs_m=25.0, h_ut_m=1.5), PropagationModel)

    def test_los_less_than_nlos(self):
        los = GPP38901UMa(h_bs_m=25.0, h_ut_m=1.5, los_mode="force_los")
        nlos = GPP38901UMa(h_bs_m=25.0, h_ut_m=1.5, los_mode="force_nlos")
        assert los(500.0, 3.5e9).path_loss_db < nlos(500.0, 3.5e9).path_loss_db

    def test_los_short_distance_matches_pl1_formula(self):
        """At d_2D = 100 m (pre-breakpoint at 3.5 GHz), UMa LOS PL_1 formula.

        PL_LOS = 28.0 + 22*log10(d_3D) + 20*log10(f_c_GHz)
        d_3D ≈ sqrt(100² + (25-1.5)²) ≈ 102.72 m
        """
        model = GPP38901UMa(h_bs_m=25.0, h_ut_m=1.5, los_mode="force_los")
        d_3d = math.sqrt(100.0**2 + (25.0 - 1.5) ** 2)
        expected = 28.0 + 22 * math.log10(d_3d) + 20 * math.log10(3.5)
        # Path loss includes shadow fading. Separate it:
        result_seeded = model(100.0, 3.5e9, seed=0)
        assert math.isclose(
            result_seeded.path_loss_db - result_seeded.shadow_fading_db,
            expected,
            rel_tol=1e-3,
        )

    def test_farther_distance_more_loss(self):
        m = GPP38901UMa(h_bs_m=25.0, h_ut_m=1.5, los_mode="force_los")
        # Use force_los to avoid stochasticity in LOS/NLOS switching
        assert m(1000.0, 3.5e9, seed=0).path_loss_db > m(100.0, 3.5e9, seed=0).path_loss_db

    def test_higher_frequency_more_loss_los(self):
        m = GPP38901UMa(h_bs_m=25.0, h_ut_m=1.5, los_mode="force_los")
        # Separate shadow fading (same seed → same realization)
        r1 = m(500.0, 2e9, seed=0)
        r2 = m(500.0, 28e9, seed=0)
        assert (r2.path_loss_db - r2.shadow_fading_db) > (r1.path_loss_db - r1.shadow_fading_db)

    def test_los_populates_k_factor(self):
        m = GPP38901UMa(h_bs_m=25.0, h_ut_m=1.5, los_mode="force_los")
        result = m(500.0, 3.5e9, seed=0)
        assert result.k_factor_db is not None

    def test_nlos_k_factor_is_none(self):
        m = GPP38901UMa(h_bs_m=25.0, h_ut_m=1.5, los_mode="force_nlos")
        result = m(500.0, 3.5e9, seed=0)
        assert result.k_factor_db is None

    def test_populates_delay_spread(self):
        m = GPP38901UMa(h_bs_m=25.0, h_ut_m=1.5, los_mode="force_los")
        result = m(500.0, 3.5e9, seed=0)
        assert result.rms_delay_spread_s is not None
        assert 1e-9 < result.rms_delay_spread_s < 1e-5

    def test_populates_angular_spread(self):
        m = GPP38901UMa(h_bs_m=25.0, h_ut_m=1.5, los_mode="force_los")
        result = m(500.0, 3.5e9, seed=0)
        assert result.angular_spread_deg is not None
        assert 0.0 < result.angular_spread_deg < 360.0

    def test_seed_reproducibility(self):
        m = GPP38901UMa(h_bs_m=25.0, h_ut_m=1.5)
        r1 = m(500.0, 3.5e9, seed=42)
        r2 = m(500.0, 3.5e9, seed=42)
        assert r1.path_loss_db == r2.path_loss_db
        assert r1.shadow_fading_db == r2.shadow_fading_db
        assert r1.rms_delay_spread_s == r2.rms_delay_spread_s

    def test_different_seeds_different_shadow(self):
        m = GPP38901UMa(h_bs_m=25.0, h_ut_m=1.5, los_mode="force_los")
        r1 = m(500.0, 3.5e9, seed=1)
        r2 = m(500.0, 3.5e9, seed=2)
        assert r1.shadow_fading_db != r2.shadow_fading_db

    def test_stochastic_los_probability_at_short_range_is_1(self):
        # Per TR 38.901 Table 7.4.2-1, UMa: P_LOS = 1 for d_2D <= 18 m
        m = GPP38901UMa(h_bs_m=25.0, h_ut_m=1.5, los_mode="stochastic")
        # At 10 m, LOS should always occur → k_factor should be populated
        result = m(10.0, 3.5e9, seed=42)
        assert result.k_factor_db is not None

    def test_freq_out_of_range_raises(self):
        m = GPP38901UMa(h_bs_m=25.0, h_ut_m=1.5)
        with pytest.raises(ValueError, match="freq"):
            m(500.0, 200e6)  # below 500 MHz

    def test_distance_out_of_range_raises(self):
        m = GPP38901UMa(h_bs_m=25.0, h_ut_m=1.5)
        with pytest.raises(ValueError, match="distance"):
            m(1.0, 3.5e9)  # below 10 m

    def test_invalid_los_mode_raises(self):
        m = GPP38901UMa(h_bs_m=25.0, h_ut_m=1.5, los_mode="bogus")  # ty: ignore[invalid-argument-type] # intentional invalid Literal
        with pytest.raises(ValueError, match="los_mode"):
            m(500.0, 3.5e9, seed=0)


class TestGPP38901UMi:
    def test_is_propagation_model(self):
        assert isinstance(GPP38901UMi(h_bs_m=10.0, h_ut_m=1.5), PropagationModel)

    def test_los_less_than_nlos(self):
        los = GPP38901UMi(h_bs_m=10.0, h_ut_m=1.5, los_mode="force_los")
        nlos = GPP38901UMi(h_bs_m=10.0, h_ut_m=1.5, los_mode="force_nlos")
        assert los(300.0, 28e9, seed=0).path_loss_db < nlos(300.0, 28e9, seed=0).path_loss_db

    def test_los_formula_short_distance(self):
        """UMi LOS PL_1: PL = 32.4 + 21*log10(d_3D) + 20*log10(f_c_GHz)."""
        m = GPP38901UMi(h_bs_m=10.0, h_ut_m=1.5, los_mode="force_los")
        d_3d = math.sqrt(100.0**2 + (10.0 - 1.5) ** 2)
        expected = 32.4 + 21.0 * math.log10(d_3d) + 20.0 * math.log10(3.5)
        r = m(100.0, 3.5e9, seed=0)
        assert math.isclose(r.path_loss_db - r.shadow_fading_db, expected, rel_tol=1e-3)

    def test_populates_multipath_fields(self):
        m = GPP38901UMi(h_bs_m=10.0, h_ut_m=1.5, los_mode="force_los")
        r = m(300.0, 28e9, seed=0)
        assert r.rms_delay_spread_s is not None
        assert r.k_factor_db is not None
        assert r.angular_spread_deg is not None

    def test_seed_reproducibility(self):
        m = GPP38901UMi(h_bs_m=10.0, h_ut_m=1.5)
        r1 = m(300.0, 28e9, seed=42)
        r2 = m(300.0, 28e9, seed=42)
        assert r1.path_loss_db == r2.path_loss_db


class TestGPP38901RMa:
    def test_is_propagation_model(self):
        m = GPP38901RMa(h_bs_m=35.0, h_ut_m=1.5)
        assert isinstance(m, PropagationModel)

    def test_los_less_than_nlos(self):
        los = GPP38901RMa(h_bs_m=35.0, h_ut_m=1.5, los_mode="force_los")
        nlos = GPP38901RMa(h_bs_m=35.0, h_ut_m=1.5, los_mode="force_nlos")
        assert los(1000.0, 700e6, seed=0).path_loss_db < nlos(1000.0, 700e6, seed=0).path_loss_db

    def test_accepts_building_and_street_params(self):
        m = GPP38901RMa(h_bs_m=35.0, h_ut_m=1.5, h_building_m=10.0, w_street_m=30.0)
        r = m(1000.0, 700e6, seed=0)
        assert isinstance(r, PathLossResult)

    def test_populates_multipath_fields(self):
        m = GPP38901RMa(h_bs_m=35.0, h_ut_m=1.5, los_mode="force_los")
        r = m(1000.0, 700e6, seed=0)
        assert r.rms_delay_spread_s is not None
        assert r.k_factor_db is not None
        assert r.angular_spread_deg is not None

    def test_distance_envelope_10km(self):
        m = GPP38901RMa(h_bs_m=35.0, h_ut_m=1.5)
        # Should accept up to 10 km
        m(9000.0, 700e6, seed=0)
        # Should reject above
        with pytest.raises(ValueError):
            m(11000.0, 700e6)

    def test_freq_envelope_30ghz(self):
        m = GPP38901RMa(h_bs_m=35.0, h_ut_m=1.5)
        m(1000.0, 30e9, seed=0)  # OK
        with pytest.raises(ValueError):
            m(1000.0, 60e9)  # Above 30 GHz


class TestGPP38901InH:
    def test_is_propagation_model(self):
        m = GPP38901InH(h_bs_m=3.0, h_ut_m=1.0)
        assert isinstance(m, PropagationModel)

    def test_los_less_than_nlos(self):
        los = GPP38901InH(h_bs_m=3.0, h_ut_m=1.0, los_mode="force_los")
        nlos = GPP38901InH(h_bs_m=3.0, h_ut_m=1.0, los_mode="force_nlos")
        assert los(30.0, 3.5e9, seed=0).path_loss_db < nlos(30.0, 3.5e9, seed=0).path_loss_db

    def test_mixed_office_variant(self):
        m = GPP38901InH(h_bs_m=3.0, h_ut_m=1.0, variant="mixed_office")
        r = m(30.0, 3.5e9, seed=0)
        assert isinstance(r, PathLossResult)

    def test_open_office_variant(self):
        m = GPP38901InH(h_bs_m=3.0, h_ut_m=1.0, variant="open_office")
        r = m(30.0, 3.5e9, seed=0)
        assert isinstance(r, PathLossResult)

    def test_variants_differ_in_nlos(self):
        mixed = GPP38901InH(h_bs_m=3.0, h_ut_m=1.0, variant="mixed_office", los_mode="force_nlos")
        openp = GPP38901InH(h_bs_m=3.0, h_ut_m=1.0, variant="open_office", los_mode="force_nlos")
        # The LOS probability formulas differ but force_nlos bypasses that.
        # NLOS PL formula is the same — check with stochastic mode and seeds that
        # the LOS probabilities differ.
        mix_los_p = mixed._los_probability(30.0)
        open_los_p = openp._los_probability(30.0)
        assert mix_los_p != open_los_p

    def test_invalid_variant_raises(self):
        with pytest.raises(ValueError, match="variant"):
            GPP38901InH(h_bs_m=3.0, h_ut_m=1.0, variant="industrial")

    def test_distance_envelope_150m(self):
        m = GPP38901InH(h_bs_m=3.0, h_ut_m=1.0)
        m(140.0, 3.5e9, seed=0)  # OK
        with pytest.raises(ValueError):
            m(200.0, 3.5e9)  # Above 150 m


class TestITU_R_P1411:
    def test_is_propagation_model(self):
        m = ITU_R_P1411(environment="urban_high_rise")
        assert isinstance(m, PropagationModel)

    def test_los_less_than_nlos(self):
        los = ITU_R_P1411(environment="urban_high_rise", los_mode="force_los")
        nlos = ITU_R_P1411(environment="urban_high_rise", los_mode="force_nlos")
        assert los(200.0, 2.4e9, seed=0).path_loss_db < nlos(200.0, 2.4e9, seed=0).path_loss_db

    def test_all_three_environments(self):
        envs: list[Literal["urban_high_rise", "urban_low_rise_suburban", "residential"]] = [
            "urban_high_rise",
            "urban_low_rise_suburban",
            "residential",
        ]
        for env in envs:
            m = ITU_R_P1411(environment=env)
            r = m(200.0, 2.4e9, seed=0)
            assert isinstance(r, PathLossResult)

    def test_invalid_environment_raises(self):
        with pytest.raises(ValueError, match="environment"):
            ITU_R_P1411(environment="rural")  # ty: ignore[invalid-argument-type] # intentional invalid Literal

    def test_farther_distance_more_loss(self):
        m = ITU_R_P1411(environment="urban_high_rise", los_mode="force_los")
        r1 = m(100.0, 2.4e9, seed=0)
        r2 = m(1000.0, 2.4e9, seed=0)
        assert (r2.path_loss_db - r2.shadow_fading_db) > (r1.path_loss_db - r1.shadow_fading_db)

    def test_multipath_fields_none(self):
        m = ITU_R_P1411(environment="urban_high_rise")
        r = m(200.0, 2.4e9, seed=0)
        assert r.rms_delay_spread_s is None
        assert r.k_factor_db is None

    def test_freq_envelope(self):
        m = ITU_R_P1411(environment="urban_high_rise")
        with pytest.raises(ValueError, match="freq"):
            m(200.0, 100e6)  # below 300 MHz

    def test_distance_envelope(self):
        m = ITU_R_P1411(environment="urban_high_rise")
        with pytest.raises(ValueError, match="distance"):
            m(10.0, 2.4e9)  # below 50 m

    def test_seed_reproducibility(self):
        m = ITU_R_P1411(environment="urban_high_rise")
        r1 = m(200.0, 2.4e9, seed=42)
        r2 = m(200.0, 2.4e9, seed=42)
        assert r1.path_loss_db == r2.path_loss_db
        assert r1.shadow_fading_db == r2.shadow_fading_db

    def test_shadow_fading_within_2_sigma(self):
        """Repeated draws should have std dev close to the tabulated sigma."""
        m = ITU_R_P1411(environment="urban_high_rise", los_mode="force_los")
        shadows = [m(200.0, 2.4e9, seed=i).shadow_fading_db for i in range(200)]
        import statistics

        std = statistics.stdev(shadows)
        # Urban high-rise LOS sigma should be in [2, 5] dB
        assert 1.5 < std < 6.0
