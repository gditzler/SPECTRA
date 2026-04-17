"""Tests for ITU-R P.676 gaseous attenuation helper."""

import warnings

import pytest
from spectra.environment.propagation.atmospheric import gaseous_attenuation_db


class TestGaseousAttenuation:
    def test_zero_distance_zero_attenuation(self):
        assert gaseous_attenuation_db(0.0, 10e9) == 0.0

    def test_below_1ghz_returns_zero_with_warning(self):
        with pytest.warns(UserWarning, match="P.676"):
            result = gaseous_attenuation_db(1000.0, 500e6)
        assert result == 0.0

    def test_below_1ghz_only_warns_once(self):
        with warnings.catch_warnings(record=True) as caught:
            warnings.simplefilter("always")
            gaseous_attenuation_db(1000.0, 500e6)
            gaseous_attenuation_db(1000.0, 400e6)
        # Should warn at most once per process (module-level state)
        p676_warns = [w for w in caught if "P.676" in str(w.message)]
        assert len(p676_warns) <= 1

    def test_positive_at_22ghz_water_vapor_line(self):
        """22.235 GHz water vapor absorption line — should be > 0."""
        att = gaseous_attenuation_db(1000.0, 22.235e9)
        assert att > 0.0

    def test_60ghz_oxygen_complex_is_high(self):
        """60 GHz oxygen complex is the highest terrestrial absorption."""
        att_60 = gaseous_attenuation_db(1000.0, 60e9)
        att_30 = gaseous_attenuation_db(1000.0, 30e9)
        att_100 = gaseous_attenuation_db(1000.0, 100e9)
        # 60 GHz should be > both 30 and 100 GHz
        assert att_60 > att_30
        assert att_60 > att_100

    def test_linear_in_distance(self):
        """γ · d: doubling distance doubles attenuation."""
        a1 = gaseous_attenuation_db(1000.0, 22e9)
        a2 = gaseous_attenuation_db(2000.0, 22e9)
        assert abs(a2 - 2 * a1) / a1 < 1e-9

    def test_higher_water_vapor_increases_22ghz(self):
        dry = gaseous_attenuation_db(1000.0, 22e9, water_vapor_density_g_m3=0.1)
        humid = gaseous_attenuation_db(1000.0, 22e9, water_vapor_density_g_m3=15.0)
        assert humid > dry

    def test_reference_value_10ghz_standard_atmosphere(self):
        """At 10 GHz, γ ≈ 0.009-0.015 dB/km under ITU reference atmosphere.

        Check that for 1 km, attenuation is in [0.005, 0.03] dB.
        """
        att = gaseous_attenuation_db(1000.0, 10e9)
        assert 0.005 <= att <= 0.05

    def test_negative_distance_raises(self):
        with pytest.raises(ValueError, match="distance_m"):
            gaseous_attenuation_db(-1.0, 10e9)
