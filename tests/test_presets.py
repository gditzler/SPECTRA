"""Tests for propagation model presets."""

from spectra.environment.presets import propagation_presets
from spectra.environment.propagation import PropagationModel


class TestPresets:
    def test_all_presets_are_propagation_models(self):
        for name, model in propagation_presets.items():
            assert isinstance(model, PropagationModel), f"Preset '{name}' is not a PropagationModel"

    def test_expected_presets_exist(self):
        expected = {"free_space", "urban_macro", "suburban", "indoor_office", "cost231_urban"}
        assert expected.issubset(set(propagation_presets.keys()))

    def test_all_presets_callable(self):
        # Each preset model has its own validity envelope; pick a
        # (distance, frequency) pair that is inside each model's range.
        per_preset_params = {
            "free_space": (100.0, 1800e6),
            "urban_macro": (100.0, 1800e6),
            "suburban": (100.0, 1800e6),
            "indoor_office": (100.0, 1800e6),
            "cost231_urban": (1000.0, 1800e6),
            "urban_macro_5g": (100.0, 3500e6),
            "urban_micro_mmwave": (100.0, 28e9),
            "rural_macro_5g": (1000.0, 1800e6),
            "indoor_office_5g": (20.0, 3500e6),
            "urban_hata_4g": (5000.0, 900e6),
            "short_range_urban": (500.0, 2000e6),
        }
        for name, model in propagation_presets.items():
            distance_m, freq_hz = per_preset_params[name]
            result = model(distance_m=distance_m, freq_hz=freq_hz)
            assert result.path_loss_db > 0, f"Preset '{name}' returned non-positive path loss"


def test_new_5g_presets_present():
    from spectra.environment.presets import propagation_presets
    expected = {
        "urban_macro_5g",
        "urban_micro_mmwave",
        "rural_macro_5g",
        "indoor_office_5g",
        "urban_hata_4g",
        "short_range_urban",
    }
    assert expected.issubset(set(propagation_presets.keys()))


def test_urban_macro_5g_is_uma():
    from spectra.environment.presets import propagation_presets
    from spectra.environment.propagation import GPP38901UMa
    assert isinstance(propagation_presets["urban_macro_5g"], GPP38901UMa)


def test_short_range_urban_is_p1411():
    from spectra.environment.presets import propagation_presets
    from spectra.environment.propagation import ITU_R_P1411
    assert isinstance(propagation_presets["short_range_urban"], ITU_R_P1411)
