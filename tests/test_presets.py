"""Tests for propagation model presets."""

from spectra.environment.presets import propagation_presets
from spectra.environment.propagation import PropagationModel


class TestPresets:
    def test_all_presets_are_propagation_models(self):
        for name, model in propagation_presets.items():
            assert isinstance(model, PropagationModel), f"Preset '{name}' is not a PropagationModel"

    def test_expected_presets_exist(self):
        expected = {"free_space", "urban_macro", "suburban", "indoor_office", "cost231_urban"}
        assert set(propagation_presets.keys()) == expected

    def test_all_presets_callable(self):
        for name, model in propagation_presets.items():
            result = model(distance_m=100.0, freq_hz=1800e6)
            assert result.path_loss_db > 0, f"Preset '{name}' returned non-positive path loss"
