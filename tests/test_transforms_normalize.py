import numpy as np
import torch
import pytest


class TestSpectrogramNormalize:
    def test_db_normalize_output_range(self):
        from spectra.transforms.normalize import SpectrogramNormalize

        # [1, freq, time] tensor
        spec = torch.rand(1, 64, 32) * 100.0
        norm = SpectrogramNormalize(mode="db")
        result = norm(spec)
        assert result.min() >= -0.01
        assert result.max() <= 1.01

    def test_db_normalize_shape_preserved(self):
        from spectra.transforms.normalize import SpectrogramNormalize

        spec = torch.rand(1, 128, 64)
        result = SpectrogramNormalize(mode="db")(spec)
        assert result.shape == spec.shape

    def test_standardize_zero_mean(self):
        from spectra.transforms.normalize import SpectrogramNormalize

        spec = torch.rand(1, 64, 32) * 50.0 + 10.0
        result = SpectrogramNormalize(mode="standardize")(spec)
        assert abs(result.mean().item()) < 0.01

    def test_standardize_unit_variance(self):
        from spectra.transforms.normalize import SpectrogramNormalize

        spec = torch.rand(1, 64, 32) * 50.0 + 10.0
        result = SpectrogramNormalize(mode="standardize")(spec)
        assert abs(result.std().item() - 1.0) < 0.05

    def test_standardize_shape_preserved(self):
        from spectra.transforms.normalize import SpectrogramNormalize

        spec = torch.rand(1, 128, 64)
        result = SpectrogramNormalize(mode="standardize")(spec)
        assert result.shape == spec.shape

    def test_output_dtype_float32(self):
        from spectra.transforms.normalize import SpectrogramNormalize

        spec = torch.rand(1, 64, 32)
        result = SpectrogramNormalize(mode="db")(spec)
        assert result.dtype == torch.float32

    def test_invalid_mode_raises(self):
        from spectra.transforms.normalize import SpectrogramNormalize

        with pytest.raises(ValueError, match="mode"):
            SpectrogramNormalize(mode="invalid")

    def test_no_nans_on_zero_input(self):
        from spectra.transforms.normalize import SpectrogramNormalize

        spec = torch.zeros(1, 64, 32)
        result_db = SpectrogramNormalize(mode="db")(spec)
        result_std = SpectrogramNormalize(mode="standardize")(spec)
        assert not torch.any(torch.isnan(result_db))
        assert not torch.any(torch.isnan(result_std))
