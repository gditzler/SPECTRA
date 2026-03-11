import numpy as np
import pytest
import torch


class TestNormalize:
    def test_zero_mean_unit_var(self):
        from spectra.transforms import Normalize

        iq = (np.random.randn(1024) + 1j * np.random.randn(1024)).astype(np.complex64)
        normed = Normalize()(iq)
        assert np.abs(np.mean(normed)) < 0.1
        assert np.abs(np.std(normed) - 1.0) < 0.1

    def test_preserves_dtype(self):
        from spectra.transforms import Normalize

        iq = np.ones(100, dtype=np.complex64)
        result = Normalize()(iq)
        assert result.dtype == np.complex64


class TestComplexTo2D:
    def test_shape(self):
        from spectra.transforms import ComplexTo2D

        iq = (np.random.randn(512) + 1j * np.random.randn(512)).astype(np.complex64)
        tensor = ComplexTo2D()(iq)
        assert tensor.shape == (2, 512)

    def test_channels_match_iq(self):
        from spectra.transforms import ComplexTo2D

        iq = np.array([1 + 2j, 3 + 4j], dtype=np.complex64)
        tensor = ComplexTo2D()(iq)
        assert tensor[0, 0].item() == pytest.approx(1.0)
        assert tensor[1, 0].item() == pytest.approx(2.0)


class TestSpectrogram:
    def test_output_shape(self):
        from spectra.transforms import Spectrogram

        iq = (np.random.randn(1024) + 1j * np.random.randn(1024)).astype(np.complex64)
        spec = Spectrogram(nfft=256, hop_length=64)(iq)
        assert spec.ndim == 3
        assert spec.shape[0] == 1
        assert spec.shape[1] == 256  # complex STFT returns nfft bins

    def test_db_scale(self):
        from spectra.transforms import Spectrogram

        iq = (np.random.randn(1024) + 1j * np.random.randn(1024)).astype(np.complex64)
        Spectrogram(nfft=256, hop_length=64, db_scale=False)(iq)
        spec_db = Spectrogram(nfft=256, hop_length=64, db_scale=True)(iq)
        # dB values should generally be negative (for small magnitude signals)
        assert torch.any(spec_db < 0)
