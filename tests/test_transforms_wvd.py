"""Tests for WVD transform."""
import numpy as np
import pytest
import torch

from spectra.transforms.wvd import WVD


class TestWVDShape:
    """Output shape tests."""

    def test_magnitude_shape(self):
        iq = np.exp(1j * 2 * np.pi * 0.1 * np.arange(512)).astype(np.complex64)
        wvd = WVD(nfft=256, output_format="magnitude")
        out = wvd(iq)
        assert out.shape == (1, 512, 256)
        assert out.dtype == torch.float32

    def test_real_imag_shape(self):
        iq = np.exp(1j * 2 * np.pi * 0.1 * np.arange(512)).astype(np.complex64)
        wvd = WVD(nfft=256, output_format="real_imag")
        out = wvd(iq)
        assert out.shape == (2, 512, 256)

    def test_log_magnitude_shape(self):
        iq = np.exp(1j * 2 * np.pi * 0.1 * np.arange(512)).astype(np.complex64)
        wvd = WVD(nfft=128, output_format="log_magnitude")
        out = wvd(iq)
        assert out.shape == (1, 512, 128)

    def test_n_time_subsampling(self):
        iq = np.exp(1j * 2 * np.pi * 0.1 * np.arange(1024)).astype(np.complex64)
        wvd = WVD(nfft=256, n_time=64, output_format="magnitude")
        out = wvd(iq)
        assert out.shape == (1, 64, 256)


class TestWVDContent:
    """Content / signal property tests."""

    def test_pure_tone_concentrated_frequency(self):
        """Pure tone should have energy concentrated at a single frequency bin."""
        f0 = 0.1  # normalized frequency
        nfft = 256
        iq = np.exp(1j * 2 * np.pi * f0 * np.arange(256)).astype(np.complex64)
        wvd = WVD(nfft=nfft, output_format="magnitude")
        out = wvd(iq)
        # For a time slice in the middle (avoiding edges), check that
        # energy is concentrated: the peak bin holds a large fraction of total energy
        mid_slice = out[0, 128, :]
        peak_val = torch.max(mid_slice).item()
        mean_val = torch.mean(mid_slice).item()
        # Peak should be significantly above average for a concentrated tone
        assert peak_val > 5 * mean_val

    def test_dtype_float32(self):
        iq = np.exp(1j * 2 * np.pi * 0.1 * np.arange(128)).astype(np.complex64)
        wvd = WVD(nfft=64, output_format="magnitude")
        out = wvd(iq)
        assert out.dtype == torch.float32

    def test_invalid_format(self):
        with pytest.raises(ValueError):
            WVD(output_format="bad_format")
