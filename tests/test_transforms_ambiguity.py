"""Tests for Ambiguity Function transform."""

import numpy as np
import pytest
import torch
from spectra.transforms.ambiguity import AmbiguityFunction


class TestAmbiguityShape:
    """Output shape tests."""

    def test_magnitude_shape(self):
        iq = np.exp(1j * 2 * np.pi * 0.1 * np.arange(512)).astype(np.complex64)
        af = AmbiguityFunction(max_lag=64, n_doppler=128, output_format="magnitude")
        out = af(iq)
        assert out.shape == (1, 128, 129)  # 2*64+1 = 129
        assert out.dtype == torch.float32

    def test_mag_phase_shape(self):
        iq = np.exp(1j * 2 * np.pi * 0.1 * np.arange(512)).astype(np.complex64)
        af = AmbiguityFunction(max_lag=32, n_doppler=64, output_format="mag_phase")
        out = af(iq)
        assert out.shape == (2, 64, 65)  # 2*32+1 = 65

    def test_real_imag_shape(self):
        iq = np.exp(1j * 2 * np.pi * 0.1 * np.arange(512)).astype(np.complex64)
        af = AmbiguityFunction(max_lag=32, n_doppler=64, output_format="real_imag")
        out = af(iq)
        assert out.shape == (2, 64, 65)


class TestAmbiguityContent:
    """Content / signal property tests."""

    def test_peak_at_zero_lag(self):
        """Noise-like signal ambiguity function should peak at tau=0."""
        rng = np.random.default_rng(42)
        iq = (rng.standard_normal(512) + 1j * rng.standard_normal(512)).astype(np.complex64)
        af = AmbiguityFunction(max_lag=64, n_doppler=128, output_format="magnitude")
        out = af(iq)
        # For a noise-like signal, the overall peak should be near tau=0
        mag = out[0]
        # Find the global peak location
        flat_idx = torch.argmax(mag).item()
        peak_lag = flat_idx % mag.shape[1]
        center_lag = 64  # max_lag index corresponds to tau=0
        assert abs(peak_lag - center_lag) <= 2

    def test_dtype_float32(self):
        iq = np.exp(1j * 2 * np.pi * 0.1 * np.arange(256)).astype(np.complex64)
        af = AmbiguityFunction(max_lag=32, n_doppler=64, output_format="magnitude")
        out = af(iq)
        assert out.dtype == torch.float32

    def test_invalid_format(self):
        with pytest.raises(ValueError):
            AmbiguityFunction(output_format="bad_format")
