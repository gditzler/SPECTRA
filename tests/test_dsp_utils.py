import numpy as np
import numpy.testing as npt
import pytest


class TestLowPass:
    def test_returns_array(self):
        from spectra.utils import low_pass

        taps = low_pass(101, 0.5)
        assert isinstance(taps, np.ndarray)
        assert len(taps) == 101

    def test_unit_dc_gain(self):
        from spectra.utils import low_pass

        taps = low_pass(101, 0.5)
        assert np.sum(taps) == pytest.approx(1.0, abs=0.01)


class TestGaussianTaps:
    def test_returns_array(self):
        from spectra.utils import gaussian_taps

        taps = gaussian_taps(0.3, 4, 8)
        assert isinstance(taps, np.ndarray)

    def test_sums_to_one(self):
        from spectra.utils import gaussian_taps

        taps = gaussian_taps(0.3, 4, 8)
        assert np.sum(taps) == pytest.approx(1.0, abs=0.01)


class TestFrequencyShift:
    def test_shape_preserved(self):
        from spectra.utils import frequency_shift

        iq = np.ones(100, dtype=np.complex64)
        result = frequency_shift(iq, 1000.0, 1e6)
        assert len(result) == 100


class TestUpsample:
    def test_length(self):
        from spectra.utils import upsample

        sig = np.array([1, 2, 3], dtype=np.complex64)
        result = upsample(sig, 4)
        assert len(result) == 12


class TestConvolve:
    def test_basic(self):
        from spectra.utils import convolve

        sig = np.array([1, 0, 0, 0], dtype=np.complex64)
        taps = np.array([1, 2, 3], dtype=np.float32)
        result = convolve(sig, taps)
        npt.assert_allclose(result[:3].real, [1, 2, 3], atol=1e-5)


class TestNoiseGenerator:
    def test_white_power(self):
        from spectra.utils import noise_generator

        noise = noise_generator(10000, power=2.0, color="white", seed=42)
        power = np.mean(np.abs(noise) ** 2)
        assert power == pytest.approx(2.0, abs=0.2)

    def test_pink_generates(self):
        from spectra.utils import noise_generator

        noise = noise_generator(1024, power=1.0, color="pink", seed=42)
        assert len(noise) == 1024


class TestComputeSpectrogram:
    def test_shape(self):
        from spectra.utils import compute_spectrogram

        iq = np.ones(1024, dtype=np.complex64)
        spec = compute_spectrogram(iq, nfft=256, hop=64)
        assert spec.ndim == 2
        assert spec.shape[0] == 256


class TestHelpers:
    def test_center_freq(self):
        from spectra.utils import center_freq_from_bounds

        assert center_freq_from_bounds(-100, 300) == pytest.approx(100.0)

    def test_bandwidth(self):
        from spectra.utils import bandwidth_from_bounds

        assert bandwidth_from_bounds(-100, 300) == pytest.approx(400.0)


class TestMultistageResampler:
    def test_length(self):
        from spectra.utils import multistage_resampler

        sig = np.ones(100, dtype=np.complex64)
        result = multistage_resampler(sig, up=2, down=1)
        assert len(result) == 200
