import numpy as np
import pytest
from spectra.impairments.timing import FractionalDelay, SamplingJitter
from spectra.scene.signal_desc import SignalDescription


@pytest.fixture
def desc():
    return SignalDescription(
        t_start=0.0,
        t_stop=1.0,
        f_low=-500e3,
        f_high=500e3,
        label="test",
        snr=20.0,
    )


class TestFractionalDelay:
    def test_output_shape_dtype(self, desc):
        iq = np.exp(2j * np.pi * 0.1 * np.arange(1024)).astype(np.complex64)
        fd = FractionalDelay(delay=0.5)
        out, _ = fd(iq, desc)
        assert out.shape == iq.shape
        assert out.dtype == np.complex64

    def test_half_sample_delay_phase(self, desc):
        f = 0.01  # normalized frequency
        t = np.arange(1024)
        iq = np.exp(2j * np.pi * f * t).astype(np.complex64)
        fd = FractionalDelay(delay=0.5)
        out, _ = fd(iq, desc)
        # 0.5 sample delay should shift phase by pi*f
        mid = slice(100, 900)  # avoid edges
        phase_diff = np.angle(out[mid] * np.conj(iq[mid]))
        expected = -2 * np.pi * f * 0.5
        assert np.abs(np.mean(phase_diff) - expected) < 0.1

    def test_random_delay(self, desc):
        iq = np.exp(2j * np.pi * 0.05 * np.arange(512)).astype(np.complex64)
        fd = FractionalDelay(max_delay=1.0)
        out, _ = fd(iq, desc)
        assert out.shape == iq.shape

    def test_power_preserved(self, desc):
        iq = (np.random.randn(1024) + 1j * np.random.randn(1024)).astype(np.complex64)
        fd = FractionalDelay(delay=0.3)
        out, _ = fd(iq, desc)
        in_power = np.mean(np.abs(iq) ** 2)
        out_power = np.mean(np.abs(out) ** 2)
        assert abs(out_power / in_power - 1.0) < 0.1


class TestSamplingJitter:
    def test_output_shape_dtype(self, desc):
        iq = (np.random.randn(1024) + 1j * np.random.randn(1024)).astype(np.complex64)
        sj = SamplingJitter(std_samples=0.01)
        out, _ = sj(iq, desc)
        assert out.shape == iq.shape
        assert out.dtype == np.complex64

    def test_output_differs(self, desc):
        iq = np.exp(2j * np.pi * 0.1 * np.arange(1024)).astype(np.complex64)
        sj = SamplingJitter(std_samples=0.1)
        out, _ = sj(iq, desc)
        assert not np.array_equal(out, iq)

    def test_power_approximately_preserved(self, desc):
        iq = (np.random.randn(1024) + 1j * np.random.randn(1024)).astype(np.complex64)
        sj = SamplingJitter(std_samples=0.01)
        out, _ = sj(iq, desc)
        in_power = np.mean(np.abs(iq) ** 2)
        out_power = np.mean(np.abs(out) ** 2)
        assert abs(out_power / in_power - 1.0) < 0.2
