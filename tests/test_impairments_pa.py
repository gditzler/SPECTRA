import numpy as np
import pytest

from spectra.impairments.power_amplifier import RappPA, SalehPA
from spectra.scene.signal_desc import SignalDescription


@pytest.fixture
def desc():
    return SignalDescription(t_start=0.0, t_stop=1.0, f_low=-500e3, f_high=500e3, label="test", snr=20.0)


class TestRappPA:
    def test_output_shape_dtype(self, desc):
        iq = (np.random.randn(1024) + 1j * np.random.randn(1024)).astype(np.complex64) * 0.5
        pa = RappPA(smoothness=3.0, saturation=1.0)
        out, _ = pa(iq, desc)
        assert out.shape == iq.shape
        assert out.dtype == np.complex64

    def test_amplitude_never_exceeds_saturation(self, desc):
        iq = (np.random.randn(1024) + 1j * np.random.randn(1024)).astype(np.complex64) * 2.0
        pa = RappPA(smoothness=3.0, saturation=1.0)
        out, _ = pa(iq, desc)
        assert np.all(np.abs(out) <= 1.0 + 1e-6)

    def test_small_signals_pass_through(self, desc):
        iq = (np.random.randn(1024) + 1j * np.random.randn(1024)).astype(np.complex64) * 0.01
        pa = RappPA(smoothness=3.0, saturation=1.0)
        out, _ = pa(iq, desc)
        np.testing.assert_allclose(np.abs(out), np.abs(iq), rtol=0.01)

    def test_compression(self, desc):
        iq = (np.random.randn(1024) + 1j * np.random.randn(1024)).astype(np.complex64) * 1.0
        pa = RappPA(smoothness=3.0, saturation=1.0)
        out, _ = pa(iq, desc)
        assert np.mean(np.abs(out) ** 2) < np.mean(np.abs(iq) ** 2)


class TestSalehPA:
    def test_output_shape_dtype(self, desc):
        iq = (np.random.randn(1024) + 1j * np.random.randn(1024)).astype(np.complex64) * 0.5
        pa = SalehPA()
        out, _ = pa(iq, desc)
        assert out.shape == iq.shape
        assert out.dtype == np.complex64

    def test_am_pm_introduces_phase(self, desc):
        iq = (np.random.randn(1024) + 1j * np.random.randn(1024)).astype(np.complex64) * 0.5
        pa = SalehPA(alpha_p=1.0, beta_p=1.0)
        out, _ = pa(iq, desc)
        phase_in = np.angle(iq)
        phase_out = np.angle(out)
        # Phase should differ for non-zero amplitude signals
        assert not np.allclose(phase_in, phase_out, atol=0.01)

    def test_small_signals_nearly_linear(self, desc):
        iq = (np.random.randn(1024) + 1j * np.random.randn(1024)).astype(np.complex64) * 0.01
        pa = SalehPA(alpha_a=2.0, beta_a=1.0, alpha_p=1.0, beta_p=1.0)
        out, _ = pa(iq, desc)
        # For small r: A(r) ~ alpha_a * r, so output ~ alpha_a * input
        expected_amp = 2.0 * np.abs(iq)
        np.testing.assert_allclose(np.abs(out), expected_amp, rtol=0.05)

    def test_compression(self, desc):
        # At large amplitudes, Saleh model compresses
        iq = (np.random.randn(1024) + 1j * np.random.randn(1024)).astype(np.complex64) * 2.0
        pa = SalehPA(alpha_a=2.0, beta_a=1.0)
        out, _ = pa(iq, desc)
        # Saleh AM/AM peaks at r=1/sqrt(beta_a) then decreases
        # Large signals should have lower gain than alpha_a
        mean_gain = np.mean(np.abs(out)) / np.mean(np.abs(iq))
        assert mean_gain < 2.0  # less than alpha_a
