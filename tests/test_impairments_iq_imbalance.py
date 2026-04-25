import numpy as np
import numpy.testing as npt
from conftest import make_signal_description


class TestIQImbalance:
    def test_no_imbalance_preserves_signal(self):
        from spectra.impairments.iq_imbalance import IQImbalance

        rng = np.random.default_rng(42)
        iq = (rng.standard_normal(1024) + 1j * rng.standard_normal(1024)).astype(np.complex64)
        desc = make_signal_description()
        result, _ = IQImbalance(amplitude_imbalance_db=0.0, phase_imbalance_deg=0.0)(iq, desc)
        npt.assert_allclose(result, iq, atol=1e-5)

    def test_amplitude_imbalance_modifies_signal(self):
        from spectra.impairments.iq_imbalance import IQImbalance

        iq = np.array([1 + 1j, 1 - 1j, -1 + 1j], dtype=np.complex64)
        desc = make_signal_description()
        result, _ = IQImbalance(amplitude_imbalance_db=6.0, phase_imbalance_deg=0.0)(iq, desc)
        # With amplitude imbalance, I channel is scaled
        assert not np.allclose(result.real, iq.real, atol=0.01)

    def test_output_shape_and_dtype(self):
        from spectra.impairments.iq_imbalance import IQImbalance

        iq = np.ones(256, dtype=np.complex64)
        desc = make_signal_description()
        result, _ = IQImbalance(amplitude_imbalance_db=1.0, phase_imbalance_deg=5.0)(iq, desc)
        assert result.shape == iq.shape
        assert result.dtype == np.complex64

    def test_phase_imbalance_modifies_signal(self):
        from spectra.impairments.iq_imbalance import IQImbalance

        iq = np.ones(512, dtype=np.complex64) * (1 + 1j)
        desc = make_signal_description()
        result, _ = IQImbalance(amplitude_imbalance_db=0.0, phase_imbalance_deg=10.0)(iq, desc)
        # Phase imbalance should change the I channel
        assert not np.allclose(result.real, iq.real, atol=0.01)

    def test_desc_unchanged(self):
        from spectra.impairments.iq_imbalance import IQImbalance

        iq = np.ones(256, dtype=np.complex64)
        desc = make_signal_description()
        _, new_desc = IQImbalance(amplitude_imbalance_db=1.0, phase_imbalance_deg=5.0)(iq, desc)
        assert new_desc.f_low == desc.f_low
        assert new_desc.f_high == desc.f_high

    def test_default_params(self):
        from spectra.impairments.iq_imbalance import IQImbalance

        imb = IQImbalance()
        assert imb._amp_db == 1.0
        assert imb._phase_deg == 5.0

    def test_zero_amplitude_zero_phase_identity(self):
        from spectra.impairments.iq_imbalance import IQImbalance

        rng = np.random.default_rng(99)
        iq = (rng.standard_normal(256) + 1j * rng.standard_normal(256)).astype(np.complex64)
        desc = make_signal_description()
        result, _ = IQImbalance(amplitude_imbalance_db=0.0, phase_imbalance_deg=0.0)(iq, desc)
        npt.assert_allclose(result, iq, atol=1e-5)
