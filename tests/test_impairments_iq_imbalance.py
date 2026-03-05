import numpy as np
import numpy.testing as npt
import pytest

from spectra.scene.signal_desc import SignalDescription


def _make_desc():
    return SignalDescription(0.0, 0.001, -5e3, 5e3, "QPSK", 20.0)


class TestIQImbalance:
    def test_no_imbalance_preserves_signal(self):
        from spectra.impairments.iq_imbalance import IQImbalance

        rng = np.random.default_rng(42)
        iq = (rng.standard_normal(1024) + 1j * rng.standard_normal(1024)).astype(
            np.complex64
        )
        desc = _make_desc()
        result, _ = IQImbalance(gain_imbalance=1.0, phase_imbalance=0.0)(iq, desc)
        npt.assert_allclose(result, iq, atol=1e-5)

    def test_gain_only_scales_q(self):
        from spectra.impairments.iq_imbalance import IQImbalance

        iq = np.array([1 + 1j, 1 - 1j, -1 + 1j], dtype=np.complex64)
        desc = _make_desc()
        result, _ = IQImbalance(gain_imbalance=2.0, phase_imbalance=0.0)(iq, desc)
        # I channel unchanged, Q channel scaled by 2
        npt.assert_allclose(result.real, iq.real, atol=1e-6)
        npt.assert_allclose(result.imag, 2.0 * iq.imag, atol=1e-6)

    def test_output_shape_and_dtype(self):
        from spectra.impairments.iq_imbalance import IQImbalance

        iq = np.ones(256, dtype=np.complex64)
        desc = _make_desc()
        result, _ = IQImbalance(gain_imbalance=1.1, phase_imbalance=0.05)(iq, desc)
        assert result.shape == iq.shape
        assert result.dtype == np.complex64

    def test_range_randomizes(self):
        from spectra.impairments.iq_imbalance import IQImbalance

        iq = np.ones(512, dtype=np.complex64) * (1 + 1j)
        desc = _make_desc()
        imb = IQImbalance(
            gain_imbalance_range=(0.8, 1.2), phase_imbalance_range=(-0.1, 0.1)
        )
        results = [imb(iq.copy(), desc)[0] for _ in range(20)]
        # At least some variation expected
        qs = [r.imag[0] for r in results]
        assert max(qs) - min(qs) > 0.01

    def test_desc_unchanged(self):
        from spectra.impairments.iq_imbalance import IQImbalance

        iq = np.ones(256, dtype=np.complex64)
        desc = _make_desc()
        _, new_desc = IQImbalance(gain_imbalance=1.1, phase_imbalance=0.05)(iq, desc)
        assert new_desc.f_low == desc.f_low
        assert new_desc.f_high == desc.f_high

    def test_requires_gain_param(self):
        from spectra.impairments.iq_imbalance import IQImbalance

        with pytest.raises(ValueError):
            IQImbalance(phase_imbalance=0.1)

    def test_requires_phase_param(self):
        from spectra.impairments.iq_imbalance import IQImbalance

        with pytest.raises(ValueError):
            IQImbalance(gain_imbalance=1.0)
