import numpy as np
import numpy.testing as npt
import pytest

from tests.helpers import make_signal_description


class TestDCOffset:
    def test_fixed_offset_adds_dc(self):
        from spectra.impairments.dc_offset import DCOffset

        iq = np.zeros(1024, dtype=np.complex64)
        desc = make_signal_description()
        result, _ = DCOffset(offset=0.1 + 0.2j)(iq, desc)
        npt.assert_allclose(result.real, 0.1, atol=1e-6)
        npt.assert_allclose(result.imag, 0.2, atol=1e-6)

    def test_zero_offset_preserves_signal(self):
        from spectra.impairments.dc_offset import DCOffset

        iq = np.ones(512, dtype=np.complex64)
        desc = make_signal_description()
        result, _ = DCOffset(offset=0.0 + 0.0j)(iq, desc)
        npt.assert_allclose(result, iq, atol=1e-6)

    def test_max_offset_randomizes(self):
        from spectra.impairments.dc_offset import DCOffset

        iq = np.zeros(256, dtype=np.complex64)
        desc = make_signal_description()
        dc = DCOffset(max_offset=0.5)
        means = [np.mean(dc(iq.copy(), desc)[0]) for _ in range(20)]
        reals = [m.real for m in means]
        assert max(reals) - min(reals) > 0.01

    def test_output_shape_and_dtype(self):
        from spectra.impairments.dc_offset import DCOffset

        iq = np.ones(256, dtype=np.complex64)
        desc = make_signal_description()
        result, _ = DCOffset(offset=0.05 + 0.05j)(iq, desc)
        assert result.shape == iq.shape
        assert result.dtype == np.complex64

    def test_desc_unchanged(self):
        from spectra.impairments.dc_offset import DCOffset

        iq = np.ones(256, dtype=np.complex64)
        desc = make_signal_description()
        _, new_desc = DCOffset(offset=0.1 + 0.0j)(iq, desc)
        assert new_desc.f_low == desc.f_low

    def test_requires_param(self):
        from spectra.impairments.dc_offset import DCOffset

        with pytest.raises(ValueError):
            DCOffset()
