import numpy as np
import numpy.testing as npt
import pytest

from conftest import make_signal_description


class TestSampleRateOffset:
    def test_zero_ppm_preserves_signal(self):
        from spectra.impairments.sample_rate_offset import SampleRateOffset

        iq = np.exp(1j * np.linspace(0, 10 * np.pi, 1024)).astype(np.complex64)
        desc = make_signal_description()
        result, _ = SampleRateOffset(ppm=0.0)(iq, desc)
        npt.assert_allclose(result, iq, atol=1e-4)

    def test_output_length_preserved(self):
        from spectra.impairments.sample_rate_offset import SampleRateOffset

        iq = np.ones(512, dtype=np.complex64)
        desc = make_signal_description()
        result, _ = SampleRateOffset(ppm=100.0)(iq, desc)
        assert len(result) == len(iq)

    def test_output_dtype(self):
        from spectra.impairments.sample_rate_offset import SampleRateOffset

        iq = np.ones(256, dtype=np.complex64)
        desc = make_signal_description()
        result, _ = SampleRateOffset(ppm=50.0)(iq, desc)
        assert result.dtype == np.complex64

    def test_nonzero_ppm_modifies_signal(self):
        from spectra.impairments.sample_rate_offset import SampleRateOffset

        iq = np.exp(1j * np.linspace(0, 20 * np.pi, 1024)).astype(np.complex64)
        desc = make_signal_description()
        result, _ = SampleRateOffset(ppm=500.0)(iq, desc)
        assert not np.allclose(result, iq, atol=1e-3)

    def test_max_ppm_randomizes(self):
        from spectra.impairments.sample_rate_offset import SampleRateOffset

        iq = np.exp(1j * np.linspace(0, 10 * np.pi, 512)).astype(np.complex64)
        desc = make_signal_description()
        sro = SampleRateOffset(max_ppm=200.0)
        results = [sro(iq.copy(), desc)[0] for _ in range(20)]
        diffs = [np.max(np.abs(results[i] - results[i + 1])) for i in range(19)]
        assert not all(d < 1e-6 for d in diffs)

    def test_desc_unchanged(self):
        from spectra.impairments.sample_rate_offset import SampleRateOffset

        iq = np.ones(256, dtype=np.complex64)
        desc = make_signal_description()
        _, new_desc = SampleRateOffset(ppm=10.0)(iq, desc)
        assert new_desc.f_low == desc.f_low

    def test_requires_param(self):
        from spectra.impairments.sample_rate_offset import SampleRateOffset

        with pytest.raises(ValueError):
            SampleRateOffset()
