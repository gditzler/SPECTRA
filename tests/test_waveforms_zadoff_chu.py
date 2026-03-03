import numpy as np
import numpy.testing as npt
import pytest


class TestZadoffChu:
    def test_generate_returns_valid_iq(self, assert_valid_iq, sample_rate):
        from spectra.waveforms.zadoff_chu import ZadoffChu

        waveform = ZadoffChu(length=63, root=25)
        iq = waveform.generate(num_symbols=5, sample_rate=sample_rate)
        assert_valid_iq(iq)

    def test_label(self):
        from spectra.waveforms.zadoff_chu import ZadoffChu

        assert ZadoffChu(length=63, root=25).label == "ZadoffChu"

    def test_unit_magnitude(self, sample_rate):
        from spectra.waveforms.zadoff_chu import ZadoffChu

        waveform = ZadoffChu(length=31, root=7)
        iq = waveform.generate(num_symbols=3, sample_rate=sample_rate)
        npt.assert_allclose(np.abs(iq), 1.0, atol=1e-5)

    def test_bandwidth(self, sample_rate):
        from spectra.waveforms.zadoff_chu import ZadoffChu

        spc = 16
        waveform = ZadoffChu(length=63, root=25, samples_per_chip=spc)
        assert waveform.bandwidth(sample_rate) == pytest.approx(sample_rate / spc)

    def test_output_length(self, sample_rate):
        from spectra.waveforms.zadoff_chu import ZadoffChu

        length, spc, n_sym = 31, 8, 4
        waveform = ZadoffChu(length=length, root=7, samples_per_chip=spc)
        iq = waveform.generate(num_symbols=n_sym, sample_rate=sample_rate)
        assert len(iq) == n_sym * length * spc

    def test_deterministic(self, sample_rate):
        from spectra.waveforms.zadoff_chu import ZadoffChu

        waveform = ZadoffChu(length=63, root=25)
        iq1 = waveform.generate(num_symbols=3, sample_rate=sample_rate, seed=0)
        iq2 = waveform.generate(num_symbols=3, sample_rate=sample_rate, seed=99)
        npt.assert_array_equal(iq1, iq2)  # Deterministic, seed irrelevant

    def test_different_roots_differ(self, sample_rate):
        from spectra.waveforms.zadoff_chu import ZadoffChu

        iq1 = ZadoffChu(length=31, root=1).generate(3, sample_rate)
        iq2 = ZadoffChu(length=31, root=5).generate(3, sample_rate)
        assert not np.allclose(iq1, iq2)

    def test_invalid_root_raises(self):
        from spectra.waveforms.zadoff_chu import ZadoffChu

        with pytest.raises(ValueError):
            ZadoffChu(length=31, root=0)

    def test_root_equals_length_raises(self):
        from spectra.waveforms.zadoff_chu import ZadoffChu

        with pytest.raises(ValueError):
            ZadoffChu(length=31, root=31)
