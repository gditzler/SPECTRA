import numpy as np
import numpy.testing as npt
import pytest


class TestBarkerCode:
    def test_generate_returns_valid_iq(self, assert_valid_iq, sample_rate):
        from spectra.waveforms.barker import BarkerCode

        waveform = BarkerCode(length=13)
        iq = waveform.generate(num_symbols=10, sample_rate=sample_rate)
        assert_valid_iq(iq)

    def test_label(self):
        from spectra.waveforms.barker import BarkerCode

        assert BarkerCode(length=13).label == "Barker"

    def test_unit_magnitude(self, sample_rate):
        from spectra.waveforms.barker import BarkerCode

        waveform = BarkerCode(length=7)
        iq = waveform.generate(num_symbols=5, sample_rate=sample_rate)
        npt.assert_allclose(np.abs(iq), 1.0, atol=1e-6)

    def test_bandwidth(self, sample_rate):
        from spectra.waveforms.barker import BarkerCode

        spc = 16
        waveform = BarkerCode(length=13, samples_per_chip=spc)
        assert waveform.bandwidth(sample_rate) == pytest.approx(sample_rate / spc)

    def test_output_length(self, sample_rate):
        from spectra.waveforms.barker import BarkerCode

        length, spc, n_sym = 13, 8, 5
        waveform = BarkerCode(length=length, samples_per_chip=spc)
        iq = waveform.generate(num_symbols=n_sym, sample_rate=sample_rate)
        assert len(iq) == n_sym * length * spc

    def test_deterministic(self, sample_rate):
        from spectra.waveforms.barker import BarkerCode

        waveform = BarkerCode(length=13)
        iq1 = waveform.generate(num_symbols=5, sample_rate=sample_rate, seed=0)
        iq2 = waveform.generate(num_symbols=5, sample_rate=sample_rate, seed=99)
        npt.assert_array_equal(iq1, iq2)  # Deterministic code, seed irrelevant

    @pytest.mark.parametrize("length", [2, 3, 4, 5, 7, 11, 13])
    def test_all_valid_lengths(self, length, assert_valid_iq, sample_rate):
        from spectra.waveforms.barker import BarkerCode

        waveform = BarkerCode(length=length)
        iq = waveform.generate(num_symbols=3, sample_rate=sample_rate)
        assert_valid_iq(iq)

    def test_invalid_length_raises(self):
        from spectra.waveforms.barker import BarkerCode

        with pytest.raises(ValueError, match="Barker"):
            BarkerCode(length=6)
