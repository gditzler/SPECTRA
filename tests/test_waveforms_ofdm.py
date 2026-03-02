import numpy as np
import numpy.testing as npt
import pytest


class TestOFDMWaveform:
    def test_generate_returns_valid_iq(self, assert_valid_iq, sample_rate):
        from spectra.waveforms.ofdm import OFDM
        waveform = OFDM()
        iq = waveform.generate(num_symbols=10, sample_rate=sample_rate)
        assert_valid_iq(iq)

    def test_output_length(self, sample_rate):
        from spectra.waveforms.ofdm import OFDM
        nsc, fft, cp = 64, 256, 16
        waveform = OFDM(num_subcarriers=nsc, fft_size=fft, cp_length=cp)
        iq = waveform.generate(num_symbols=5, sample_rate=sample_rate)
        assert len(iq) == 5 * (fft + cp)

    def test_label(self):
        from spectra.waveforms.ofdm import OFDM
        assert OFDM().label == "OFDM"

    def test_bandwidth(self, sample_rate):
        from spectra.waveforms.ofdm import OFDM
        nsc, fft = 64, 256
        waveform = OFDM(num_subcarriers=nsc, fft_size=fft)
        expected_bw = nsc * sample_rate / fft
        assert waveform.bandwidth(sample_rate) == pytest.approx(expected_bw)

    def test_samples_per_symbol_attribute(self):
        from spectra.waveforms.ofdm import OFDM
        waveform = OFDM(fft_size=256, cp_length=16)
        assert waveform.samples_per_symbol == 272

    def test_deterministic_with_seed(self, sample_rate):
        from spectra.waveforms.ofdm import OFDM
        waveform = OFDM()
        iq1 = waveform.generate(num_symbols=5, sample_rate=sample_rate, seed=42)
        iq2 = waveform.generate(num_symbols=5, sample_rate=sample_rate, seed=42)
        npt.assert_array_equal(iq1, iq2)

    def test_no_cyclic_prefix(self, assert_valid_iq, sample_rate):
        from spectra.waveforms.ofdm import OFDM
        waveform = OFDM(cp_length=0)
        iq = waveform.generate(num_symbols=5, sample_rate=sample_rate)
        assert_valid_iq(iq)
        assert len(iq) == 5 * 256  # default fft_size, no CP
