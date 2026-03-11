import numpy.testing as npt
import pytest


class TestQAM16Waveform:
    def test_generate_returns_valid_iq(self, assert_valid_iq, sample_rate):
        from spectra.waveforms.qam import QAM16

        waveform = QAM16()
        iq = waveform.generate(num_symbols=128, sample_rate=sample_rate)
        assert_valid_iq(iq)

    def test_label(self):
        from spectra.waveforms.qam import QAM16

        assert QAM16().label == "16QAM"

    def test_bandwidth(self, sample_rate):
        from spectra.waveforms.qam import QAM16

        sps, rolloff = 8, 0.35
        waveform = QAM16(samples_per_symbol=sps, rolloff=rolloff)
        expected_bw = (sample_rate / sps) * (1.0 + rolloff)
        assert waveform.bandwidth(sample_rate) == pytest.approx(expected_bw)

    def test_deterministic_with_seed(self, sample_rate):
        from spectra.waveforms.qam import QAM16

        waveform = QAM16()
        iq1 = waveform.generate(num_symbols=64, sample_rate=sample_rate, seed=42)
        iq2 = waveform.generate(num_symbols=64, sample_rate=sample_rate, seed=42)
        npt.assert_array_equal(iq1, iq2)


class TestQAM64Waveform:
    def test_generate_returns_valid_iq(self, assert_valid_iq, sample_rate):
        from spectra.waveforms.qam import QAM64

        iq = QAM64().generate(num_symbols=128, sample_rate=sample_rate)
        assert_valid_iq(iq)

    def test_label(self):
        from spectra.waveforms.qam import QAM64

        assert QAM64().label == "64QAM"


class TestQAM256Waveform:
    def test_generate_returns_valid_iq(self, assert_valid_iq, sample_rate):
        from spectra.waveforms.qam import QAM256

        iq = QAM256().generate(num_symbols=128, sample_rate=sample_rate)
        assert_valid_iq(iq)

    def test_label(self):
        from spectra.waveforms.qam import QAM256

        assert QAM256().label == "256QAM"
