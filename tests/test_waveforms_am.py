import numpy as np
import pytest


class TestAMDSBSC:
    def test_generate_returns_valid_iq(self, assert_valid_iq, sample_rate):
        from spectra.waveforms import AMDSB_SC

        iq = AMDSB_SC().generate(num_symbols=4096, sample_rate=sample_rate, seed=42)
        assert_valid_iq(iq)

    def test_label(self):
        from spectra.waveforms import AMDSB_SC

        assert AMDSB_SC().label == "AM-DSB-SC"

    def test_bandwidth(self, sample_rate):
        from spectra.waveforms import AMDSB_SC

        wf = AMDSB_SC(audio_bw_fraction=0.1)
        expected = 2.0 * sample_rate * 0.1
        assert wf.bandwidth(sample_rate) == pytest.approx(expected)

    def test_deterministic(self, sample_rate):
        from spectra.waveforms import AMDSB_SC

        wf = AMDSB_SC()
        iq1 = wf.generate(4096, sample_rate, seed=42)
        iq2 = wf.generate(4096, sample_rate, seed=42)
        np.testing.assert_array_equal(iq1, iq2)


class TestAMDSB:
    def test_generate_returns_valid_iq(self, assert_valid_iq, sample_rate):
        from spectra.waveforms import AMDSB

        iq = AMDSB().generate(num_symbols=4096, sample_rate=sample_rate, seed=42)
        assert_valid_iq(iq)

    def test_label(self):
        from spectra.waveforms import AMDSB

        assert AMDSB().label == "AM-DSB"


class TestAMLSB:
    def test_generate_returns_valid_iq(self, assert_valid_iq, sample_rate):
        from spectra.waveforms import AMLSB

        iq = AMLSB().generate(num_symbols=4096, sample_rate=sample_rate, seed=42)
        assert_valid_iq(iq)

    def test_label(self):
        from spectra.waveforms import AMLSB

        assert AMLSB().label == "AM-LSB"

    def test_ssb_bandwidth(self, sample_rate):
        from spectra.waveforms import AMLSB

        wf = AMLSB(audio_bw_fraction=0.1)
        expected = sample_rate * 0.1
        assert wf.bandwidth(sample_rate) == pytest.approx(expected)


class TestAMUSB:
    def test_generate_returns_valid_iq(self, assert_valid_iq, sample_rate):
        from spectra.waveforms import AMUSB

        iq = AMUSB().generate(num_symbols=4096, sample_rate=sample_rate, seed=42)
        assert_valid_iq(iq)

    def test_label(self):
        from spectra.waveforms import AMUSB

        assert AMUSB().label == "AM-USB"
