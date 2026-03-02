import numpy as np
import numpy.testing as npt
import pytest


class TestChirpSSWaveform:
    def test_generate_returns_valid_iq(self, assert_valid_iq, sample_rate):
        from spectra.waveforms import ChirpSS

        iq = ChirpSS().generate(num_symbols=10, sample_rate=sample_rate, seed=42)
        assert_valid_iq(iq)

    def test_label(self):
        from spectra.waveforms import ChirpSS

        assert ChirpSS().label == "ChirpSS"

    def test_constant_envelope(self, sample_rate):
        from spectra.waveforms import ChirpSS

        iq = ChirpSS().generate(10, sample_rate, seed=42)
        npt.assert_allclose(np.abs(iq), 1.0, atol=1e-5)

    def test_correct_symbol_count(self, sample_rate):
        from spectra.waveforms import ChirpSS

        sf = 7
        num_symbols = 10
        wf = ChirpSS(spreading_factor=sf)
        iq = wf.generate(num_symbols, sample_rate, seed=42)
        expected_length = num_symbols * (2**sf)
        assert len(iq) == expected_length

    def test_bandwidth(self, sample_rate):
        from spectra.waveforms import ChirpSS

        bw_frac = 0.5
        wf = ChirpSS(bandwidth_fraction=bw_frac)
        assert wf.bandwidth(sample_rate) == pytest.approx(sample_rate * bw_frac)

    def test_deterministic(self, sample_rate):
        from spectra.waveforms import ChirpSS

        wf = ChirpSS()
        iq1 = wf.generate(10, sample_rate, seed=42)
        iq2 = wf.generate(10, sample_rate, seed=42)
        npt.assert_array_equal(iq1, iq2)
