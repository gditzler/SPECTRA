import numpy as np
import numpy.testing as npt
import pytest


class TestFMWaveform:
    def test_generate_returns_valid_iq(self, assert_valid_iq, sample_rate):
        from spectra.waveforms import FM

        iq = FM().generate(num_symbols=4096, sample_rate=sample_rate, seed=42)
        assert_valid_iq(iq)

    def test_label(self):
        from spectra.waveforms import FM

        assert FM().label == "FM"

    def test_constant_envelope(self, sample_rate):
        from spectra.waveforms import FM

        iq = FM().generate(4096, sample_rate, seed=42)
        npt.assert_allclose(np.abs(iq), 1.0, atol=1e-5)

    def test_bandwidth_carsons_rule(self, sample_rate):
        from spectra.waveforms import FM

        dev = 0.1
        audio = 0.05
        wf = FM(deviation_fraction=dev, audio_bw_fraction=audio)
        expected = 2.0 * (sample_rate * dev + sample_rate * audio)
        assert wf.bandwidth(sample_rate) == pytest.approx(expected)

    def test_deterministic(self, sample_rate):
        from spectra.waveforms import FM

        wf = FM()
        iq1 = wf.generate(4096, sample_rate, seed=42)
        iq2 = wf.generate(4096, sample_rate, seed=42)
        npt.assert_array_equal(iq1, iq2)
