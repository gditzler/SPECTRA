import numpy as np
import numpy.testing as npt
import pytest


class TestNoise:
    def test_generate_returns_valid_iq(self, assert_valid_iq, sample_rate):
        from spectra.waveforms.noise import Noise

        waveform = Noise()
        iq = waveform.generate(num_symbols=128, sample_rate=sample_rate)
        assert_valid_iq(iq)

    def test_label(self):
        from spectra.waveforms.noise import Noise

        assert Noise().label == "Noise"

    def test_bandwidth(self, sample_rate):
        from spectra.waveforms.noise import Noise

        bw_frac = 0.3
        waveform = Noise(bandwidth_fraction=bw_frac)
        assert waveform.bandwidth(sample_rate) == pytest.approx(bw_frac * sample_rate)

    def test_deterministic_with_seed(self, sample_rate):
        from spectra.waveforms.noise import Noise

        waveform = Noise()
        iq1 = waveform.generate(num_symbols=64, sample_rate=sample_rate, seed=42)
        iq2 = waveform.generate(num_symbols=64, sample_rate=sample_rate, seed=42)
        npt.assert_array_equal(iq1, iq2)

    def test_different_seeds_differ(self, sample_rate):
        from spectra.waveforms.noise import Noise

        waveform = Noise()
        iq1 = waveform.generate(num_symbols=64, sample_rate=sample_rate, seed=1)
        iq2 = waveform.generate(num_symbols=64, sample_rate=sample_rate, seed=2)
        assert not np.allclose(iq1, iq2)

    def test_output_is_complex_noise(self, sample_rate):
        from spectra.waveforms.noise import Noise

        waveform = Noise()
        iq = waveform.generate(num_symbols=256, sample_rate=sample_rate, seed=0)
        # Both I and Q should have non-trivial variance
        assert np.std(iq.real) > 0.01
        assert np.std(iq.imag) > 0.01

    def test_unit_power(self, sample_rate):
        from spectra.waveforms.noise import Noise

        waveform = Noise()
        iq = waveform.generate(num_symbols=1024, sample_rate=sample_rate, seed=0)
        power = np.mean(np.abs(iq) ** 2)
        npt.assert_allclose(power, 1.0, atol=0.2)
