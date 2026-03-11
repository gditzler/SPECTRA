import numpy as np
import numpy.testing as npt
import pytest


class TestLFMWaveform:
    def test_generate_returns_valid_iq(self, assert_valid_iq, sample_rate):
        from spectra.waveforms.lfm import LFM

        waveform = LFM()
        iq = waveform.generate(num_symbols=4, sample_rate=sample_rate)
        assert_valid_iq(iq)

    def test_output_length(self, sample_rate):
        from spectra.waveforms.lfm import LFM

        spp = 256
        waveform = LFM(samples_per_pulse=spp)
        iq = waveform.generate(num_symbols=3, sample_rate=sample_rate)
        assert len(iq) == 3 * spp

    def test_label(self):
        from spectra.waveforms.lfm import LFM

        assert LFM().label == "LFM"

    def test_bandwidth(self, sample_rate):
        from spectra.waveforms.lfm import LFM

        waveform = LFM(bandwidth_fraction=0.4)
        assert waveform.bandwidth(sample_rate) == pytest.approx(0.4 * sample_rate)

    def test_unit_magnitude(self, sample_rate):
        from spectra.waveforms.lfm import LFM

        waveform = LFM()
        iq = waveform.generate(num_symbols=2, sample_rate=sample_rate)
        npt.assert_allclose(np.abs(iq), 1.0, atol=1e-5)

    def test_samples_per_symbol_attribute(self):
        from spectra.waveforms.lfm import LFM

        waveform = LFM(samples_per_pulse=512)
        assert waveform.samples_per_symbol == 512

    def test_deterministic(self, sample_rate):
        from spectra.waveforms.lfm import LFM

        waveform = LFM()
        iq1 = waveform.generate(num_symbols=4, sample_rate=sample_rate, seed=42)
        iq2 = waveform.generate(num_symbols=4, sample_rate=sample_rate, seed=42)
        npt.assert_array_equal(iq1, iq2)
