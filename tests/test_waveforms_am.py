import numpy as np
import numpy.testing as npt
import pytest


class TestAM:
    def test_generate_returns_valid_iq(self, assert_valid_iq, sample_rate):
        from spectra.waveforms.am import AM

        waveform = AM()
        iq = waveform.generate(num_symbols=128, sample_rate=sample_rate)
        assert_valid_iq(iq)

    def test_label(self):
        from spectra.waveforms.am import AM

        assert AM().label == "AM"

    def test_bandwidth(self, sample_rate):
        from spectra.waveforms.am import AM

        msg_bw = 5e3
        waveform = AM(message_bandwidth=msg_bw)
        assert waveform.bandwidth(sample_rate) == pytest.approx(2.0 * msg_bw)

    def test_deterministic_with_seed(self, sample_rate):
        from spectra.waveforms.am import AM

        waveform = AM()
        iq1 = waveform.generate(num_symbols=64, sample_rate=sample_rate, seed=42)
        iq2 = waveform.generate(num_symbols=64, sample_rate=sample_rate, seed=42)
        npt.assert_array_equal(iq1, iq2)

    def test_different_seeds_differ(self, sample_rate):
        from spectra.waveforms.am import AM

        waveform = AM()
        iq1 = waveform.generate(num_symbols=64, sample_rate=sample_rate, seed=1)
        iq2 = waveform.generate(num_symbols=64, sample_rate=sample_rate, seed=2)
        assert not np.allclose(iq1, iq2)

    def test_envelope_positive(self, sample_rate):
        from spectra.waveforms.am import AM

        waveform = AM(mod_index=0.5)
        iq = waveform.generate(num_symbols=128, sample_rate=sample_rate, seed=0)
        # With mod_index <= 1.0, envelope (1 + m*msg) should be non-negative
        # since |msg| <= 1 and m <= 1
        assert np.all(iq.real >= -0.1)  # small tolerance for filtering edge effects

    def test_custom_parameters(self, assert_valid_iq, sample_rate):
        from spectra.waveforms.am import AM

        waveform = AM(mod_index=0.8, message_bandwidth=10e3, samples_per_symbol=16)
        iq = waveform.generate(num_symbols=64, sample_rate=sample_rate)
        assert_valid_iq(iq)
