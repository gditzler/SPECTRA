import numpy as np
import numpy.testing as npt


class TestToneWaveform:
    def test_generate_returns_valid_iq(self, assert_valid_iq, sample_rate):
        from spectra.waveforms import Tone

        waveform = Tone()
        iq = waveform.generate(num_symbols=1024, sample_rate=sample_rate)
        assert_valid_iq(iq)

    def test_label(self):
        from spectra.waveforms import Tone

        assert Tone().label == "Tone"

    def test_bandwidth_is_zero(self, sample_rate):
        from spectra.waveforms import Tone

        assert Tone().bandwidth(sample_rate) == 0.0

    def test_constant_envelope(self, sample_rate):
        from spectra.waveforms import Tone

        iq = Tone(frequency=1000.0).generate(1024, sample_rate)
        magnitudes = np.abs(iq)
        npt.assert_allclose(magnitudes, magnitudes[0], atol=1e-5)

    def test_length(self, sample_rate):
        from spectra.waveforms import Tone

        iq = Tone().generate(num_symbols=512, sample_rate=sample_rate)
        assert len(iq) == 512
