import numpy as np
import numpy.testing as npt


class TestFSKWaveform:
    def test_generate_returns_valid_iq(self, assert_valid_iq, sample_rate):
        from spectra.waveforms.fsk import FSK

        waveform = FSK()
        iq = waveform.generate(num_symbols=128, sample_rate=sample_rate)
        assert_valid_iq(iq)

    def test_output_length(self, sample_rate):
        from spectra.waveforms.fsk import FSK

        sps = 8
        waveform = FSK(samples_per_symbol=sps)
        iq = waveform.generate(num_symbols=100, sample_rate=sample_rate)
        assert len(iq) == 100 * sps

    def test_label(self):
        from spectra.waveforms.fsk import FSK

        assert FSK().label == "FSK"

    def test_unit_magnitude(self, sample_rate):
        """CPFSK produces constant-envelope signals."""
        from spectra.waveforms.fsk import FSK

        waveform = FSK()
        iq = waveform.generate(num_symbols=64, sample_rate=sample_rate)
        npt.assert_allclose(np.abs(iq), 1.0, atol=1e-5)

    def test_deterministic_with_seed(self, sample_rate):
        from spectra.waveforms.fsk import FSK

        waveform = FSK()
        iq1 = waveform.generate(num_symbols=64, sample_rate=sample_rate, seed=42)
        iq2 = waveform.generate(num_symbols=64, sample_rate=sample_rate, seed=42)
        npt.assert_array_equal(iq1, iq2)

    def test_4fsk(self, assert_valid_iq, sample_rate):
        from spectra.waveforms.fsk import FSK

        waveform = FSK(order=4)
        iq = waveform.generate(num_symbols=64, sample_rate=sample_rate)
        assert_valid_iq(iq)


class TestMSKWaveform:
    def test_generate_returns_valid_iq(self, assert_valid_iq, sample_rate):
        from spectra.waveforms.fsk import MSK

        waveform = MSK()
        iq = waveform.generate(num_symbols=128, sample_rate=sample_rate)
        assert_valid_iq(iq)

    def test_label(self):
        from spectra.waveforms.fsk import MSK

        assert MSK().label == "MSK"

    def test_unit_magnitude(self, sample_rate):
        from spectra.waveforms.fsk import MSK

        iq = MSK().generate(num_symbols=64, sample_rate=sample_rate)
        npt.assert_allclose(np.abs(iq), 1.0, atol=1e-5)


class TestGMSKWaveform:
    def test_generate_returns_valid_iq(self, assert_valid_iq, sample_rate):
        from spectra.waveforms.fsk import GMSK

        waveform = GMSK()
        iq = waveform.generate(num_symbols=128, sample_rate=sample_rate)
        assert_valid_iq(iq)

    def test_label(self):
        from spectra.waveforms.fsk import GMSK

        assert GMSK().label == "GMSK"

    def test_deterministic_with_seed(self, sample_rate):
        from spectra.waveforms.fsk import GMSK

        waveform = GMSK()
        iq1 = waveform.generate(num_symbols=64, sample_rate=sample_rate, seed=42)
        iq2 = waveform.generate(num_symbols=64, sample_rate=sample_rate, seed=42)
        npt.assert_array_equal(iq1, iq2)

    def test_near_unit_magnitude(self, sample_rate):
        """GMSK is approximately constant-envelope after Gaussian filtering."""
        from spectra.waveforms.fsk import GMSK

        iq = GMSK().generate(num_symbols=128, sample_rate=sample_rate)
        # Gaussian filtering makes it not exactly constant-envelope
        # but magnitudes should be close to 1
        npt.assert_allclose(np.abs(iq), 1.0, atol=0.15)
