import numpy as np
import numpy.testing as npt
import pytest


class TestFrankCodeWaveform:
    def test_generate_returns_valid_iq(self, assert_valid_iq, sample_rate):
        from spectra.waveforms.polyphase import FrankCode

        waveform = FrankCode()
        iq = waveform.generate(num_symbols=2, sample_rate=sample_rate)
        assert_valid_iq(iq)

    def test_output_length(self, sample_rate):
        from spectra.waveforms.polyphase import FrankCode

        order, spc = 4, 8
        waveform = FrankCode(code_order=order, samples_per_chip=spc)
        iq = waveform.generate(num_symbols=3, sample_rate=sample_rate)
        assert len(iq) == 3 * order * order * spc

    def test_label(self):
        from spectra.waveforms.polyphase import FrankCode

        assert FrankCode().label == "Frank"

    def test_bandwidth(self, sample_rate):
        from spectra.waveforms.polyphase import FrankCode

        spc = 8
        waveform = FrankCode(samples_per_chip=spc)
        assert waveform.bandwidth(sample_rate) == pytest.approx(sample_rate / spc)

    def test_unit_magnitude(self, sample_rate):
        from spectra.waveforms.polyphase import FrankCode

        waveform = FrankCode()
        iq = waveform.generate(num_symbols=1, sample_rate=sample_rate)
        npt.assert_allclose(np.abs(iq), 1.0, atol=1e-5)

    def test_samples_per_symbol_attribute(self):
        from spectra.waveforms.polyphase import FrankCode

        waveform = FrankCode(code_order=5, samples_per_chip=4)
        assert waveform.samples_per_symbol == 5 * 5 * 4


class TestP1CodeWaveform:
    def test_generate_returns_valid_iq(self, assert_valid_iq, sample_rate):
        from spectra.waveforms.polyphase import P1Code

        waveform = P1Code()
        iq = waveform.generate(num_symbols=2, sample_rate=sample_rate)
        assert_valid_iq(iq)

    def test_label(self):
        from spectra.waveforms.polyphase import P1Code

        assert P1Code().label == "P1"

    def test_output_length(self, sample_rate):
        from spectra.waveforms.polyphase import P1Code

        waveform = P1Code(code_order=4, samples_per_chip=8)
        iq = waveform.generate(num_symbols=2, sample_rate=sample_rate)
        assert len(iq) == 2 * 16 * 8

    def test_unit_magnitude(self, sample_rate):
        from spectra.waveforms.polyphase import P1Code

        iq = P1Code().generate(num_symbols=1, sample_rate=sample_rate)
        npt.assert_allclose(np.abs(iq), 1.0, atol=1e-5)


class TestP2CodeWaveform:
    def test_generate_returns_valid_iq(self, assert_valid_iq, sample_rate):
        from spectra.waveforms.polyphase import P2Code

        waveform = P2Code()
        iq = waveform.generate(num_symbols=2, sample_rate=sample_rate)
        assert_valid_iq(iq)

    def test_label(self):
        from spectra.waveforms.polyphase import P2Code

        assert P2Code().label == "P2"

    def test_rejects_odd_order(self):
        from spectra.waveforms.polyphase import P2Code

        with pytest.raises(ValueError, match="even"):
            P2Code(code_order=5)

    def test_unit_magnitude(self, sample_rate):
        from spectra.waveforms.polyphase import P2Code

        iq = P2Code().generate(num_symbols=1, sample_rate=sample_rate)
        npt.assert_allclose(np.abs(iq), 1.0, atol=1e-5)


class TestP3CodeWaveform:
    def test_generate_returns_valid_iq(self, assert_valid_iq, sample_rate):
        from spectra.waveforms.polyphase import P3Code

        waveform = P3Code()
        iq = waveform.generate(num_symbols=2, sample_rate=sample_rate)
        assert_valid_iq(iq)

    def test_label(self):
        from spectra.waveforms.polyphase import P3Code

        assert P3Code().label == "P3"

    def test_output_length(self, sample_rate):
        from spectra.waveforms.polyphase import P3Code

        waveform = P3Code(code_length=25, samples_per_chip=4)
        iq = waveform.generate(num_symbols=3, sample_rate=sample_rate)
        assert len(iq) == 3 * 25 * 4

    def test_unit_magnitude(self, sample_rate):
        from spectra.waveforms.polyphase import P3Code

        iq = P3Code().generate(num_symbols=1, sample_rate=sample_rate)
        npt.assert_allclose(np.abs(iq), 1.0, atol=1e-5)

    def test_arbitrary_length(self, assert_valid_iq, sample_rate):
        from spectra.waveforms.polyphase import P3Code

        for n in [9, 25, 64, 100]:
            waveform = P3Code(code_length=n)
            iq = waveform.generate(num_symbols=1, sample_rate=sample_rate)
            assert_valid_iq(iq)


class TestP4CodeWaveform:
    def test_generate_returns_valid_iq(self, assert_valid_iq, sample_rate):
        from spectra.waveforms.polyphase import P4Code

        waveform = P4Code()
        iq = waveform.generate(num_symbols=2, sample_rate=sample_rate)
        assert_valid_iq(iq)

    def test_label(self):
        from spectra.waveforms.polyphase import P4Code

        assert P4Code().label == "P4"

    def test_output_length(self, sample_rate):
        from spectra.waveforms.polyphase import P4Code

        waveform = P4Code(code_length=36, samples_per_chip=4)
        iq = waveform.generate(num_symbols=2, sample_rate=sample_rate)
        assert len(iq) == 2 * 36 * 4

    def test_unit_magnitude(self, sample_rate):
        from spectra.waveforms.polyphase import P4Code

        iq = P4Code().generate(num_symbols=1, sample_rate=sample_rate)
        npt.assert_allclose(np.abs(iq), 1.0, atol=1e-5)
