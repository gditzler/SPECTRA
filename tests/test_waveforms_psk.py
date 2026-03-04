import numpy as np
import numpy.testing as npt
import pytest


class TestQPSKWaveform:
    def test_generate_returns_valid_iq(self, assert_valid_iq, sample_rate):
        from spectra.waveforms import QPSK
        waveform = QPSK()
        iq = waveform.generate(num_symbols=128, sample_rate=sample_rate)
        assert_valid_iq(iq)

    def test_default_parameters(self):
        from spectra.waveforms import QPSK
        waveform = QPSK()
        assert waveform.samples_per_symbol == 8
        assert waveform.rolloff == 0.35

    def test_label(self):
        from spectra.waveforms import QPSK
        assert QPSK().label == "QPSK"

    def test_bandwidth(self, sample_rate):
        from spectra.waveforms import QPSK
        sps = 8
        rolloff = 0.35
        waveform = QPSK(samples_per_symbol=sps, rolloff=rolloff)
        expected_bw = (sample_rate / sps) * (1.0 + rolloff)
        assert waveform.bandwidth(sample_rate) == pytest.approx(expected_bw)

    def test_custom_parameters(self, assert_valid_iq, sample_rate):
        from spectra.waveforms import QPSK
        waveform = QPSK(samples_per_symbol=4, rolloff=0.5, filter_span=6)
        iq = waveform.generate(num_symbols=64, sample_rate=sample_rate)
        assert_valid_iq(iq)

    def test_deterministic_with_seed(self, sample_rate):
        from spectra.waveforms import QPSK
        waveform = QPSK()
        iq1 = waveform.generate(num_symbols=64, sample_rate=sample_rate, seed=42)
        iq2 = waveform.generate(num_symbols=64, sample_rate=sample_rate, seed=42)
        npt.assert_array_equal(iq1, iq2)

    @pytest.mark.parametrize("num_symbols", [1, 16, 256])
    def test_various_lengths(self, num_symbols, assert_valid_iq, sample_rate):
        from spectra.waveforms import QPSK
        waveform = QPSK(samples_per_symbol=4)
        iq = waveform.generate(num_symbols=num_symbols, sample_rate=sample_rate)
        assert_valid_iq(iq)


class TestBPSKWaveform:
    def test_generate_returns_valid_iq(self, assert_valid_iq, sample_rate):
        from spectra.waveforms import BPSK
        waveform = BPSK()
        iq = waveform.generate(num_symbols=128, sample_rate=sample_rate)
        assert_valid_iq(iq)

    def test_label(self):
        from spectra.waveforms import BPSK
        assert BPSK().label == "BPSK"

    def test_constellation_is_real_axis(self, sample_rate):
        """BPSK symbols should lie on the real axis before filtering."""
        from spectra._rust import generate_bpsk_symbols
        symbols = generate_bpsk_symbols(1000, seed=0)
        npt.assert_allclose(symbols.imag, 0.0, atol=1e-6)


class TestPSK8Waveform:
    def test_generate_returns_valid_iq(self, assert_valid_iq, sample_rate):
        from spectra.waveforms.psk import PSK8
        waveform = PSK8()
        iq = waveform.generate(num_symbols=128, sample_rate=sample_rate)
        assert_valid_iq(iq)

    def test_label(self):
        from spectra.waveforms.psk import PSK8
        assert PSK8().label == "8PSK"

    def test_bandwidth(self, sample_rate):
        from spectra.waveforms.psk import PSK8
        sps, rolloff = 8, 0.35
        waveform = PSK8(samples_per_symbol=sps, rolloff=rolloff)
        expected_bw = (sample_rate / sps) * (1.0 + rolloff)
        assert waveform.bandwidth(sample_rate) == pytest.approx(expected_bw)

    def test_deterministic_with_seed(self, sample_rate):
        from spectra.waveforms.psk import PSK8
        waveform = PSK8()
        iq1 = waveform.generate(num_symbols=64, sample_rate=sample_rate, seed=42)
        iq2 = waveform.generate(num_symbols=64, sample_rate=sample_rate, seed=42)
        npt.assert_array_equal(iq1, iq2)

    def test_constellation_on_unit_circle(self):
        """8PSK symbols should lie on the unit circle before filtering."""
        from spectra._rust import generate_8psk_symbols
        symbols = generate_8psk_symbols(1000, seed=0)
        npt.assert_allclose(np.abs(symbols), 1.0, atol=1e-6)
