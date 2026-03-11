import numpy as np
import numpy.testing as npt
import pytest


# ---------------------------------------------------------------------------
# ADS-B
# ---------------------------------------------------------------------------


class TestADSBWaveform:
    def test_generate_returns_valid_iq(self, assert_valid_iq, sample_rate):
        from spectra.waveforms import ADSB

        waveform = ADSB()
        iq = waveform.generate(num_symbols=2, sample_rate=sample_rate)
        assert_valid_iq(iq)

    def test_label(self):
        from spectra.waveforms import ADSB

        assert ADSB().label == "ADSB"

    def test_bandwidth(self, sample_rate):
        from spectra.waveforms import ADSB

        spc = 10
        waveform = ADSB(samples_per_chip=spc)
        expected_bw = sample_rate / spc
        assert waveform.bandwidth(sample_rate) == pytest.approx(expected_bw)

    def test_deterministic_with_seed(self, sample_rate):
        from spectra.waveforms import ADSB

        waveform = ADSB()
        iq1 = waveform.generate(num_symbols=2, sample_rate=sample_rate, seed=42)
        iq2 = waveform.generate(num_symbols=2, sample_rate=sample_rate, seed=42)
        npt.assert_array_equal(iq1, iq2)

    def test_ppm_structure(self, sample_rate):
        """PPM preamble should have non-zero samples at pulse positions."""
        from spectra.waveforms import ADSB

        spc = 10
        waveform = ADSB(samples_per_chip=spc)
        iq = waveform.generate(num_symbols=1, sample_rate=sample_rate, seed=0)
        # Preamble pulses at chip positions 0, 2, 7, 9
        # Each chip is spc samples
        for chip_pos in [0, 2, 7, 9]:
            sample_idx = chip_pos * spc
            assert np.abs(iq[sample_idx]) > 0, (
                f"Expected non-zero at preamble chip {chip_pos}"
            )
        # Chip 1 should be zero (no pulse)
        sample_idx = 1 * spc
        assert np.abs(iq[sample_idx]) == 0, "Expected zero between preamble pulses"


# ---------------------------------------------------------------------------
# Mode S
# ---------------------------------------------------------------------------


class TestModeSWaveform:
    def test_generate_returns_valid_iq(self, assert_valid_iq, sample_rate):
        from spectra.waveforms import ModeS

        waveform = ModeS()
        iq = waveform.generate(num_symbols=2, sample_rate=sample_rate)
        assert_valid_iq(iq)

    def test_label(self):
        from spectra.waveforms import ModeS

        assert ModeS().label == "ModeS"

    def test_message_length_56(self, assert_valid_iq, sample_rate):
        from spectra.waveforms import ModeS

        waveform = ModeS(message_length=56)
        iq = waveform.generate(num_symbols=1, sample_rate=sample_rate, seed=42)
        assert_valid_iq(iq)

    def test_message_length_112(self, assert_valid_iq, sample_rate):
        from spectra.waveforms import ModeS

        waveform = ModeS(message_length=112)
        iq = waveform.generate(num_symbols=1, sample_rate=sample_rate, seed=42)
        assert_valid_iq(iq)

    def test_invalid_message_length(self):
        from spectra.waveforms import ModeS

        with pytest.raises(ValueError):
            ModeS(message_length=64)

    def test_deterministic_with_seed(self, sample_rate):
        from spectra.waveforms import ModeS

        waveform = ModeS()
        iq1 = waveform.generate(num_symbols=2, sample_rate=sample_rate, seed=42)
        iq2 = waveform.generate(num_symbols=2, sample_rate=sample_rate, seed=42)
        npt.assert_array_equal(iq1, iq2)

    def test_56_shorter_than_112(self, sample_rate):
        from spectra.waveforms import ModeS

        spc = 10
        w56 = ModeS(message_length=56, samples_per_chip=spc)
        w112 = ModeS(message_length=112, samples_per_chip=spc)
        iq56 = w56.generate(num_symbols=1, sample_rate=sample_rate, seed=42)
        iq112 = w112.generate(num_symbols=1, sample_rate=sample_rate, seed=42)
        assert len(iq56) < len(iq112)


# ---------------------------------------------------------------------------
# AIS
# ---------------------------------------------------------------------------


class TestAISWaveform:
    def test_generate_returns_valid_iq(self, assert_valid_iq, sample_rate):
        from spectra.waveforms import AIS

        waveform = AIS()
        iq = waveform.generate(num_symbols=2, sample_rate=sample_rate)
        assert_valid_iq(iq)

    def test_label(self):
        from spectra.waveforms import AIS

        assert AIS().label == "AIS"

    def test_bandwidth(self, sample_rate):
        from spectra.waveforms import AIS

        sps = 8
        bt = 0.4
        waveform = AIS(samples_per_symbol=sps, bt=bt)
        symbol_rate = sample_rate / sps
        expected_bw = symbol_rate * (1.0 + bt)
        assert waveform.bandwidth(sample_rate) == pytest.approx(expected_bw)

    def test_deterministic_with_seed(self, sample_rate):
        from spectra.waveforms import AIS

        waveform = AIS()
        iq1 = waveform.generate(num_symbols=2, sample_rate=sample_rate, seed=42)
        iq2 = waveform.generate(num_symbols=2, sample_rate=sample_rate, seed=42)
        npt.assert_array_equal(iq1, iq2)

    def test_gmsk_constant_envelope(self, sample_rate):
        """GMSK should produce roughly constant-envelope signal."""
        from spectra.waveforms import AIS

        waveform = AIS()
        iq = waveform.generate(num_symbols=4, sample_rate=sample_rate, seed=42)
        magnitudes = np.abs(iq)
        # GMSK is constant-envelope, magnitudes should all be ~1.0
        npt.assert_allclose(magnitudes, 1.0, atol=1e-5)


# ---------------------------------------------------------------------------
# ACARS
# ---------------------------------------------------------------------------


class TestACARSWaveform:
    def test_generate_returns_valid_iq(self, assert_valid_iq, sample_rate):
        from spectra.waveforms import ACARS

        waveform = ACARS()
        iq = waveform.generate(num_symbols=2, sample_rate=sample_rate)
        assert_valid_iq(iq)

    def test_label(self):
        from spectra.waveforms import ACARS

        assert ACARS().label == "ACARS"

    def test_bandwidth(self, sample_rate):
        from spectra.waveforms import ACARS

        sps = 8
        waveform = ACARS(samples_per_symbol=sps)
        symbol_rate = sample_rate / sps
        expected_bw = symbol_rate * 1.5
        assert waveform.bandwidth(sample_rate) == pytest.approx(expected_bw)

    def test_deterministic_with_seed(self, sample_rate):
        from spectra.waveforms import ACARS

        waveform = ACARS()
        iq1 = waveform.generate(num_symbols=2, sample_rate=sample_rate, seed=42)
        iq2 = waveform.generate(num_symbols=2, sample_rate=sample_rate, seed=42)
        npt.assert_array_equal(iq1, iq2)

    def test_msk_constant_envelope(self, sample_rate):
        """MSK should produce constant-envelope signal."""
        from spectra.waveforms import ACARS

        waveform = ACARS()
        iq = waveform.generate(num_symbols=4, sample_rate=sample_rate, seed=42)
        magnitudes = np.abs(iq)
        npt.assert_allclose(magnitudes, 1.0, atol=1e-5)


# ---------------------------------------------------------------------------
# DME
# ---------------------------------------------------------------------------


class TestDMEWaveform:
    def test_generate_returns_valid_iq(self, assert_valid_iq, sample_rate):
        from spectra.waveforms import DME

        waveform = DME()
        iq = waveform.generate(num_symbols=3, sample_rate=sample_rate)
        assert_valid_iq(iq)

    def test_label(self):
        from spectra.waveforms import DME

        assert DME().label == "DME"

    def test_deterministic_with_seed(self, sample_rate):
        from spectra.waveforms import DME

        waveform = DME()
        iq1 = waveform.generate(num_symbols=3, sample_rate=sample_rate, seed=42)
        iq2 = waveform.generate(num_symbols=3, sample_rate=sample_rate, seed=42)
        npt.assert_array_equal(iq1, iq2)

    def test_pulse_pair_structure(self, sample_rate):
        """DME signal should have pulse pairs with the expected spacing."""
        from spectra.waveforms import DME

        pulse_spacing_us = 12.0
        samples_per_us = 10
        waveform = DME(
            pulse_spacing_us=pulse_spacing_us,
            samples_per_us=samples_per_us,
        )
        iq = waveform.generate(num_symbols=1, sample_rate=sample_rate, seed=42)
        envelope = np.abs(iq.real)
        # Find peaks (above half max)
        threshold = np.max(envelope) * 0.5
        above = envelope > threshold
        # Should have two distinct pulse regions
        transitions = np.diff(above.astype(int))
        rises = np.where(transitions == 1)[0]
        assert len(rises) >= 2, "DME should have at least 2 pulse regions"


# ---------------------------------------------------------------------------
# ILS Localizer
# ---------------------------------------------------------------------------


class TestILSLocalizerWaveform:
    def test_generate_returns_valid_iq(self, assert_valid_iq, sample_rate):
        from spectra.waveforms import ILS_Localizer

        waveform = ILS_Localizer()
        iq = waveform.generate(num_symbols=64, sample_rate=sample_rate)
        assert_valid_iq(iq)

    def test_label(self):
        from spectra.waveforms import ILS_Localizer

        assert ILS_Localizer().label == "ILS_Localizer"

    def test_bandwidth(self, sample_rate):
        from spectra.waveforms import ILS_Localizer

        waveform = ILS_Localizer()
        assert waveform.bandwidth(sample_rate) == pytest.approx(300.0)

    def test_deterministic_with_seed(self, sample_rate):
        from spectra.waveforms import ILS_Localizer

        waveform = ILS_Localizer()
        iq1 = waveform.generate(num_symbols=32, sample_rate=sample_rate, seed=42)
        iq2 = waveform.generate(num_symbols=32, sample_rate=sample_rate, seed=42)
        npt.assert_array_equal(iq1, iq2)

    def test_90_and_150_hz_tones(self, sample_rate):
        """ILS Localizer should contain 90 Hz and 150 Hz tones."""
        from spectra.waveforms import ILS_Localizer

        # Need enough samples for spectral resolution at 90/150 Hz
        # With 1 MHz sample rate, use enough symbols
        sps = 256
        # Need at least 1/90 s ~= 11.1 ms, so at 1 MHz = 11100 samples
        # 256 samples_per_symbol * 64 symbols = 16384 samples
        waveform = ILS_Localizer(samples_per_symbol=sps, modulation_depth=0.3)
        iq = waveform.generate(num_symbols=64, sample_rate=sample_rate, seed=42)

        # FFT analysis
        spectrum = np.fft.fft(iq.real)
        freqs = np.fft.fftfreq(len(iq), d=1.0 / sample_rate)
        magnitude = np.abs(spectrum)

        # Find spectral bins near 90 Hz and 150 Hz
        idx_90 = np.argmin(np.abs(freqs - 90.0))
        idx_150 = np.argmin(np.abs(freqs - 150.0))

        # DC component should be dominant (carrier)
        # Tones at 90 and 150 should be above noise floor
        noise_floor = np.median(magnitude)
        assert magnitude[idx_90] > noise_floor * 5, (
            "90 Hz tone should be above noise floor"
        )
        assert magnitude[idx_150] > noise_floor * 5, (
            "150 Hz tone should be above noise floor"
        )


# ---------------------------------------------------------------------------
# Rust frame generator smoke tests
# ---------------------------------------------------------------------------


class TestRustFrameGenerators:
    @pytest.mark.rust
    def test_adsb_frame_basic(self):
        from spectra._rust import generate_adsb_frame

        frame = np.array(generate_adsb_frame(seed=42))
        assert len(frame) == 14
        assert frame[0] >> 3 == 17  # DF=17

    @pytest.mark.rust
    def test_ais_frame_basic(self):
        from spectra._rust import generate_ais_frame

        frame = np.array(generate_ais_frame(seed=42))
        assert len(frame) > 0
        assert frame[0] == 0x7E  # HDLC start flag

    @pytest.mark.rust
    def test_acars_frame_basic(self):
        from spectra._rust import generate_acars_frame

        frame = np.array(generate_acars_frame(seed=42))
        assert frame[0] == 0xAA  # preamble
        assert frame[1] == 0xAA
        assert frame[2] == 0x2B  # sync '+'
        assert frame[3] == 0x2A  # sync '*'

    @pytest.mark.rust
    def test_deterministic(self):
        from spectra._rust import generate_adsb_frame

        f1 = np.array(generate_adsb_frame(seed=99))
        f2 = np.array(generate_adsb_frame(seed=99))
        npt.assert_array_equal(f1, f2)

    @pytest.mark.rust
    def test_mode_s_frame_df11_for_56bit(self):
        from spectra._rust import generate_mode_s_frame

        frame = np.array(generate_mode_s_frame(message_length=56, seed=42))
        assert len(frame) == 7
        assert frame[0] >> 3 == 11, f"56-bit Mode S frame should have DF=11, got {frame[0] >> 3}"

    @pytest.mark.rust
    def test_mode_s_frame_df17_for_112bit(self):
        from spectra._rust import generate_mode_s_frame

        frame = np.array(generate_mode_s_frame(message_length=112, seed=42))
        assert len(frame) == 14
        assert frame[0] >> 3 == 17, f"112-bit Mode S frame should have DF=17, got {frame[0] >> 3}"
