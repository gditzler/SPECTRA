import numpy as np
import numpy.testing as npt
import pytest


class TestDSSS_QPSK:
    def test_generate_returns_valid_iq(self, assert_valid_iq, sample_rate):
        from spectra.waveforms import DSSS_QPSK

        iq = DSSS_QPSK().generate(num_symbols=10, sample_rate=sample_rate, seed=42)
        assert_valid_iq(iq)

    def test_label(self):
        from spectra.waveforms import DSSS_QPSK

        assert DSSS_QPSK().label == "DSSS_QPSK"

    def test_bandwidth(self, sample_rate):
        from spectra.waveforms import DSSS_QPSK

        spc = 4
        wf = DSSS_QPSK(samples_per_chip=spc)
        assert wf.bandwidth(sample_rate) == pytest.approx(sample_rate / spc)

    def test_deterministic(self, sample_rate):
        from spectra.waveforms import DSSS_QPSK

        wf = DSSS_QPSK()
        iq1 = wf.generate(10, sample_rate, seed=42)
        iq2 = wf.generate(10, sample_rate, seed=42)
        npt.assert_array_equal(iq1, iq2)

    @pytest.mark.parametrize("code_type", ["msequence", "gold", "kasami"])
    def test_code_types(self, assert_valid_iq, sample_rate, code_type):
        from spectra.waveforms import DSSS_QPSK

        # kasami requires even order
        order = 6 if code_type == "kasami" else 5
        wf = DSSS_QPSK(code_type=code_type, code_order=order)
        iq = wf.generate(10, sample_rate, seed=42)
        assert_valid_iq(iq)

    def test_correct_length_msequence(self, sample_rate):
        from spectra.waveforms import DSSS_QPSK

        order = 5
        spc = 4
        num_symbols = 10
        wf = DSSS_QPSK(code_type="msequence", code_order=order, samples_per_chip=spc)
        iq = wf.generate(num_symbols, sample_rate, seed=42)
        code_len = (2**order) - 1
        expected_len = num_symbols * code_len * spc
        assert len(iq) == expected_len

    def test_kasami_odd_order_raises(self):
        from spectra.waveforms import DSSS_QPSK

        with pytest.raises(ValueError, match="even"):
            DSSS_QPSK(code_type="kasami", code_order=5)


class TestFHSS:
    def test_generate_returns_valid_iq(self, assert_valid_iq, sample_rate):
        from spectra.waveforms import FHSS

        iq = FHSS().generate(num_symbols=10, sample_rate=sample_rate, seed=42)
        assert_valid_iq(iq)

    def test_label(self):
        from spectra.waveforms import FHSS

        assert FHSS().label == "FHSS"

    def test_bandwidth(self, sample_rate):
        from spectra.waveforms import FHSS

        wf = FHSS()
        assert wf.bandwidth(sample_rate) == pytest.approx(sample_rate * 0.8)

    def test_deterministic(self, sample_rate):
        from spectra.waveforms import FHSS

        wf = FHSS()
        iq1 = wf.generate(10, sample_rate, seed=42)
        iq2 = wf.generate(10, sample_rate, seed=42)
        npt.assert_array_equal(iq1, iq2)

    def test_correct_length(self, sample_rate):
        from spectra.waveforms import FHSS

        num_hops = 10
        dwell = 64
        wf = FHSS(dwell_samples=dwell)
        iq = wf.generate(num_hops, sample_rate, seed=42)
        assert len(iq) == num_hops * dwell

    def test_frequency_content(self, sample_rate):
        from spectra.waveforms import FHSS

        # Generate FHSS with many hops — energy should spread across frequencies
        wf = FHSS(num_channels=8, hop_pattern="random", dwell_samples=256)
        iq = wf.generate(100, sample_rate, seed=42)
        spectrum = np.abs(np.fft.fft(iq)) ** 2
        # Energy should not be concentrated in just one frequency bin
        max_bin_power = np.max(spectrum)
        total_power = np.sum(spectrum)
        assert max_bin_power < 0.5 * total_power, "Energy too concentrated for FHSS"

    @pytest.mark.parametrize("pattern", ["random", "linear", "costas"])
    def test_hop_patterns(self, assert_valid_iq, sample_rate, pattern):
        from spectra.waveforms import FHSS

        wf = FHSS(hop_pattern=pattern)
        iq = wf.generate(10, sample_rate, seed=42)
        assert_valid_iq(iq)


class TestCDMA_Forward:
    def test_generate_returns_valid_iq(self, assert_valid_iq, sample_rate):
        from spectra.waveforms import CDMA_Forward

        iq = CDMA_Forward().generate(num_symbols=10, sample_rate=sample_rate, seed=42)
        assert_valid_iq(iq)

    def test_label(self):
        from spectra.waveforms import CDMA_Forward

        assert CDMA_Forward().label == "CDMA_Forward"

    def test_bandwidth(self, sample_rate):
        from spectra.waveforms import CDMA_Forward

        wf = CDMA_Forward()
        assert wf.bandwidth(sample_rate) == pytest.approx(sample_rate)

    def test_deterministic(self, sample_rate):
        from spectra.waveforms import CDMA_Forward

        wf = CDMA_Forward()
        iq1 = wf.generate(10, sample_rate, seed=42)
        iq2 = wf.generate(10, sample_rate, seed=42)
        npt.assert_array_equal(iq1, iq2)

    def test_correct_length(self, sample_rate):
        from spectra.waveforms import CDMA_Forward

        sf = 64
        num_symbols = 10
        wf = CDMA_Forward(spreading_factor=sf)
        iq = wf.generate(num_symbols, sample_rate, seed=42)
        assert len(iq) == num_symbols * sf

    def test_multi_user_combining(self, sample_rate):
        from spectra.waveforms import CDMA_Forward

        # More users should change the signal
        wf1 = CDMA_Forward(num_users=1)
        wf4 = CDMA_Forward(num_users=4)
        iq1 = wf1.generate(10, sample_rate, seed=42)
        iq4 = wf4.generate(10, sample_rate, seed=42)
        assert not np.allclose(iq1, iq4)


class TestCDMA_Reverse:
    def test_generate_returns_valid_iq(self, assert_valid_iq, sample_rate):
        from spectra.waveforms import CDMA_Reverse

        iq = CDMA_Reverse().generate(num_symbols=10, sample_rate=sample_rate, seed=42)
        assert_valid_iq(iq)

    def test_label(self):
        from spectra.waveforms import CDMA_Reverse

        assert CDMA_Reverse().label == "CDMA_Reverse"

    def test_bandwidth(self, sample_rate):
        from spectra.waveforms import CDMA_Reverse

        wf = CDMA_Reverse()
        assert wf.bandwidth(sample_rate) == pytest.approx(sample_rate)

    def test_deterministic(self, sample_rate):
        from spectra.waveforms import CDMA_Reverse

        wf = CDMA_Reverse()
        iq1 = wf.generate(10, sample_rate, seed=42)
        iq2 = wf.generate(10, sample_rate, seed=42)
        npt.assert_array_equal(iq1, iq2)

    def test_correct_length(self, sample_rate):
        from spectra.waveforms import CDMA_Reverse

        sf = 64
        num_symbols = 10
        wf = CDMA_Reverse(spreading_factor=sf)
        iq = wf.generate(num_symbols, sample_rate, seed=42)
        assert len(iq) == num_symbols * sf

    def test_multi_user_combining(self, sample_rate):
        from spectra.waveforms import CDMA_Reverse

        wf1 = CDMA_Reverse(num_users=1)
        wf4 = CDMA_Reverse(num_users=4)
        iq1 = wf1.generate(10, sample_rate, seed=42)
        iq4 = wf4.generate(10, sample_rate, seed=42)
        assert not np.allclose(iq1, iq4)


class TestTHSS:
    def test_generate_returns_valid_iq(self, assert_valid_iq, sample_rate):
        from spectra.waveforms import THSS

        iq = THSS().generate(num_symbols=10, sample_rate=sample_rate, seed=42)
        assert_valid_iq(iq)

    def test_label(self):
        from spectra.waveforms import THSS

        assert THSS().label == "THSS"

    def test_bandwidth(self, sample_rate):
        from spectra.waveforms import THSS

        ps = 16
        wf = THSS(pulse_samples=ps)
        assert wf.bandwidth(sample_rate) == pytest.approx(sample_rate / ps)

    def test_deterministic(self, sample_rate):
        from spectra.waveforms import THSS

        wf = THSS()
        iq1 = wf.generate(10, sample_rate, seed=42)
        iq2 = wf.generate(10, sample_rate, seed=42)
        npt.assert_array_equal(iq1, iq2)

    def test_correct_length(self, sample_rate):
        from spectra.waveforms import THSS

        n_frames = 32
        slots = 8
        ps = 16
        num_symbols = 10
        wf = THSS(num_frames=n_frames, slots_per_frame=slots, pulse_samples=ps)
        iq = wf.generate(num_symbols, sample_rate, seed=42)
        assert len(iq) == num_symbols * n_frames * slots * ps

    def test_sparsity(self, sample_rate):
        from spectra.waveforms import THSS

        wf = THSS(num_frames=32, slots_per_frame=8, pulse_samples=16)
        iq = wf.generate(10, sample_rate, seed=42)
        # Most samples should be zero (only 1 pulse per frame out of slots_per_frame slots)
        zero_fraction = np.sum(np.abs(iq) < 1e-10) / len(iq)
        # Expected: (slots_per_frame - 1) / slots_per_frame = 7/8 = 0.875
        assert zero_fraction > 0.5, f"THSS should be sparse, got {zero_fraction:.2f} zero fraction"

    @pytest.mark.parametrize("pulse_shape", ["gaussian", "rect"])
    def test_pulse_shapes(self, assert_valid_iq, sample_rate, pulse_shape):
        from spectra.waveforms import THSS

        wf = THSS(pulse_shape=pulse_shape)
        iq = wf.generate(10, sample_rate, seed=42)
        assert_valid_iq(iq)
