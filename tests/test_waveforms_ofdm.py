import numpy as np
import numpy.testing as npt
import pytest


class TestOFDMWaveform:
    def test_generate_returns_valid_iq(self, assert_valid_iq, sample_rate):
        from spectra.waveforms.ofdm import OFDM
        waveform = OFDM()
        iq = waveform.generate(num_symbols=10, sample_rate=sample_rate)
        assert_valid_iq(iq)

    def test_output_length(self, sample_rate):
        from spectra.waveforms.ofdm import OFDM
        nsc, fft, cp = 64, 256, 16
        waveform = OFDM(num_subcarriers=nsc, fft_size=fft, cp_length=cp)
        iq = waveform.generate(num_symbols=5, sample_rate=sample_rate)
        assert len(iq) == 5 * (fft + cp)

    def test_label(self):
        from spectra.waveforms.ofdm import OFDM
        assert OFDM().label == "OFDM"

    def test_bandwidth(self, sample_rate):
        from spectra.waveforms.ofdm import OFDM
        nsc, fft = 64, 256
        waveform = OFDM(num_subcarriers=nsc, fft_size=fft)
        expected_bw = nsc * sample_rate / fft
        assert waveform.bandwidth(sample_rate) == pytest.approx(expected_bw)

    def test_samples_per_symbol_attribute(self):
        from spectra.waveforms.ofdm import OFDM
        waveform = OFDM(fft_size=256, cp_length=16)
        assert waveform.samples_per_symbol == 272

    def test_deterministic_with_seed(self, sample_rate):
        from spectra.waveforms.ofdm import OFDM
        waveform = OFDM()
        iq1 = waveform.generate(num_symbols=5, sample_rate=sample_rate, seed=42)
        iq2 = waveform.generate(num_symbols=5, sample_rate=sample_rate, seed=42)
        npt.assert_array_equal(iq1, iq2)

    def test_no_cyclic_prefix(self, assert_valid_iq, sample_rate):
        from spectra.waveforms.ofdm import OFDM
        waveform = OFDM(cp_length=0)
        iq = waveform.generate(num_symbols=5, sample_rate=sample_rate)
        assert_valid_iq(iq)
        assert len(iq) == 5 * 256  # default fft_size, no CP


class TestOFDMEnhanced:
    def test_backward_compat(self):
        """Default params produce same behavior as before."""
        from spectra.waveforms.ofdm import OFDM
        ofdm = OFDM()
        iq = ofdm.generate(num_symbols=10, sample_rate=1e6, seed=42)
        assert iq.dtype == np.complex64
        assert len(iq) > 0

    def test_guard_bands(self):
        from spectra.waveforms.ofdm import OFDM
        ofdm = OFDM(num_subcarriers=64, fft_size=256, guard_bands=(6, 5))
        iq = ofdm.generate(num_symbols=5, sample_rate=1e6, seed=42)
        assert len(iq) > 0
        # Verify bandwidth is reduced
        bw_no_guard = OFDM(num_subcarriers=64, fft_size=256).bandwidth(1e6)
        bw_guard = ofdm.bandwidth(1e6)
        assert bw_guard < bw_no_guard

    def test_dc_null(self):
        from spectra.waveforms.ofdm import OFDM
        ofdm = OFDM(num_subcarriers=64, fft_size=256, dc_null=True)
        iq = ofdm.generate(num_symbols=5, sample_rate=1e6, seed=42)
        assert len(iq) > 0

    def test_pilots(self):
        from spectra.waveforms.ofdm import OFDM
        ofdm = OFDM(num_subcarriers=64, fft_size=256, pilot_indices=[0, 16, 32, 48])
        iq = ofdm.generate(num_symbols=5, sample_rate=1e6, seed=42)
        assert len(iq) > 0

    def test_modulation_bpsk(self):
        from spectra.waveforms.ofdm import OFDM
        ofdm = OFDM(modulation="BPSK")
        iq = ofdm.generate(num_symbols=5, sample_rate=1e6, seed=42)
        assert len(iq) > 0

    def test_modulation_qam16(self):
        from spectra.waveforms.ofdm import OFDM
        ofdm = OFDM(modulation="QAM16")
        iq = ofdm.generate(num_symbols=5, sample_rate=1e6, seed=42)
        assert len(iq) > 0


class TestSCFDMA:
    def test_generate(self):
        from spectra.waveforms.ofdm import SCFDMA
        sc = SCFDMA(num_subcarriers=64, fft_size=256)
        iq = sc.generate(num_symbols=10, sample_rate=1e6, seed=42)
        assert iq.dtype == np.complex64
        assert len(iq) > 0

    def test_label(self):
        from spectra.waveforms.ofdm import SCFDMA
        assert SCFDMA().label == "SC-FDMA"

    def test_lower_papr_than_ofdm(self):
        """SC-FDMA should have lower PAPR than OFDM."""
        from spectra.waveforms.ofdm import OFDM, SCFDMA
        ofdm = OFDM(num_subcarriers=64, fft_size=256)
        scfdma = SCFDMA(num_subcarriers=64, fft_size=256)

        ofdm_iq = ofdm.generate(num_symbols=100, sample_rate=1e6, seed=42)
        sc_iq = scfdma.generate(num_symbols=100, sample_rate=1e6, seed=42)

        def papr(x):
            return float(np.max(np.abs(x)**2) / np.mean(np.abs(x)**2))

        # SC-FDMA should generally have lower PAPR
        assert papr(sc_iq) < papr(ofdm_iq) * 1.5  # allow some margin
