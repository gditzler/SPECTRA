import numpy as np
import numpy.testing as npt
import pytest


class TestNROFDM:
    def test_generate_returns_valid_iq(self, assert_valid_iq, sample_rate):
        from spectra.waveforms.nr import NR_OFDM

        waveform = NR_OFDM()
        iq = waveform.generate(num_symbols=2, sample_rate=sample_rate)
        assert_valid_iq(iq)

    def test_label(self):
        from spectra.waveforms.nr import NR_OFDM

        assert NR_OFDM().label == "NR_OFDM"

    def test_bandwidth(self):
        from spectra.waveforms.nr import NR_OFDM

        # mu=1 -> 30 kHz SCS, 25 RBs -> 25*12*30000 = 9 MHz
        waveform = NR_OFDM(numerology=1, num_resource_blocks=25)
        assert waveform.bandwidth(1e6) == pytest.approx(9_000_000)

    def test_deterministic_with_seed(self, sample_rate):
        from spectra.waveforms.nr import NR_OFDM

        waveform = NR_OFDM()
        iq1 = waveform.generate(num_symbols=2, sample_rate=sample_rate, seed=42)
        iq2 = waveform.generate(num_symbols=2, sample_rate=sample_rate, seed=42)
        npt.assert_array_equal(iq1, iq2)

    @pytest.mark.parametrize("mu", [0, 1, 2, 3, 4])
    def test_numerologies(self, assert_valid_iq, sample_rate, mu):
        from spectra.waveforms.nr import NR_OFDM

        waveform = NR_OFDM(numerology=mu, num_resource_blocks=10, fft_size=256)
        iq = waveform.generate(num_symbols=1, sample_rate=sample_rate, seed=99)
        assert_valid_iq(iq)
        assert len(iq) > 0

    def test_fr1_20mhz_preset(self, assert_valid_iq, sample_rate):
        from spectra.waveforms.nr import NR_OFDM

        waveform = NR_OFDM.fr1_20mhz()
        assert waveform._numerology == 1
        assert waveform._num_rbs == 51
        assert waveform._fft_size == 1024
        iq = waveform.generate(num_symbols=1, sample_rate=sample_rate, seed=1)
        assert_valid_iq(iq)

    def test_invalid_numerology(self):
        from spectra.waveforms.nr import NR_OFDM

        with pytest.raises(ValueError):
            NR_OFDM(numerology=5)

    def test_qam16_modulation(self, assert_valid_iq, sample_rate):
        from spectra.waveforms.nr import NR_OFDM

        waveform = NR_OFDM(modulation="qam16")
        iq = waveform.generate(num_symbols=1, sample_rate=sample_rate, seed=7)
        assert_valid_iq(iq)


class TestNRSSB:
    def test_generate_returns_valid_iq(self, assert_valid_iq, sample_rate):
        from spectra.waveforms.nr import NR_SSB

        waveform = NR_SSB()
        iq = waveform.generate(num_symbols=2, sample_rate=sample_rate, seed=42)
        assert_valid_iq(iq)

    def test_label(self):
        from spectra.waveforms.nr import NR_SSB

        assert NR_SSB().label == "NR_SSB"

    def test_bandwidth(self):
        from spectra.waveforms.nr import NR_SSB

        # SSB is 240 subcarriers * 30 kHz SCS = 7.2 MHz
        waveform = NR_SSB(numerology=1)
        assert waveform.bandwidth(1e6) == pytest.approx(7_200_000)

    def test_deterministic_with_seed(self, sample_rate):
        from spectra.waveforms.nr import NR_SSB

        waveform = NR_SSB()
        iq1 = waveform.generate(num_symbols=1, sample_rate=sample_rate, seed=42)
        iq2 = waveform.generate(num_symbols=1, sample_rate=sample_rate, seed=42)
        npt.assert_array_equal(iq1, iq2)

    def test_pss_sss_present(self, sample_rate):
        """Verify that SSB generation uses PSS/SSS (non-zero output)."""
        from spectra.waveforms.nr import NR_SSB

        waveform = NR_SSB(n_id_1=100, n_id_2=1)
        iq = waveform.generate(num_symbols=1, sample_rate=sample_rate, seed=42)
        # Should have non-trivial signal content
        assert np.mean(np.abs(iq) ** 2) > 0

    def test_n78_preset(self, assert_valid_iq, sample_rate):
        from spectra.waveforms.nr import NR_SSB

        waveform = NR_SSB.n78()
        iq = waveform.generate(num_symbols=1, sample_rate=sample_rate, seed=1)
        assert_valid_iq(iq)

    def test_invalid_n_id(self):
        from spectra.waveforms.nr import NR_SSB

        with pytest.raises(ValueError):
            NR_SSB(n_id_1=336)
        with pytest.raises(ValueError):
            NR_SSB(n_id_2=3)


class TestNRPDSCH:
    def test_generate_returns_valid_iq(self, assert_valid_iq, sample_rate):
        from spectra.waveforms.nr import NR_PDSCH

        waveform = NR_PDSCH()
        iq = waveform.generate(num_symbols=1, sample_rate=sample_rate, seed=42)
        assert_valid_iq(iq)

    def test_label(self):
        from spectra.waveforms.nr import NR_PDSCH

        assert NR_PDSCH().label == "NR_PDSCH"

    def test_bandwidth(self):
        from spectra.waveforms.nr import NR_PDSCH

        waveform = NR_PDSCH(numerology=1, num_resource_blocks=25)
        assert waveform.bandwidth(1e6) == pytest.approx(9_000_000)

    def test_deterministic_with_seed(self, sample_rate):
        from spectra.waveforms.nr import NR_PDSCH

        waveform = NR_PDSCH()
        iq1 = waveform.generate(num_symbols=1, sample_rate=sample_rate, seed=42)
        iq2 = waveform.generate(num_symbols=1, sample_rate=sample_rate, seed=42)
        npt.assert_array_equal(iq1, iq2)

    def test_qam256_modulation(self, assert_valid_iq, sample_rate):
        from spectra.waveforms.nr import NR_PDSCH

        waveform = NR_PDSCH(modulation="qam256")
        iq = waveform.generate(num_symbols=1, sample_rate=sample_rate, seed=7)
        assert_valid_iq(iq)

    def test_dmrs_symbols_count(self):
        from spectra.waveforms.nr import NR_PDSCH

        waveform = NR_PDSCH(num_dmrs_symbols=4)
        assert len(waveform._dmrs_symbol_indices) == 4


class TestNRPUSCH:
    def test_generate_returns_valid_iq(self, assert_valid_iq, sample_rate):
        from spectra.waveforms.nr import NR_PUSCH

        waveform = NR_PUSCH()
        iq = waveform.generate(num_symbols=1, sample_rate=sample_rate, seed=42)
        assert_valid_iq(iq)

    def test_label(self):
        from spectra.waveforms.nr import NR_PUSCH

        assert NR_PUSCH().label == "NR_PUSCH"

    def test_bandwidth(self):
        from spectra.waveforms.nr import NR_PUSCH

        waveform = NR_PUSCH(numerology=1, num_resource_blocks=25)
        assert waveform.bandwidth(1e6) == pytest.approx(9_000_000)

    def test_deterministic_with_seed(self, sample_rate):
        from spectra.waveforms.nr import NR_PUSCH

        waveform = NR_PUSCH()
        iq1 = waveform.generate(num_symbols=1, sample_rate=sample_rate, seed=42)
        iq2 = waveform.generate(num_symbols=1, sample_rate=sample_rate, seed=42)
        npt.assert_array_equal(iq1, iq2)

    def test_transform_precoding(self, assert_valid_iq, sample_rate):
        from spectra.waveforms.nr import NR_PUSCH

        waveform_cp = NR_PUSCH(transform_precoding=False)
        waveform_tp = NR_PUSCH(transform_precoding=True)

        iq_cp = waveform_cp.generate(num_symbols=1, sample_rate=sample_rate, seed=42)
        iq_tp = waveform_tp.generate(num_symbols=1, sample_rate=sample_rate, seed=42)

        assert_valid_iq(iq_cp)
        assert_valid_iq(iq_tp)
        assert len(iq_cp) == len(iq_tp)
        # Transform precoding should produce different samples
        assert not np.allclose(iq_cp, iq_tp)

    def test_transform_precoding_lower_papr(self, sample_rate):
        """SC-FDMA (transform precoding) should generally have lower PAPR."""
        from spectra.waveforms.nr import NR_PUSCH

        waveform_ofdm = NR_PUSCH(transform_precoding=False, num_resource_blocks=10)
        waveform_sc = NR_PUSCH(transform_precoding=True, num_resource_blocks=10)

        iq_ofdm = waveform_ofdm.generate(num_symbols=5, sample_rate=sample_rate, seed=42)
        iq_sc = waveform_sc.generate(num_symbols=5, sample_rate=sample_rate, seed=42)

        def papr(x):
            return float(np.max(np.abs(x) ** 2) / np.mean(np.abs(x) ** 2))

        # With proper DFT normalization, SC-FDMA should have lower PAPR
        assert papr(iq_sc) < papr(iq_ofdm) * 1.1


class TestNRPRACH:
    def test_generate_returns_valid_iq(self, assert_valid_iq, sample_rate):
        from spectra.waveforms.nr import NR_PRACH

        waveform = NR_PRACH()
        iq = waveform.generate(num_symbols=2, sample_rate=sample_rate)
        assert_valid_iq(iq)

    def test_label(self):
        from spectra.waveforms.nr import NR_PRACH

        assert NR_PRACH().label == "NR_PRACH"

    def test_deterministic(self, sample_rate):
        from spectra.waveforms.nr import NR_PRACH

        waveform = NR_PRACH()
        iq1 = waveform.generate(num_symbols=2, sample_rate=sample_rate, seed=42)
        iq2 = waveform.generate(num_symbols=2, sample_rate=sample_rate, seed=42)
        npt.assert_array_equal(iq1, iq2)

    def test_format_0_preset(self, assert_valid_iq, sample_rate):
        from spectra.waveforms.nr import NR_PRACH

        waveform = NR_PRACH.format_0()
        assert waveform._preamble_format == 0
        iq = waveform.generate(num_symbols=1, sample_rate=sample_rate)
        assert_valid_iq(iq)

    def test_cyclic_shift(self, sample_rate):
        from spectra.waveforms.nr import NR_PRACH

        waveform_no_cs = NR_PRACH(cyclic_shift=0)
        waveform_cs = NR_PRACH(cyclic_shift=10)

        iq1 = waveform_no_cs.generate(num_symbols=1, sample_rate=sample_rate)
        iq2 = waveform_cs.generate(num_symbols=1, sample_rate=sample_rate)
        # Different cyclic shifts -> different signals
        assert not np.allclose(iq1, iq2)

    def test_invalid_root(self):
        from spectra.waveforms.nr import NR_PRACH

        with pytest.raises(ValueError):
            NR_PRACH(root_index=0)
        with pytest.raises(ValueError):
            NR_PRACH(root_index=839)


class TestNRImports:
    """Verify NR waveforms are accessible from the package."""

    def test_import_from_waveforms(self):
        from spectra.waveforms import NR_OFDM, NR_SSB, NR_PDSCH, NR_PUSCH, NR_PRACH

        assert NR_OFDM is not None
        assert NR_SSB is not None
        assert NR_PDSCH is not None
        assert NR_PUSCH is not None
        assert NR_PRACH is not None
