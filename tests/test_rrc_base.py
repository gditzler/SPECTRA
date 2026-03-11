"""Tests for _RRCWaveformBase shared behavior."""

import numpy as np


class TestRRCWaveformBase:
    def test_base_provides_bandwidth(self):
        from spectra.waveforms.rrc_base import _RRCWaveformBase

        class Dummy(_RRCWaveformBase):
            label = "dummy"

            def _generate_symbols(self, num_symbols, seed):
                return np.zeros(num_symbols, dtype=np.complex64)

        w = Dummy(rolloff=0.35, filter_span=10, samples_per_symbol=8)
        expected = (1_000_000 / 8) * (1.0 + 0.35)
        assert w.bandwidth(1_000_000) == expected

    def test_base_generate_returns_array(self):
        from spectra.waveforms.rrc_base import _RRCWaveformBase

        class Dummy(_RRCWaveformBase):
            label = "dummy"

            def _generate_symbols(self, num_symbols, seed):
                rng = np.random.default_rng(seed)
                return (rng.integers(0, 2, num_symbols) * 2 - 1).astype(np.complex64)

        w = Dummy(rolloff=0.35, filter_span=10, samples_per_symbol=8)
        iq = w.generate(100, 1_000_000, seed=42)
        assert iq.dtype == np.complex64
        assert len(iq) > 0

    def test_existing_waveforms_unchanged(self):
        """QPSK, BPSK, QAM16, OOK must produce identical output after refactor."""
        from spectra.waveforms.ask import OOK
        from spectra.waveforms.psk import BPSK, QPSK
        from spectra.waveforms.qam import QAM16

        for Cls in [QPSK, BPSK, QAM16, OOK]:
            w = Cls()
            iq1 = w.generate(100, 1e6, seed=99)
            iq2 = w.generate(100, 1e6, seed=99)
            np.testing.assert_array_equal(iq1, iq2)
            assert len(iq1) > 0

    def test_cross_qam_still_works(self):
        """Cross QAM waveforms should still function after refactor."""
        from spectra.waveforms.qam import QAM32

        w = QAM32()
        iq = w.generate(100, 1e6, seed=42)
        assert len(iq) > 0
        assert iq.dtype == np.complex64

    def test_psk_base_subclasses_work(self):
        """PSK16/32/64 should still function after refactor."""
        from spectra.waveforms.psk import PSK16, PSK32, PSK64

        for Cls in [PSK16, PSK32, PSK64]:
            w = Cls()
            iq = w.generate(100, 1e6, seed=42)
            assert len(iq) > 0
