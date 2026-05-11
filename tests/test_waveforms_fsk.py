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


class TestGMSKModulationIndex:
    """Regression test: GMSK steady-state per-symbol phase change.

    Standard MSK / GMSK uses modulation index h = 0.5, so a constant
    +1 bit stream drives the phase by π·0.5 = π/2 rad per symbol.
    A prior implementation used zero-insertion upsampling with a
    sum-normalised Gaussian, producing h_eff = 0.5/sps = 0.0625 (a
    factor-of-sps error). This test guards against regression.
    """

    def test_constant_bit_stream_gives_h_one_half(self, monkeypatch):
        import numpy as np
        from spectra import _rust
        from spectra.waveforms.fsk import GMSK

        sps = 8
        num_symbols = 256

        # Force the underlying BPSK generator to return all +1 so the
        # GMSK input is a constant bit stream. After the Gaussian filter
        # settles, every per-symbol phase increment should equal π·h = π/2.
        def all_plus_one(n, seed=0):
            return np.ones(n, dtype=np.complex64)

        monkeypatch.setattr(_rust, "generate_bpsk_symbols", all_plus_one)
        # Also patch the symbol it's imported under in fsk.py:
        from spectra.waveforms import fsk as fsk_mod

        monkeypatch.setattr(fsk_mod, "generate_bpsk_symbols", all_plus_one)

        wf = GMSK(bt=0.3, samples_per_symbol=sps)
        iq = wf.generate(num_symbols=num_symbols, sample_rate=1.0e6, seed=0)

        # Steady-state per-symbol phase change. Skip the first and last
        # 16 symbols to avoid Gaussian-filter transients.
        phase = np.unwrap(np.angle(iq))
        per_symbol = phase[sps::sps] - phase[:-sps:sps]
        inner = per_symbol[16:-16]
        median_step = float(np.median(np.abs(inner)))

        expected = np.pi * 0.5  # h = 0.5
        # 1 % relative tolerance; the Gaussian filter's amplitude response
        # at DC is exactly 1 for a sum-normalised kernel, so the residual
        # error is float32 round-off.
        assert abs(median_step - expected) <= 0.01 * expected, (
            f"steady-state |Δφ|/symbol = {median_step:.4f} rad, "
            f"expected {expected:.4f} rad (h = 0.5)"
        )
