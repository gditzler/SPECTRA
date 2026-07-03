import warnings

import numpy as np
import numpy.testing as npt
import pytest


def _occupied_bandwidth_99(iq: np.ndarray, sample_rate: float, nperseg: int = 1024) -> float:
    """99% occupied bandwidth from a Welch PSD (Hann window, 50% overlap)."""
    window = np.hanning(nperseg)
    step = nperseg // 2
    num_segments = (len(iq) - nperseg) // step + 1
    psd = np.zeros(nperseg)
    for i in range(num_segments):
        seg = iq[i * step : i * step + nperseg] * window
        psd += np.abs(np.fft.fft(seg)) ** 2
    psd = np.fft.fftshift(psd)
    freqs = np.fft.fftshift(np.fft.fftfreq(nperseg, d=1.0 / sample_rate))
    cum = np.cumsum(psd) / np.sum(psd)
    f_lo = freqs[np.searchsorted(cum, 0.005)]
    f_hi = freqs[np.searchsorted(cum, 0.995)]
    return f_hi - f_lo


class TestFSKSymbolLevels:
    """FSK frequency symbols follow the standard CPFSK convention:
    odd-integer levels ±1, ±3, ..., ±(M-1), giving adjacent-tone
    spacing h·Rs when the per-sample phase step is π·h·a_k/sps."""

    @pytest.mark.rust
    @pytest.mark.parametrize("order", [2, 4, 8])
    def test_levels_are_odd_integers(self, order):
        from spectra._rust import generate_fsk_symbols

        symbols = generate_fsk_symbols(1000, order, seed=42)
        expected = set(range(-(order - 1), order, 2))
        observed = set(np.unique(symbols).astype(int).tolist())
        npt.assert_allclose(np.unique(symbols), np.round(np.unique(symbols)), atol=1e-6)
        assert observed <= expected
        # With 1000 draws all levels should appear
        assert observed == expected


class TestFSKOccupiedBandwidth:
    """The 99% occupied bandwidth measured from a Welch PSD must be
    consistent with the Carson-rule claim of bandwidth(): no more than
    the claim, and not smaller by more than ~2x (which would indicate a
    modulation-index or symbol-level scaling bug)."""

    @pytest.mark.parametrize(
        "cls_name, kwargs",
        [
            ("FSK", {"order": 2, "mod_index": 1.0}),
            ("FSK", {"order": 4, "mod_index": 1.0}),
            ("MSK4", {}),
            ("GFSK", {"order": 2}),
            ("GFSK", {"order": 4}),
            ("GMSK4", {}),
        ],
        ids=["FSK2", "FSK4", "MSK4", "GFSK2", "GFSK4", "GMSK4"],
    )
    def test_obw_matches_bandwidth_claim(self, cls_name, kwargs):
        from spectra.waveforms import fsk as fsk_mod

        fs = 10e6
        waveform = getattr(fsk_mod, cls_name)(**kwargs)
        iq = waveform.generate(num_symbols=8192, sample_rate=fs, seed=42)
        obw = _occupied_bandwidth_99(iq, fs)
        claim = waveform.bandwidth(fs)
        assert obw <= 1.05 * claim, (
            f"99% OBW {obw / 1e6:.2f} MHz exceeds claim {claim / 1e6:.2f} MHz"
        )
        assert obw >= 0.45 * claim, (
            f"99% OBW {obw / 1e6:.2f} MHz far below claim {claim / 1e6:.2f} MHz "
            "(modulation-index or symbol-level scaling bug)"
        )


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


class TestFSKNyquistDefaults:
    """High-order FSK/GFSK defaults must fit within the sampling bandwidth.

    With odd-integer CPFSK levels ±1..±(M-1), the Carson-rule band edge is
    ((M-1)·h + 2)·R_s. At the old default sps=8 this exceeded the sample
    rate for M=8 and M=16 (outermost 16FSK tones at ±0.94·fs — aliased).
    """

    @pytest.mark.parametrize(
        "cls_name", ["FSK", "FSK4", "FSK8", "FSK16", "GFSK", "GFSK4", "GFSK8", "GFSK16"]
    )
    def test_default_bandwidth_claim_fits_sample_rate(self, cls_name, sample_rate):
        import spectra.waveforms.fsk as fsk_mod

        waveform = getattr(fsk_mod, cls_name)()
        assert waveform.bandwidth(sample_rate) <= sample_rate, (
            f"{cls_name} default config claims bandwidth "
            f"{waveform.bandwidth(sample_rate):.0f} Hz > sample rate {sample_rate:.0f} Hz"
        )

    @pytest.mark.parametrize("cls_name", ["FSK8", "FSK16"])
    def test_default_measured_obw_within_claim(self, cls_name, sample_rate):
        """Empirical check: 99% OBW is below the Carson claim, not alias-saturated."""
        import spectra.waveforms.fsk as fsk_mod

        waveform = getattr(fsk_mod, cls_name)()
        iq = waveform.generate(num_symbols=4096, sample_rate=sample_rate, seed=7)
        measured = _occupied_bandwidth_99(iq, sample_rate)
        claimed = waveform.bandwidth(sample_rate)
        assert measured <= claimed, (
            f"{cls_name}: measured 99% OBW {measured:.0f} Hz exceeds claim {claimed:.0f} Hz"
        )
        # An aliased signal saturates the whole capture (~0.99·fs).
        assert measured <= 0.75 * sample_rate, (
            f"{cls_name}: measured 99% OBW {measured:.0f} Hz saturates the capture "
            f"bandwidth ({sample_rate:.0f} Hz) — spectrum is aliased"
        )


class TestFSKAliasWarning:
    def test_fsk_generate_warns_when_carson_exceeds_sample_rate(self, sample_rate):
        from spectra.waveforms.fsk import FSK

        waveform = FSK(order=8, mod_index=1.0, samples_per_symbol=8)
        with pytest.warns(UserWarning, match="alias"):
            waveform.generate(num_symbols=64, sample_rate=sample_rate)

    def test_fsk16_sps8_warns(self, sample_rate):
        from spectra.waveforms.fsk import FSK

        waveform = FSK(order=16, mod_index=1.0, samples_per_symbol=8)
        with pytest.warns(UserWarning, match="alias"):
            waveform.generate(num_symbols=64, sample_rate=sample_rate)

    def test_gfsk_generate_warns_when_carson_exceeds_sample_rate(self, sample_rate):
        from spectra.waveforms.fsk import GFSK

        waveform = GFSK(order=16, mod_index=1.0, samples_per_symbol=8)
        with pytest.warns(UserWarning, match="alias"):
            waveform.generate(num_symbols=64, sample_rate=sample_rate)

    @pytest.mark.parametrize(
        "cls_name",
        ["FSK", "FSK4", "FSK8", "FSK16", "MSK4", "MSK8", "GFSK", "GFSK4", "GFSK8", "GFSK16"],
    )
    def test_default_configs_do_not_warn(self, cls_name, sample_rate):
        import spectra.waveforms.fsk as fsk_mod

        waveform = getattr(fsk_mod, cls_name)()
        with warnings.catch_warnings():
            warnings.simplefilter("error")
            waveform.generate(num_symbols=64, sample_rate=sample_rate)


class TestGFSKModulationIndex:
    """Regression test: GFSK steady-state per-symbol phase change.

    A constant +1 symbol stream must drive the phase by π·mod_index per
    symbol. A prior implementation used zero-insertion upsampling with the
    sum-normalised Rust gaussian_taps kernel, attenuating the frequency-pulse
    train by sps and producing h_eff = mod_index/sps (a factor-of-sps error).
    This mirrors TestGMSKModulationIndex, which guards the same bug in GMSK.
    """

    def test_constant_symbol_stream_gives_full_mod_index(self, monkeypatch):
        from spectra.waveforms import fsk as fsk_mod
        from spectra.waveforms.fsk import GFSK

        sps = 8
        num_symbols = 256

        def all_plus_one(n, order, seed=0):
            return np.ones(n, dtype=np.float32)

        monkeypatch.setattr(fsk_mod, "generate_fsk_symbols", all_plus_one)

        wf = GFSK(order=2, mod_index=1.0, bt=0.3, samples_per_symbol=sps)
        iq = wf.generate(num_symbols=num_symbols, sample_rate=1.0e6, seed=0)

        # Steady-state per-symbol phase change; skip filter transients.
        phase = np.unwrap(np.angle(iq))
        per_symbol = phase[sps::sps] - phase[:-sps:sps]
        inner = per_symbol[16:-16]
        median_step = float(np.median(np.abs(inner)))

        expected = np.pi * 1.0  # h = mod_index = 1.0
        assert abs(median_step - expected) <= 0.01 * expected, (
            f"steady-state |Δφ|/symbol = {median_step:.4f} rad, "
            f"expected {expected:.4f} rad (h = 1.0)"
        )
