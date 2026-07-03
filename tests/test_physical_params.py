"""Tests for physical-unit waveform parameterization."""

import numpy as np
import pytest
from spectra.waveforms.physical import resample_to_rate, resolve_symbol_rate


class TestResolveSymbolRate:
    def test_exact_integer_sps(self):
        # 10 MHz / 1.25 MBd = exactly 8 sps, no resampling
        sps, up, down = resolve_symbol_rate(10e6, 1.25e6)
        assert (sps, up, down) == (8, 1, 1)

    def test_rounds_within_tolerance(self):
        # 10 MHz / 1.24 MBd = 8.0645 -> rounds to 8 (0.8% error <= 1%)
        sps, up, down = resolve_symbol_rate(10e6, 1.24e6)
        assert (sps, up, down) == (8, 1, 1)

    def test_resamples_beyond_tolerance(self):
        # 10 MHz / 1.5 MBd = 6.667 -> 5% from 7, needs resampling.
        sps, up, down = resolve_symbol_rate(10e6, 1.5e6)
        assert sps == 7               # ceil(6.667)
        assert up < down              # downsampling: generated rate > target
        # after resampling by up/down, samples-per-symbol matches the exact
        # ratio to <0.1%:  7 * 20/21 = 6.667
        eff_sps = sps * up / down
        assert abs(eff_sps - 10e6 / 1.5e6) / (10e6 / 1.5e6) < 1e-3

    def test_symbol_rate_above_nyquist_raises(self):
        with pytest.raises(ValueError, match="symbol_rate"):
            resolve_symbol_rate(10e6, 6e6)   # sps would be < 2

    def test_nonpositive_symbol_rate_raises(self):
        with pytest.raises(ValueError, match="symbol_rate"):
            resolve_symbol_rate(10e6, 0.0)


class TestResampleToRate:
    def test_identity_when_up_down_one(self):
        x = np.arange(64, dtype=np.complex64)
        out = resample_to_rate(x, 1, 1)
        assert out is x

    def test_length_scales_by_ratio(self):
        x = np.exp(2j * np.pi * 0.01 * np.arange(8000)).astype(np.complex64)
        out = resample_to_rate(x, 20, 21)
        assert abs(len(out) - len(x) * 20 / 21) <= 21
        assert out.dtype == np.complex64

    def test_preserves_tone_frequency(self):
        # A tone at normalized f=0.05 resampled by 20/21 must appear at
        # 0.05 * 21/20 = 0.0525 of the new rate
        n = 16384
        x = np.exp(2j * np.pi * 0.05 * np.arange(n)).astype(np.complex64)
        out = resample_to_rate(x, 20, 21)
        spec = np.abs(np.fft.fft(out * np.hanning(len(out))))
        peak = np.argmax(spec[: len(out) // 2]) / len(out)
        assert abs(peak - 0.0525) < 0.001
