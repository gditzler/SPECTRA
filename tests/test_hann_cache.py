"""Tests for Hann window caching in STFT and Spectrogram."""

import numpy as np


class TestHannWindowCaching:
    def test_stft_window_computed_once(self):
        """Window object identity should be same across calls."""
        from spectra.transforms.stft import STFT

        t = STFT(nfft=128, hop_length=32)
        assert hasattr(t, "_window")
        w1 = t._window
        iq = np.random.default_rng(0).standard_normal(256).astype(np.complex64)
        _ = t(iq)
        w2 = t._window
        assert w1 is w2  # Same object, not recomputed

    def test_spectrogram_window_computed_once(self):
        from spectra.transforms.spectrogram import Spectrogram

        t = Spectrogram(nfft=128, hop_length=32)
        assert hasattr(t, "_window")
