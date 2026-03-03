import numpy as np
import torch
import pytest


class TestPSD:
    def test_output_shape(self):
        from spectra.transforms.psd import PSD

        iq = np.exp(1j * np.linspace(0, 100, 4096)).astype(np.complex64)
        psd = PSD(nfft=256)
        result = psd(iq)
        assert result.shape == (1, 256)

    def test_output_dtype(self):
        from spectra.transforms.psd import PSD

        iq = np.ones(2048, dtype=np.complex64)
        psd = PSD(nfft=128)
        result = psd(iq)
        assert result.dtype == torch.float32

    def test_output_is_tensor(self):
        from spectra.transforms.psd import PSD

        iq = np.ones(2048, dtype=np.complex64)
        result = PSD(nfft=256)(iq)
        assert isinstance(result, torch.Tensor)

    def test_no_nans_or_infs(self):
        from spectra.transforms.psd import PSD

        rng = np.random.default_rng(42)
        iq = (rng.standard_normal(4096) + 1j * rng.standard_normal(4096)).astype(
            np.complex64
        )
        result = PSD(nfft=256)(iq)
        assert not torch.any(torch.isnan(result))
        assert not torch.any(torch.isinf(result))

    def test_tone_has_peak(self):
        from spectra.transforms.psd import PSD

        fs = 1e6
        f_tone = 100e3
        n = 8192
        t = np.arange(n) / fs
        iq = np.exp(1j * 2 * np.pi * f_tone * t).astype(np.complex64)
        result = PSD(nfft=256)(iq)
        # Peak should be near the tone frequency bin
        peak_bin = torch.argmax(result[0]).item()
        # The tone at 100kHz with fs=1MHz should be at bin ~0.1*256 from center
        # After fftshift, DC is at center (bin 128)
        expected_bin = 128 + int(0.1 * 256)
        assert abs(peak_bin - expected_bin) <= 3

    def test_custom_overlap(self):
        from spectra.transforms.psd import PSD

        iq = np.ones(4096, dtype=np.complex64)
        result = PSD(nfft=256, overlap=128)(iq)
        assert result.shape == (1, 256)

    def test_short_signal(self):
        from spectra.transforms.psd import PSD

        # Signal shorter than nfft should still work (zero-padded)
        iq = np.ones(64, dtype=np.complex64)
        result = PSD(nfft=256)(iq)
        assert result.shape == (1, 256)
