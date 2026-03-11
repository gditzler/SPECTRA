"""Tests for the Reassigned Gabor Transform."""

import numpy as np
import pytest
import torch
from spectra.transforms.reassigned_gabor import ReassignedGabor


def _make_tone(f0: float, n: int) -> np.ndarray:
    """Pure complex tone at normalized frequency f0 ∈ (0, 1)."""
    return np.exp(1j * 2 * np.pi * f0 * np.arange(n)).astype(np.complex64)


def _make_noise(n: int, seed: int = 0) -> np.ndarray:
    rng = np.random.default_rng(seed)
    return (rng.standard_normal(n) + 1j * rng.standard_normal(n)).astype(np.complex64)


class TestReassignedGaborShape:
    """Output shape and dtype tests."""

    def test_shape_basic(self):
        # n_frames = 1 + (512 - 128) // 64 = 7
        iq = _make_tone(0.1, 512)
        rgt = ReassignedGabor(nfft=128, hop_length=64, sigma=16.0)
        out = rgt(iq)
        assert out.shape == (1, 128, 7)
        assert out.dtype == torch.float32

    def test_shape_default_sigma(self):
        # sigma defaults to nfft / 4
        iq = _make_tone(0.1, 512)
        rgt = ReassignedGabor(nfft=128, hop_length=64)
        out = rgt(iq)
        assert out.shape == (1, 128, 7)

    def test_shape_longer_signal(self):
        # n_frames = 1 + (4096 - 256) // 64 = 61
        iq = _make_noise(4096)
        rgt = ReassignedGabor(nfft=256, hop_length=64)
        out = rgt(iq)
        assert out.shape == (1, 256, 61)

    def test_single_frame(self):
        # Signal length == nfft → exactly 1 frame
        iq = _make_tone(0.1, 256)
        rgt = ReassignedGabor(nfft=256, hop_length=64)
        out = rgt(iq)
        assert out.shape == (1, 256, 1)


class TestReassignedGaborContent:
    """Numerical correctness and signal property tests."""

    def test_non_negative(self):
        """Output is power (|Sx|²), must be non-negative everywhere."""
        iq = _make_noise(1024)
        rgt = ReassignedGabor(nfft=128, hop_length=32)
        out = rgt(iq)
        assert (out >= 0).all()

    def test_no_nan_or_inf(self):
        iq = _make_noise(1024)
        rgt = ReassignedGabor(nfft=128, hop_length=32)
        out = rgt(iq)
        assert not torch.any(torch.isnan(out))
        assert not torch.any(torch.isinf(out))

    def test_deterministic(self):
        iq = _make_noise(1024)
        rgt = ReassignedGabor(nfft=128, hop_length=32)
        r1 = rgt(iq)
        r2 = rgt(iq)
        assert torch.equal(r1, r2)

    def test_pure_tone_concentrated(self):
        """
        A pure tone should concentrate energy at a single frequency band.
        Compare: peak row energy >> mean row energy across the spectrum.
        """
        nfft = 256
        f0 = 0.1  # normalized frequency
        iq = _make_tone(f0, 2048)
        rgt = ReassignedGabor(nfft=nfft, hop_length=64, sigma=32.0)
        out = rgt(iq)  # [1, nfft, n_frames]
        # Sum power across time for each frequency bin → [nfft]
        freq_power = out[0].sum(dim=1)
        peak_val = freq_power.max().item()
        mean_val = freq_power.mean().item()
        # Reassigned spectrogram should be much more concentrated than average
        assert peak_val > 10 * mean_val, (
            f"Expected concentrated peak; peak={peak_val:.4f}, mean={mean_val:.4f}"
        )

    def test_accepts_float64_input(self):
        """Python wrapper should cast float64 input to complex64."""
        iq = _make_tone(0.1, 512).astype(np.complex128)
        rgt = ReassignedGabor(nfft=128, hop_length=64)
        out = rgt(iq)
        assert out.dtype == torch.float32


class TestReassignedGaborValidation:
    """Constructor validation."""

    def test_invalid_nfft(self):
        with pytest.raises(ValueError):
            ReassignedGabor(nfft=0)

    def test_invalid_hop(self):
        with pytest.raises(ValueError):
            ReassignedGabor(hop_length=0)

    def test_invalid_sigma(self):
        with pytest.raises(ValueError):
            ReassignedGabor(sigma=-1.0)
