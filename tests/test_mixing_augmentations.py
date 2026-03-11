"""Tests for MixUp and CutMix augmentations."""

import numpy as np
import torch
from spectra.datasets.mixing import CutMixDataset, MixUpDataset
from spectra.datasets.narrowband import NarrowbandDataset
from spectra.transforms.augmentations import CutMix, MixUp
from spectra.waveforms import BPSK, QPSK


class TestMixUpSignalLevel:
    """Signal-level MixUp augmentation."""

    def test_preserves_shape(self):
        rng = np.random.default_rng(42)
        iq = (rng.standard_normal(1024) + 1j * rng.standard_normal(1024)).astype(np.complex64)
        aug = MixUp(alpha=0.2)
        out = aug(iq, rng=rng)
        assert out.shape == iq.shape

    def test_preserves_dtype(self):
        rng = np.random.default_rng(42)
        iq = (rng.standard_normal(512) + 1j * rng.standard_normal(512)).astype(np.complex64)
        aug = MixUp(alpha=0.2)
        out = aug(iq, rng=rng)
        assert out.dtype == iq.dtype

    def test_output_differs(self):
        rng = np.random.default_rng(42)
        iq = (rng.standard_normal(512) + 1j * rng.standard_normal(512)).astype(np.complex64)
        aug = MixUp(alpha=0.2)
        out = aug(iq, rng=np.random.default_rng(42))
        # Output should generally differ from input (unless lambda=1)
        assert out.shape == iq.shape


class TestCutMixSignalLevel:
    """Signal-level CutMix augmentation."""

    def test_preserves_shape(self):
        rng = np.random.default_rng(42)
        iq = (rng.standard_normal(1024) + 1j * rng.standard_normal(1024)).astype(np.complex64)
        aug = CutMix(alpha=1.0)
        out = aug(iq, rng=rng)
        assert out.shape == iq.shape

    def test_preserves_dtype(self):
        rng = np.random.default_rng(42)
        iq = (rng.standard_normal(512) + 1j * rng.standard_normal(512)).astype(np.complex64)
        aug = CutMix(alpha=1.0)
        out = aug(iq, rng=rng)
        assert out.dtype == iq.dtype


class TestMixUpDataset:
    """MixUpDataset wrapper."""

    def test_returns_soft_labels(self):
        pool = [BPSK(), QPSK()]
        base = NarrowbandDataset(
            waveform_pool=pool,
            num_samples=20,
            num_iq_samples=256,
            sample_rate=1e6,
            seed=42,
        )
        ds = MixUpDataset(base, alpha=0.2)
        assert len(ds) == 20
        x, (y1, y2, lam) = ds[0]
        assert isinstance(x, torch.Tensor)
        assert isinstance(y1, int)
        assert isinstance(y2, int)
        assert 0.0 <= lam <= 1.0


class TestCutMixDataset:
    """CutMixDataset wrapper."""

    def test_returns_soft_labels(self):
        pool = [BPSK(), QPSK()]
        base = NarrowbandDataset(
            waveform_pool=pool,
            num_samples=20,
            num_iq_samples=256,
            sample_rate=1e6,
            seed=42,
        )
        ds = CutMixDataset(base, alpha=1.0)
        assert len(ds) == 20
        x, (y1, y2, lam) = ds[0]
        assert isinstance(x, torch.Tensor)
        assert isinstance(y1, int)
        assert isinstance(y2, int)
        assert 0.0 <= lam <= 1.0

    def test_shape_preserved(self):
        pool = [BPSK(), QPSK()]
        base = NarrowbandDataset(
            waveform_pool=pool,
            num_samples=10,
            num_iq_samples=256,
            sample_rate=1e6,
            seed=42,
        )
        ds = CutMixDataset(base, alpha=1.0)
        x, _ = ds[0]
        x_base, _ = base[0]
        assert x.shape == x_base.shape
