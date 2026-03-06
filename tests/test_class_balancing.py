"""Tests for class balancing utilities."""
import numpy as np
import pytest
import torch

from spectra.datasets.narrowband import NarrowbandDataset
from spectra.datasets.sampler import balanced_sampler
from spectra.waveforms import BPSK, QPSK, FM


class TestClassWeights:
    """NarrowbandDataset class_weights parameter."""

    def test_weighted_selection_single_class(self):
        """class_weights=[1, 0, 0] should produce only class 0."""
        pool = [BPSK(), QPSK(), FM()]
        ds = NarrowbandDataset(
            waveform_pool=pool,
            num_samples=100,
            num_iq_samples=256,
            sample_rate=1e6,
            class_weights=[1.0, 0.0, 0.0],
            seed=42,
        )
        labels = [ds[i][1] for i in range(100)]
        assert all(l == 0 for l in labels)

    def test_uniform_when_none(self):
        """class_weights=None should match default uniform behavior."""
        pool = [BPSK(), QPSK(), FM()]
        ds_default = NarrowbandDataset(
            waveform_pool=pool,
            num_samples=50,
            num_iq_samples=256,
            sample_rate=1e6,
            seed=42,
        )
        ds_none = NarrowbandDataset(
            waveform_pool=pool,
            num_samples=50,
            num_iq_samples=256,
            sample_rate=1e6,
            class_weights=None,
            seed=42,
        )
        for i in range(50):
            assert ds_default[i][1] == ds_none[i][1]

    def test_weighted_produces_multiple_classes(self):
        """Balanced weights should produce multiple classes over many samples."""
        pool = [BPSK(), QPSK(), FM()]
        ds = NarrowbandDataset(
            waveform_pool=pool,
            num_samples=200,
            num_iq_samples=256,
            sample_rate=1e6,
            class_weights=[1.0, 1.0, 1.0],
            seed=42,
        )
        labels = set(ds[i][1] for i in range(200))
        assert len(labels) >= 2


class TestBalancedSampler:
    """balanced_sampler utility function."""

    def test_returns_weighted_sampler(self):
        pool = [BPSK(), QPSK()]
        ds = NarrowbandDataset(
            waveform_pool=pool,
            num_samples=100,
            num_iq_samples=256,
            sample_rate=1e6,
            seed=42,
        )
        sampler = balanced_sampler(ds, num_classes=2)
        assert isinstance(sampler, torch.utils.data.WeightedRandomSampler)
        assert sampler.num_samples == 100

    def test_custom_num_samples(self):
        pool = [BPSK(), QPSK()]
        ds = NarrowbandDataset(
            waveform_pool=pool,
            num_samples=100,
            num_iq_samples=256,
            sample_rate=1e6,
            seed=42,
        )
        sampler = balanced_sampler(ds, num_classes=2, num_samples=50)
        assert sampler.num_samples == 50
