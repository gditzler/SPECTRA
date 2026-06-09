"""Tests for SpectrogramDataset (spectra.datasets.spectrogram_dataset)."""

import numpy as np
import pytest
import torch
from spectra import AWGN, BPSK, QPSK, Compose, NarrowbandDataset
from spectra.datasets import SpectrogramDataset
from spectra.transforms import STFT


@pytest.fixture
def iq_dataset():
    return NarrowbandDataset(
        waveform_pool=[QPSK(), BPSK()],
        num_samples=32,
        num_iq_samples=1024,
        sample_rate=1e6,
        impairments=Compose([AWGN(snr=10.0)]),
        seed=42,
    )


# ---------------------------------------------------------------------------
# Basic shape verification
# ---------------------------------------------------------------------------


def test_spectrogram_shape(iq_dataset):
    spec_ds = SpectrogramDataset(iq_dataset, transform=STFT(nfft=256, hop_length=64))
    assert len(spec_ds) == len(iq_dataset)
    spec, label = spec_ds[0]
    assert isinstance(spec, torch.Tensor)
    assert spec.dim() == 3
    assert spec.shape[0] == 1  # single channel
    assert spec.shape[1] == 256  # nfft (full FFT, incl. negative freqs)
    assert spec.shape[2] == 17  # ~1024 / 64 + 1


def test_spectrogram_dtype(iq_dataset):
    spec_ds = SpectrogramDataset(iq_dataset, transform=STFT(nfft=256))
    spec, _ = spec_ds[0]
    assert spec.dtype == torch.float32


# ---------------------------------------------------------------------------
# Cache behaviour
# ---------------------------------------------------------------------------


def test_memory_cache(iq_dataset):
    spec_ds = SpectrogramDataset(
        iq_dataset, transform=STFT(nfft=256), cache="memory"
    )
    spec1, label1 = spec_ds[0]
    spec2, label2 = spec_ds[0]
    assert torch.equal(spec1, spec2)
    assert label1 == label2


def test_no_cache(iq_dataset):
    spec_ds = SpectrogramDataset(iq_dataset, transform=STFT(nfft=256), cache=None)
    spec1, _ = spec_ds[0]
    spec2, _ = spec_ds[0]
    assert spec1.shape == spec2.shape
    assert spec1.dtype == spec2.dtype
    assert torch.isfinite(spec1).all()


def test_cache_size_limit(iq_dataset):
    spec_ds = SpectrogramDataset(
        iq_dataset, transform=STFT(nfft=256), cache="memory", cache_size=2
    )
    # Touch first two items
    _ = spec_ds[0]
    _ = spec_ds[1]
    assert len(spec_ds.cache) == 2
    # Touch third; should evict first
    _ = spec_ds[2]
    assert len(spec_ds.cache) == 2
    assert 0 not in spec_ds.cache
    assert 1 in spec_ds.cache


def test_clear_cache(iq_dataset):
    spec_ds = SpectrogramDataset(
        iq_dataset, transform=STFT(nfft=256), cache="memory"
    )
    _ = spec_ds[0]
    spec_ds.clear_cache()
    assert len(spec_ds.cache) == 0


# ---------------------------------------------------------------------------
# freq_bins / time_bins properties
# ---------------------------------------------------------------------------


def test_freq_time_bins(iq_dataset):
    spec_ds = SpectrogramDataset(iq_dataset, transform=STFT(nfft=256))
    with pytest.raises(RuntimeError):
        _ = spec_ds.freq_bins
    _ = spec_ds[0]
    assert spec_ds.freq_bins == 256
    assert spec_ds.time_bins > 0


# ---------------------------------------------------------------------------
# Tensor input path
# ---------------------------------------------------------------------------


def test_tensor_iq_input():
    """A dataset that returns [2, N] real tensors."""

    class FakeDataset:
        def __len__(self):
            return 4

        def __getitem__(self, idx):
            return torch.randn(2, 512), idx

    ds = SpectrogramDataset(FakeDataset(), transform=STFT(nfft=256))
    spec, label = ds[0]
    assert spec.dim() == 3
    assert spec.shape[0] == 1


# ---------------------------------------------------------------------------
# Error handling
# ---------------------------------------------------------------------------


def test_bad_transform_output():
    class BadTransform:
        def __call__(self, iq):
            # Returns a 1-D vector — not acceptable
            return torch.tensor([1.0, 2.0, 3.0])

    class FakeDataset:
        def __len__(self):
            return 1

        def __getitem__(self, idx):
            return np.ones(64, dtype=np.complex64), 0

    ds = SpectrogramDataset(FakeDataset(), transform=BadTransform())
    with pytest.raises(RuntimeError, match="expected transform output"):
        ds[0]
