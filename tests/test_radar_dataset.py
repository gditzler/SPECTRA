# tests/test_radar_dataset.py
"""Tests for RadarDataset."""
from typing import Any

import numpy as np
import torch
from spectra.waveforms import LFM, BarkerCodedPulse


def _make_ds(**kwargs):
    from spectra.datasets.radar import RadarDataset
    defaults: dict[str, Any] = dict(
        waveform_pool=[LFM(), BarkerCodedPulse()],
        num_range_bins=256,
        sample_rate=1e6,
        snr_range=(5.0, 20.0),
        num_targets_range=(1, 2),
        num_samples=20,
        seed=42,
    )
    defaults.update(kwargs)
    return RadarDataset(**defaults)


def test_radar_dataset_len():
    ds = _make_ds(num_samples=30)
    assert len(ds) == 30


def test_radar_dataset_output_shape():
    from spectra.datasets.radar import RadarTarget
    ds = _make_ds()
    data, target = ds[0]
    assert isinstance(data, torch.Tensor)
    assert data.shape == (256,)
    assert isinstance(target, RadarTarget)


def test_radar_target_fields():
    ds = _make_ds(num_targets_range=(1, 1))
    _, target = ds[0]
    assert target.num_targets >= 0
    assert len(target.range_bins) == target.num_targets
    assert len(target.snrs) == target.num_targets
    assert isinstance(target.waveform_label, str)


def test_radar_dataset_deterministic():
    ds = _make_ds()
    d1, t1 = ds[5]
    d2, t2 = ds[5]
    assert torch.allclose(d1, d2)
    if t1.num_targets > 0:
        assert np.allclose(t1.range_bins, t2.range_bins)


def test_radar_dataset_zero_targets():
    """Dataset must handle num_targets=0 (noise-only)."""
    from spectra.datasets.radar import RadarDataset
    ds = RadarDataset(
        waveform_pool=[LFM()],
        num_range_bins=128,
        sample_rate=1e6,
        snr_range=(10.0, 20.0),
        num_targets_range=(0, 0),
        num_samples=5,
        seed=0,
    )
    data, target = ds[0]
    assert target.num_targets == 0
    assert data.shape == (128,)


def test_radar_dataset_dataloader():
    """Must work with PyTorch DataLoader using a simple collate_fn."""
    from torch.utils.data import DataLoader
    ds = _make_ds(num_targets_range=(1, 1), num_samples=8)

    def collate_fn(batch):
        data = torch.stack([b[0] for b in batch])
        targets = [b[1] for b in batch]
        return data, targets

    loader = DataLoader(ds, batch_size=4, collate_fn=collate_fn)
    batch_data, batch_targets = next(iter(loader))
    assert batch_data.shape == (4, 256)
