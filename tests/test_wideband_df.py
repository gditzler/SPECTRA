# tests/test_wideband_df.py
"""Tests for WidebandDirectionFindingDataset."""
import numpy as np
import torch
from typing import Any
from spectra.arrays.array import ula
from spectra.waveforms import BPSK, QPSK


def _make_ds(**kwargs):
    from spectra.datasets.wideband_df import WidebandDirectionFindingDataset
    arr = ula(num_elements=4, spacing=0.5, frequency=2.4e9)
    defaults: dict[str, Any] = dict(
        array=arr,
        signal_pool=[BPSK(samples_per_symbol=4), QPSK(samples_per_symbol=4)],
        num_signals=2,
        num_snapshots=128,
        sample_rate=10e6,           # 10 MHz wideband capture
        capture_bandwidth=8e6,      # ±4 MHz around DC
        snr_range=(10.0, 20.0),
        azimuth_range=(np.deg2rad(20), np.deg2rad(160)),
        elevation_range=(0.0, 0.0),
        min_freq_separation=1e6,    # 1 MHz min separation
        min_angular_separation=np.deg2rad(15),
        num_samples=20,
        seed=0,
    )
    defaults.update(kwargs)
    return WidebandDirectionFindingDataset(**defaults)


def test_wbdf_len():
    ds = _make_ds(num_samples=25)
    assert len(ds) == 25


def test_wbdf_output_shape():
    from spectra.datasets.wideband_df import WidebandDFTarget
    ds = _make_ds()
    data, target = ds[0]
    assert isinstance(data, torch.Tensor)
    assert data.shape == (4, 2, 128)   # (N_elem, 2, T)
    assert isinstance(target, WidebandDFTarget)


def test_wbdf_target_fields():
    ds = _make_ds(num_signals=2)
    _, target = ds[0]
    assert target.num_signals == 2
    assert len(target.azimuths) == 2
    assert len(target.center_freqs) == 2
    assert len(target.snrs) == 2
    assert len(target.labels) == 2


def test_wbdf_deterministic():
    ds = _make_ds()
    d1, t1 = ds[3]
    d2, t2 = ds[3]
    assert torch.allclose(d1, d2)
    assert np.allclose(t1.azimuths, t2.azimuths)
    assert np.allclose(t1.center_freqs, t2.center_freqs)


def test_wbdf_center_freqs_in_range():
    """All center frequencies must lie within ±capture_bandwidth/2."""
    bw = 8e6
    ds = _make_ds(capture_bandwidth=bw, num_samples=30)
    for i in range(len(ds)):
        _, target = ds[i]
        for f in target.center_freqs:
            assert abs(f) <= bw / 2, f"Center freq {f/1e6:.2f} MHz out of ±{bw/2/1e6:.0f} MHz"


def test_wbdf_freq_separation():
    """Signals must be at least min_freq_separation apart."""
    ds = _make_ds(min_freq_separation=1.5e6, num_signals=2, num_samples=20)
    for i in range(len(ds)):
        _, target = ds[i]
        if target.num_signals >= 2:
            freqs = sorted(target.center_freqs)
            sep = freqs[1] - freqs[0]
            assert sep >= 1.5e6 - 1.0, f"Freq separation {sep/1e6:.3f} MHz < 1.5 MHz"


def test_wbdf_variable_num_signals():
    from spectra.datasets.wideband_df import WidebandDirectionFindingDataset
    arr = ula(num_elements=4, spacing=0.5, frequency=2.4e9)
    ds = WidebandDirectionFindingDataset(
        array=arr,
        signal_pool=[BPSK(samples_per_symbol=4), QPSK(samples_per_symbol=4)],
        num_signals=(1, 3),
        num_snapshots=64,
        sample_rate=10e6,
        capture_bandwidth=8e6,
        snr_range=(10.0, 20.0),
        num_samples=30,
        seed=42,
    )
    counts = {ds[i][1].num_signals for i in range(30)}
    assert len(counts) > 1, "With variable num_signals, should see different counts"
