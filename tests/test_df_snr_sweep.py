# tests/test_df_snr_sweep.py
import numpy as np
import torch
from spectra.arrays.array import ula
from spectra.waveforms import BPSK, QPSK


def _make_ds(snr_levels=None):
    from spectra.datasets.df_snr_sweep import DirectionFindingSNRSweepDataset
    arr = ula(num_elements=4, spacing=0.5, frequency=1e9)
    if snr_levels is None:
        snr_levels = [0.0, 10.0, 20.0]
    return DirectionFindingSNRSweepDataset(
        array=arr,
        signal_pool=[BPSK(samples_per_symbol=4), QPSK(samples_per_symbol=4)],
        snr_levels=snr_levels,
        samples_per_snr=5,
        num_signals=1,
        num_snapshots=64,
        sample_rate=1e6,
        azimuth_range=(np.deg2rad(30), np.deg2rad(150)),
        seed=0,
    )


def test_df_snr_sweep_len():
    ds = _make_ds(snr_levels=[0.0, 10.0, 20.0])
    assert len(ds) == 3 * 5  # 3 SNR levels × 5 samples each


def test_df_snr_sweep_output_shape():
    ds = _make_ds()
    data, target, snr_db = ds[0]
    assert isinstance(data, torch.Tensor)
    assert data.shape == (4, 2, 64)  # (n_elements, 2, num_snapshots)
    assert isinstance(snr_db, float)


def test_df_snr_sweep_snr_matches():
    """Returned snr_db must match the SNR level for that cell."""
    ds = _make_ds(snr_levels=[5.0, 15.0])
    _, _, snr0 = ds[0]   # first cell: SNR=5.0
    _, _, snr1 = ds[5]   # second cell: SNR=15.0
    assert abs(snr0 - 5.0) < 1e-6
    assert abs(snr1 - 15.0) < 1e-6


def test_df_snr_sweep_deterministic():
    ds = _make_ds()
    d1, t1, s1 = ds[3]
    d2, t2, s2 = ds[3]
    assert torch.allclose(d1, d2)
    assert np.allclose(t1.azimuths, t2.azimuths)
