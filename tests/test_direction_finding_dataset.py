import numpy as np
import pytest
import torch
from torch.utils.data import DataLoader


def _make_dataset(**kwargs):
    """Helper to create a minimal DirectionFindingDataset for testing."""
    from spectra.arrays.array import ula
    from spectra.datasets.direction_finding import DirectionFindingDataset
    from spectra.waveforms import BPSK

    defaults = dict(
        array=ula(num_elements=4, frequency=2.4e9),
        signal_pool=[BPSK(samples_per_symbol=4)],
        num_signals=1,
        num_snapshots=128,
        sample_rate=1e6,
        snr_range=(10.0, 20.0),
        num_samples=50,
        seed=42,
    )
    defaults.update(kwargs)
    return DirectionFindingDataset(**defaults)


def test_dataset_len():
    ds = _make_dataset(num_samples=100)
    assert len(ds) == 100


def test_dataset_getitem_types():
    ds = _make_dataset()
    data, target = ds[0]
    assert isinstance(data, torch.Tensor)
    from spectra.datasets.direction_finding import DirectionFindingTarget
    assert isinstance(target, DirectionFindingTarget)


def test_output_tensor_shape():
    ds = _make_dataset(num_snapshots=128)
    data, target = ds[0]
    assert data.shape == (4, 2, 128)  # 4 elements, 2 channels (I/Q), 128 snapshots
    assert data.dtype == torch.float32


def test_deterministic():
    ds = _make_dataset(seed=7)
    d1, t1 = ds[0]
    d2, t2 = ds[0]
    torch.testing.assert_close(d1, d2)
    np.testing.assert_array_equal(t1.azimuths, t2.azimuths)


def test_different_indices_differ():
    ds = _make_dataset(seed=42)
    d0, _ = ds[0]
    d1, _ = ds[1]
    assert not torch.equal(d0, d1)


def test_target_fields():
    ds = _make_dataset(num_signals=2)
    _, target = ds[0]
    assert target.num_sources == 2
    assert len(target.azimuths) == 2
    assert len(target.elevations) == 2
    assert len(target.snrs) == 2
    assert len(target.labels) == 2
    assert len(target.signal_descs) == 2


def test_target_signal_desc_has_doa():
    ds = _make_dataset(num_signals=1)
    _, target = ds[0]
    desc = target.signal_descs[0]
    assert "doa" in desc.modulation_params
    doa = desc.modulation_params["doa"]
    assert "azimuth_rad" in doa
    assert "elevation_rad" in doa
    assert doa["azimuth_spread_rad"] is None
    assert doa["elevation_spread_rad"] is None


def test_num_signals_range():
    ds = _make_dataset(num_signals=(1, 3))
    for i in range(20):
        _, target = ds[i]
        assert 1 <= target.num_sources <= 3


def test_azimuth_in_range():
    az_range = (0.0, np.pi)
    ds = _make_dataset(azimuth_range=az_range, num_signals=1)
    for i in range(20):
        _, target = ds[i]
        assert az_range[0] <= target.azimuths[0] <= az_range[1]


def test_snr_in_range():
    snr_range = (5.0, 15.0)
    ds = _make_dataset(snr_range=snr_range)
    for i in range(20):
        _, target = ds[i]
        for snr in target.snrs:
            assert snr_range[0] <= snr <= snr_range[1]


def test_with_dataloader():
    ds = _make_dataset(num_samples=16, num_snapshots=64)

    def _collate(batch):
        data = torch.stack([x for x, _ in batch])
        targets = [t for _, t in batch]
        return data, targets

    loader = DataLoader(ds, batch_size=4, collate_fn=_collate)
    batch_data, batch_targets = next(iter(loader))
    assert batch_data.shape == (4, 4, 2, 64)
    assert len(batch_targets) == 4


def test_with_calibration_errors():
    from spectra.arrays.calibration import CalibrationErrors

    cal = CalibrationErrors.random(num_elements=4, rng=np.random.default_rng(0))
    ds = _make_dataset(calibration_errors=cal)
    data, target = ds[0]
    assert data.shape == (4, 2, 128)


def test_with_impairments():
    from spectra.impairments.awgn import AWGN
    from spectra.impairments.compose import Compose

    pipeline = Compose([AWGN(snr=20.0)])
    ds = _make_dataset(impairments=pipeline)
    data, target = ds[0]
    assert data.shape == (4, 2, 128)


def test_with_transform():
    from spectra.transforms.snapshot import ToSnapshotMatrix

    # ToSnapshotMatrix expects numpy input [N, 2, T] but dataset returns torch.Tensor
    # Use a lambda to convert tensor → numpy → snapshot matrix → back to tensor
    def snapshot_transform(x):
        import numpy as np
        arr = x.numpy()
        return torch.from_numpy(arr[:, 0, :] + 1j * arr[:, 1, :])

    ds = _make_dataset(transform=snapshot_transform)
    data, _ = ds[0]
    assert data.shape == (4, 128)
    assert data.is_complex()


def test_min_angular_separation():
    ds = _make_dataset(num_signals=2, min_angular_separation=np.deg2rad(10))
    _, target = ds[0]
    assert target.num_sources == 2  # just verify it runs without error


def test_no_nan_in_output():
    ds = _make_dataset()
    for i in range(5):
        data, _ = ds[i]
        assert not torch.any(torch.isnan(data))
        assert not torch.any(torch.isinf(data))


def test_public_export_from_datasets():
    from spectra.datasets import DirectionFindingDataset, DirectionFindingTarget

    assert DirectionFindingDataset is not None
    assert DirectionFindingTarget is not None
