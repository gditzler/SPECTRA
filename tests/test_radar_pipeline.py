"""Tests for RadarPipelineDataset."""
import numpy as np
import pytest
import torch


def _make_pipeline_ds(**kwargs):
    from spectra.datasets.radar_pipeline import RadarPipelineDataset
    from spectra.targets.trajectory import ConstantVelocity
    from spectra.impairments.clutter import RadarClutter
    from spectra.waveforms import LFM

    defaults = dict(
        waveform_pool=[LFM()],
        trajectory_pool=[
            ConstantVelocity(initial_range=100.0, velocity=0.5, dt=1.0),
        ],
        swerling_cases=[0],
        clutter_presets=[RadarClutter.ground(sample_rate=1e6, terrain="rural")],
        num_range_bins=256,
        sample_rate=1e6,
        carrier_frequency=10e9,
        pri=1e-3,
        snr_range=(10.0, 20.0),
        num_targets_range=(1, 2),
        sequence_length=5,
        pulses_per_cpi=16,
        apply_mti=True,
        cfar_type="ca",
        num_samples=10,
        seed=42,
    )
    defaults.update(kwargs)
    return RadarPipelineDataset(**defaults)


def test_pipeline_len():
    ds = _make_pipeline_ds(num_samples=15)
    assert len(ds) == 15


def test_pipeline_output_shape():
    ds = _make_pipeline_ds(sequence_length=5, num_range_bins=128)
    data, target = ds[0]
    assert isinstance(data, torch.Tensor)
    assert data.shape == (5, 128)


def test_pipeline_single_frame():
    ds = _make_pipeline_ds(sequence_length=1, num_range_bins=64)
    data, target = ds[0]
    assert data.shape == (1, 64)


def test_pipeline_target_fields():
    from spectra.datasets.radar_pipeline import RadarPipelineTarget
    ds = _make_pipeline_ds(sequence_length=3)
    _, target = ds[0]
    assert isinstance(target, RadarPipelineTarget)
    assert target.true_ranges.shape[0] == 3
    assert target.true_velocities.shape[0] == 3
    assert target.rcs_amplitudes.shape[0] == 3
    assert len(target.detections) == 3
    assert target.kf_states.shape[0] == 3
    assert target.num_targets >= 1


def test_pipeline_deterministic():
    ds = _make_pipeline_ds()
    d1, t1 = ds[3]
    d2, t2 = ds[3]
    assert torch.allclose(d1, d2)
    assert np.allclose(t1.true_ranges, t2.true_ranges)


def test_pipeline_no_mti():
    ds = _make_pipeline_ds(apply_mti=False)
    data, target = ds[0]
    assert isinstance(data, torch.Tensor)


def test_pipeline_os_cfar():
    ds = _make_pipeline_ds(cfar_type="os")
    data, target = ds[0]
    assert isinstance(data, torch.Tensor)


def test_pipeline_tensor_normalized():
    ds = _make_pipeline_ds()
    data, _ = ds[0]
    assert data.min() >= 0.0
    assert data.max() <= 1.0


def test_pipeline_dataloader():
    from spectra.datasets import collate_fn
    ds = _make_pipeline_ds(num_samples=8)
    loader = torch.utils.data.DataLoader(ds, batch_size=4, collate_fn=collate_fn)
    batch_data, batch_targets = next(iter(loader))
    assert batch_data.shape[0] == 4
    assert len(batch_targets) == 4


def test_pipeline_track_doppler_output():
    ds = _make_pipeline_ds(track_doppler=True, sequence_length=5)
    data, target = ds[0]
    assert isinstance(data, torch.Tensor)
    assert target.doppler_detections is not None
    assert len(target.doppler_detections) == 5
    assert target.kf_states.shape[0] == 5
    assert target.kf_states.shape[2] == 2


def test_pipeline_track_doppler_false_backward_compat():
    ds = _make_pipeline_ds(track_doppler=False)
    _, target = ds[0]
    assert target.doppler_detections is None


def test_pipeline_track_doppler_default_is_false():
    ds = _make_pipeline_ds()
    _, target = ds[0]
    assert target.doppler_detections is None


def test_pipeline_track_doppler_deterministic():
    ds = _make_pipeline_ds(track_doppler=True)
    d1, t1 = ds[3]
    d2, t2 = ds[3]
    assert torch.allclose(d1, d2)
    assert np.allclose(t1.kf_states, t2.kf_states)
    for dd1, dd2 in zip(t1.doppler_detections, t2.doppler_detections):
        assert np.array_equal(dd1, dd2)
