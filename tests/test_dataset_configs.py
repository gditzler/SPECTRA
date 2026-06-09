"""Tests for dataset configuration dataclasses (spectra.datasets.configs)."""

import os
import tempfile

import pytest
from spectra import (
    NarrowbandConfig,
    RadarConfig,
    WidebandConfig,
)
from spectra.datasets.configs import (
    BaseDatasetConfig,
    _restore_tuples,
    _sanitize_for_yaml,
    resolve_config,
)

# ---------------------------------------------------------------------------
# Sanity: round-trip dict
# ---------------------------------------------------------------------------


def test_base_config_roundtrip():
    cfg = BaseDatasetConfig(num_samples=500, sample_rate=2e6, seed=7)
    d = cfg.to_dict()
    cfg2 = BaseDatasetConfig.from_dict(d)
    assert cfg == cfg2


def test_narrowband_config_roundtrip():
    cfg = NarrowbandConfig(
        num_samples=1000,
        num_iq_samples=2048,
        snr_range=(-5.0, 30.0),
        class_weights=[2.0, 1.0],
    )
    d = cfg.to_dict()
    cfg2 = NarrowbandConfig.from_dict(d)
    assert cfg == cfg2


def test_wideband_config_roundtrip():
    cfg = WidebandConfig(
        num_signals=(1, 5),
        capture_bandwidth=10e6,
        capture_duration=1e-3,
    )
    d = cfg.to_dict()
    cfg2 = WidebandConfig.from_dict(d)
    assert cfg == cfg2


def test_radar_config_roundtrip():
    cfg = RadarConfig(
        num_range_bins=512,
        snr_range=(5.0, 25.0),
        num_targets_range=(1, 3),
    )
    d = cfg.to_dict()
    cfg2 = RadarConfig.from_dict(d)
    assert cfg == cfg2


# ---------------------------------------------------------------------------
# YAML round-trip
# ---------------------------------------------------------------------------


def test_narrowband_yaml_roundtrip():
    cfg = NarrowbandConfig(num_samples=256, num_iq_samples=1024, seed=99)
    with tempfile.NamedTemporaryFile(mode="w", suffix=".yaml", delete=False) as f:
        path = f.name
    try:
        cfg.to_yaml(path)
        cfg2 = BaseDatasetConfig.from_yaml(path)
        assert isinstance(cfg2, NarrowbandConfig)
        assert cfg2.num_samples == 256
        assert cfg2.seed == 99
    finally:
        os.unlink(path)


def test_wideband_yaml_roundtrip():
    cfg = WidebandConfig(
        num_signals=(2, 4),
        capture_bandwidth=5e6,
        allow_overlap=False,
    )
    with tempfile.NamedTemporaryFile(mode="w", suffix=".yaml", delete=False) as f:
        path = f.name
    try:
        cfg.to_yaml(path)
        cfg2 = BaseDatasetConfig.from_yaml(path)
        assert isinstance(cfg2, WidebandConfig)
        assert cfg2.num_signals == (2, 4)
        assert cfg2.allow_overlap is False
    finally:
        os.unlink(path)


# ---------------------------------------------------------------------------
# Validation errors
# ---------------------------------------------------------------------------


def test_narrowband_bad_iq_samples():
    with pytest.raises(ValueError, match="num_iq_samples"):
        NarrowbandConfig(num_iq_samples=0)


def test_narrowband_bad_snr_range():
    with pytest.raises(ValueError, match="snr_range"):
        NarrowbandConfig(snr_range=(0.0,))


def test_wideband_bad_capture_bandwidth():
    with pytest.raises(ValueError, match="capture_bandwidth"):
        WidebandConfig(capture_bandwidth=-1e6)


def test_wideband_bad_num_signals():
    with pytest.raises(ValueError, match="num_signals"):
        WidebandConfig(num_signals=(5, 1))


def test_radar_bad_targets_range():
    with pytest.raises(ValueError, match="num_targets_range"):
        RadarConfig(num_targets_range=(-1, 2))


# ---------------------------------------------------------------------------
# Registry
# ---------------------------------------------------------------------------


def test_resolve_registered():
    assert resolve_config("narrowband") is NarrowbandConfig
    assert resolve_config("wideband") is WidebandConfig


def test_resolve_unknown_raises():
    with pytest.raises(ValueError, match="Unknown config type"):
        resolve_config("not_a_config")


# ---------------------------------------------------------------------------
# Factory build_dataset requires waveform pool
# ---------------------------------------------------------------------------


@pytest.mark.slow
def test_narrowband_build_dataset():
    from spectra import AWGN, BPSK, QPSK, Compose

    cfg = NarrowbandConfig(num_samples=16, num_iq_samples=512)
    ds = cfg.build_dataset(
        waveform_pool=[QPSK(), BPSK()],
        impairments=Compose([AWGN(snr=10.0)]),
    )
    assert len(ds) == 16
    iq, label = ds[0]
    assert iq.shape == (2, 512)


# ---------------------------------------------------------------------------
# Sanitise helpers
# ---------------------------------------------------------------------------


def test_sanitize_tuple_to_list():
    d = {"a": (1, 2), "b": ["x", "y"], "c": {"d": (3, 4)}}
    out = _sanitize_for_yaml(d)
    assert out["a"] == [1, 2]
    assert isinstance(out["a"], list)
    assert out["c"]["d"] == [3, 4]


def test_restore_tuples():
    d = _restore_tuples({"a": [1, 2]}, fields=["a"])
    assert d["a"] == (1, 2)
