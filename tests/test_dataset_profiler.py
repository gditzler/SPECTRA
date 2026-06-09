"""Tests for DatasetProfiler (spectra.datasets.profiling)."""

import pytest
from spectra import AWGN, BPSK, QPSK, Compose, NarrowbandDataset
from spectra.datasets import DatasetProfile, DatasetProfiler
from spectra.datasets.profiling import _extract_snr, _narrowband_label


@pytest.fixture
def narrowband_dataset():
    return NarrowbandDataset(
        waveform_pool=[QPSK(), BPSK()],
        num_samples=100,
        num_iq_samples=1024,
        sample_rate=1e6,
        impairments=Compose([AWGN(snr=15.0)]),
        seed=42,
    )


# ---------------------------------------------------------------------------
# Profile run
# ---------------------------------------------------------------------------


def test_profiler_basic(narrowband_dataset):
    profiler = DatasetProfiler(narrowband_dataset, max_samples=16)
    profile = profiler.run()
    assert isinstance(profile, DatasetProfile)
    assert profile.num_samples == 16
    assert len(profile.class_counts) > 0
    assert profile.average_power_db < 100.0  # rough sanity


def test_profiler_all_samples(narrowband_dataset):
    profiler = DatasetProfiler(narrowband_dataset)  # max_samples=None
    profile = profiler.run()
    assert profile.num_samples == len(narrowband_dataset)
    assert sum(profile.class_counts.values()) == profile.num_samples


# ---------------------------------------------------------------------------
# Caching / idempotency
# ---------------------------------------------------------------------------


def test_profiler_repeatable(narrowband_dataset):
    p1 = DatasetProfiler(narrowband_dataset, max_samples=8).run()
    p2 = DatasetProfiler(narrowband_dataset, max_samples=8).run()
    assert p1.num_samples == p2.num_samples
    assert p1.class_counts == p2.class_counts


# ---------------------------------------------------------------------------
# Preset auto-detection
# ---------------------------------------------------------------------------


def test_from_dataset_preset(narrowband_dataset):
    profiler = DatasetProfiler.from_dataset(narrowband_dataset, max_samples=8)
    profile = profiler.run()
    # narrowband labels become "class_0" / "class_1"
    assert any(k.startswith("class_") for k in profile.class_counts)


# ---------------------------------------------------------------------------
# Label extractors
# ---------------------------------------------------------------------------


def test_narrowband_label():
    assert _narrowband_label(0) == "class_0"
    assert _narrowband_label("foo") == "foo"


def test_extract_snr_dict():
    from dataclasses import dataclass

    @dataclass
    class FakeDesc:
        snr: float

    assert _extract_snr(FakeDesc(snr=10.0)) == 10.0
    assert _extract_snr({"snr": 5.0}) == 5.0
    assert _extract_snr({}) is None


# ---------------------------------------------------------------------------
# Profile repr / summary
# ---------------------------------------------------------------------------


def test_profile_summary(narrowband_dataset):
    profile = DatasetProfiler(narrowband_dataset, max_samples=4).run()
    s = profile.summary()
    assert "samples=" in s
    assert "avg_pwr=" in s


def test_profile_repr(narrowband_dataset):
    profile = DatasetProfiler(narrowband_dataset, max_samples=4).run()
    assert "DatasetProfile" in repr(profile)


# ---------------------------------------------------------------------------
# Histogram path (SNRSweepDataset returns float snr label)
# ---------------------------------------------------------------------------


def test_histogram_not_empty_on_snr():
    from spectra import SNRSweepDataset

    def impairments_fn(snr):
        return Compose([AWGN(snr=snr)])

    ds = SNRSweepDataset(
        waveform_pool=[QPSK(), BPSK()],
        snr_levels=[0.0, 10.0],
        samples_per_cell=4,
        num_iq_samples=256,
        sample_rate=1e6,
        impairments_fn=impairments_fn,
        seed=0,
    )
    profiler = DatasetProfiler(ds, max_samples=8, label_extractor=lambda x: str(x))
    profile = profiler.run()
    assert len(profile.snr_histogram.get("counts", [])) > 0
