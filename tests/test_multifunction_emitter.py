"""Tests for spectra.waveforms.multifunction — the multi-function emitter."""

import pytest
from spectra.waveforms import PulsedRadar

# ── Task 2: SegmentSpec + ModeSpec ───────────────────────────────────────────


def test_segment_spec_required_fields():
    from spectra.waveforms.multifunction.segment import SegmentSpec

    spec = SegmentSpec(
        waveform=PulsedRadar(),
        duration_samples=1024,
        mode="search",
    )
    assert spec.duration_samples == 1024
    assert spec.mode == "search"
    assert spec.power_offset_db == 0.0
    assert spec.freq_offset_hz == 0.0
    assert spec.gain_offset_db == 0.0
    assert spec.metadata == {}


def test_segment_spec_optional_fields():
    from spectra.waveforms.multifunction.segment import SegmentSpec

    spec = SegmentSpec(
        waveform=PulsedRadar(),
        duration_samples=1024,
        mode="track",
        power_offset_db=3.0,
        freq_offset_hz=50e3,
        gain_offset_db=2.0,
        metadata={"priority": "high"},
    )
    assert spec.power_offset_db == 3.0
    assert spec.freq_offset_hz == 50e3
    assert spec.gain_offset_db == 2.0
    assert spec.metadata == {"priority": "high"}


def test_mode_spec_fields():
    from spectra.waveforms.multifunction.segment import ModeSpec

    def factory(rng):
        return PulsedRadar()

    spec = ModeSpec(
        waveform_factory=factory,
        duration_samples=(512, 2048),
        power_offset_db=1.5,
        freq_offset_hz=(-100e3, 100e3),
        metadata={"kind": "search"},
    )
    assert spec.waveform_factory is factory
    assert spec.duration_samples == (512, 2048)
    assert spec.power_offset_db == 1.5
    assert spec.freq_offset_hz == (-100e3, 100e3)
    assert spec.metadata == {"kind": "search"}


# ── Task 3: Schedule ABC + StaticSchedule ────────────────────────────────────


def test_schedule_is_abstract():
    from spectra.waveforms.multifunction.schedule import Schedule

    with pytest.raises(TypeError):
        Schedule()  # cannot instantiate abstract class


def test_static_schedule_yields_in_order():
    from spectra.waveforms.multifunction.schedule import StaticSchedule
    from spectra.waveforms.multifunction.segment import SegmentSpec

    s1 = SegmentSpec(waveform=PulsedRadar(), duration_samples=100, mode="a")
    s2 = SegmentSpec(waveform=PulsedRadar(), duration_samples=200, mode="b")
    sched = StaticSchedule(segments=[s1, s2], loop=False)

    out = list(sched.iter_segments(total_samples=500, sample_rate=1e6, seed=0))
    assert [seg.mode for seg in out] == ["a", "b"]


def test_static_schedule_loops_until_total_samples():
    from spectra.waveforms.multifunction.schedule import StaticSchedule
    from spectra.waveforms.multifunction.segment import SegmentSpec

    s1 = SegmentSpec(waveform=PulsedRadar(), duration_samples=100, mode="a")
    s2 = SegmentSpec(waveform=PulsedRadar(), duration_samples=100, mode="b")
    sched = StaticSchedule(segments=[s1, s2], loop=True)

    out = list(sched.iter_segments(total_samples=450, sample_rate=1e6, seed=0))
    # Each iteration adds 100 samples. We need ≥ 450 samples covered.
    # After yielding a,b,a,b,a the cumulative is 500 ≥ 450 → stop.
    assert [seg.mode for seg in out] == ["a", "b", "a", "b", "a"]


def test_static_schedule_no_loop_stops_when_exhausted():
    from spectra.waveforms.multifunction.schedule import StaticSchedule
    from spectra.waveforms.multifunction.segment import SegmentSpec

    s1 = SegmentSpec(waveform=PulsedRadar(), duration_samples=100, mode="a")
    sched = StaticSchedule(segments=[s1], loop=False)

    out = list(sched.iter_segments(total_samples=500, sample_rate=1e6, seed=0))
    assert [seg.mode for seg in out] == ["a"]  # list exhausted after one yield


def test_static_schedule_empty_list_is_empty_iterator():
    from spectra.waveforms.multifunction.schedule import StaticSchedule

    sched = StaticSchedule(segments=[], loop=True)
    out = list(sched.iter_segments(total_samples=100, sample_rate=1e6, seed=0))
    assert out == []


def test_static_schedule_rejects_zero_duration_segment():
    from spectra.waveforms.multifunction.schedule import StaticSchedule
    from spectra.waveforms.multifunction.segment import SegmentSpec

    bad = SegmentSpec(waveform=PulsedRadar(), duration_samples=0, mode="x")
    with pytest.raises(ValueError, match="duration_samples"):
        StaticSchedule(segments=[bad], loop=True)
