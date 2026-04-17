"""Tests for spectra.waveforms.multifunction — the multi-function emitter."""

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
