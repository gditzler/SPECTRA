"""Tests for spectra.waveforms.multifunction — the multi-function emitter."""

import numpy as np
import pytest
from spectra.scene.signal_desc import SignalDescription
from spectra.waveforms import QPSK, PulsedRadar

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


# ── Task 4: ScheduledWaveform orchestration ─────────────────────────────────

FS = 1e6


def _small_radar(pri=128, num_pulses=4, pw=16):
    return PulsedRadar(
        pulse_width_samples=pw,
        pri_samples=pri,
        num_pulses=num_pulses,
    )


def test_scheduled_waveform_output_length_exact(assert_valid_iq):
    from spectra.waveforms.multifunction.schedule import StaticSchedule
    from spectra.waveforms.multifunction.scheduled_waveform import ScheduledWaveform
    from spectra.waveforms.multifunction.segment import SegmentSpec

    seg = SegmentSpec(waveform=_small_radar(), duration_samples=512, mode="m")
    sw = ScheduledWaveform(StaticSchedule([seg], loop=True), label="TestMFR")

    for n in (300, 512, 1000, 2050):
        iq = sw.generate(num_symbols=n, sample_rate=FS, seed=1)
        assert_valid_iq(iq, expected_length=n)


def test_scheduled_waveform_generate_matches_generate_with_segments():
    from spectra.waveforms.multifunction.schedule import StaticSchedule
    from spectra.waveforms.multifunction.scheduled_waveform import ScheduledWaveform
    from spectra.waveforms.multifunction.segment import SegmentSpec

    seg = SegmentSpec(waveform=_small_radar(), duration_samples=256, mode="m")
    sw = ScheduledWaveform(StaticSchedule([seg], loop=True))

    iq_a = sw.generate(num_symbols=1024, sample_rate=FS, seed=7)
    iq_b, segments = sw.generate_with_segments(num_samples=1024, sample_rate=FS, seed=7)
    assert np.array_equal(iq_a, iq_b)
    assert len(segments) >= 1
    assert all(isinstance(s, SignalDescription) for s in segments)
    assert all(s.mode == "m" for s in segments)


def test_scheduled_waveform_segment_timeline_covers_output():
    from spectra.waveforms.multifunction.schedule import StaticSchedule
    from spectra.waveforms.multifunction.scheduled_waveform import ScheduledWaveform
    from spectra.waveforms.multifunction.segment import SegmentSpec

    seg_a = SegmentSpec(waveform=_small_radar(), duration_samples=256, mode="a")
    seg_b = SegmentSpec(waveform=_small_radar(), duration_samples=256, mode="b")
    sw = ScheduledWaveform(StaticSchedule([seg_a, seg_b], loop=True))

    _, segments = sw.generate_with_segments(num_samples=1024, sample_rate=FS, seed=0)

    # Segments back-to-back, cover exactly 0..1024 samples (1024/fs seconds).
    assert segments[0].t_start == 0.0
    last_stop_samples = round(segments[-1].t_stop * FS)
    assert last_stop_samples == 1024
    for prev, nxt in zip(segments[:-1], segments[1:]):
        assert prev.t_stop == pytest.approx(nxt.t_start)


def test_scheduled_waveform_truncates_final_segment():
    from spectra.waveforms.multifunction.schedule import StaticSchedule
    from spectra.waveforms.multifunction.scheduled_waveform import ScheduledWaveform
    from spectra.waveforms.multifunction.segment import SegmentSpec

    seg = SegmentSpec(waveform=_small_radar(), duration_samples=500, mode="m")
    sw = ScheduledWaveform(StaticSchedule([seg], loop=True))

    iq, segments = sw.generate_with_segments(num_samples=1200, sample_rate=FS, seed=0)
    assert len(iq) == 1200
    # 2 full segments (1000 samples) plus a final truncated segment of 200.
    assert len(segments) == 3
    durations = [round((s.t_stop - s.t_start) * FS) for s in segments]
    assert durations == [500, 500, 200]


def test_scheduled_waveform_power_offset_applied():
    """+6 dB power offset ≈ 4× power vs baseline."""
    from spectra.waveforms.multifunction.schedule import StaticSchedule
    from spectra.waveforms.multifunction.scheduled_waveform import ScheduledWaveform
    from spectra.waveforms.multifunction.segment import SegmentSpec

    wf = QPSK(samples_per_symbol=8)
    base = SegmentSpec(waveform=wf, duration_samples=2048, mode="base")
    boosted = SegmentSpec(
        waveform=wf,
        duration_samples=2048,
        mode="boost",
        power_offset_db=6.0,
    )

    sw_base = ScheduledWaveform(StaticSchedule([base], loop=False))
    sw_boost = ScheduledWaveform(StaticSchedule([boosted], loop=False))

    iq_base = sw_base.generate(num_symbols=2048, sample_rate=FS, seed=42)
    iq_boost = sw_boost.generate(num_symbols=2048, sample_rate=FS, seed=42)
    p_base = np.mean(np.abs(iq_base[500:1500]) ** 2)
    p_boost = np.mean(np.abs(iq_boost[500:1500]) ** 2)
    # 6 dB = 10^(6/10) ≈ 3.98. Allow ±10%.
    ratio = p_boost / p_base
    assert 3.6 < ratio < 4.4, f"expected ~4x, got {ratio:.2f}"


def test_scheduled_waveform_freq_offset_applied():
    """A segment with freq_offset_hz=+F puts most energy in the positive half-spectrum.

    We compare total spectral energy above DC to total energy below DC. A pure
    positive shift must make the positive half dominate — we don't rely on
    argmax peak position, which is unstable for a finite random-symbol burst.
    """
    from spectra.waveforms.multifunction.schedule import StaticSchedule
    from spectra.waveforms.multifunction.scheduled_waveform import ScheduledWaveform
    from spectra.waveforms.multifunction.segment import SegmentSpec

    wf = QPSK(samples_per_symbol=8)
    offset = 100e3  # 100 kHz at fs=1 MHz
    seg = SegmentSpec(waveform=wf, duration_samples=8192, mode="m", freq_offset_hz=offset)
    sw = ScheduledWaveform(StaticSchedule([seg], loop=False))

    iq = sw.generate(num_symbols=8192, sample_rate=FS, seed=0)
    spec_power = np.abs(np.fft.fftshift(np.fft.fft(iq))) ** 2
    freqs = np.fft.fftshift(np.fft.fftfreq(len(iq), d=1.0 / FS))
    pos_energy = spec_power[freqs > 0].sum()
    neg_energy = spec_power[freqs < 0].sum()
    # QPSK is symmetric around DC at baseband, so without the shift the two halves
    # are equal. After +100 kHz shift (well below fs/2=500 kHz), the positive half
    # must dominate by a large margin.
    assert pos_energy > 5.0 * neg_energy, (
        f"expected positive-half energy > 5x negative-half; got "
        f"pos={pos_energy:.2e}, neg={neg_energy:.2e}"
    )


def test_scheduled_waveform_label_is_user_supplied():
    from spectra.waveforms.multifunction.schedule import StaticSchedule
    from spectra.waveforms.multifunction.scheduled_waveform import ScheduledWaveform
    from spectra.waveforms.multifunction.segment import SegmentSpec

    seg = SegmentSpec(waveform=_small_radar(), duration_samples=100, mode="m")
    sw = ScheduledWaveform(StaticSchedule([seg], loop=True), label="MyMFR")
    assert sw.label == "MyMFR"


def test_scheduled_waveform_bandwidth_with_default():
    from spectra.waveforms.multifunction.schedule import StaticSchedule
    from spectra.waveforms.multifunction.scheduled_waveform import ScheduledWaveform
    from spectra.waveforms.multifunction.segment import SegmentSpec

    seg = SegmentSpec(waveform=_small_radar(), duration_samples=100, mode="m")
    sw = ScheduledWaveform(
        StaticSchedule([seg], loop=True),
        default_bandwidth_hz=1e6,
    )
    assert sw.bandwidth(sample_rate=FS) == 1e6


def test_scheduled_waveform_bandwidth_heuristic():
    """Without default, bandwidth() reflects the widest child + its offset."""
    from spectra.waveforms.multifunction.schedule import StaticSchedule
    from spectra.waveforms.multifunction.scheduled_waveform import ScheduledWaveform
    from spectra.waveforms.multifunction.segment import SegmentSpec

    narrow = SegmentSpec(
        waveform=_small_radar(pw=32),
        duration_samples=256,
        mode="n",
    )
    wide = SegmentSpec(
        waveform=_small_radar(pw=4),
        duration_samples=256,
        mode="w",
        freq_offset_hz=50e3,
    )
    sw = ScheduledWaveform(StaticSchedule([narrow, wide], loop=True))
    # Wide child: fs / pw = 1e6/4 = 250 kHz. Plus 2*|offset| = 100 kHz.
    bw = sw.bandwidth(sample_rate=FS)
    assert bw >= 250e3 + 100e3 - 1  # ≥ 350 kHz (allow 1 Hz float slop)


def test_scheduled_waveform_samples_per_symbol_is_one():
    from spectra.waveforms.multifunction.schedule import StaticSchedule
    from spectra.waveforms.multifunction.scheduled_waveform import ScheduledWaveform
    from spectra.waveforms.multifunction.segment import SegmentSpec

    seg = SegmentSpec(waveform=_small_radar(), duration_samples=100, mode="m")
    sw = ScheduledWaveform(StaticSchedule([seg], loop=True))
    # Composer uses `getattr(w, "samples_per_symbol", 8)`; for us this means
    # num_symbols == num_samples.
    assert sw.samples_per_symbol == 1


# ── Task 5: StochasticSchedule ──────────────────────────────────────────────


def test_stochastic_schedule_validates_unknown_mode():
    from spectra.waveforms.multifunction.schedule import StochasticSchedule
    from spectra.waveforms.multifunction.segment import ModeSpec

    def _factory(r):
        return _small_radar()

    modes = {"a": ModeSpec(waveform_factory=_factory, duration_samples=100)}
    with pytest.raises(ValueError, match="unknown"):
        StochasticSchedule(
            modes=modes,
            transitions={"a": {"a": 0.5, "b": 0.5}},  # 'b' not in modes
            initial_mode="a",
        )


def test_stochastic_schedule_validates_row_sum():
    from spectra.waveforms.multifunction.schedule import StochasticSchedule
    from spectra.waveforms.multifunction.segment import ModeSpec

    def _factory(r):
        return _small_radar()

    modes = {
        "a": ModeSpec(waveform_factory=_factory, duration_samples=100),
        "b": ModeSpec(waveform_factory=_factory, duration_samples=100),
    }
    with pytest.raises(ValueError, match="sum"):
        StochasticSchedule(
            modes=modes,
            transitions={"a": {"a": 0.5, "b": 0.2}, "b": {"a": 0.5, "b": 0.5}},
            initial_mode="a",
        )


def test_stochastic_schedule_validates_non_positive_duration():
    from spectra.waveforms.multifunction.schedule import StochasticSchedule
    from spectra.waveforms.multifunction.segment import ModeSpec

    def _factory(r):
        return _small_radar()

    bad_scalar = {"a": ModeSpec(waveform_factory=_factory, duration_samples=0)}
    with pytest.raises(ValueError, match="duration_samples"):
        StochasticSchedule(modes=bad_scalar, transitions={"a": {"a": 1.0}}, initial_mode="a")

    bad_tuple = {"a": ModeSpec(waveform_factory=_factory, duration_samples=(0, 100))}
    with pytest.raises(ValueError, match="duration range"):
        StochasticSchedule(modes=bad_tuple, transitions={"a": {"a": 1.0}}, initial_mode="a")


def test_stochastic_schedule_validates_initial_mode_unknown():
    from spectra.waveforms.multifunction.schedule import StochasticSchedule
    from spectra.waveforms.multifunction.segment import ModeSpec

    def _factory(r):
        return _small_radar()

    modes = {"a": ModeSpec(waveform_factory=_factory, duration_samples=100)}
    with pytest.raises(ValueError, match="initial_mode"):
        StochasticSchedule(
            modes=modes,
            transitions={"a": {"a": 1.0}},
            initial_mode="b",
        )


def test_stochastic_schedule_determinism():
    """Same seed -> identical segment list."""
    from spectra.waveforms.multifunction.schedule import StochasticSchedule
    from spectra.waveforms.multifunction.segment import ModeSpec

    def _factory_a(r):
        return _small_radar(pri=128)

    def _factory_b(r):
        return _small_radar(pri=256)

    modes = {
        "a": ModeSpec(waveform_factory=_factory_a, duration_samples=(100, 400)),
        "b": ModeSpec(waveform_factory=_factory_b, duration_samples=(100, 400)),
    }
    sched = StochasticSchedule(
        modes=modes,
        transitions={"a": {"a": 0.5, "b": 0.5}, "b": {"a": 0.5, "b": 0.5}},
        initial_mode="a",
    )

    run1 = [(s.mode, s.duration_samples) for s in sched.iter_segments(5000, 1e6, seed=123)]
    run2 = [(s.mode, s.duration_samples) for s in sched.iter_segments(5000, 1e6, seed=123)]
    assert run1 == run2
    assert len(run1) >= 2  # enough segments in 5000 samples


def test_stochastic_schedule_different_seeds_differ():
    from spectra.waveforms.multifunction.schedule import StochasticSchedule
    from spectra.waveforms.multifunction.segment import ModeSpec

    def _factory_a(r):
        return _small_radar()

    def _factory_b(r):
        return _small_radar(pri=256)

    modes = {
        "a": ModeSpec(waveform_factory=_factory_a, duration_samples=(100, 400)),
        "b": ModeSpec(waveform_factory=_factory_b, duration_samples=(100, 400)),
    }
    sched = StochasticSchedule(
        modes=modes,
        transitions={"a": {"a": 0.3, "b": 0.7}, "b": {"a": 0.7, "b": 0.3}},
        initial_mode={"a": 0.5, "b": 0.5},
    )

    run1 = [(s.mode, s.duration_samples) for s in sched.iter_segments(10_000, 1e6, seed=1)]
    run2 = [(s.mode, s.duration_samples) for s in sched.iter_segments(10_000, 1e6, seed=2)]
    assert run1 != run2


def test_stochastic_schedule_duration_callable():
    from spectra.waveforms.multifunction.schedule import StochasticSchedule
    from spectra.waveforms.multifunction.segment import ModeSpec

    def _factory(r):
        return _small_radar()

    def _dur(r):
        return 250

    modes = {
        "fixed": ModeSpec(
            waveform_factory=_factory,
            duration_samples=_dur,
        ),
    }
    sched = StochasticSchedule(
        modes=modes,
        transitions={"fixed": {"fixed": 1.0}},
        initial_mode="fixed",
    )
    out = list(sched.iter_segments(1000, 1e6, seed=0))
    assert all(s.duration_samples == 250 for s in out)


def test_stochastic_schedule_freq_and_power_offsets_drawn_from_range():
    from spectra.waveforms.multifunction.schedule import StochasticSchedule
    from spectra.waveforms.multifunction.segment import ModeSpec

    def _factory(r):
        return _small_radar()

    modes = {
        "m": ModeSpec(
            waveform_factory=_factory,
            duration_samples=200,
            power_offset_db=(-2.0, 2.0),
            freq_offset_hz=(-50e3, 50e3),
        ),
    }
    sched = StochasticSchedule(
        modes=modes,
        transitions={"m": {"m": 1.0}},
        initial_mode="m",
    )
    segs = list(sched.iter_segments(5000, 1e6, seed=0))
    assert len(segs) > 1
    for s in segs:
        assert -2.0 <= s.power_offset_db <= 2.0
        assert -50e3 <= s.freq_offset_hz <= 50e3


def test_scheduled_waveform_with_stochastic_determinism(assert_valid_iq):
    """ScheduledWaveform IQ is byte-identical with same seed using stochastic."""
    from spectra.waveforms.multifunction.schedule import StochasticSchedule
    from spectra.waveforms.multifunction.scheduled_waveform import ScheduledWaveform
    from spectra.waveforms.multifunction.segment import ModeSpec

    def _factory_a(r):
        return _small_radar(pri=128)

    def _factory_b(r):
        return _small_radar(pri=256)

    modes = {
        "a": ModeSpec(waveform_factory=_factory_a, duration_samples=(200, 400)),
        "b": ModeSpec(waveform_factory=_factory_b, duration_samples=(200, 400)),
    }
    sw = ScheduledWaveform(
        StochasticSchedule(
            modes=modes,
            transitions={"a": {"a": 0.3, "b": 0.7}, "b": {"a": 0.7, "b": 0.3}},
            initial_mode="a",
        )
    )

    iq1 = sw.generate(num_symbols=5000, sample_rate=1e6, seed=42)
    iq2 = sw.generate(num_symbols=5000, sample_rate=1e6, seed=42)
    assert_valid_iq(iq1, expected_length=5000)
    assert np.array_equal(iq1, iq2)


# ── Task 6: CognitiveSchedule ABC ───────────────────────────────────────────


def test_cognitive_schedule_cannot_instantiate_directly():
    from spectra.waveforms.multifunction.schedule import CognitiveSchedule

    with pytest.raises(TypeError):
        CognitiveSchedule()


def test_cognitive_schedule_subclass_works():
    """A trivial deterministic subclass validates the seam."""
    from spectra.waveforms.multifunction.schedule import CognitiveSchedule
    from spectra.waveforms.multifunction.segment import SegmentSpec

    class AlternatingCognitive(CognitiveSchedule):
        def next_segment(self, history, state):
            mode = "b" if history and history[-1].mode == "a" else "a"
            return SegmentSpec(
                waveform=_small_radar(),
                duration_samples=100,
                mode=mode,
            )

    sched = AlternatingCognitive()
    modes = [s.mode for s in sched.iter_segments(400, 1e6, seed=0)]
    assert modes == ["a", "b", "a", "b"]


def test_cognitive_schedule_returning_none_ends_schedule():
    from spectra.waveforms.multifunction.schedule import CognitiveSchedule
    from spectra.waveforms.multifunction.segment import SegmentSpec

    class FixedLength(CognitiveSchedule):
        def __init__(self, n):
            self._n = n
            self._count = 0

        def next_segment(self, history, state):
            if self._count >= self._n:
                return None
            self._count += 1
            return SegmentSpec(
                waveform=_small_radar(),
                duration_samples=100,
                mode="m",
            )

    sched = FixedLength(3)
    out = list(sched.iter_segments(10_000, 1e6, seed=0))
    assert len(out) == 3


# ── Task 7: segments_to_mode_mask helper ─────────────────────────────────────


def test_segments_to_mode_mask_basic():
    from spectra.waveforms.multifunction import segments_to_mode_mask

    segs = [
        SignalDescription(
            t_start=0.0,
            t_stop=0.001,
            f_low=0,
            f_high=1,
            label="x",
            snr=0.0,
            mode="a",
        ),
        SignalDescription(
            t_start=0.001,
            t_stop=0.003,
            f_low=0,
            f_high=1,
            label="x",
            snr=0.0,
            mode="b",
        ),
    ]
    mask = segments_to_mode_mask(
        segments=segs,
        total_samples=3000,
        sample_rate=1e6,
        mode_to_index={"a": 0, "b": 1},
        fill_index=-1,
    )
    assert mask.shape == (3000,)
    assert mask[0] == 0
    assert mask[999] == 0
    assert mask[1000] == 1
    assert mask[2999] == 1


def test_segments_to_mode_mask_fill_index_for_uncovered():
    from spectra.waveforms.multifunction import segments_to_mode_mask

    segs = [
        SignalDescription(
            t_start=0.0,
            t_stop=0.001,
            f_low=0,
            f_high=1,
            label="x",
            snr=0.0,
            mode="a",
        ),
    ]
    mask = segments_to_mode_mask(
        segments=segs,
        total_samples=3000,
        sample_rate=1e6,
        mode_to_index={"a": 0},
        fill_index=-1,
    )
    assert mask[0] == 0
    assert mask[999] == 0
    assert mask[1000] == -1
    assert mask[2999] == -1


def test_segments_to_mode_mask_unknown_mode_raises():
    from spectra.waveforms.multifunction import segments_to_mode_mask

    segs = [
        SignalDescription(
            t_start=0.0,
            t_stop=0.001,
            f_low=0,
            f_high=1,
            label="x",
            snr=0.0,
            mode="unknown",
        ),
    ]
    with pytest.raises(KeyError):
        segments_to_mode_mask(
            segments=segs,
            total_samples=1000,
            sample_rate=1e6,
            mode_to_index={"a": 0},
        )


# ── Task 8: top-level re-exports ─────────────────────────────────────────────


def test_public_names_reexported_from_spectra_waveforms():
    import spectra.waveforms as wf

    expected = [
        "ScheduledWaveform",
        "Schedule",
        "StaticSchedule",
        "StochasticSchedule",
        "CognitiveSchedule",
        "SegmentSpec",
        "ModeSpec",
        "segments_to_mode_mask",
    ]
    for name in expected:
        assert hasattr(wf, name), f"spectra.waveforms is missing {name!r}"


# ── Task 9: Example 1 — MFR search/track ────────────────────────────────────


def test_multifunction_search_track_radar_produces_two_modes(assert_valid_iq):
    from spectra.waveforms import multifunction_search_track_radar

    sw = multifunction_search_track_radar()
    iq, segs = sw.generate_with_segments(num_samples=20_000, sample_rate=20e6, seed=0)

    assert_valid_iq(iq, expected_length=20_000)
    modes = {s.mode for s in segs}
    assert "search" in modes
    assert "track" in modes


def test_multifunction_search_track_radar_track_is_louder():
    from spectra.waveforms import multifunction_search_track_radar

    sw = multifunction_search_track_radar()
    _, segs = sw.generate_with_segments(num_samples=20_000, sample_rate=20e6, seed=0)

    tracks = [s for s in segs if s.mode == "track"]
    assert tracks
    assert all(s.modulation_params["power_offset_db"] == pytest.approx(3.0) for s in tracks)


def test_multifunction_search_track_radar_deterministic():
    from spectra.waveforms import multifunction_search_track_radar

    iq1 = multifunction_search_track_radar().generate(
        num_symbols=50_000,
        sample_rate=20e6,
        seed=7,
    )
    iq2 = multifunction_search_track_radar().generate(
        num_symbols=50_000,
        sample_rate=20e6,
        seed=7,
    )
    assert np.array_equal(iq1, iq2)


# ── Task 10: Example 2 — multi-PRF pulse-Doppler ─────────────────────────────


def test_multi_prf_pulse_doppler_produces_all_three_modes(assert_valid_iq):
    from spectra.waveforms import multi_prf_pulse_doppler_radar

    sw = multi_prf_pulse_doppler_radar()
    iq, segs = sw.generate_with_segments(num_samples=100_000, sample_rate=20e6, seed=3)

    assert_valid_iq(iq, expected_length=100_000)
    modes = {s.mode for s in segs}
    assert modes.issubset({"low_prf", "medium_prf", "high_prf"})
    assert len(modes) >= 2


def test_multi_prf_pulse_doppler_deterministic():
    from spectra.waveforms import multi_prf_pulse_doppler_radar

    iq1 = multi_prf_pulse_doppler_radar().generate(
        num_symbols=50_000,
        sample_rate=20e6,
        seed=11,
    )
    iq2 = multi_prf_pulse_doppler_radar().generate(
        num_symbols=50_000,
        sample_rate=20e6,
        seed=11,
    )
    assert np.array_equal(iq1, iq2)


# ── Task 11: Example 3 — frequency-agile stepped-PRI ────────────────────────


def test_frequency_agile_produces_varying_freq_offsets(assert_valid_iq):
    from spectra.waveforms import frequency_agile_stepped_pri_radar

    sw = frequency_agile_stepped_pri_radar()
    iq, segs = sw.generate_with_segments(num_samples=60_000, sample_rate=20e6, seed=5)

    assert_valid_iq(iq, expected_length=60_000)
    assert len(segs) >= 3
    offsets = [s.modulation_params["freq_offset_hz"] for s in segs]
    assert len(set(offsets)) >= 2, "expected frequency agility across dwells"
    assert all(-200e3 <= f <= 200e3 for f in offsets)


def test_frequency_agile_deterministic():
    from spectra.waveforms import frequency_agile_stepped_pri_radar

    iq1 = frequency_agile_stepped_pri_radar().generate(
        num_symbols=30_000,
        sample_rate=20e6,
        seed=12,
    )
    iq2 = frequency_agile_stepped_pri_radar().generate(
        num_symbols=30_000,
        sample_rate=20e6,
        seed=12,
    )
    assert np.array_equal(iq1, iq2)


# ── Task 12: Example 4 — RadCom ──────────────────────────────────────────────


def test_radcom_emitter_produces_both_families(assert_valid_iq):
    from spectra.waveforms import radcom_emitter

    sw = radcom_emitter()
    iq, segs = sw.generate_with_segments(num_samples=60_000, sample_rate=20e6, seed=0)

    assert_valid_iq(iq, expected_length=60_000)
    modes = {s.mode for s in segs}
    assert "radar" in modes
    assert "comms" in modes

    labels = {s.label for s in segs}
    assert "PulsedRadar" in labels
    assert "QPSK" in labels


def test_radcom_emitter_deterministic():
    from spectra.waveforms import radcom_emitter

    iq1 = radcom_emitter().generate(num_symbols=30_000, sample_rate=20e6, seed=9)
    iq2 = radcom_emitter().generate(num_symbols=30_000, sample_rate=20e6, seed=9)
    assert np.array_equal(iq1, iq2)
