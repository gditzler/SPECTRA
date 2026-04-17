# Multi-Function Emitter Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Add a `ScheduledWaveform` abstraction to SPECTRA that lets a single emitter interleave multiple waveform modes over time, with four reference examples (multi-function radar, multi-PRF pulse-Doppler, frequency-agile, radar-comms).

**Architecture:** A new subpackage `python/spectra/waveforms/multifunction/` houses a `Schedule` strategy family (`StaticSchedule`, `StochasticSchedule` with Markov transitions, and a `CognitiveSchedule` ABC stub). `ScheduledWaveform(Waveform)` iterates the schedule, asks each sub-waveform for IQ, optionally applies per-segment `power_offset_db` and `freq_offset_hz` at baseband, and emits a list of `SignalDescription`s (which gain a new optional `mode: str | None` field). `Emitter`, `Environment`, and `Composer` are untouched — a `ScheduledWaveform` satisfies the existing `Waveform` contract.

**Tech Stack:** Python 3.10+, NumPy, existing SPECTRA Rust primitives (unchanged). No new Rust, no new external dependencies.

**Spec:** `docs/superpowers/specs/2026-04-17-multifunction-emitter-design.md`

---

## File Structure

| File | Responsibility |
|------|---------------|
| `python/spectra/scene/signal_desc.py` | Modify: add optional `mode: str \| None` field |
| `python/spectra/waveforms/multifunction/__init__.py` | Package init + re-exports + `segments_to_mode_mask` helper |
| `python/spectra/waveforms/multifunction/segment.py` | `SegmentSpec` dataclass, `ModeSpec` dataclass |
| `python/spectra/waveforms/multifunction/schedule.py` | `Schedule` ABC, `StaticSchedule`, `StochasticSchedule`, `CognitiveSchedule` |
| `python/spectra/waveforms/multifunction/scheduled_waveform.py` | `ScheduledWaveform(Waveform)` orchestrator |
| `python/spectra/waveforms/multifunction/examples.py` | Four factory functions |
| `python/spectra/waveforms/__init__.py` | Modify: re-export new public names |
| `tests/test_multifunction_emitter.py` | All unit + integration tests |
| `docs/user-guide/multifunction-emitters.md` | User-facing docs page |
| `mkdocs.yml` | Modify: add nav entry |
| `README.md` | Modify: mention new subpackage in architecture section |
| `CLAUDE.md` | Modify: mention `waveforms/multifunction/` in architecture section |

---

## Task 1: Add `mode` field to `SignalDescription`

Everything else depends on this field existing. Small and self-contained.

**Files:**
- Modify: `python/spectra/scene/signal_desc.py`
- Test: `tests/test_signal_description_mode.py` (new small file; keeps the change discoverable)

- [ ] **Step 1: Write the failing test**

```python
# tests/test_signal_description_mode.py
"""Tests for the optional `mode` field on SignalDescription."""
from spectra.scene.signal_desc import SignalDescription


def test_mode_defaults_to_none():
    desc = SignalDescription(
        t_start=0.0, t_stop=1e-3, f_low=-5e3, f_high=5e3,
        label="QPSK", snr=10.0,
    )
    assert desc.mode is None


def test_mode_can_be_set():
    desc = SignalDescription(
        t_start=0.0, t_stop=1e-3, f_low=-5e3, f_high=5e3,
        label="PulsedRadar", snr=10.0, mode="track",
    )
    assert desc.mode == "track"


def test_mode_is_optional_positional_not_required():
    # Existing callers that construct SignalDescription positionally with the
    # first six required fields must still work. This is the back-compat check.
    desc = SignalDescription(0.0, 1e-3, -5e3, 5e3, "QPSK", 10.0)
    assert desc.mode is None
    assert desc.label == "QPSK"
```

- [ ] **Step 2: Run test to verify it fails**

Run: `pytest tests/test_signal_description_mode.py -v`
Expected: FAIL (`TypeError: unexpected keyword argument 'mode'` on `test_mode_can_be_set`).

- [ ] **Step 3: Add the field to the dataclass**

Modify `python/spectra/scene/signal_desc.py`:

```python
from dataclasses import dataclass, field
from typing import Dict, Optional


@dataclass
class SignalDescription:
    """Ground-truth record for a single signal in a scene or impairment pipeline.

    Frequencies are in Hz relative to DC (baseband). A signal centered at
    100 kHz with 50 kHz bandwidth has ``f_low=75_000``, ``f_high=125_000``.
    Times are in seconds from the start of the capture.

    Attributes:
        t_start: Start time of the signal in seconds.
        t_stop: End time of the signal in seconds.
        f_low: Lower edge of the signal's occupied bandwidth in Hz.
        f_high: Upper edge of the signal's occupied bandwidth in Hz.
        label: Modulation/waveform class string (e.g., ``"QPSK"``).
        snr: Signal-to-noise ratio in dB at the point this description was created.
        modulation_params: Reserved dict for waveform-specific metadata.
        mode: Optional operating-mode label for emitter-internal timelines
            (e.g., ``"search"``, ``"track"``, ``"comms"``). ``None`` for
            single-mode signals.
    """

    t_start: float
    t_stop: float
    f_low: float
    f_high: float
    label: str
    snr: float
    modulation_params: Dict = field(default_factory=dict)
    mode: Optional[str] = None

    @property
    def f_center(self) -> float:
        return (self.f_low + self.f_high) / 2.0

    @property
    def bandwidth(self) -> float:
        return self.f_high - self.f_low

    @property
    def duration(self) -> float:
        return self.t_stop - self.t_start
```

- [ ] **Step 4: Run test to verify it passes**

Run: `pytest tests/test_signal_description_mode.py -v`
Expected: PASS — all 3 tests.

- [ ] **Step 5: Run the full existing test suite to confirm no regressions**

Run: `pytest -x -q`
Expected: PASS. If any test fails because it uses positional construction that conflicts with the new field, stop and investigate — do not proceed until the new field is truly back-compatible.

- [ ] **Step 6: Commit**

```bash
git add python/spectra/scene/signal_desc.py tests/test_signal_description_mode.py
git commit -m "feat(scene): add optional mode field to SignalDescription"
```

---

## Task 2: `SegmentSpec` and `ModeSpec` dataclasses

Pure data types, no logic. Used by all subsequent tasks.

**Files:**
- Create: `python/spectra/waveforms/multifunction/__init__.py` (empty placeholder for now)
- Create: `python/spectra/waveforms/multifunction/segment.py`
- Test: `tests/test_multifunction_emitter.py` (new)

- [ ] **Step 1: Write the failing tests**

```python
# tests/test_multifunction_emitter.py
"""Tests for spectra.waveforms.multifunction — the multi-function emitter."""
import numpy as np
import pytest

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

    factory = lambda rng: PulsedRadar()
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
```

- [ ] **Step 2: Run test to verify it fails**

Run: `pytest tests/test_multifunction_emitter.py -v -k segment_spec or test_mode_spec`
Expected: FAIL — `ModuleNotFoundError: No module named 'spectra.waveforms.multifunction'`.

- [ ] **Step 3: Create the subpackage scaffold**

Create `python/spectra/waveforms/multifunction/__init__.py` with a single line:

```python
"""Multi-function emitter subpackage — schedule-driven multi-mode waveforms."""
```

Create `python/spectra/waveforms/multifunction/segment.py`:

```python
"""Dataclasses for describing segments of a multi-function emitter's timeline."""
from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any, Callable, Union

import numpy as np

from spectra.waveforms.base import Waveform


@dataclass
class SegmentSpec:
    """One segment of a scheduled emitter timeline.

    Used directly by :class:`StaticSchedule` and produced dynamically by
    :class:`StochasticSchedule` and :class:`CognitiveSchedule`.

    Attributes:
        waveform: The sub-waveform to play during this segment.
        duration_samples: Exact duration of the segment in samples.
        mode: User-facing label for this segment's mode (e.g., ``"search"``).
        power_offset_db: Baseband amplitude delta applied as
            ``iq * 10 ** (power_offset_db / 20)``. Default ``0.0``.
        freq_offset_hz: Baseband frequency shift (relative to the emitter's
            carrier) applied to the segment IQ. Default ``0.0``.
        gain_offset_db: Recorded in the segment's SignalDescription metadata;
            **not** applied to IQ (handled downstream if needed). Default ``0.0``.
        metadata: Free-form dict copied into each segment's SignalDescription.
    """

    waveform: Waveform
    duration_samples: int
    mode: str
    power_offset_db: float = 0.0
    freq_offset_hz: float = 0.0
    gain_offset_db: float = 0.0
    metadata: dict = field(default_factory=dict)


@dataclass
class ModeSpec:
    """Describes one Markov mode in a :class:`StochasticSchedule`.

    Attributes:
        waveform_factory: Callable ``(rng) -> Waveform`` producing a fresh
            sub-waveform per dwell. ``rng`` is a ``numpy.random.Generator``
            owned by the schedule and advanced across dwells.
        duration_samples: Either a fixed int, a ``(min, max)`` tuple
            (uniform-int range, inclusive), or a callable ``(rng) -> int``.
        power_offset_db: Scalar or ``(min, max)`` tuple (uniform float range).
        freq_offset_hz: Scalar or ``(min, max)`` tuple (uniform float range).
        metadata: Free-form dict copied into each resulting SegmentSpec.
    """

    waveform_factory: Callable[[np.random.Generator], Waveform]
    duration_samples: Union[int, tuple[int, int], Callable[[np.random.Generator], int]]
    power_offset_db: Union[float, tuple[float, float]] = 0.0
    freq_offset_hz: Union[float, tuple[float, float]] = 0.0
    metadata: dict = field(default_factory=dict)
```

- [ ] **Step 4: Run test to verify it passes**

Run: `pytest tests/test_multifunction_emitter.py -v -k segment_spec or mode_spec`
Expected: PASS — 3 tests.

- [ ] **Step 5: Commit**

```bash
git add python/spectra/waveforms/multifunction/__init__.py \
        python/spectra/waveforms/multifunction/segment.py \
        tests/test_multifunction_emitter.py
git commit -m "feat(waveforms): add SegmentSpec and ModeSpec dataclasses"
```

---

## Task 3: `Schedule` ABC + `StaticSchedule`

The simplest schedule type. Validates the `iter_segments` contract.

**Files:**
- Create: `python/spectra/waveforms/multifunction/schedule.py`
- Modify: `tests/test_multifunction_emitter.py`

- [ ] **Step 1: Write the failing tests (append to `tests/test_multifunction_emitter.py`)**

```python
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
    # Each iteration adds 200 samples. We need ≥ 450 samples covered.
    # First pass: 200 samples (a, b). Second pass: 400 (a, b, a, b). Third iter
    # starts at 400, yields a (500 >= 450, done). Expect 5 segments: a,b,a,b,a.
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
```

- [ ] **Step 2: Run tests to verify they fail**

Run: `pytest tests/test_multifunction_emitter.py -v -k schedule`
Expected: FAIL — `ImportError: cannot import name 'Schedule'` or similar.

- [ ] **Step 3: Write `schedule.py` with the ABC and `StaticSchedule`**

Create `python/spectra/waveforms/multifunction/schedule.py`:

```python
"""Schedule strategies for multi-function emitters."""
from __future__ import annotations

from abc import ABC, abstractmethod
from typing import Iterator

from spectra.waveforms.multifunction.segment import SegmentSpec


class Schedule(ABC):
    """Abstract base for anything that yields a stream of :class:`SegmentSpec`.

    Subclasses implement ``iter_segments(total_samples, sample_rate, seed)``
    which yields ``SegmentSpec`` instances until the caller has accumulated
    enough samples. The schedule does not need to track cumulative samples
    itself — the caller (typically :class:`ScheduledWaveform`) stops iterating
    once cumulative duration reaches ``total_samples``.
    """

    @abstractmethod
    def iter_segments(
        self,
        total_samples: int,
        sample_rate: float,
        seed: int,
    ) -> Iterator[SegmentSpec]:
        ...


class StaticSchedule(Schedule):
    """Deterministic timeline of segments, optionally looped.

    Args:
        segments: Ordered list of :class:`SegmentSpec` instances.
        loop: If ``True``, wraps to the start when the list is exhausted. If
            ``False``, stops when the list is exhausted (the caller will
            zero-pad any remaining samples). Default ``True``.
    """

    def __init__(self, segments: list[SegmentSpec], loop: bool = True):
        self._segments = list(segments)
        self._loop = loop

    def iter_segments(self, total_samples, sample_rate, seed):
        if not self._segments:
            return
        cumulative = 0
        idx = 0
        while cumulative < total_samples:
            seg = self._segments[idx]
            yield seg
            cumulative += seg.duration_samples
            idx += 1
            if idx >= len(self._segments):
                if self._loop:
                    idx = 0
                else:
                    return
```

- [ ] **Step 4: Run tests to verify they pass**

Run: `pytest tests/test_multifunction_emitter.py -v -k schedule`
Expected: PASS — 5 tests.

- [ ] **Step 5: Commit**

```bash
git add python/spectra/waveforms/multifunction/schedule.py tests/test_multifunction_emitter.py
git commit -m "feat(waveforms): add Schedule ABC and StaticSchedule"
```

---

## Task 4: `ScheduledWaveform` core orchestration

The heart of the design. Build and test with `StaticSchedule` (already implemented).

**Files:**
- Create: `python/spectra/waveforms/multifunction/scheduled_waveform.py`
- Modify: `tests/test_multifunction_emitter.py`

- [ ] **Step 1: Write the failing tests (append to `tests/test_multifunction_emitter.py`)**

```python
# ── Task 4: ScheduledWaveform orchestration ─────────────────────────────────

from spectra.scene.signal_desc import SignalDescription

FS = 1e6


def _small_radar(pri=128, num_pulses=4, pw=16):
    return PulsedRadar(
        pulse_width_samples=pw, pri_samples=pri, num_pulses=num_pulses,
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
    iq_b, segments = sw.generate_with_segments(
        num_samples=1024, sample_rate=FS, seed=7
    )
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
        waveform=wf, duration_samples=2048, mode="boost", power_offset_db=6.0,
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
    """A segment with freq_offset_hz=F shows a spectral peak near F."""
    from spectra.waveforms.multifunction.schedule import StaticSchedule
    from spectra.waveforms.multifunction.scheduled_waveform import ScheduledWaveform
    from spectra.waveforms.multifunction.segment import SegmentSpec

    wf = QPSK(samples_per_symbol=8)
    offset = 100e3  # 100 kHz at fs=1 MHz
    seg = SegmentSpec(waveform=wf, duration_samples=8192, mode="m", freq_offset_hz=offset)
    sw = ScheduledWaveform(StaticSchedule([seg], loop=False))

    iq = sw.generate(num_symbols=8192, sample_rate=FS, seed=0)
    spec = np.abs(np.fft.fftshift(np.fft.fft(iq)))
    freqs = np.fft.fftshift(np.fft.fftfreq(len(iq), d=1.0 / FS))
    peak_freq = freqs[np.argmax(spec)]
    assert abs(peak_freq - offset) < 10e3, f"expected peak near {offset}, got {peak_freq}"


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
        StaticSchedule([seg], loop=True), default_bandwidth_hz=1e6,
    )
    assert sw.bandwidth(sample_rate=FS) == 1e6


def test_scheduled_waveform_bandwidth_heuristic():
    """Without default, bandwidth() reflects the widest child + its offset."""
    from spectra.waveforms.multifunction.schedule import StaticSchedule
    from spectra.waveforms.multifunction.scheduled_waveform import ScheduledWaveform
    from spectra.waveforms.multifunction.segment import SegmentSpec

    narrow = SegmentSpec(
        waveform=_small_radar(pw=32), duration_samples=256, mode="n",
    )
    wide = SegmentSpec(
        waveform=_small_radar(pw=4), duration_samples=256, mode="w",
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
```

- [ ] **Step 2: Run tests to verify they fail**

Run: `pytest tests/test_multifunction_emitter.py -v -k scheduled_waveform`
Expected: FAIL — `ImportError: cannot import name 'ScheduledWaveform'`.

- [ ] **Step 3: Implement `scheduled_waveform.py`**

Create `python/spectra/waveforms/multifunction/scheduled_waveform.py`:

```python
"""ScheduledWaveform: a Waveform that plays a schedule of sub-waveforms."""
from __future__ import annotations

import math
from typing import Optional

import numpy as np

from spectra.scene.signal_desc import SignalDescription
from spectra.utils.dsp import frequency_shift
from spectra.waveforms.base import Waveform
from spectra.waveforms.multifunction.schedule import Schedule

_PREVIEW_DURATION_SEC = 0.1
_FALLBACK_BANDWIDTH_FRACTION = 1.0 / 16.0  # used when preview yields nothing


class ScheduledWaveform(Waveform):
    """A Waveform that iterates a :class:`Schedule` and concatenates child IQ.

    Per-segment ``power_offset_db`` and ``freq_offset_hz`` are applied to the
    child IQ at baseband before concatenation. The emitter carrier, antenna
    gain, and receiver-side link budget are handled downstream by
    :class:`spectra.environment.Emitter` and :class:`spectra.environment.Environment`
    exactly as for any other waveform.

    Args:
        schedule: The schedule driving segment emission.
        label: User-facing classification label (returned by ``label``).
            Default ``"multifunction"``.
        default_bandwidth_hz: If set, overrides the automatic bandwidth
            heuristic. Recommended when the heuristic underestimates (e.g.,
            rare wide-band modes in a stochastic schedule).
    """

    samples_per_symbol: int = 1  # so num_symbols == num_samples in callers

    def __init__(
        self,
        schedule: Schedule,
        label: str = "multifunction",
        default_bandwidth_hz: Optional[float] = None,
    ):
        self._schedule = schedule
        self._label = label
        self._default_bandwidth_hz = default_bandwidth_hz

    @property
    def label(self) -> str:
        return self._label

    def generate(
        self,
        num_symbols: int,
        sample_rate: float,
        seed: Optional[int] = None,
    ) -> np.ndarray:
        iq, _ = self.generate_with_segments(num_symbols, sample_rate, seed)
        return iq

    def generate_with_segments(
        self,
        num_samples: int,
        sample_rate: float,
        seed: Optional[int] = None,
    ) -> tuple[np.ndarray, list[SignalDescription]]:
        """Generate IQ plus a list of per-segment :class:`SignalDescription`.

        Args:
            num_samples: Total output sample count.
            sample_rate: Sample rate in Hz.
            seed: Master seed. Same seed → byte-identical output.

        Returns:
            ``(iq, segments)`` where ``iq`` is a complex64 array of exactly
            ``num_samples`` samples, and ``segments`` is a list of
            :class:`SignalDescription` entries, one per emitted segment.
        """
        s = seed if seed is not None else int(np.random.randint(0, 2**32))
        rng = np.random.default_rng(s)
        seed_for_schedule = int(rng.integers(0, 2**32))

        out = np.zeros(num_samples, dtype=np.complex64)
        segments: list[SignalDescription] = []
        cursor = 0

        for spec in self._schedule.iter_segments(
            total_samples=num_samples,
            sample_rate=sample_rate,
            seed=seed_for_schedule,
        ):
            if cursor >= num_samples:
                break
            remaining = num_samples - cursor
            dur = min(int(spec.duration_samples), remaining)
            if dur <= 0:
                continue

            child_sps = max(1, int(getattr(spec.waveform, "samples_per_symbol", 1)))
            child_num_symbols = max(1, math.ceil(dur / child_sps))
            child_seed = int(rng.integers(0, 2**32))
            child_iq = spec.waveform.generate(
                num_symbols=child_num_symbols,
                sample_rate=sample_rate,
                seed=child_seed,
            )
            child_iq = np.asarray(child_iq, dtype=np.complex64)

            # Truncate or zero-pad to exactly `dur` samples.
            if len(child_iq) >= dur:
                child_iq = child_iq[:dur]
            else:
                pad = np.zeros(dur - len(child_iq), dtype=np.complex64)
                child_iq = np.concatenate([child_iq, pad])

            # Apply per-segment power offset (amplitude).
            if spec.power_offset_db != 0.0:
                child_iq = (child_iq * (10.0 ** (spec.power_offset_db / 20.0))).astype(
                    np.complex64
                )

            # Apply per-segment frequency offset at baseband.
            if spec.freq_offset_hz != 0.0:
                child_iq = frequency_shift(child_iq, spec.freq_offset_hz, sample_rate)

            out[cursor : cursor + dur] = child_iq

            child_bw = float(spec.waveform.bandwidth(sample_rate))
            segments.append(
                SignalDescription(
                    t_start=cursor / sample_rate,
                    t_stop=(cursor + dur) / sample_rate,
                    f_low=spec.freq_offset_hz - child_bw / 2.0,
                    f_high=spec.freq_offset_hz + child_bw / 2.0,
                    label=spec.waveform.label,
                    snr=0.0,
                    modulation_params={
                        **dict(spec.metadata),
                        "gain_offset_db": float(spec.gain_offset_db),
                        "power_offset_db": float(spec.power_offset_db),
                        "freq_offset_hz": float(spec.freq_offset_hz),
                    },
                    mode=spec.mode,
                )
            )
            cursor += dur

        return out, segments

    def bandwidth(self, sample_rate: float) -> float:
        """Return the multi-function emitter's aggregate bandwidth envelope.

        If ``default_bandwidth_hz`` was provided to the constructor, that is
        returned. Otherwise a preview run of the schedule over
        ~``_PREVIEW_DURATION_SEC`` seconds is inspected and the maximum of
        ``child.bandwidth(sample_rate) + 2*|freq_offset_hz|`` is returned.
        """
        if self._default_bandwidth_hz is not None:
            return float(self._default_bandwidth_hz)

        total_samples = int(sample_rate * _PREVIEW_DURATION_SEC)
        max_bw = 0.0
        for spec in self._schedule.iter_segments(
            total_samples=total_samples, sample_rate=sample_rate, seed=0
        ):
            child_bw = float(spec.waveform.bandwidth(sample_rate))
            envelope = child_bw + 2.0 * abs(float(spec.freq_offset_hz))
            if envelope > max_bw:
                max_bw = envelope
        if max_bw == 0.0:
            return float(sample_rate * _FALLBACK_BANDWIDTH_FRACTION)
        return max_bw
```

- [ ] **Step 4: Run tests to verify they pass**

Run: `pytest tests/test_multifunction_emitter.py -v -k scheduled_waveform`
Expected: PASS — 10 tests.

- [ ] **Step 5: Commit**

```bash
git add python/spectra/waveforms/multifunction/scheduled_waveform.py tests/test_multifunction_emitter.py
git commit -m "feat(waveforms): add ScheduledWaveform orchestrator"
```

---

## Task 5: `StochasticSchedule` (Markov)

Seeded Markov model with per-mode duration/parameter distributions.

**Files:**
- Modify: `python/spectra/waveforms/multifunction/schedule.py`
- Modify: `tests/test_multifunction_emitter.py`

- [ ] **Step 1: Write the failing tests (append to `tests/test_multifunction_emitter.py`)**

```python
# ── Task 5: StochasticSchedule ──────────────────────────────────────────────

def test_stochastic_schedule_validates_unknown_mode():
    from spectra.waveforms.multifunction.schedule import StochasticSchedule
    from spectra.waveforms.multifunction.segment import ModeSpec

    modes = {"a": ModeSpec(waveform_factory=lambda r: _small_radar(), duration_samples=100)}
    with pytest.raises(ValueError, match="unknown"):
        StochasticSchedule(
            modes=modes,
            transitions={"a": {"a": 0.5, "b": 0.5}},  # 'b' not in modes
            initial_mode="a",
        )


def test_stochastic_schedule_validates_row_sum():
    from spectra.waveforms.multifunction.schedule import StochasticSchedule
    from spectra.waveforms.multifunction.segment import ModeSpec

    modes = {
        "a": ModeSpec(waveform_factory=lambda r: _small_radar(), duration_samples=100),
        "b": ModeSpec(waveform_factory=lambda r: _small_radar(), duration_samples=100),
    }
    with pytest.raises(ValueError, match="sum"):
        StochasticSchedule(
            modes=modes,
            transitions={"a": {"a": 0.5, "b": 0.2}, "b": {"a": 0.5, "b": 0.5}},
            initial_mode="a",
        )


def test_stochastic_schedule_validates_initial_mode_unknown():
    from spectra.waveforms.multifunction.schedule import StochasticSchedule
    from spectra.waveforms.multifunction.segment import ModeSpec

    modes = {"a": ModeSpec(waveform_factory=lambda r: _small_radar(), duration_samples=100)}
    with pytest.raises(ValueError, match="initial_mode"):
        StochasticSchedule(
            modes=modes, transitions={"a": {"a": 1.0}}, initial_mode="b",
        )


def test_stochastic_schedule_determinism():
    """Same seed → identical segment list."""
    from spectra.waveforms.multifunction.schedule import StochasticSchedule
    from spectra.waveforms.multifunction.segment import ModeSpec

    modes = {
        "a": ModeSpec(waveform_factory=lambda r: _small_radar(pri=128), duration_samples=(100, 400)),
        "b": ModeSpec(waveform_factory=lambda r: _small_radar(pri=256), duration_samples=(100, 400)),
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

    modes = {
        "a": ModeSpec(waveform_factory=lambda r: _small_radar(), duration_samples=(100, 400)),
        "b": ModeSpec(waveform_factory=lambda r: _small_radar(pri=256), duration_samples=(100, 400)),
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

    modes = {
        "fixed": ModeSpec(
            waveform_factory=lambda r: _small_radar(),
            duration_samples=lambda r: 250,
        ),
    }
    sched = StochasticSchedule(
        modes=modes, transitions={"fixed": {"fixed": 1.0}}, initial_mode="fixed",
    )
    out = [s for s in sched.iter_segments(1000, 1e6, seed=0)]
    assert all(s.duration_samples == 250 for s in out)


def test_stochastic_schedule_freq_and_power_offsets_drawn_from_range():
    from spectra.waveforms.multifunction.schedule import StochasticSchedule
    from spectra.waveforms.multifunction.segment import ModeSpec

    modes = {
        "m": ModeSpec(
            waveform_factory=lambda r: _small_radar(),
            duration_samples=200,
            power_offset_db=(-2.0, 2.0),
            freq_offset_hz=(-50e3, 50e3),
        ),
    }
    sched = StochasticSchedule(
        modes=modes, transitions={"m": {"m": 1.0}}, initial_mode="m",
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

    modes = {
        "a": ModeSpec(waveform_factory=lambda r: _small_radar(pri=128), duration_samples=(200, 400)),
        "b": ModeSpec(waveform_factory=lambda r: _small_radar(pri=256), duration_samples=(200, 400)),
    }
    sw = ScheduledWaveform(StochasticSchedule(
        modes=modes,
        transitions={"a": {"a": 0.3, "b": 0.7}, "b": {"a": 0.7, "b": 0.3}},
        initial_mode="a",
    ))

    iq1 = sw.generate(num_symbols=5000, sample_rate=1e6, seed=42)
    iq2 = sw.generate(num_symbols=5000, sample_rate=1e6, seed=42)
    assert_valid_iq(iq1, expected_length=5000)
    assert np.array_equal(iq1, iq2)
```

- [ ] **Step 2: Run tests to verify they fail**

Run: `pytest tests/test_multifunction_emitter.py -v -k stochastic`
Expected: FAIL — `ImportError: cannot import name 'StochasticSchedule'`.

- [ ] **Step 3: Add `StochasticSchedule` to `schedule.py`**

Append to `python/spectra/waveforms/multifunction/schedule.py`:

```python
# At top of file, add:
from typing import Callable, Union

import numpy as np

from spectra.waveforms.multifunction.segment import ModeSpec


_ROW_SUM_TOL = 1e-6


def _draw_int_range(
    val: Union[int, tuple[int, int], Callable[[np.random.Generator], int]],
    rng: np.random.Generator,
) -> int:
    """Resolve an int-or-range-or-callable into a concrete int."""
    if callable(val):
        return int(val(rng))
    if isinstance(val, tuple):
        lo, hi = val
        return int(rng.integers(int(lo), int(hi) + 1))
    return int(val)


def _draw_float_range(
    val: Union[float, tuple[float, float]], rng: np.random.Generator
) -> float:
    """Resolve a float-or-range into a concrete float."""
    if isinstance(val, tuple):
        lo, hi = val
        return float(rng.uniform(float(lo), float(hi)))
    return float(val)


class StochasticSchedule(Schedule):
    """Markov-modelled schedule with per-mode parameter distributions.

    Args:
        modes: Mapping of mode name to :class:`ModeSpec`.
        transitions: Markov transition matrix as ``{from_mode: {to_mode: prob}}``.
            Every row must sum to 1.0 (± 1e-6) and reference only known modes.
        initial_mode: Either a single mode name (fixed initial state) or a
            distribution ``{mode_name: prob}`` (drawn at ``iter_segments`` start).

    Raises:
        ValueError: If any mode name is unknown, any row doesn't sum to 1,
            or ``initial_mode`` references an unknown mode.
    """

    def __init__(
        self,
        modes: dict[str, ModeSpec],
        transitions: dict[str, dict[str, float]],
        initial_mode: Union[str, dict[str, float]],
    ):
        self._modes = dict(modes)
        self._transitions = {k: dict(v) for k, v in transitions.items()}
        self._initial_mode = (
            initial_mode if isinstance(initial_mode, str) else dict(initial_mode)
        )
        self._validate()

    def _validate(self):
        names = set(self._modes)
        for from_mode, row in self._transitions.items():
            if from_mode not in names:
                raise ValueError(f"transitions from unknown mode {from_mode!r}")
            for to_mode in row:
                if to_mode not in names:
                    raise ValueError(f"transitions to unknown mode {to_mode!r}")
            total = sum(row.values())
            if abs(total - 1.0) > _ROW_SUM_TOL:
                raise ValueError(
                    f"transitions[{from_mode!r}] row sum is {total}, expected 1.0"
                )
        if isinstance(self._initial_mode, str):
            if self._initial_mode not in names:
                raise ValueError(f"initial_mode {self._initial_mode!r} is unknown")
        else:
            for name in self._initial_mode:
                if name not in names:
                    raise ValueError(f"initial_mode {name!r} is unknown")
            total = sum(self._initial_mode.values())
            if abs(total - 1.0) > _ROW_SUM_TOL:
                raise ValueError(f"initial_mode distribution sum is {total}, expected 1.0")
        # Every mode must have a transition row so the chain never stalls.
        for name in names:
            if name not in self._transitions:
                raise ValueError(f"mode {name!r} has no transition row")

    def _draw_mode(
        self, distribution: dict[str, float], rng: np.random.Generator
    ) -> str:
        names = list(distribution.keys())
        probs = np.array([distribution[n] for n in names], dtype=np.float64)
        probs = probs / probs.sum()  # renormalize for float slop
        idx = int(rng.choice(len(names), p=probs))
        return names[idx]

    def iter_segments(self, total_samples, sample_rate, seed):
        rng = np.random.default_rng(int(seed))

        if isinstance(self._initial_mode, str):
            current = self._initial_mode
        else:
            current = self._draw_mode(self._initial_mode, rng)

        cumulative = 0
        while cumulative < total_samples:
            mode_spec = self._modes[current]
            dur = _draw_int_range(mode_spec.duration_samples, rng)
            power = _draw_float_range(mode_spec.power_offset_db, rng)
            freq = _draw_float_range(mode_spec.freq_offset_hz, rng)
            waveform = mode_spec.waveform_factory(rng)

            yield SegmentSpec(
                waveform=waveform,
                duration_samples=dur,
                mode=current,
                power_offset_db=power,
                freq_offset_hz=freq,
                metadata=dict(mode_spec.metadata),
            )
            cumulative += dur
            current = self._draw_mode(self._transitions[current], rng)
```

Also add this import near the top of `schedule.py`:

```python
from spectra.waveforms.multifunction.segment import SegmentSpec
```

(The file already imports `SegmentSpec` via the `Schedule` signature — confirm it's present; if not, add it.)

- [ ] **Step 4: Run tests to verify they pass**

Run: `pytest tests/test_multifunction_emitter.py -v -k stochastic`
Expected: PASS — 8 tests.

- [ ] **Step 5: Commit**

```bash
git add python/spectra/waveforms/multifunction/schedule.py tests/test_multifunction_emitter.py
git commit -m "feat(waveforms): add StochasticSchedule with Markov transitions"
```

---

## Task 6: `CognitiveSchedule` ABC stub

Defines the closed-loop seam; ships with no concrete subclass. A trivial test
subclass validates the interface.

**Files:**
- Modify: `python/spectra/waveforms/multifunction/schedule.py`
- Modify: `tests/test_multifunction_emitter.py`

- [ ] **Step 1: Write the failing tests (append)**

```python
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
                waveform=_small_radar(), duration_samples=100, mode=mode,
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
                waveform=_small_radar(), duration_samples=100, mode="m",
            )

    sched = FixedLength(3)
    out = list(sched.iter_segments(10_000, 1e6, seed=0))
    assert len(out) == 3
```

- [ ] **Step 2: Run tests to verify they fail**

Run: `pytest tests/test_multifunction_emitter.py -v -k cognitive`
Expected: FAIL — `cannot import name 'CognitiveSchedule'`.

- [ ] **Step 3: Add `CognitiveSchedule` to `schedule.py`**

Append to `python/spectra/waveforms/multifunction/schedule.py`:

```python
from typing import Any, Optional


class CognitiveSchedule(Schedule):
    """Abstract base for closed-loop / event-driven schedules.

    Subclasses implement :meth:`next_segment`, which is called repeatedly with
    the history of emitted segments and an opaque ``state`` object. Returning
    ``None`` ends the schedule (the caller zero-pads any remaining samples).

    Actual wiring to scene/target state (the reason to have this ABC at all)
    is deferred to a future spec. This version ships with no concrete
    subclass; the abstract contract lets us validate the seam in tests and
    commit to the interface shape.
    """

    @abstractmethod
    def next_segment(
        self,
        history: list[SegmentSpec],
        state: Any,
    ) -> Optional[SegmentSpec]:
        """Return the next segment, or ``None`` to end the schedule."""
        ...

    def iter_segments(self, total_samples, sample_rate, seed):
        history: list[SegmentSpec] = []
        state: Any = None  # subclasses may use this; unused in v1
        cumulative = 0
        while cumulative < total_samples:
            nxt = self.next_segment(history, state)
            if nxt is None:
                return
            yield nxt
            history.append(nxt)
            cumulative += int(nxt.duration_samples)
```

- [ ] **Step 4: Run tests to verify they pass**

Run: `pytest tests/test_multifunction_emitter.py -v -k cognitive`
Expected: PASS — 3 tests.

- [ ] **Step 5: Commit**

```bash
git add python/spectra/waveforms/multifunction/schedule.py tests/test_multifunction_emitter.py
git commit -m "feat(waveforms): add CognitiveSchedule ABC (closed-loop stub)"
```

---

## Task 7: `segments_to_mode_mask` helper

Per-sample mode-index array for segmentation tasks.

**Files:**
- Modify: `python/spectra/waveforms/multifunction/__init__.py`
- Modify: `tests/test_multifunction_emitter.py`

- [ ] **Step 1: Write the failing tests (append)**

```python
# ── Task 7: segments_to_mode_mask helper ─────────────────────────────────────

def test_segments_to_mode_mask_basic():
    from spectra.waveforms.multifunction import segments_to_mode_mask

    segs = [
        SignalDescription(t_start=0.0, t_stop=0.001, f_low=0, f_high=1,
                          label="x", snr=0.0, mode="a"),
        SignalDescription(t_start=0.001, t_stop=0.003, f_low=0, f_high=1,
                          label="x", snr=0.0, mode="b"),
    ]
    mask = segments_to_mode_mask(
        segments=segs, total_samples=3000, sample_rate=1e6,
        mode_to_index={"a": 0, "b": 1}, fill_index=-1,
    )
    assert mask.shape == (3000,)
    assert mask[0] == 0
    assert mask[999] == 0
    assert mask[1000] == 1
    assert mask[2999] == 1


def test_segments_to_mode_mask_fill_index_for_uncovered():
    from spectra.waveforms.multifunction import segments_to_mode_mask

    segs = [
        SignalDescription(t_start=0.0, t_stop=0.001, f_low=0, f_high=1,
                          label="x", snr=0.0, mode="a"),
    ]
    mask = segments_to_mode_mask(
        segments=segs, total_samples=3000, sample_rate=1e6,
        mode_to_index={"a": 0}, fill_index=-1,
    )
    assert mask[0] == 0
    assert mask[999] == 0
    assert mask[1000] == -1
    assert mask[2999] == -1


def test_segments_to_mode_mask_unknown_mode_raises():
    from spectra.waveforms.multifunction import segments_to_mode_mask

    segs = [
        SignalDescription(t_start=0.0, t_stop=0.001, f_low=0, f_high=1,
                          label="x", snr=0.0, mode="unknown"),
    ]
    with pytest.raises(KeyError):
        segments_to_mode_mask(
            segments=segs, total_samples=1000, sample_rate=1e6,
            mode_to_index={"a": 0},
        )
```

- [ ] **Step 2: Run tests to verify they fail**

Run: `pytest tests/test_multifunction_emitter.py -v -k segments_to_mode_mask`
Expected: FAIL — `ImportError: cannot import name 'segments_to_mode_mask'`.

- [ ] **Step 3: Implement the helper and finalize package exports**

Replace `python/spectra/waveforms/multifunction/__init__.py` contents with:

```python
"""Multi-function emitter subpackage — schedule-driven multi-mode waveforms."""
from __future__ import annotations

import numpy as np

from spectra.scene.signal_desc import SignalDescription
from spectra.waveforms.multifunction.schedule import (
    CognitiveSchedule,
    Schedule,
    StaticSchedule,
    StochasticSchedule,
)
from spectra.waveforms.multifunction.scheduled_waveform import ScheduledWaveform
from spectra.waveforms.multifunction.segment import ModeSpec, SegmentSpec


def segments_to_mode_mask(
    segments: list[SignalDescription],
    total_samples: int,
    sample_rate: float,
    mode_to_index: dict[str, int],
    fill_index: int = -1,
) -> np.ndarray:
    """Return a per-sample int array with the active mode index per sample.

    Samples not covered by any segment receive ``fill_index``.

    Args:
        segments: Per-segment SignalDescription list (from
            :meth:`ScheduledWaveform.generate_with_segments`).
        total_samples: Length of the output mask array.
        sample_rate: Sample rate in Hz (to convert segment times to indices).
        mode_to_index: Mapping of mode string to int index.
        fill_index: Value for samples not covered by any segment.

    Returns:
        int64 array of shape ``(total_samples,)``.

    Raises:
        KeyError: If a segment's ``mode`` is not in ``mode_to_index``.
    """
    mask = np.full(total_samples, fill_index, dtype=np.int64)
    for seg in segments:
        if seg.mode is None:
            continue
        idx = mode_to_index[seg.mode]  # KeyError if unknown — intentional
        start = max(0, int(round(seg.t_start * sample_rate)))
        stop = min(total_samples, int(round(seg.t_stop * sample_rate)))
        if stop > start:
            mask[start:stop] = idx
    return mask


__all__ = [
    "CognitiveSchedule",
    "ModeSpec",
    "ScheduledWaveform",
    "Schedule",
    "SegmentSpec",
    "StaticSchedule",
    "StochasticSchedule",
    "segments_to_mode_mask",
]
```

- [ ] **Step 4: Run tests to verify they pass**

Run: `pytest tests/test_multifunction_emitter.py -v -k segments_to_mode_mask`
Expected: PASS — 3 tests.

- [ ] **Step 5: Commit**

```bash
git add python/spectra/waveforms/multifunction/__init__.py tests/test_multifunction_emitter.py
git commit -m "feat(waveforms): add segments_to_mode_mask helper and package exports"
```

---

## Task 8: Re-export from `spectra.waveforms`

Top-level convenience so users write `from spectra.waveforms import ScheduledWaveform`.

**Files:**
- Modify: `python/spectra/waveforms/__init__.py`
- Modify: `tests/test_multifunction_emitter.py`

- [ ] **Step 1: Write the failing test (append)**

```python
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
```

- [ ] **Step 2: Run test to verify it fails**

Run: `pytest tests/test_multifunction_emitter.py -v -k public_names_reexported`
Expected: FAIL — missing attributes.

- [ ] **Step 3: Edit `python/spectra/waveforms/__init__.py`**

Add at the bottom of the file, after the existing imports (before `__all__`):

```python
from spectra.waveforms.multifunction import (
    CognitiveSchedule,
    ModeSpec,
    ScheduledWaveform,
    Schedule,
    SegmentSpec,
    StaticSchedule,
    StochasticSchedule,
    segments_to_mode_mask,
)
```

Then extend the existing `__all__` list to include these names. The `__all__` list is sorted; add these entries in alphabetical order within the list. The final `__all__` must include exactly these additions (no duplicates):

```python
"CognitiveSchedule",
"ModeSpec",
"ScheduledWaveform",
"Schedule",
"SegmentSpec",
"StaticSchedule",
"StochasticSchedule",
"segments_to_mode_mask",
```

- [ ] **Step 4: Run test to verify it passes**

Run: `pytest tests/test_multifunction_emitter.py -v -k public_names_reexported`
Expected: PASS.

- [ ] **Step 5: Run full existing test suite**

Run: `pytest -x -q`
Expected: PASS (we haven't broken anything yet).

- [ ] **Step 6: Commit**

```bash
git add python/spectra/waveforms/__init__.py tests/test_multifunction_emitter.py
git commit -m "feat(waveforms): re-export multi-function emitter public API"
```

---

## Task 9: Example 1 — `multifunction_search_track_radar` (Static)

Alternating search/track MFR. Exercises static scheduling with per-segment
power and frequency offsets.

**Files:**
- Create: `python/spectra/waveforms/multifunction/examples.py`
- Modify: `python/spectra/waveforms/multifunction/__init__.py` (export the factory)
- Modify: `python/spectra/waveforms/__init__.py` (top-level re-export)
- Modify: `tests/test_multifunction_emitter.py`

- [ ] **Step 1: Write the failing tests (append)**

```python
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
    """The track dwell has power_offset_db=+3, so its modulation_params shows it."""
    from spectra.waveforms import multifunction_search_track_radar

    sw = multifunction_search_track_radar()
    _, segs = sw.generate_with_segments(num_samples=20_000, sample_rate=20e6, seed=0)

    tracks = [s for s in segs if s.mode == "track"]
    assert tracks
    assert all(s.modulation_params["power_offset_db"] == pytest.approx(3.0) for s in tracks)


def test_multifunction_search_track_radar_deterministic():
    from spectra.waveforms import multifunction_search_track_radar

    iq1 = multifunction_search_track_radar().generate(
        num_symbols=50_000, sample_rate=20e6, seed=7,
    )
    iq2 = multifunction_search_track_radar().generate(
        num_symbols=50_000, sample_rate=20e6, seed=7,
    )
    assert np.array_equal(iq1, iq2)
```

- [ ] **Step 2: Run tests to verify they fail**

Run: `pytest tests/test_multifunction_emitter.py -v -k search_track`
Expected: FAIL — `cannot import name 'multifunction_search_track_radar'`.

- [ ] **Step 3: Create `examples.py` with the first factory**

Create `python/spectra/waveforms/multifunction/examples.py`:

```python
"""Reference multi-function emitter factory functions."""
from __future__ import annotations

from spectra.waveforms.multifunction.schedule import StaticSchedule
from spectra.waveforms.multifunction.scheduled_waveform import ScheduledWaveform
from spectra.waveforms.multifunction.segment import SegmentSpec
from spectra.waveforms.radar import PulsedRadar


def multifunction_search_track_radar() -> ScheduledWaveform:
    """Canonical multi-function radar alternating search and track dwells.

    Search dwell: long CPI (16 pulses of 2048-sample PRI, 256-sample Hamming
    pulse), representative of a wide-area surveillance scan.
    Track dwell: shorter CPI (8 pulses of 512-sample PRI, 64-sample rect
    pulse), offset by +50 kHz, +3 dB — representative of a dedicated track
    beam pointed at a designated target.

    Returns:
        A :class:`ScheduledWaveform` that alternates search/track forever.
    """
    search = SegmentSpec(
        waveform=PulsedRadar(
            pulse_width_samples=256, pri_samples=2048, num_pulses=16,
            pulse_shape="hamming",
        ),
        duration_samples=2048 * 16,  # 32_768 samples = one full CPI
        mode="search",
    )
    track = SegmentSpec(
        waveform=PulsedRadar(
            pulse_width_samples=64, pri_samples=512, num_pulses=8,
            pulse_shape="rect",
        ),
        duration_samples=512 * 8,  # 4_096 samples = one CPI
        mode="track",
        power_offset_db=3.0,
        freq_offset_hz=50e3,
    )
    schedule = StaticSchedule(segments=[search, track, search, track], loop=True)
    return ScheduledWaveform(schedule=schedule, label="MFR_SearchTrack")
```

- [ ] **Step 4: Export the factory from the package**

Edit `python/spectra/waveforms/multifunction/__init__.py` — add to the
imports section:

```python
from spectra.waveforms.multifunction.examples import (
    multifunction_search_track_radar,
)
```

And add `"multifunction_search_track_radar"` to `__all__` (alphabetical-ish
position near the other names).

- [ ] **Step 5: Re-export from top-level waveforms**

Edit `python/spectra/waveforms/__init__.py` — extend the existing multifunction
import block added in Task 8:

```python
from spectra.waveforms.multifunction import (
    CognitiveSchedule,
    ModeSpec,
    ScheduledWaveform,
    Schedule,
    SegmentSpec,
    StaticSchedule,
    StochasticSchedule,
    multifunction_search_track_radar,
    segments_to_mode_mask,
)
```

Add `"multifunction_search_track_radar"` to the top-level `__all__`.

- [ ] **Step 6: Run tests to verify they pass**

Run: `pytest tests/test_multifunction_emitter.py -v -k search_track`
Expected: PASS — 3 tests.

- [ ] **Step 7: Commit**

```bash
git add python/spectra/waveforms/multifunction/examples.py \
        python/spectra/waveforms/multifunction/__init__.py \
        python/spectra/waveforms/__init__.py \
        tests/test_multifunction_emitter.py
git commit -m "feat(waveforms): add multifunction_search_track_radar example"
```

---

## Task 10: Example 2 — `multi_prf_pulse_doppler_radar` (Stochastic)

Markov chain over {low, medium, high} PRF modes.

**Files:**
- Modify: `python/spectra/waveforms/multifunction/examples.py`
- Modify: `python/spectra/waveforms/multifunction/__init__.py`
- Modify: `python/spectra/waveforms/__init__.py`
- Modify: `tests/test_multifunction_emitter.py`

- [ ] **Step 1: Write the failing tests (append)**

```python
# ── Task 10: Example 2 — multi-PRF pulse-Doppler ─────────────────────────────

def test_multi_prf_pulse_doppler_produces_all_three_modes(assert_valid_iq):
    from spectra.waveforms import multi_prf_pulse_doppler_radar

    sw = multi_prf_pulse_doppler_radar()
    # Use a long enough run and a favorable seed to see all 3 modes.
    iq, segs = sw.generate_with_segments(num_samples=100_000, sample_rate=20e6, seed=3)

    assert_valid_iq(iq, expected_length=100_000)
    modes = {s.mode for s in segs}
    assert modes.issubset({"low_prf", "medium_prf", "high_prf"})
    assert len(modes) >= 2  # statistical: expect multiple modes over 100k samples


def test_multi_prf_pulse_doppler_deterministic():
    from spectra.waveforms import multi_prf_pulse_doppler_radar

    iq1 = multi_prf_pulse_doppler_radar().generate(
        num_symbols=50_000, sample_rate=20e6, seed=11,
    )
    iq2 = multi_prf_pulse_doppler_radar().generate(
        num_symbols=50_000, sample_rate=20e6, seed=11,
    )
    assert np.array_equal(iq1, iq2)
```

- [ ] **Step 2: Run tests to verify they fail**

Run: `pytest tests/test_multifunction_emitter.py -v -k multi_prf_pulse_doppler`
Expected: FAIL — import error.

- [ ] **Step 3: Append factory to `examples.py`**

Add to `python/spectra/waveforms/multifunction/examples.py`:

```python
from spectra.waveforms.multifunction.schedule import StochasticSchedule
from spectra.waveforms.multifunction.segment import ModeSpec


def multi_prf_pulse_doppler_radar() -> ScheduledWaveform:
    """Multi-PRF pulse-Doppler radar using Markov-model dwell sequencing.

    Three modes (``low_prf``, ``medium_prf``, ``high_prf``) differ by PRI
    and CPI length. Transitions favor mode diversity (self-loop probability
    0.2; others 0.4 each). Dwell duration is uniform in ``(2048, 16384)``
    samples.

    Returns:
        A :class:`ScheduledWaveform` wrapping a :class:`StochasticSchedule`.
    """
    modes = {
        "low_prf": ModeSpec(
            waveform_factory=lambda rng: PulsedRadar(
                pulse_width_samples=128, pri_samples=4096, num_pulses=8,
            ),
            duration_samples=(2048, 16384),
        ),
        "medium_prf": ModeSpec(
            waveform_factory=lambda rng: PulsedRadar(
                pulse_width_samples=64, pri_samples=1024, num_pulses=16,
            ),
            duration_samples=(2048, 16384),
        ),
        "high_prf": ModeSpec(
            waveform_factory=lambda rng: PulsedRadar(
                pulse_width_samples=16, pri_samples=256, num_pulses=32,
            ),
            duration_samples=(2048, 16384),
        ),
    }
    transitions = {
        "low_prf": {"low_prf": 0.2, "medium_prf": 0.4, "high_prf": 0.4},
        "medium_prf": {"low_prf": 0.4, "medium_prf": 0.2, "high_prf": 0.4},
        "high_prf": {"low_prf": 0.4, "medium_prf": 0.4, "high_prf": 0.2},
    }
    schedule = StochasticSchedule(
        modes=modes,
        transitions=transitions,
        initial_mode={"low_prf": 1.0 / 3, "medium_prf": 1.0 / 3, "high_prf": 1.0 / 3},
    )
    return ScheduledWaveform(schedule=schedule, label="PulseDopplerMultiPRF")
```

- [ ] **Step 4: Export from `__init__`s**

Edit `python/spectra/waveforms/multifunction/__init__.py` — add to the examples import:

```python
from spectra.waveforms.multifunction.examples import (
    multi_prf_pulse_doppler_radar,
    multifunction_search_track_radar,
)
```

Add `"multi_prf_pulse_doppler_radar"` to `__all__`.

Edit `python/spectra/waveforms/__init__.py` — add `multi_prf_pulse_doppler_radar`
to the import block and to `__all__`.

- [ ] **Step 5: Run tests to verify they pass**

Run: `pytest tests/test_multifunction_emitter.py -v -k multi_prf_pulse_doppler`
Expected: PASS — 2 tests.

- [ ] **Step 6: Commit**

```bash
git add python/spectra/waveforms/multifunction/examples.py \
        python/spectra/waveforms/multifunction/__init__.py \
        python/spectra/waveforms/__init__.py \
        tests/test_multifunction_emitter.py
git commit -m "feat(waveforms): add multi_prf_pulse_doppler_radar example"
```

---

## Task 11: Example 3 — `frequency_agile_stepped_pri_radar`

Single mode with a parameterized waveform factory drawing PRI and carrier offset per dwell.

**Files:**
- Modify: `python/spectra/waveforms/multifunction/examples.py`
- Modify: `python/spectra/waveforms/multifunction/__init__.py`
- Modify: `python/spectra/waveforms/__init__.py`
- Modify: `tests/test_multifunction_emitter.py`

- [ ] **Step 1: Write the failing tests (append)**

```python
# ── Task 11: Example 3 — frequency-agile stepped-PRI ────────────────────────

def test_frequency_agile_produces_varying_freq_offsets(assert_valid_iq):
    from spectra.waveforms import frequency_agile_stepped_pri_radar

    sw = frequency_agile_stepped_pri_radar()
    iq, segs = sw.generate_with_segments(num_samples=60_000, sample_rate=20e6, seed=5)

    assert_valid_iq(iq, expected_length=60_000)
    assert len(segs) >= 3  # at least 3 dwells over 60k samples
    offsets = [s.modulation_params["freq_offset_hz"] for s in segs]
    assert len(set(offsets)) >= 2, "expected frequency agility across dwells"
    assert all(-200e3 <= f <= 200e3 for f in offsets)


def test_frequency_agile_deterministic():
    from spectra.waveforms import frequency_agile_stepped_pri_radar
    iq1 = frequency_agile_stepped_pri_radar().generate(
        num_symbols=30_000, sample_rate=20e6, seed=12,
    )
    iq2 = frequency_agile_stepped_pri_radar().generate(
        num_symbols=30_000, sample_rate=20e6, seed=12,
    )
    assert np.array_equal(iq1, iq2)
```

- [ ] **Step 2: Run tests to verify they fail**

Run: `pytest tests/test_multifunction_emitter.py -v -k frequency_agile`
Expected: FAIL.

- [ ] **Step 3: Append factory**

Add to `python/spectra/waveforms/multifunction/examples.py`:

```python
def frequency_agile_stepped_pri_radar() -> ScheduledWaveform:
    """Frequency-agile, PRI-stepped pulsed radar.

    Single mode ``"agile"``; each dwell instantiates a fresh
    :class:`PulsedRadar` with PRI drawn from ``{512, 768, 1024, 1536}`` and
    carrier offset drawn uniformly in ``(-200 kHz, +200 kHz)``.

    Returns:
        A :class:`ScheduledWaveform` wrapping a :class:`StochasticSchedule`
        with a single self-looping mode.
    """
    _PRI_CHOICES = (512, 768, 1024, 1536)

    def _agile_factory(rng):
        pri = int(rng.choice(_PRI_CHOICES))
        return PulsedRadar(pulse_width_samples=64, pri_samples=pri, num_pulses=8)

    modes = {
        "agile": ModeSpec(
            waveform_factory=_agile_factory,
            duration_samples=(2048, 8192),
            freq_offset_hz=(-200e3, 200e3),
        ),
    }
    schedule = StochasticSchedule(
        modes=modes,
        transitions={"agile": {"agile": 1.0}},
        initial_mode="agile",
    )
    return ScheduledWaveform(schedule=schedule, label="FreqAgileRadar")
```

- [ ] **Step 4: Export from `__init__`s**

Update `python/spectra/waveforms/multifunction/__init__.py` examples import:

```python
from spectra.waveforms.multifunction.examples import (
    frequency_agile_stepped_pri_radar,
    multi_prf_pulse_doppler_radar,
    multifunction_search_track_radar,
)
```

Add `"frequency_agile_stepped_pri_radar"` to `__all__`.
Do the same in `python/spectra/waveforms/__init__.py`.

- [ ] **Step 5: Run tests to verify they pass**

Run: `pytest tests/test_multifunction_emitter.py -v -k frequency_agile`
Expected: PASS — 2 tests.

- [ ] **Step 6: Commit**

```bash
git add python/spectra/waveforms/multifunction/examples.py \
        python/spectra/waveforms/multifunction/__init__.py \
        python/spectra/waveforms/__init__.py \
        tests/test_multifunction_emitter.py
git commit -m "feat(waveforms): add frequency_agile_stepped_pri_radar example"
```

---

## Task 12: Example 4 — `radcom_emitter` (Cross-family)

Radar dwells interleaved with QPSK comms bursts on the same carrier.

**Files:**
- Modify: `python/spectra/waveforms/multifunction/examples.py`
- Modify: `python/spectra/waveforms/multifunction/__init__.py`
- Modify: `python/spectra/waveforms/__init__.py`
- Modify: `tests/test_multifunction_emitter.py`

- [ ] **Step 1: Write the failing tests (append)**

```python
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
```

- [ ] **Step 2: Run tests to verify they fail**

Run: `pytest tests/test_multifunction_emitter.py -v -k radcom`
Expected: FAIL.

- [ ] **Step 3: Append factory**

Add to `python/spectra/waveforms/multifunction/examples.py`:

```python
from spectra.waveforms.psk import QPSK


def radcom_emitter() -> ScheduledWaveform:
    """Joint radar-communications emitter alternating radar and comms dwells.

    Radar dwells use a short-pulse :class:`PulsedRadar`; comms bursts use
    :class:`QPSK` with RRC pulse shaping. Both share the emitter's carrier.

    Returns:
        A :class:`ScheduledWaveform` with a static radar/comms interleave.
    """
    radar = SegmentSpec(
        waveform=PulsedRadar(
            pulse_width_samples=64, pri_samples=512, num_pulses=8,
        ),
        duration_samples=512 * 8,  # 4_096
        mode="radar",
    )
    comms = SegmentSpec(
        waveform=QPSK(samples_per_symbol=8, rolloff=0.35),
        duration_samples=2048,
        mode="comms",
    )
    schedule = StaticSchedule(
        segments=[radar, comms, radar, radar, comms, radar, comms], loop=True,
    )
    return ScheduledWaveform(schedule=schedule, label="RadCom")
```

- [ ] **Step 4: Export from `__init__`s**

Update `python/spectra/waveforms/multifunction/__init__.py` examples import:

```python
from spectra.waveforms.multifunction.examples import (
    frequency_agile_stepped_pri_radar,
    multi_prf_pulse_doppler_radar,
    multifunction_search_track_radar,
    radcom_emitter,
)
```

Add `"radcom_emitter"` to `__all__`. Same in `python/spectra/waveforms/__init__.py`.

- [ ] **Step 5: Run tests to verify they pass**

Run: `pytest tests/test_multifunction_emitter.py -v -k radcom`
Expected: PASS — 2 tests.

- [ ] **Step 6: Commit**

```bash
git add python/spectra/waveforms/multifunction/examples.py \
        python/spectra/waveforms/multifunction/__init__.py \
        python/spectra/waveforms/__init__.py \
        tests/test_multifunction_emitter.py
git commit -m "feat(waveforms): add radcom_emitter example"
```

---

## Task 13: End-to-end `Composer` integration test

Proves a `ScheduledWaveform` drops into `SceneConfig.signal_pool` unchanged.

**Files:**
- Modify: `tests/test_multifunction_emitter.py`

- [ ] **Step 1: Write the test (append)**

```python
# ── Task 13: Composer integration ────────────────────────────────────────────

def test_scheduled_waveform_in_composer(assert_valid_iq):
    """A ScheduledWaveform plugs into SceneConfig.signal_pool unchanged."""
    from spectra.scene.composer import Composer, SceneConfig
    from spectra.waveforms import multifunction_search_track_radar, QPSK

    config = SceneConfig(
        capture_duration=0.005,         # 5 ms
        capture_bandwidth=10e6,
        sample_rate=20e6,
        num_signals=2,
        signal_pool=[multifunction_search_track_radar(), QPSK()],
        snr_range=(10.0, 20.0),
        allow_overlap=True,
    )
    composer = Composer(config)
    iq, descriptions = composer.generate(seed=0)

    assert_valid_iq(iq)
    assert len(descriptions) == 2
    # At least one of them carries the MFR label we gave ScheduledWaveform.
    labels = {d.label for d in descriptions}
    assert "MFR_SearchTrack" in labels or "QPSK" in labels
```

- [ ] **Step 2: Run test to verify it passes (should pass immediately — no code change needed)**

Run: `pytest tests/test_multifunction_emitter.py -v -k scheduled_waveform_in_composer`
Expected: PASS. If it fails with a bandwidth/center-freq error, investigate
before patching: the preview-based bandwidth heuristic may be under-estimating
for this config. The fix is to pass `default_bandwidth_hz=…` when constructing
the MFR in the test, which signals a real issue in the heuristic to track in a
follow-up (don't silence it by only patching the test).

- [ ] **Step 3: Commit**

```bash
git add tests/test_multifunction_emitter.py
git commit -m "test(waveforms): end-to-end ScheduledWaveform through Composer"
```

---

## Task 14: Reproducibility hash test

CI-grade sanity check to catch silent behavior drift in the four examples.

**Files:**
- Modify: `tests/test_multifunction_emitter.py`

- [ ] **Step 1: Write the test (append)**

```python
# ── Task 14: Reproducibility hashes ──────────────────────────────────────────

import hashlib


def _iq_hash(iq: np.ndarray) -> str:
    return hashlib.sha256(iq.tobytes()).hexdigest()[:16]


@pytest.mark.parametrize("factory_name,seed", [
    ("multifunction_search_track_radar", 0),
    ("multi_prf_pulse_doppler_radar", 0),
    ("frequency_agile_stepped_pri_radar", 0),
    ("radcom_emitter", 0),
])
def test_example_reproducibility_hash(factory_name, seed):
    """Hash the IQ output of each example; if this changes, behavior drifted."""
    import spectra.waveforms as wf
    factory = getattr(wf, factory_name)
    sw = factory()
    iq = sw.generate(num_symbols=20_000, sample_rate=20e6, seed=seed)
    h = _iq_hash(iq)

    # First run: print(h) and commit the resulting hash into EXPECTED_HASHES.
    # Subsequent runs must match.
    EXPECTED_HASHES = {
        "multifunction_search_track_radar": "REPLACE_ON_FIRST_RUN",
        "multi_prf_pulse_doppler_radar": "REPLACE_ON_FIRST_RUN",
        "frequency_agile_stepped_pri_radar": "REPLACE_ON_FIRST_RUN",
        "radcom_emitter": "REPLACE_ON_FIRST_RUN",
    }
    expected = EXPECTED_HASHES[factory_name]
    if expected == "REPLACE_ON_FIRST_RUN":
        pytest.skip(
            f"first-run: add {factory_name!r} -> {h!r} to EXPECTED_HASHES"
        )
    assert h == expected, (
        f"IQ output drifted for {factory_name!r}: expected {expected}, got {h}"
    )
```

- [ ] **Step 2: Run the test to capture the real hashes**

Run: `pytest tests/test_multifunction_emitter.py -v -k example_reproducibility_hash`
Expected: 4 SKIPPED tests. Each skip message prints the actual hash.

- [ ] **Step 3: Copy the printed hashes into `EXPECTED_HASHES`**

For each of the four factories, replace `"REPLACE_ON_FIRST_RUN"` with the
hash printed in the skip message (16 hex chars). After this edit the
dictionary should look like:

```python
EXPECTED_HASHES = {
    "multifunction_search_track_radar": "<16-hex-chars>",
    "multi_prf_pulse_doppler_radar":    "<16-hex-chars>",
    "frequency_agile_stepped_pri_radar": "<16-hex-chars>",
    "radcom_emitter":                   "<16-hex-chars>",
}
```

- [ ] **Step 4: Re-run — now the test must pass, not skip**

Run: `pytest tests/test_multifunction_emitter.py -v -k example_reproducibility_hash`
Expected: PASS — 4 tests, 0 skipped. If any still skips, that factory's hash
wasn't pasted in correctly.

- [ ] **Step 5: Run the full test file to confirm everything is green**

Run: `pytest tests/test_multifunction_emitter.py -v`
Expected: PASS — all tests in the file.

- [ ] **Step 6: Commit**

```bash
git add tests/test_multifunction_emitter.py
git commit -m "test(waveforms): add reproducibility hashes for MFR examples"
```

---

## Task 15: User-guide documentation page

**Files:**
- Create: `docs/user-guide/multifunction-emitters.md`
- Modify: `mkdocs.yml`

- [ ] **Step 1: Create the docs page**

Create `docs/user-guide/multifunction-emitters.md`:

````markdown
# Multi-Function Emitters

A **multi-function emitter** is a single emitter that interleaves multiple
waveform modes over time — a multi-function radar doing search/track dwells,
a multi-PRF pulse-Doppler radar, a frequency-agile ECCM radar, a joint
radar-communications emitter, and so on.

SPECTRA models this with a new `Waveform` subclass, `ScheduledWaveform`,
driven by a `Schedule` strategy.

## Concepts

- **`SegmentSpec`** — one entry in the timeline: a sub-waveform, a duration
  (in samples), a mode name, and optional baseband offsets.
- **`Schedule`** — a strategy that yields `SegmentSpec`s on demand.
- **`ScheduledWaveform`** — a `Waveform` that iterates a schedule and
  concatenates child IQ.
- **`SignalDescription.mode`** — each emitted segment comes with a
  `SignalDescription` carrying its mode label, time bounds, frequency bounds,
  and waveform label.

## Schedule types

### `StaticSchedule`

Deterministic, optionally looped list of `SegmentSpec`s. Use when the
timeline is fixed.

```python
from spectra.waveforms import (
    PulsedRadar, ScheduledWaveform, StaticSchedule, SegmentSpec,
)

search = SegmentSpec(
    waveform=PulsedRadar(pulse_width_samples=256, pri_samples=2048, num_pulses=16),
    duration_samples=2048 * 16, mode="search",
)
track = SegmentSpec(
    waveform=PulsedRadar(pulse_width_samples=64, pri_samples=512, num_pulses=8),
    duration_samples=512 * 8, mode="track",
    power_offset_db=3.0, freq_offset_hz=50e3,
)
mfr = ScheduledWaveform(StaticSchedule([search, track], loop=True), label="MFR")
iq, segments = mfr.generate_with_segments(num_samples=200_000, sample_rate=20e6, seed=0)
```

### `StochasticSchedule`

Seeded Markov chain over named modes with per-mode duration and parameter
distributions. Use for random dwell sequences.

```python
from spectra.waveforms import (
    ModeSpec, PulsedRadar, ScheduledWaveform, StochasticSchedule,
)

modes = {
    "low_prf":    ModeSpec(waveform_factory=lambda r: PulsedRadar(pri_samples=4096), duration_samples=(2048, 16384)),
    "medium_prf": ModeSpec(waveform_factory=lambda r: PulsedRadar(pri_samples=1024), duration_samples=(2048, 16384)),
    "high_prf":   ModeSpec(waveform_factory=lambda r: PulsedRadar(pri_samples=256),  duration_samples=(2048, 16384)),
}
transitions = {
    "low_prf":    {"low_prf": 0.2, "medium_prf": 0.4, "high_prf": 0.4},
    "medium_prf": {"low_prf": 0.4, "medium_prf": 0.2, "high_prf": 0.4},
    "high_prf":   {"low_prf": 0.4, "medium_prf": 0.4, "high_prf": 0.2},
}
```

### `CognitiveSchedule`

Abstract base for closed-loop / scene-coupled scheduling. Ships with no
concrete subclass in v1 — see the design doc for future plans.

## Built-in examples

SPECTRA ships four reference factories:

| Factory | Schedule type | What it exercises |
|---------|---------------|-------------------|
| `multifunction_search_track_radar` | Static | search/track dwells, per-segment power and freq offsets |
| `multi_prf_pulse_doppler_radar`    | Stochastic | Markov over 3 PRF modes |
| `frequency_agile_stepped_pri_radar` | Stochastic | parametric `waveform_factory`, per-dwell offset draws |
| `radcom_emitter`                   | Static (cross-family) | radar + QPSK on same carrier |

Each returns a `ScheduledWaveform` that can be dropped straight into a
`Composer`, a `NarrowbandDataset`, or any other code that accepts a
`Waveform`.

## Mode timelines and masks

`generate_with_segments(num_samples, sample_rate, seed)` returns `(iq,
list[SignalDescription])`. The optional `mode` field on each description
identifies the active mode for that segment.

For segmentation tasks, `segments_to_mode_mask(...)` converts the list into a
per-sample integer array.

## Scope and limitations in v1

- Single carrier only. Multi-band platforms (simultaneous transmission on
  distinct carriers — e.g., radar + IFF + datalink) are handled by using
  multiple co-located `Emitter` instances today; a first-class
  `MultiFunctionEmitter` wrapper is a follow-on spec.
- No closed-loop / scene-driven scheduling. `CognitiveSchedule` is an ABC
  stub only.
- No YAML serialization of schedules. Construct programmatically.

See [the design spec](https://github.com/gditzler/SPECTRA/blob/main/docs/superpowers/specs/2026-04-17-multifunction-emitter-design.md)
for full details.
````

- [ ] **Step 2: Add the page to mkdocs nav**

Edit `mkdocs.yml`. In the `nav:` section, under `User Guide:`, add a new entry
after `Waveforms` so the order is:

```yaml
  - User Guide:
    - Waveforms: user-guide/waveforms.md
    - Multi-Function Emitters: user-guide/multifunction-emitters.md
    - Impairments: user-guide/impairments.md
    ...
```

- [ ] **Step 3: (Optional) Validate the docs build locally**

If `mkdocs` is installed (`pip install 'spectra[docs]'`), run:
`mkdocs build --strict`
Expected: build succeeds, no warnings.
If `mkdocs` is not installed in this environment, skip this step.

- [ ] **Step 4: Commit**

```bash
git add docs/user-guide/multifunction-emitters.md mkdocs.yml
git commit -m "docs(user-guide): add multi-function emitter page"
```

---

## Task 16: Update `README.md` and `CLAUDE.md` architecture sections

**Files:**
- Modify: `README.md`
- Modify: `CLAUDE.md`

- [ ] **Step 1: Find the waveforms bullet in each file**

```bash
grep -n "waveforms/" README.md CLAUDE.md
```
Expected: a line in each file that enumerates the waveform subdirectories or
top-level waveform families.

- [ ] **Step 2: Edit `CLAUDE.md`**

In the bullet that describes `python/spectra/waveforms/` (the line that
begins `python/spectra/waveforms/` —  "`Waveform` ABC with …"), append a
mention of the new subpackage. Specifically, append this sentence to that
bullet (find the one currently describing the `waveforms/` package, after
the list of existing families):

> Also `waveforms/multifunction/` — `ScheduledWaveform` and the `Schedule`
> strategy family (`StaticSchedule`, `StochasticSchedule`, `CognitiveSchedule`)
> for multi-function emitters that interleave multiple modes over time, with
> four built-in factories: `multifunction_search_track_radar`,
> `multi_prf_pulse_doppler_radar`, `frequency_agile_stepped_pri_radar`,
> and `radcom_emitter`.

- [ ] **Step 3: Edit `README.md` similarly**

Locate the analogous section in `README.md` (it should also enumerate the
waveform families). Add a one-sentence mention of multi-function emitters
with a link to the user-guide page:

> **Multi-function emitters** (`ScheduledWaveform`) — single emitters that
> interleave multiple modes on a schedule (MFR search/track, multi-PRF
> pulse-Doppler, frequency-agile, RadCom); see
> [the user-guide page](docs/user-guide/multifunction-emitters.md).

If `README.md` does not have a waveforms enumeration, add the bullet to the
most appropriate feature list (e.g., under an "Additional waveforms" or
"Features" header). Do not create new top-level sections.

- [ ] **Step 4: Sanity-check by running the full test suite once more**

Run: `pytest -x -q`
Expected: PASS.

- [ ] **Step 5: Commit**

```bash
git add README.md CLAUDE.md
git commit -m "docs: mention multi-function emitter subpackage in README and CLAUDE"
```

---

## Final verification

After all 16 tasks are complete:

- [ ] **Run the full suite**

```bash
pytest -x -q
```

Expected: all green.

- [ ] **Lint check**

```bash
ruff check python/spectra/waveforms/multifunction tests/test_multifunction_emitter.py
ruff format --check python/spectra/waveforms/multifunction tests/test_multifunction_emitter.py
```

Expected: no issues. If `ruff format` reports differences, run
`ruff format python/spectra/waveforms/multifunction tests/test_multifunction_emitter.py`,
re-run the full suite, and commit as `style: apply ruff format`.

- [ ] **Verify the acceptance criteria from the spec**

1. ✅ `pytest tests/test_multifunction_emitter.py -v` all green.
2. ✅ Four example factories return working `ScheduledWaveform` instances.
3. ✅ `ScheduledWaveform` plugs into `SceneConfig.signal_pool` without changes to `Composer` (Task 13).
4. ✅ `SignalDescription.mode` field added with no broken callers (Task 1 Step 5).
5. ✅ `docs/user-guide/multifunction-emitters.md` exists; referenced from `mkdocs.yml` nav (Task 15).
6. ✅ `README.md` and `CLAUDE.md` mention the new `waveforms/multifunction/` subpackage (Task 16).
