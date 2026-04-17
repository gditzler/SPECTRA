# Multi-Function Emitter — Design Spec

**Date:** 2026-04-17
**Status:** Approved for implementation planning

## Goal

Add a first-class abstraction for emitters that interleave multiple waveform modes over time — e.g., multi-function radars (MFR) doing search/track dwells, multi-PRF pulse-Doppler radars, frequency-agile (ECCM) radars, and joint radar-communications (RadCom) emitters. Today SPECTRA has the raw pieces (a rich single-mode waveform library, `Emitter`, `Composer`) but no explicit concept of a single emitter that sequences several waveforms on a schedule. This spec introduces that concept as a new `Waveform` subclass plus a small `Schedule` strategy family and ships four reference examples.

## Non-Goals

- **Multi-carrier / multi-band platforms.** One platform transmitting simultaneously on distinct RF carriers (surveillance radar + IFF + datalink, each on its own center frequency) is deferred to a follow-on spec. The v1 abstraction is same-carrier only. Multi-band platforms can still be approximated today with multiple co-located `Emitter` instances.
- **Closed-loop / scene-coupled scheduling.** The ABC hook (`CognitiveSchedule`) exists and is tested with a trivial deterministic subclass, but no concrete closed-loop scheduler is included. Wiring the scheduler to `Environment` / target state (so e.g. a radar can switch search→track when a target enters a beam) is a future spec.
- **YAML round-trip for schedules.** Multi-function emitters in v1 are constructed programmatically (via the four factory functions or user-defined schedules). YAML serialization is deferred until the schedule-config shape stabilizes.
- **Automatic unwrapping of per-segment `SignalDescription`s into `Composer` scenes.** A `ScheduledWaveform` drops into `SceneConfig.signal_pool` as a single waveform; a helper to flatten its per-segment timeline into the scene's `SignalDescription` list is a follow-on.
- **New Rust primitives.** All scheduling and segment orchestration is Python; the sub-waveforms reuse existing Rust primitives unchanged.

## Design Decisions (Q&A Summary)

1. **Scope of "multi-function emitter":** general — must cover MFR, multi-system-on-one-carrier, RadCom, and cognitive/adaptive under one scheduling abstraction.
2. **Schedule drivers:** static (deterministic timeline) + stochastic (Markov transitions + duration distributions) land in v1. Closed-loop is defined as an ABC stub only; no concrete implementation.
3. **Same-carrier only for v1.** Multi-carrier platforms are a separate follow-on spec (the abstraction for that would wrap `ScheduledWaveform` inside a new `MultiFunctionEmitter`-at-the-emitter-level class).
4. **Ground truth:** per-segment list is primary; per-sample mode mask is a helper. Reuse the existing `SignalDescription` dataclass with a new optional `mode: str | None` field rather than introducing a parallel type.
5. **Four shipping examples:** search/track MFR (static), multi-PRF pulse-Doppler (stochastic Markov), frequency-agile stepped-PRI (stochastic with parameterized waveform factory), RadCom (cross-family static).
6. **Integration layer:** `ScheduledWaveform(Waveform)` with optional per-segment `power_offset_db` and `freq_offset_hz` applied at baseband. `Emitter` / `Environment` / `Composer` stay untouched.

## Architecture

### Module Layout

New subpackage under the existing waveforms tree:

```
python/spectra/waveforms/multifunction/
├── __init__.py              # public exports + segments_to_mode_mask helper
├── schedule.py              # Schedule ABC, StaticSchedule, StochasticSchedule, CognitiveSchedule
├── segment.py               # SegmentSpec dataclass + ModeSpec dataclass
├── scheduled_waveform.py    # ScheduledWaveform(Waveform) orchestrator
└── examples.py              # Four factory functions for reference emitters
```

Re-exported from `spectra.waveforms` so users write:

```python
from spectra.waveforms import (
    ScheduledWaveform, StaticSchedule, StochasticSchedule, CognitiveSchedule,
    SegmentSpec, ModeSpec,
    multifunction_search_track_radar, multi_prf_pulse_doppler_radar,
    frequency_agile_stepped_pri_radar, radcom_emitter,
)
```

`SignalDescription` in `python/spectra/scene/signal_desc.py` gains one optional field appended at the end of the existing `@dataclass`:

```python
mode: str | None = None
```

Existing fields (`t_start`, `t_stop`, `f_low`, `f_high`, `label`, `snr`, `modulation_params`) and derived properties (`f_center`, `bandwidth`, `duration`) are unchanged. All existing constructors and callers continue to work because the new field is optional with a default.

### Core Types

```python
# segment.py
@dataclass
class SegmentSpec:
    waveform: Waveform
    duration_samples: int
    mode: str
    power_offset_db: float = 0.0       # baseband amplitude delta
    freq_offset_hz: float = 0.0        # baseband frequency shift (relative to emitter carrier)
    gain_offset_db: float = 0.0        # recorded in metadata only; not applied to IQ
    metadata: dict = field(default_factory=dict)

@dataclass
class ModeSpec:
    """Used by StochasticSchedule to describe a Markov mode."""
    waveform_factory: Callable[[np.random.Generator], Waveform]
    duration_samples: int | tuple[int, int] | Callable[[np.random.Generator], int]
    power_offset_db: float | tuple[float, float] = 0.0
    freq_offset_hz: float | tuple[float, float] = 0.0
    metadata: dict = field(default_factory=dict)
```

### `Schedule` Family

Single abstract method driving all subclasses:

```python
class Schedule(ABC):
    @abstractmethod
    def iter_segments(
        self,
        total_samples: int,
        sample_rate: float,
        seed: int,
    ) -> Iterator[SegmentSpec]:
        """Yield SegmentSpecs until the caller has accumulated total_samples."""
```

The caller (`ScheduledWaveform`) is responsible for stopping once cumulative duration reaches `total_samples`. Schedules do not need to be bounded themselves — `StaticSchedule(loop=True)` and `StochasticSchedule` are conceptually infinite streams.

**`StaticSchedule`** — deterministic list, optionally looped.

```python
StaticSchedule(segments: list[SegmentSpec], loop: bool = True)
```

- `loop=True`: list wraps until `total_samples` reached.
- `loop=False`: list is played once; if shorter than `total_samples`, IQ is zero-padded to fill.
- `seed` argument is accepted but ignored (by design).

**`StochasticSchedule`** — seeded Markov model + per-mode parameter distributions.

```python
StochasticSchedule(
    modes: dict[str, ModeSpec],
    transitions: dict[str, dict[str, float]],
    initial_mode: str | dict[str, float],
)
```

- `__post_init__` validates: every mode referenced in `transitions` is in `modes`; every row of the transition matrix sums to 1.0 ± 1e-6; `initial_mode` (if a dict) sums to 1.0 and references only known modes.
- `iter_segments(total_samples, fs, seed)` creates `rng = np.random.default_rng(seed)`, draws the initial mode, then per-segment: draws `duration_samples` (from int / tuple range / callable), draws `power_offset_db` and `freq_offset_hz` (scalar or uniform from tuple), calls `mode_spec.waveform_factory(rng)` to build a per-dwell sub-waveform, yields the resulting `SegmentSpec`, and draws the next mode from `transitions[current]`.
- Reproducibility: same `(config, seed, total_samples, fs)` → byte-identical segment list.

**`CognitiveSchedule`** — ABC stub, v1 ships with no concrete subclass.

```python
class CognitiveSchedule(Schedule, ABC):
    @abstractmethod
    def next_segment(
        self,
        history: list[SegmentSpec],
        state: Any,
    ) -> SegmentSpec | None:
        """Return next segment, or None to end schedule."""

    def iter_segments(self, total_samples, sample_rate, seed):
        # Concrete loop: call next_segment() with history + state, yield,
        # accumulate duration, stop on None or when enough samples produced.
        ...
```

A test demonstrates subclassing with a trivial deterministic rule to validate the seam. Actual scene-state coupling is deferred.

### `ScheduledWaveform`

```python
class ScheduledWaveform(Waveform):
    samples_per_symbol: int = 1

    def __init__(
        self,
        schedule: Schedule,
        label: str = "multifunction",
        default_bandwidth_hz: float | None = None,
    ):
        self._schedule = schedule
        self._label = label
        self._default_bandwidth_hz = default_bandwidth_hz

    def generate(self, num_symbols, sample_rate, seed=None) -> np.ndarray:
        iq, _ = self.generate_with_segments(num_symbols, sample_rate, seed)
        return iq

    def generate_with_segments(
        self, num_samples: int, sample_rate: float, seed: int | None = None,
    ) -> tuple[np.ndarray, list[SignalDescription]]:
        ...

    def bandwidth(self, sample_rate: float) -> float:
        ...

    @property
    def label(self) -> str:
        return self._label
```

**Orchestration (`generate_with_segments`).** Pseudocode:

```
rng = default_rng(seed if seed is not None else 0)
seed_for_schedule = int(rng.integers(0, 2**32))
out = np.zeros(num_samples, dtype=complex64)
segments: list[SignalDescription] = []
cursor = 0

for spec in schedule.iter_segments(num_samples, sample_rate, seed_for_schedule):
    remaining = num_samples - cursor
    if remaining <= 0: break
    dur = min(spec.duration_samples, remaining)

    child_sps = getattr(spec.waveform, "samples_per_symbol", 1)
    child_num_symbols = max(1, math.ceil(dur / child_sps))
    child_seed = int(rng.integers(0, 2**32))
    child_iq = spec.waveform.generate(
        num_symbols=child_num_symbols,
        sample_rate=sample_rate,
        seed=child_seed,
    ).astype(np.complex64)

    # 1. Truncate or zero-pad to exactly dur samples.
    if len(child_iq) >= dur:
        child_iq = child_iq[:dur]
    else:
        pad = np.zeros(dur - len(child_iq), dtype=np.complex64)
        child_iq = np.concatenate([child_iq, pad])

    # 2. Power offset (amplitude scale).
    if spec.power_offset_db != 0.0:
        child_iq = child_iq * (10.0 ** (spec.power_offset_db / 20.0))

    # 3. Frequency offset at baseband.
    if spec.freq_offset_hz != 0.0:
        child_iq = frequency_shift(child_iq, spec.freq_offset_hz, sample_rate)

    out[cursor:cursor + dur] = child_iq

    child_bw = spec.waveform.bandwidth(sample_rate)
    segments.append(SignalDescription(
        t_start=cursor / sample_rate,
        t_stop=(cursor + dur) / sample_rate,
        f_low=spec.freq_offset_hz - child_bw / 2.0,   # relative to emitter carrier
        f_high=spec.freq_offset_hz + child_bw / 2.0,
        label=spec.waveform.label,
        snr=0.0,                                      # placeholder; set downstream by Composer/Environment
        modulation_params={
            **spec.metadata,
            "gain_offset_db": spec.gain_offset_db,
            "power_offset_db": spec.power_offset_db,
            "freq_offset_hz": spec.freq_offset_hz,
        },
        mode=spec.mode,
    ))
    cursor += dur

return out[:num_samples], segments
```

Key properties:

- Output length is exactly `num_samples` (final segment truncated if needed).
- `generate()` bit-exact equals the IQ portion of `generate_with_segments()` for same seed — ensured by not calling the RNG anywhere else.
- Child `samples_per_symbol > 1` handled via ceil + truncate.
- Seeding is interleaved: each call to `rng.integers(0, 2**32)` draws the next child seed; the schedule has its own top-level seed drawn once up front.

**`bandwidth(sample_rate)`.**

- If `default_bandwidth_hz` is set, return it.
- Otherwise, call `schedule.iter_segments(total_samples = int(sample_rate * 0.1), sample_rate, seed=0)` as a preview, and return `max(spec.waveform.bandwidth(sample_rate) + abs(spec.freq_offset_hz) * 2 for spec in preview)`. (Factor of 2 covers both sides of the center frequency.)
- `Composer` uses this to place the signal in frequency, so an overestimate is safer than an underestimate.

**`label` property** returns the user-supplied string. It does not change per segment; per-segment labels live in each `SignalDescription.mode`.

### Per-Sample Mode Mask Helper

```python
# multifunction/__init__.py
def segments_to_mode_mask(
    segments: list[SignalDescription],
    total_samples: int,
    sample_rate: float,
    mode_to_index: dict[str, int],
    fill_index: int = -1,
) -> np.ndarray:
    """Return int array of length total_samples with mode index per sample.

    Samples not covered by any segment get ``fill_index``.
    """
```

Useful for segmentation tasks. Not on the hot path of `generate()`.

### Integration Points

- **`Emitter`** — unchanged. A `ScheduledWaveform` is just a `Waveform`.
- **`Environment.compute()`** — unchanged. It operates on `Emitter.waveform.bandwidth()` and the emitter's single carrier, which is still well-defined.
- **`Composer`** — unchanged. `ScheduledWaveform` plugs into `SceneConfig.signal_pool` and is treated as a single signal in the scene. Per-segment timelines are accessible via `generate_with_segments()` but not automatically unwrapped into the scene's `SignalDescription` list (future enhancement).
- **`SignalDescription`** — gains one optional `mode: str | None = None` field in `python/spectra/scene/signal_desc.py`. No existing callers need updates.
- **Datasets** — no changes in v1. Existing `NarrowbandDataset` and `WidebandDataset` continue to work with `ScheduledWaveform` as an opaque `Waveform`. A future follow-on could add a dataset that exposes the per-segment timeline as detection/segmentation ground truth.

## Four Reference Examples

All four live in `python/spectra/waveforms/multifunction/examples.py`.

### 1. `multifunction_search_track_radar()` — Static

Alternating search and track dwells, same carrier. Demonstrates static scheduling with per-segment power and frequency offsets.

- **Search dwell:** `PulsedRadar(pulse_width_samples=256, pri_samples=2048, num_pulses=16, pulse_shape="hamming")`. `mode="search"`, no offsets.
- **Track dwell:** `PulsedRadar(pulse_width_samples=64, pri_samples=512, num_pulses=8, pulse_shape="rect")`. `mode="track"`, `power_offset_db=+3.0`, `freq_offset_hz=+50e3`.
- **Schedule:** `StaticSchedule([search, track, search, track], loop=True)`.

**Exercises:** static schedule, cross-parameter waveform swap, per-segment power and frequency offsets.

### 2. `multi_prf_pulse_doppler_radar()` — Stochastic (Markov)

Three PRF modes with Markov transitions. Classic ambiguity-resolution pattern.

- **Modes:** `low_prf` (PRI 4096), `medium_prf` (PRI 1024), `high_prf` (PRI 256). Each a `PulsedRadar` with matched pulse width and CPI count.
- **Duration per dwell:** `(2048, 16384)` sample range, uniform.
- **Transitions:** self-loop probability 0.2, spread uniformly across the other two modes (0.4 each).
- **Initial mode:** uniform over three modes.

**Exercises:** stochastic schedule, Markov model, variable dwell duration, seed-determined reproducibility.

### 3. `frequency_agile_stepped_pri_radar()` — Stochastic (parameterized factory)

Single mode `"agile"`, but each dwell draws a fresh `PulsedRadar` with random PRF and random carrier offset.

- **Waveform factory:** `rng → PulsedRadar(pulse_width_samples=64, pri_samples=rng.choice([512, 768, 1024, 1536]), num_pulses=8)`.
- **Duration per dwell:** `(2048, 8192)`.
- **Frequency offset:** uniform in `(-200e3, +200e3)` Hz.

**Exercises:** `waveform_factory` receiving an RNG, sub-waveform parameter randomization, per-segment frequency offset drawn from a range.

### 4. `radcom_emitter()` — Static (cross-family)

Radar dwells interleaved with short QPSK comms bursts on the same carrier.

- **Radar dwell:** `PulsedRadar(...)`, `mode="radar"`.
- **Comms burst:** `QPSKWaveform(samples_per_symbol=8, rrc_alpha=0.35)`, shorter duration, `mode="comms"`.
- **Schedule:** `StaticSchedule([radar, comms, radar, radar, comms, radar, comms], loop=True)`.

**Exercises:** cross-family scheduling (radar `Waveform` + PSK `Waveform` together), different `bandwidth()` per child, different `samples_per_symbol` per child, `label` preserved across families.

## Testing Strategy

All tests in `tests/test_multifunction_emitter.py`. No existing markers changed.

### Unit — `Schedule` family

- `test_static_schedule_loops` — `StaticSchedule(loop=True)` cycles; `loop=False` stops and pads.
- `test_static_schedule_exact_truncation` — final segment truncated so total output is exactly `total_samples`.
- `test_stochastic_schedule_determinism` — same seed → byte-identical IQ and segment list.
- `test_stochastic_schedule_different_seeds` — different seeds produce different sequences (statistical check, not byte-identical).
- `test_stochastic_schedule_markov_validation` — invalid transition matrix raises `ValueError` at construction (row doesn't sum to 1; reference to unknown mode).
- `test_cognitive_schedule_abstract` — cannot instantiate `CognitiveSchedule` directly; a trivial deterministic subclass drives the expected sequence.

### Unit — `ScheduledWaveform` orchestration

- `test_generate_matches_generate_with_segments` — IQ bit-exact for same seed.
- `test_output_length_exact` — exactly `num_symbols` samples for several values, including non-segment-boundary counts.
- `test_power_offset_db_applied` — `+6.0 dB` segment has ~4× power vs baseline (±10%).
- `test_freq_offset_hz_applied` — spectral peak at expected offset (FFT check).
- `test_child_samples_per_symbol_handled` — child with `sps > 1` receives `ceil(duration_samples / sps)` symbols; output truncated to `duration_samples`.
- `test_bandwidth_heuristic_stochastic` — `bandwidth()` for a stochastic schedule returns ≥ max observed in a preview run.
- `test_signal_description_has_mode` — returned segments carry `mode` from `SegmentSpec`.
- `test_segments_to_mode_mask` — helper produces correct per-sample int array; length matches; transitions at segment boundaries.

### Integration — the four examples

One test per example at `fs=20e6`, `num_samples=2_000_000` (~100 ms):
1. `generate_with_segments()` runs without error.
2. Output length equals `num_samples`.
3. `segments` has ≥2 distinct `mode` values.
4. `assert_valid_iq(iq)` (conftest fixture) passes.
5. Determinism under fixed seed.

### End-to-end

- `test_scheduled_waveform_in_composer` — `multifunction_search_track_radar()` goes into `SceneConfig.signal_pool` alongside a `QPSKWaveform`, `Composer.generate()` produces valid IQ and at least one `SignalDescription`.

### Reproducibility CI check

One test (not marked slow) hashes the IQ output of each example at a fixed seed and compares to a committed hash. Catches silent behavior drift.

## Acceptance Criteria

1. `pytest tests/test_multifunction_emitter.py -v` all green.
2. Four example factories return working `ScheduledWaveform` instances.
3. `ScheduledWaveform` plugs into `SceneConfig.signal_pool` without changes to `Composer`.
4. `SignalDescription.mode` field added with no broken callers — full existing test suite passes.
5. User-facing docs page `docs/user-guide/multifunction-emitters.md` exists covering the four examples and the schedule types; referenced from `mkdocs.yml` nav.
6. `README.md` and `CLAUDE.md` architecture sections mention the new `waveforms/multifunction/` subpackage.

## Risks & Mitigations

- **Bandwidth heuristic for stochastic schedules is wrong.** The 100 ms preview at `seed=0` may not observe the widest-bandwidth mode, causing `Composer` to under-allocate spectral space. Mitigation: (a) users can always pass `default_bandwidth_hz` explicitly; (b) the `+ abs(freq_offset_hz) * 2` factor covers both sides; (c) test `test_bandwidth_heuristic_stochastic` at minimum asserts the heuristic is ≥ the observed max.
- **Per-segment frequency offset semantics.** Putting `freq_offset_hz` at the baseband level (rather than the emitter carrier level) could be philosophically confusing. Mitigation: the `SignalDescription.f_low`/`f_high` (and derived `f_center`) for each segment are explicitly relative to the emitter carrier, and this is documented in `SegmentSpec`'s docstring and the user-guide page.
- **Closed-loop seam might need revision once a real cognitive scheduler is built.** Mitigation: we commit only to the abstract interface of `next_segment(history, state)` and test it with a trivial subclass. When the real scene-coupled version is designed, we accept that the ABC may need extension — this spec does not promise stability of the `CognitiveSchedule` API.
- **`generate_with_segments` is not part of the `Waveform` ABC.** Callers who want segments must `isinstance`-check. Mitigation: document this in the `ScheduledWaveform` docstring; consider a `Protocol` (`HasSegments`) in a future iteration if more waveforms start producing segment lists.
