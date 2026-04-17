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


def low_prf_factory(rng):
    return PulsedRadar(pri_samples=4096)


def medium_prf_factory(rng):
    return PulsedRadar(pri_samples=1024)


def high_prf_factory(rng):
    return PulsedRadar(pri_samples=256)


modes = {
    "low_prf":    ModeSpec(waveform_factory=low_prf_factory,    duration_samples=(2048, 16384)),
    "medium_prf": ModeSpec(waveform_factory=medium_prf_factory, duration_samples=(2048, 16384)),
    "high_prf":   ModeSpec(waveform_factory=high_prf_factory,   duration_samples=(2048, 16384)),
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

See the design spec at `docs/superpowers/specs/2026-04-17-multifunction-emitter-design.md`
for full details.
