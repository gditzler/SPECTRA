# Waveform Parameter Realism — Design

**Date:** 2026-07-02
**Status:** Approved design, pending implementation plan

## Problem

SPECTRA waveforms are parameterized in the sample domain (`samples_per_symbol=8`,
`pulse_width_samples=64`, `bandwidth_fraction=0.5`). A waveform's physical identity —
symbol rate, deviation, PRI, sweep bandwidth — is therefore an accident of the capture
sample rate. Consequences:

- At a given sample rate, every comms signal has nearly the same bandwidth (all default
  `sps=8`), so wideband scenes are unrealistically homogeneous.
- Users cannot express "P25 at 4.8 kBd" or "marine radar with 250 ns pulses" directly;
  they must hand-convert to samples for each capture configuration.
- Generated data does not statistically resemble real emitter populations, hurting
  transfer of trained models to over-the-air captures.

## Goals

1. **Physical-unit parameterization** — users specify baud, Hz, and seconds; the library
   derives sample-domain values from the sample rate.
2. **Transfer to real captures** — a curated registry of standards-referenced emitter
   profiles with *sampleable parameter distributions*, so scenes are realistic and diverse.

**Scope (phase 1):** comms core (PSK/QAM/ASK via RRC base, FSK/MSK/GMSK/GFSK, OFDM) and
radar core (PulsedRadar, FMCW/LFM/NLFM, Barker/polyphase coded). Protocol waveforms
(ADS-B, AIS frames, NR) already encode their standards and are out of scope, except that
an AIS *modulation-level* profile (GMSK 9.6 kBd) is included in the registry.

**Compatibility:** strictly additive. All existing constructors, benchmark YAMLs, and
tests keep working unchanged; legacy code paths produce byte-identical IQ.

## Non-goals

- No RF-band/carrier-frequency modeling (scenes remain baseband-relative; the
  `environment/` layer owns geometry and links).
- No protocol framing realism (bit-level content, CRCs) beyond what exists.
- No migration of existing benchmarks to profiles in this phase.

## Design

### Layer 1: Physical units on waveform classes

Optional kwargs, default `None` (legacy behavior when absent):

| Family | New kwargs | Derivation at `generate()` |
|---|---|---|
| PSK/QAM/ASK (`rrc_base`) | `symbol_rate` (baud) | `sps = fs / symbol_rate` |
| FSK/MSK/GMSK/GFSK | `symbol_rate` (baud), `deviation` (Hz, peak) | `h = 2 * deviation / symbol_rate` |
| OFDM | `subcarrier_spacing` (Hz) | `fft_size = fs / spacing` |
| PulsedRadar, Barker, polyphase | `pulse_width` (s), `pri` (s), `chip_rate` (Hz) | `samples = round(x * fs)` |
| FMCW/LFM/NLFM | `sweep_bandwidth` (Hz), `sweep_time` (s) | `fraction = B / fs` |

Rules:

- **Precedence:** a physical kwarg wins over its sample-domain counterpart. Passing a
  conflicting pair (e.g. `symbol_rate` and `samples_per_symbol`) raises `ValueError` at
  construction.
- **Fractional rates:** generate at the nearest integer sps; if the resulting symbol-rate
  error exceeds 1%, generate at a convenient integer sps and rational-resample to the
  exact rate with `utils.dsp.multistage_resampler`. Below 1%, round silently.
- **`bandwidth(fs)` becomes fs-independent** when physical params are set:
  `symbol_rate*(1+rolloff)` (RRC), `(M-1)*tone_spacing + 2*symbol_rate` (FSK, using the
  corrected level convention), `N_active * spacing` (OFDM), `sweep_bandwidth` (FMCW/LFM),
  `chip_rate` (coded). This is what makes wideband bounding boxes intrinsically correct.
- **Composer contract:** `Waveform` gains `num_symbols_for(num_samples, sample_rate)`
  with a base-class default reproducing today's
  `num_samples // samples_per_symbol`; physically-parameterized waveforms override it.
  `Composer` calls the method instead of `getattr(waveform, "samples_per_symbol", 8)`.
- **Validation at `generate()`** (fs first known there): occupied bandwidth <= fs,
  effective sps >= 2, `pri >= pulse_width`, `deviation < fs/2`, pulse width >= 1 sample
  (warn below 4 samples).

### Layer 2: Emitter-profile registry (`python/spectra/profiles/`)

- **`ParamSpec`** distribution vocabulary: `Fixed(v)`, `Choice([...])`,
  `Uniform(lo, hi)`, `LogUniform(lo, hi)`. Sampled with a caller-provided
  `numpy.random.Generator`; no new dependencies.
- **`EmitterProfile`** (frozen dataclass): `name` (registry key), `label` (dataset class
  label), `waveform_cls`, `params: dict[str, ParamSpec]`, `reference` (one-line standard
  citation).
- **`sample(rng, sample_rate) -> Waveform`**: draw params, construct the waveform with
  physical kwargs, validate representability. If the drawn set exceeds capture limits,
  re-draw from the representable subrange when non-empty; otherwise raise
  `ProfileNotRepresentable` naming the profile and minimum viable sample rate. Profiles
  never silently distort their parameters.
- **Registry API:** `spectra.profiles.get(name)`, `list_profiles()`, `register(profile)`.
  Unknown names raise with near-match suggestions. ParamSpec bounds are validated at
  import time so a typo'd profile fails loudly, not mid-training.

**Initial curated set (~16, each with a cited reference):**

- Comms: BLE 1M, BLE 2M (GFSK, h in [0.28, 0.35], BT=0.5), Bluetooth BR (GFSK),
  P25 Phase 1 C4FM (4FSK 4.8 kBd), DMR (4FSK 9.6 kBd), TETRA (pi/4-DQPSK 18 kBd,
  approximated as QPSK), POCSAG (FSK +/-4.5 kHz), AIS (GMSK 9.6 kBd BT=0.4),
  Wi-Fi-like 20 MHz OFDM (312.5 kHz spacing, 52 active), LTE-like OFDM (15 kHz spacing),
  DVB-S QPSK (rolloff 0.35).
- Radar: marine navigation pulsed, weather surveillance pulsed (WSR-88D-class),
  ATC surveillance pulsed, automotive FMCW (baseband sweep), radar altimeter FMCW,
  Barker-13 pulse compression.

### Integration

- `SceneConfig.signal_pool` accepts `Waveform | EmitterProfile`. When the Composer draws
  a profile it calls `profile.sample(rng, cfg.sample_rate)` using the scene RNG, so
  results stay deterministic under the `(seed, idx)` DataLoader-worker rule.
  `SignalDescription.label` comes from `profile.label`.
- Benchmark YAML loader accepts `- {profile: "bluetooth-le"}` alongside `{type: ...}`.

### Error handling summary

- Conflicting kwargs: `ValueError` at construction.
- Unphysical parameters: `ValueError` at `generate()` with the failed constraint.
- Unrepresentable profile at given fs: `ProfileNotRepresentable` with minimum fs.
- Registry integrity: validated at import.

### Testing

1. **Regression:** legacy constructors with fixed seeds produce byte-identical IQ
   (physical path fully dormant when kwargs absent).
2. **Unit:** derivation math (sps, h, fft_size, sample counts), precedence errors,
   validation errors, resample-threshold behavior, registry integrity, profile
   determinism under `(seed, idx)`.
3. **Spectral (marked `slow`):** for every profile and physically-parameterized
   waveform, Welch-PSD 99% occupied bandwidth must agree with `bandwidth(fs)` within a
   family-specific epsilon. This encodes the 2026-07-02 bandwidth verification harness
   as a permanent test.

### Documentation

- New user-guide page `realistic-emitters` (physical units, registry, scene example).
- `examples/datasets/wideband_scenes` gains a profile-based scene section.

## Dependencies

Builds on the two in-flight fixes: FSK level-convention/bandwidth and GFSK mod-index
attenuation (task_0885f2c0), OFDM pilot/guard bandwidth semantics (task_3ccc50b2). The
spectral test epsilon assumes corrected `bandwidth()` behavior; FSK/GMSK/OFDM profiles
should not merge before those fixes land.

## Decisions log

- Approach A chosen over profile-only layer (B) and parallel emitter hierarchy (C).
- Profiles carry distributions, not fixed nominals.
- Additive-only compatibility; no deprecations in this phase.
- Round-vs-resample threshold: 1% symbol-rate error.
- Conflicting physical/sample kwargs are a hard error, not a precedence rule.
