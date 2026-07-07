# LFM Up/Down/Random Chirp Direction — Design

**Date:** 2026-07-07
**Status:** Approved, ready for implementation planning

## Summary

Add configurable chirp direction to the `LFM` waveform so it can produce
up-chirps (current behavior), down-chirps, or a per-call randomized choice
between the two. Randomization is seeded so it remains reproducible and
DataLoader-worker-safe under SPECTRA's `(seed, idx)` convention.

## Motivation

The `LFM` waveform (`python/spectra/waveforms/lfm.py`) currently hardcodes an
up-chirp: it always sweeps `f0, f1 = -bw/2, +bw/2`. Real LFM emitters use both
up- and down-chirps, and during dataset generation it is useful to randomize
the direction as an intra-class augmentation so classifiers learn
direction-invariant LFM features.

The underlying Rust primitive `generate_chirp(duration, fs, f0, f1)` already
accepts arbitrary sweep endpoints, so a down-chirp is simply the up-chirp with
`f0` and `f1` swapped. This is therefore a Python-layer API change with no Rust
changes required, preserving the design rule that Rust functions stay stateless
and Python owns all randomness.

## API

Add one parameter to `LFM.__init__`:

```python
LFM(..., direction: str = "up")
```

Accepted values:

- `"up"` — default; preserves existing behavior exactly.
- `"down"` — reversed sweep.
- `"random"` — one seeded coin flip per `generate()` call selects up or down.

Any other value raises `ValueError` at construction time, consistent with the
existing mutually-exclusive-argument validation already done in the constructor.

## Behavior

`generate()` resolves the sweep endpoints from the bandwidth `bw`, then passes
them to the existing Rust `generate_chirp`:

- `"up"`   → `f0, f1 = -bw/2, +bw/2` (unchanged)
- `"down"` → `f0, f1 = +bw/2, -bw/2`
- `"random"` → seed a NumPy `Generator` from the `seed` argument, draw a single
  boolean per `generate()` call, and apply the resulting direction to **all**
  `num_symbols` pulses in that call.

No Rust changes. `bandwidth()`, output length, and `num_symbols_for()` are
independent of direction.

## Label

`label` remains `"LFM"` for all directions. Up and down chirps are the same
class; `"random"` is intra-class augmentation. This keeps `label` a stable
property rather than something that would have to be resolved per `generate()`
call.

## Randomness & determinism

The `"random"` coin flip derives solely from the `seed` passed to `generate()`
(e.g. `np.random.default_rng(seed)`), making it reproducible and
worker-safe under the `(seed, idx)` seeding convention used across SPECTRA
datasets. When `seed is None`, `"random"` falls back to a nondeterministic
draw, matching how sibling waveforms treat `seed=None` (to be verified against
an existing waveform during implementation).

## Testing

- `direction="up"` output is byte-identical to the current `LFM()` output
  (regression guard).
- `direction="down"` produces the frequency-reversed sweep — its instantaneous
  frequency slope is negative and mirrors the up-chirp.
- `direction="random"` with a fixed `seed` is deterministic and reproducible
  across repeated calls; different seeds can produce different directions.
- Invalid `direction` raises `ValueError`.
- `bandwidth()`, `label`, and output length are unaffected by direction.

## Out of scope (YAGNI)

- Per-pulse alternation within a single `generate()` call.
- Intra-pulse "V"/triangle (up-then-down) chirps — FMCW's `"triangle"` sweep
  already covers that use case.
- Distinct class labels for up vs. down chirps.
