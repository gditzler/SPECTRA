# LFM Up/Down/Random Chirp Direction Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Add a `direction` parameter to the `LFM` waveform so it can generate up-chirps (default, unchanged), down-chirps, or a per-call seeded-random choice between the two.

**Architecture:** Pure Python-layer change in `python/spectra/waveforms/lfm.py`. The existing Rust primitive `generate_chirp(duration, fs, f0, f1)` already accepts arbitrary sweep endpoints, so a down-chirp is the up-chirp with `f0`/`f1` swapped. `LFM.__init__` validates the new `direction` argument; `LFM.generate` resolves the endpoints (using the `seed` for the random case) before calling `generate_chirp`. No Rust changes.

**Tech Stack:** Python 3.10+, NumPy, PyO3 Rust extension (`spectra._rust`), pytest.

## Global Constraints

- No Rust changes — `generate_chirp` is used as-is.
- Backward compatibility: default `direction="up"` must produce byte-identical output to the current `LFM()`.
- `label` stays `"LFM"` for all directions.
- Randomness derives solely from the `seed` passed to `generate()`, using the repo convention: `s = seed if seed is not None else np.random.randint(0, 2**32)` then `rng = np.random.default_rng(s)`.
- Random direction is decided once per `generate()` call and applied to all `num_symbols` pulses.
- Ruff: `line-length = 100`, `target-version = "py310"`.

---

## File Structure

- Modify: `python/spectra/waveforms/lfm.py` — add `direction` param, validation, endpoint resolution.
- Modify: `tests/test_waveforms_lfm.py` — add tests for down/random/validation/regression.

Both changes are tightly coupled to one small class, so this is a single task with a TDD cycle.

---

### Task 1: Add `direction` parameter to `LFM`

**Files:**
- Modify: `python/spectra/waveforms/lfm.py`
- Test: `tests/test_waveforms_lfm.py`

**Interfaces:**
- Consumes: `spectra._rust.generate_chirp(duration: float, fs: float, f0: float, f1: float) -> np.ndarray[complex64]` (existing, unchanged).
- Produces:
  - `LFM.__init__(self, bandwidth_fraction=None, samples_per_pulse=None, sweep_bandwidth=None, pulse_duration=None, direction="up")` — raises `ValueError` if `direction` not in `{"up", "down", "random"}`.
  - `LFM.generate(self, num_symbols, sample_rate, seed=None) -> np.ndarray[complex64]` — behavior unchanged for `direction="up"`; `"down"` swaps endpoints; `"random"` picks one direction per call from `seed`.
  - `LFM.label` stays `"LFM"`.

- [ ] **Step 1: Write the failing tests**

Add these tests as methods on the existing `TestLFMWaveform` class in `tests/test_waveforms_lfm.py`, matching its indentation. Two file conventions to follow exactly:
- The module top already imports `numpy as np`, `numpy.testing as npt`, and `pytest` — reuse those; do NOT re-import them inside methods.
- Each existing test imports `LFM` locally with `from spectra.waveforms.lfm import LFM` as its first line. Follow that pattern: put `from spectra.waveforms.lfm import LFM` as the first line inside each new method below.

```python
def test_direction_default_is_up_regression(self, sample_rate):
    """Default direction must be byte-identical to legacy up-chirp output."""
    from spectra.waveforms.lfm import LFM

    legacy = LFM(samples_per_pulse=128)
    explicit_up = LFM(samples_per_pulse=128, direction="up")
    a = legacy.generate(num_symbols=3, sample_rate=sample_rate, seed=7)
    b = explicit_up.generate(num_symbols=3, sample_rate=sample_rate, seed=7)
    np.testing.assert_array_equal(a, b)

def test_down_chirp_is_frequency_reversed(self, sample_rate):
    """Down-chirp equals the conjugate of the up-chirp (mirrored instantaneous frequency)."""
    from spectra.waveforms.lfm import LFM

    up = LFM(samples_per_pulse=256, direction="up").generate(1, sample_rate, seed=1)
    down = LFM(samples_per_pulse=256, direction="down").generate(1, sample_rate, seed=1)
    np.testing.assert_allclose(down, np.conj(up), rtol=1e-5, atol=1e-5)

def test_down_chirp_has_negative_slope(self, sample_rate):
    """Instantaneous frequency of a down-chirp decreases over the pulse."""
    from spectra.waveforms.lfm import LFM

    down = LFM(samples_per_pulse=256, direction="down").generate(1, sample_rate, seed=1)
    inst_phase = np.unwrap(np.angle(down))
    inst_freq = np.diff(inst_phase)
    # Later half sweeps lower in frequency than the earlier half.
    assert inst_freq[-1] < inst_freq[0]

def test_random_direction_is_deterministic(self, sample_rate):
    """Same seed reproduces the same randomized direction/output."""
    from spectra.waveforms.lfm import LFM

    wf = LFM(samples_per_pulse=128, direction="random")
    a = wf.generate(num_symbols=2, sample_rate=sample_rate, seed=42)
    b = wf.generate(num_symbols=2, sample_rate=sample_rate, seed=42)
    np.testing.assert_array_equal(a, b)

def test_random_direction_matches_up_or_down(self, sample_rate):
    """A randomized call is exactly one of the two deterministic directions."""
    from spectra.waveforms.lfm import LFM

    rnd = LFM(samples_per_pulse=128, direction="random").generate(1, sample_rate, seed=3)
    up = LFM(samples_per_pulse=128, direction="up").generate(1, sample_rate, seed=3)
    down = LFM(samples_per_pulse=128, direction="down").generate(1, sample_rate, seed=3)
    matches_up = np.array_equal(rnd, up)
    matches_down = np.array_equal(rnd, down)
    assert matches_up or matches_down

def test_random_direction_varies_across_seeds(self, sample_rate):
    """Across many seeds, the randomized mode produces both directions."""
    from spectra.waveforms.lfm import LFM

    up_wf = LFM(samples_per_pulse=64, direction="up")
    rnd_wf = LFM(samples_per_pulse=64, direction="random")
    saw_up = saw_down = False
    for s in range(20):
        # The pulse content is seed-independent (no data bits), so any fixed
        # seed's up-chirp is the reference for "is this call an up-chirp?".
        up_ref = up_wf.generate(1, sample_rate, seed=s)
        is_up = np.array_equal(rnd_wf.generate(1, sample_rate, seed=s), up_ref)
        saw_up |= is_up
        saw_down |= not is_up
    assert saw_up and saw_down

def test_invalid_direction_raises(self):
    """Unknown direction is rejected at construction time."""
    from spectra.waveforms.lfm import LFM

    with pytest.raises(ValueError):
        LFM(direction="sideways")

def test_label_unaffected_by_direction(self):
    from spectra.waveforms.lfm import LFM

    assert LFM(direction="down").label == "LFM"
    assert LFM(direction="random").label == "LFM"
```

Note: these are methods on the existing `TestLFMWaveform` (or equivalently named) class in that file — add them inside that class, matching the existing indentation and `self`/fixture style. If the file uses module-level functions instead of a class, drop the `self` parameter and adapt accordingly; check the file first.

- [ ] **Step 2: Run the tests to verify they fail**

Run: `pytest tests/test_waveforms_lfm.py -v -k "direction or down_chirp or random or label_unaffected or invalid_direction"`
Expected: FAIL — `__init__() got an unexpected keyword argument 'direction'` (and `TypeError`/`ValueError` mismatches).

- [ ] **Step 3: Implement the `direction` parameter**

Edit `python/spectra/waveforms/lfm.py`. Add the import for the random fallback, add `direction` to `__init__` with validation, and resolve endpoints in `generate`.

Replace the constructor signature and body (currently lines 16–31) so it accepts and validates `direction`:

```python
    _VALID_DIRECTIONS = ("up", "down", "random")

    def __init__(
        self,
        bandwidth_fraction: Optional[float] = None,
        samples_per_pulse: Optional[int] = None,
        sweep_bandwidth: Optional[float] = None,
        pulse_duration: Optional[float] = None,
        direction: str = "up",
    ):
        if sweep_bandwidth is not None and bandwidth_fraction is not None:
            raise ValueError("sweep_bandwidth and bandwidth_fraction are mutually exclusive")
        if pulse_duration is not None and samples_per_pulse is not None:
            raise ValueError("pulse_duration and samples_per_pulse are mutually exclusive")
        if direction not in self._VALID_DIRECTIONS:
            raise ValueError(
                f"direction must be one of {self._VALID_DIRECTIONS}, got {direction!r}"
            )
        self._direction = direction
        self._sweep_bandwidth = sweep_bandwidth
        self._pulse_duration = pulse_duration
        self._bandwidth_fraction = 0.5 if bandwidth_fraction is None else bandwidth_fraction
        self._samples_per_pulse = 256 if samples_per_pulse is None else samples_per_pulse
        self.samples_per_symbol = self._samples_per_pulse
```

In `generate`, replace the current endpoint line (`f0, f1 = -bw / 2.0, bw / 2.0`, line 49) with direction resolution. The current up-chirp uses `f0, f1 = -bw/2, +bw/2`, so keep that mapping for `"up"`:

```python
        if self._direction == "random":
            s = seed if seed is not None else np.random.randint(0, 2**32)
            rng = np.random.default_rng(s)
            direction = "down" if rng.random() < 0.5 else "up"
        else:
            direction = self._direction

        if direction == "down":
            f0, f1 = bw / 2.0, -bw / 2.0
        else:  # "up"
            f0, f1 = -bw / 2.0, bw / 2.0
```

Leave the rest of `generate` (the `n`/`duration` computation, the `generate_chirp` loop, and the `np.concatenate`) unchanged.

- [ ] **Step 4: Run the tests to verify they pass**

Run: `pytest tests/test_waveforms_lfm.py -v`
Expected: PASS — all existing LFM tests plus the eight new ones.

- [ ] **Step 5: Lint**

Run: `ruff check python/spectra/waveforms/lfm.py tests/test_waveforms_lfm.py && ruff format --check python/spectra/waveforms/lfm.py`
Expected: no errors. (`tests/test_waveforms_lfm.py` may already be in the pre-existing format-debt set — if `ruff format --check` flags only the test file and only for pre-existing reasons, do not reformat unrelated lines; format just the lines you added. Run `ruff format` on `lfm.py` if it reports a diff there.)

- [ ] **Step 6: Commit**

```bash
git add python/spectra/waveforms/lfm.py tests/test_waveforms_lfm.py
git commit -m "feat: add up/down/random chirp direction to LFM waveform"
```

---

## Self-Review

**Spec coverage:**
- API `direction="up"|"down"|"random"` with `ValueError` on invalid → Task 1, Step 3 (`_VALID_DIRECTIONS`, validation) + `test_invalid_direction_raises`.
- `"up"` unchanged / byte-identical → `test_direction_default_is_up_regression`.
- `"down"` frequency-reversed → `test_down_chirp_is_frequency_reversed`, `test_down_chirp_has_negative_slope`.
- `"random"` per-call, seeded, deterministic → `test_random_direction_is_deterministic`, `test_random_direction_matches_up_or_down`, `test_random_direction_varies_across_seeds`.
- Label stays `"LFM"` → `test_label_unaffected_by_direction`.
- `seed=None` fallback follows repo convention → Step 3 uses `s = seed if seed is not None else np.random.randint(0, 2**32)`.
- No Rust changes → confirmed; only `lfm.py` and its test are touched.

**Placeholder scan:** No TBD/TODO; all steps show concrete code and commands.

**Type consistency:** `generate_chirp(duration, fs, f0, f1)` signature matches existing usage; `direction` is a `str`; endpoint mapping for `"up"` matches the original `f0, f1 = -bw/2, +bw/2`.

**Down-chirp verification note:** `test_down_chirp_is_frequency_reversed` asserts `down == conj(up)`. This holds because both share the same start-phase reference (the chirp starts at phase 0 at t=0) and the down-sweep `f(t) = -f_up(t)`, so the complex exponential is conjugated sample-for-sample. If floating-point differences exceed tolerance in practice, the implementer should keep the negative-slope test (`test_down_chirp_has_negative_slope`) as the primary correctness check and relax the conjugate test's tolerance rather than changing the implementation.
