# Reviewer Tutorial for the Verification Suite — Design

**Status:** Draft for review
**Date:** 2026-05-11
**Author:** brainstormed with Claude Code
**Topic:** A second, pedagogy-focused companion to `examples/verification/` aimed at a skeptical RF / communications reviewer. The existing suite proves correctness; this tutorial earns the reader's trust in how that proof is constructed.

## Motivation

The verification suite (`examples/verification/`) is a results dump: per-waveform pass/fail tables and asset PNGs. It is *evidentiary* — every claim cites the literature, every tolerance is grounded — but it is not pedagogically self-explanatory. A reviewer landing in the suite for the first time sees what passed, not *why* the methodology should be trusted.

This tutorial adds a second-layer artifact next to the suite: a narrative notebook (and companion script) that walks a skeptical reviewer through the methodology end-to-end. The argument structure is:

1. **The suite has already caught real bugs in this codebase** — open by referencing PR #4 (GMSK h_eff → h = 0.5) and PR #6 (16-QAM row-major → Gray-coded). These are not hypothetical. The framework caught these.
2. **Canonical proof walkthrough — BPSK** — every check from the existing `verify_bpsk.py`, reimplemented inline with the math beside the code and the tolerance derived from CLT / Welch variance. Then a regression catalog: deliberately corrupt the signal and watch the suite catch the corruption.
3. **Same methodology, different math — OFDM** — show the framework adapts: subcarrier orthogonality (exact equality), CP correlation (closed-form), EVM with ZF equalisation. Same regression catalog pattern.
4. **Same methodology, exact-equality reference — Barker-13** — the strictest evidence type. Single-bit corruption → measurable PSLR degradation. No statistical tolerance needed.

A reviewer who finishes this notebook should be able to (a) explain why the suite's tolerances are what they are, (b) describe the failure mode each check guards against, and (c) write a new verifier for a new waveform from scratch.

## Scope

### In scope (first cut)

- A narrative Jupyter notebook (`examples/verification/tutorial_for_reviewers.ipynb`) with prose, equations, and inline code for three waveforms: **BPSK**, **OFDM**, **Barker-13**.
- A companion Python script (`examples/verification/tutorial_for_reviewers.py`) that exposes every demonstrated check as a top-level callable function with the prose moved into docstrings. Smoke-tested against the notebook to ensure numeric equivalence.
- A regression-injection module (`examples/verification/_tutorial_regressions.py`, example-local, not part of the public package surface) with two layered mechanisms:
  - **Post-generation IQ corruption helpers** — pure-Python mutations applied to a clean `sp.X` output (phase rotation, CRC byte flip, CP sample drop, symbol scramble, pulse-shape broadening).
  - **`Buggy*` waveform subclasses** — `BuggyBPSK`, `BuggyOFDM`, `BuggyBarker13` that override `generate()` to introduce specific upstream defects (wrong rolloff, omitted CP, flipped chip).
- A `pytest --nbmake` smoke test that executes the notebook end-to-end with `FULL=False` and asserts the script and notebook agree on every numeric value.
- Light updates to `examples/verification/README.md` linking to the tutorial.

### Out of scope (deferred)

- Coverage expansion to waveforms not currently in the verification suite (FMCW, NLFM, Mode S, AIS, ACARS, DSSS, AM/FM, etc.). The original brainstorm flagged a separate "coverage" track; this spec is the pedagogy track only.
- The coherent BT=0.3 GMSK BER receiver (Laurent / Viterbi). Tracked in the parent spec's Discovered Work.
- Non-waveform verification (impairments, propagation, antennas, datasets). Out of scope for the verification suite itself, let alone this tutorial.
- A rendered HTML version. The notebook is the source of truth; if a static site comes later, it ships separately.

## Architecture

### Audience and tone

Single audience: a working RF / communications engineer doing a critical review of someone else's verification work. Knows the math, doesn't know the SPECTRA codebase, came in skeptical. Tone is professional and direct — no hand-holding, no overstatement, no "this powerful framework" marketing.

Each section follows the same micro-structure:

1. **The property** — one sentence, with citation.
2. **The closed-form reference** — equation block.
3. **The measurement** — Python code, executable in the notebook, ≤ 30 lines.
4. **The tolerance** — derivation: how was the threshold picked? CLT? Welch variance? Pure float round-off?
5. **The regression** — inject a fault, re-run the check, show what the check produces in the fault state. Confirm the fault is caught (or, instructively, that this particular check *misses* the fault — see §"Layering" below).

### Story arc

The notebook is organised as a top-down narrative:

- **§0 — How to read this tutorial.** Two paragraphs: what "verification" means here, what tier each check belongs to (`P*` property vs `S*` statistical), how to interpret pass/fail rows in the existing suite, what citation keys resolve to.
- **§1 — The suite has caught real bugs.** Two short subsections, one per merged bug fix, with concrete evidence:
  - GMSK `h_eff = 0.5/sps` → `h = 0.5` (linked to PR #4 / commit `f034fb6`).
  - 16-QAM row-major → Gray-coded (linked to PR #6 / commit `85a4154`).
  Each subsection ends with: *"the suite caught this because of `P2` / `P3` — the same kind of check we're about to walk through for BPSK."*
- **§2 — Canonical proof walkthrough: BPSK.** Full anatomy of `verify_bpsk.py` reimplemented inline. Constellation, PSD, BER. Tolerance derivations. Then a regression catalog with at least one *miss* (a fault one of the checks doesn't catch, motivating the need for layered checks).
- **§3 — Same methodology, different math: OFDM.** Subcarrier orthogonality (exact equality, no tolerance), CP correlation peak, EVM at high SNR. Regression catalog focuses on faults specific to multicarrier (dropped CP sample, missing IFFT phase ramp).
- **§4 — Same methodology, exact-equality reference: Barker-13.** Sequence equality with Levanon Tab. 6.1, PSLR exactly 13, detection rate at SNR 10 dB. Regression catalog includes single-chip flips that produce measurable PSLR degradation.
- **§5 — How to add a new verifier.** A short checklist distilled from §2–4: pick one property check with a citation, pick one performance check with a citation, expose `properties()` and `performance(full)`, write at least one regression to prove your own check is alive.

### Layering — why mixed mechanisms

The regression-injection mechanism is *deliberately layered*. The two mechanisms model different fault classes:

- **Post-generation IQ corruption** models *transmission-style* faults — channel impairments, receiver-side bugs, post-processing errors. The signal generator is correct; the IQ stream gets perturbed after the fact.
- **`Buggy*` subclasses** model *generator-side* faults — the kind of defects the suite is actually for. Wrong rolloff, omitted CP, flipped chip in a coded waveform. The fault is upstream of the IQ samples.

The reviewer needs to see both. A reviewer who only sees post-IQ corruption could reasonably conclude the suite isn't testing what its proof script claims to test. A reviewer who only sees `Buggy*` subclasses misses the point that some real generator-side bugs (like the GMSK upsample bug) produce a clean-looking IQ stream that nonetheless fails a downstream measurement.

The tutorial uses both in each waveform section, identified explicitly so the reviewer can categorise.

### Code relationship to the existing suite

Every check the tutorial demonstrates exists in the `verify_*.py` suite already. The tutorial **reimplements each check inline** rather than calling into the helpers, with one exception: at the end of each waveform section, the tutorial calls the corresponding `properties()` / `performance()` from `examples/verification/verify_<wf>.py` and asserts numeric equivalence with the inline version. This proves the tutorial isn't a separate codebase pretending to be the suite.

The duplication is intentional and is the source of the tutorial's value. A reviewer reading `verify_bpsk.py` sees `_welch_psd(iq, fs, nperseg=512)` — a black-box call. Reading the tutorial they see Welch's method implemented in eight lines with the segment-averaging variance derivation in a markdown cell above it. Two artifacts, same numbers, different purposes.

### File layout

```
examples/verification/
  tutorial_for_reviewers.ipynb        # narrative notebook (source of truth)
  tutorial_for_reviewers.py           # companion script — every check as a callable function
  _tutorial_regressions.py            # injection helpers + Buggy* subclasses (example-local)
  README.md                           # add a "Tutorial" section linking to the above
tests/verification/
  test_tutorial_for_reviewers.py      # nbmake smoke + numeric-parity assertions
```

The tutorial assets share `assets/verification/` if any new figures are saved. The existing per-waveform PNGs are reused for the reference comparisons; the tutorial does not regenerate them.

### Notebook section sizes

Approximate target word counts (prose only, not code):

- §0 — 250 words
- §1 — 200 words per bug (~400 total)
- §2 — 1500 words (the deep walkthrough)
- §3 — 800 words (compressed; reader trusts the framework by now)
- §4 — 600 words (further compressed; mostly tables)
- §5 — 400 words

Total prose ~4000 words. Plus equation blocks, code cells, and result tables.

## Components

### `tutorial_for_reviewers.ipynb`

Six top-level sections per the story arc. Each waveform section has:

- a **theoretical lead-in** (markdown, ≤ 200 words),
- one **inline check** per property (each its own code cell),
- a **regression catalog** as a single results table generated from a fault loop,
- an **equivalence assertion** against the corresponding `verify_<wf>.py`.

Notebook uses `matplotlib` (already a dev dep) for any inline plots. Figures are rendered inline with `%matplotlib inline`, not saved to disk — the existing suite PNGs cover the saved-figure use case.

### `tutorial_for_reviewers.py`

Mirror of the notebook with prose moved into docstrings. Every check the notebook demonstrates is a top-level function with a typed signature. Example:

```python
def bpsk_psd_correlation_check(iq: np.ndarray, fs: float,
                                rolloff: float = 0.35) -> tuple[float, float]:
    """Welch PSD vs squared-RRC mask; returns (measured_corr, tolerance)."""
    ...
```

Useful for:
- Reviewers who want CLI execution without a notebook runtime.
- The numeric-parity smoke test (which calls the same functions the notebook does).

The script's `__main__` block runs every check end-to-end and prints a summary table, exiting non-zero on any failure.

### `_tutorial_regressions.py`

Two clearly-separated sections.

#### Section A: post-generation IQ corruption helpers

Pure functions on `np.ndarray[complex64]`. Each has a one-line docstring naming the fault class:

```python
def rotate_phase(iq, radians): ...          # constant phase rotation
def drop_cp_sample(iq, n_fft, cp_len): ...  # remove one CP sample per OFDM symbol
def flip_chip(iq, samples_per_chip, k): ... # invert chip k in a chip-coded waveform
def broaden_pulse(iq, blur_kernel_len): ... # apply a moving-average smear
def scramble_random(iq, n_samples, seed): ...
```

#### Section B: `Buggy*` waveform subclasses

Thin subclasses of the corresponding `sp.X` that override `generate()` to introduce a specific defect. Each defect is named:

```python
class BuggyBPSK_WrongRolloff(BPSK):
    """RRC rolloff bumped from 0.35 to 0.5. PSD correlation should drop."""
    def __init__(self, **kwargs): super().__init__(rolloff=0.5, **kwargs)

class BuggyBPSK_NoRRC(BPSK):
    """Pulse-shape filter omitted. Constellation is fine; PSD is wrong."""
    def generate(self, *args, **kwargs): ...

class BuggyOFDM_MissingCP(OFDM):
    """Cyclic prefix not prepended. CP correlation peak vanishes."""
    def generate(self, *args, **kwargs): ...

class BuggyBarker13_FlippedChip(Barker13):
    """Chip 7 inverted. PSLR drops by ~1.4 dB."""
    def generate(self, *args, **kwargs): ...
```

Subclass count target: ~8–12 across the three waveforms. Each is named after the specific defect, not the waveform — naming the fault makes the regression catalog table self-explanatory.

### `test_tutorial_for_reviewers.py`

Two test methods:

1. **`test_notebook_executes`** — `pytest --nbmake examples/verification/tutorial_for_reviewers.ipynb`-style smoke. Notebook must run start to finish with `FULL=False` and exit 0.
2. **`test_script_matches_notebook`** — calls the script's `__main__`-level functions, captures their printed results, and asserts the numeric values match a small pinned reference (e.g., BPSK PSD correlation > 0.99, OFDM EVM < 2%, Barker-13 PSLR == 13.0). The reference values are robust to tolerance widths so the test isn't brittle to small RNG changes.

Both are marked `@pytest.mark.verification` and `@pytest.mark.slow` because they execute the full notebook.

## Data flow

Each waveform section follows the same flow:

```
sp.X(...).generate(seed=K) ──┬─> inline checks (notebook + script)
                              │
                              └─> _verify_helpers / verify_<wf>.properties()
                                       ↓
                                  numeric-equivalence assertion
                                       ↓
                                       ✓
```

Plus the regression branch:

```
sp.X(...).generate(seed=K) ──> _tutorial_regressions.rotate_phase(...)
                                   │
                                   └─> inline checks → expected to fail / detect
                                          ↓
                                     results table

BuggyX(...).generate(seed=K) ──> inline checks → expected to fail / detect
                                          ↓
                                     results table
```

Both branches use the same RNG seed across baseline and faulted runs so the only variable is the injected defect.

## Error handling

This is a tutorial, not a library. Error handling is:

- Notebook cells fail loudly on any unmet check (uncaught `AssertionError` is the right behavior — the reader sees what failed).
- The script's `__main__` exits non-zero on any failure and prints a summary table.
- Regression injection helpers do *not* assert their own correctness — that's the reader's job in the next cell.

## Testing

- `pytest -m verification tests/verification/test_tutorial_for_reviewers.py -v` — notebook smoke and script-notebook numeric parity.
- `pytest -m "verification and slow" tests/verification/test_tutorial_for_reviewers.py -v` — full notebook execution including any `FULL=True` paths (initially, the notebook always runs with `FULL=False`; the slow mark is forward-looking for future `--full` paths).
- Manual: `python examples/verification/tutorial_for_reviewers.py` — runs the script CLI and emits the result table.

## Risks

- **Notebook execution time.** Even with `FULL=False`, the regression catalog runs every check ~5×. Targeting ≤ 30 s on a modern laptop for the whole notebook with `FULL=False`; ≤ 5 min with `FULL=True`. If we exceed this, drop the BER regressions (the longest-running checks) and rely on PSD / structural ones.
- **Drift from the verification suite.** The tutorial reimplements checks inline. If the suite's helpers (`_welch_psd`, `measure_obw`, etc.) get edited and the tutorial doesn't, the equivalence assertion will catch the divergence — that's by design. The numeric-parity test is the regression guard.
- **Skeptical reviewers vs newcomers.** The tutorial is aimed at skeptical reviewers, not newcomers. If a contributor needs a "how to add a verifier" walkthrough only, §5 is the relevant piece; the first four sections may feel like overkill. Acceptable — adding a separate contributor doc would scope-creep this PR.
- **Asset stability.** The tutorial cross-references the existing `verify_*.py` numeric outputs. If those outputs drift (e.g., a new verifier change tightens a tolerance), the tutorial's equivalence assertion may need a minor tweak. Caught by the numeric-parity test.

## Discovered work (out of scope, not blocking)

- A static HTML rendering of the notebook for the docs site (`mkdocs` would render this if we add it). Defer to a docs sprint.
- A second tutorial aimed at *contributors* ("how to add a verifier in 30 lines") that lifts §5 and expands it. The original brainstorm noted this as a distinct audience; bundling here would muddy the tone.
- Tutorial expansion to additional waveforms (PSK family beyond BPSK, QAM family beyond what §2 hints at, radar codes beyond Barker-13). Mechanical once the pattern is established; not in this PR.
