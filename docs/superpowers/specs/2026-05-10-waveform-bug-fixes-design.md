# Waveform Bug Fixes — Design

**Status:** Draft for review
**Date:** 2026-05-10
**Author:** brainstormed with Claude Code
**Topic:** Fix two waveform-generation defects surfaced by the
`2026-05-08-signal-generation-verification` suite: (A) GMSK effective
modulation index is `0.5/sps` instead of `0.5`, and (B) 16-QAM (and all
higher-order square QAM) uses row-major labelling instead of Gray coding.
Both fixes restore textbook behaviour; both let the verification suite
revert its workaround tolerances to citation-grounded values.

## Motivation

The signal-generation verification suite landed two waveforms with
documented deviations from the published standard, accepted as known
characteristics rather than fixed:

1. **GMSK** — the `delta_phi = π·0.5·filtered/sps` expression in
   `python/spectra/waveforms/fsk.py:115`, combined with zero-insertion
   upsampling and a sum-normalised Gaussian filter, produces an effective
   modulation index `h_eff = 0.5/sps = 0.0625` (sps=8). Standard MSK / GMSK
   requires `h = 0.5`. Cascading effects: frequency deviation is 8× too
   small, the Laurent expansion for spectral BW does not apply, the MSK BER
   formula does not apply, and the 99 % OBW is 6× too narrow.
   `verify_gmsk.py` works around all of this — empirical OBW reference,
   spectral-compactness regression guard instead of a BW match, and a BER
   threshold at 40 dB instead of a theory curve over [0, 10] dB.

2. **16-QAM** — `build_qam_constellation` in `rust/src/modulators.rs:27-42`
   sweeps a 2-D index in row-major order, then maps integer symbol `k` to
   the grid point `(2·(k/side) − (side−1), 2·(k%side) − (side−1))`. This is
   not Gray-coded: adjacent integer labels that cross a row boundary differ
   by multiple bits and are not physical nearest neighbours. The BER↔SER
   relationship deviates from the standard `BER ≈ SER/log₂(M)` by up to a
   factor of `log₂(M)` at moderate-to-high SNR. `verify_qam16.py` omits the
   Gray-adjacency check entirely.

Both deviations are real bugs, not design choices. This spec defines two
ordered, independently-mergeable fixes that restore textbook behaviour,
update the verification suite to use textbook tolerances, and (for QAM)
take a clean backward-incompatible break appropriate to SPECTRA's pre-1.0
state.

## Scope

### In scope

- **Track A — GMSK h_eff fix** (lands first as PR-A):
  - Replace zero-insertion upsampling with repeat-upsampling in
    `python/spectra/waveforms/fsk.py:GMSK.generate`.
  - Revert `examples/verification/verify_gmsk.py` workarounds: P2 expected
    step, P4 spectral-BW check, P5 OBW reference; drop S1.
  - Delete the now-orphaned `assets/verification/gmsk_S1_ber.png` (no
    BER check renders a figure).
  - Update the "Notes on Findings" GMSK entry in
    `examples/verification/README.md`.
  - Add a `tests/test_waveforms_fsk.py` regression test asserting
    steady-state `|Δφ|/symbol ≈ π/2` for a constant-`+1` bit sequence.

- **Track B — 16-QAM Gray-coding fix** (lands second as PR-B,
  backward-incompatible):
  - Replace the row-major loop in
    `rust/src/modulators.rs::build_qam_constellation` with the
    Gray-bit-split construction described below.
  - Add a Rust `#[test]` asserting Gray adjacency for M ∈ {16, 64, 256}:
    every pair of constellation points that are physical nearest neighbours
    have integer labels whose Gray codes differ by exactly one bit.
  - Add a `P_gray` property check to
    `examples/verification/verify_qam16.py` and delete the inline "Gray
    adjacency check omitted" note.
  - Tighten S1 SER tolerance and add a BER-vs-SER/log₂(M) high-SNR check.
  - Audit downstream callers (see Audit section).
  - Add a BREAKING-change entry to `CHANGELOG.md` (create file if absent),
    bump version per project policy.
  - Regenerate `assets/verification/qam16_*.png`.

### Out of scope (deferred)

- The "reviewer tutorial" notebook (`tutorial_for_reviewers.ipynb`) and its
  companion script. Tracked separately; will be written *against* the
  post-fix codebase.
- Other modulators that may share the same row-major QAM order (e.g.,
  QAM64, QAM256) — same fix function handles them, but no expansion of the
  verification suite to those orders is required by this spec.
- Other Gaussian-filtered waveforms (GFSK family) — `GFSK.generate` uses
  the same zero-insertion + sum-normalised Gaussian pattern at
  `python/spectra/waveforms/fsk.py:218-224` and may have the same defect.
  This is flagged but not fixed here; see the Discovered Work section.
- Verifier expansion to non-currently-covered waveforms (see the parent
  spec's deferred list).

## Track A — GMSK fix

### Root cause

`python/spectra/waveforms/fsk.py:106-117` currently does:

```python
symbols_up = np.zeros(num_symbols * sps, dtype=np.float32)
symbols_up[::sps] = symbols.real                # impulses at multiples of sps
h = self._gaussian_taps()                       # sum-normalised: Σh = 1
filtered = np.convolve(symbols_up, h, mode="same")
delta_phi = np.pi * 0.5 * filtered / sps        # per-sample phase increment
```

A sum-1 filter preserves the average value of its input. The average of an
impulse train with magnitude ±1 at every `sps`-th sample is `±1/sps`. The
per-symbol total phase change is therefore
`Σ_n delta_phi[n] = π · 0.5 · (1/sps) · sps / sps = π · 0.5 / sps`,
yielding `h_eff = 0.5/sps`.

The neighbouring `MSK.generate` and `FSK.generate` use `np.repeat` instead
of zero-insertion, which is the textbook construction (rectangular
frequency pulse held over the symbol interval).

### Fix

Replace the upsample line; everything else stays the same:

```python
# Repeat-upsample so the frequency-pulse train averages to ±1.
symbols_up = np.repeat(symbols.real.astype(np.float32), sps)
```

After the fix, per-symbol total
`Σ delta_phi = π · 0.5 · sps · (±1) / sps = ±π/2`. Modulation index `h = 0.5`.
The Gaussian filter still shapes the rectangular pulse — that's GMSK by
construction.

### Regression test

`tests/test_waveforms_fsk.py` adds:

```python
def test_gmsk_modulation_index_steady_state():
    """Per-symbol phase change is π·0.5 = π/2 for a constant +1 bit stream."""
    # rationale and expected value documented inline
```

The test drives `sp.GMSK` with a constant input (via a monkey-patched
`generate_bpsk_symbols` returning all-+1) and asserts steady-state
`|Δφ|/symbol` within 1 % of `π/2`. It does not exercise the random-data
path because that's covered by the verifier.

### Verifier updates

`examples/verification/verify_gmsk.py`:

- P2 — expected step `π · 0.5 ≈ 1.5708 rad`, tolerance 1 % relative
  (was `π · 0.0625`). Citation key unchanged: `proakis2008:§4.4-3`.
- P4 — PSD 3-dB BW vs the BT=0.3 / h=0.5 reference `0.27 · R_s` (Laurent
  expansion, third-order term dominant), tolerance 25 % relative. Citation
  `laurent1986:§III`. Note: the `0.5 · R_s` figure that appears in some
  references is the BT = ∞ MSK main-lobe; the Gaussian filter at BT = 0.3
  compresses the main lobe to ≈ 0.27 · R_s.
- P5 — 99 % OBW vs `0.92 · R_s` (BT=0.3 GMSK, GSM/3GPP industry
  reference), tolerance 10 % relative. Citation `itu_sm_328:§3`. Note: the
  `1.5 · R_s` figure is the Carson's-rule peak-deviation bandwidth, not
  the 99 % OBW — they are distinct quantities.
- S1 — **dropped**. A per-bit coherent matched filter is the wrong receiver
  for BT = 0.3 GMSK: the Gaussian-shaped phase pulse spreads ISI over
  ~3 bit intervals, so any per-bit observation loses ~26 dB versus
  Q(√(2·Eb/N0)). The correct coherent receivers are a Laurent-decomposition
  detector or a Viterbi CPM decoder — both real engineering work, out of
  scope for the bug-fix PR. The fix is fully evidenced by P1 (constant
  envelope), P2 (exact π/2 per symbol = h = 0.5), P3 (Gaussian filter BW),
  P4 (PSD 3-dB main lobe), and P5 (99 % OBW). A proper coherent receiver
  + BER curve is tracked as discovered work for a follow-on PR.
- The five-paragraph deviation block at the top of the docstring deletes.
- `assets/verification/gmsk_S1_ber.png` is deleted (no BER row to render).

### README update

`examples/verification/README.md` — the "Notes on Findings" entry for
GMSK changes from "documented characteristic" wording to a single sentence:

> *GMSK previously produced h_eff = 0.5/sps due to a zero-insertion upsample.
> Fixed in PR-A; verifier now uses textbook MSK tolerances.*

The GMSK section's bullet list reverts to the standard MSK / GMSK
properties.

## Track B — 16-QAM Gray-coding fix

### Root cause

`rust/src/modulators.rs:27-42` builds the constellation in row-major order:

```rust
for i in 0..side {
    for j in 0..side {
        let re = 2.0 * i as f64 - (side - 1) as f64;
        let im = 2.0 * j as f64 - (side - 1) as f64;
        constellation.push(Complex32::new(re as f32, im as f32));
    }
}
```

Integer label `k = i·side + j` is placed at grid position `(I=2i−(s−1), Q=2j−(s−1))`.
At row boundaries, adjacent labels (`k=3`→`k=4` for M=16) are not physical
nearest neighbours and differ by multiple bits.

### Fix

Replace with a Gray-bit-split construction:

```rust
fn build_qam_constellation(order: usize) -> Result<Vec<Complex32>, String> {
    let side = (order as f64).sqrt() as usize;
    if side * side != order {
        return Err("QAM order must be a perfect square (16, 64, 256, ...)".to_string());
    }
    let n = (side as f64).log2() as u32;
    if (1usize << n) != side {
        return Err("QAM order must be 2^(2n) (16, 64, 256, 1024)".to_string());
    }
    let mut constellation = vec![Complex32::new(0.0, 0.0); order];
    for i in 0..side {
        for j in 0..side {
            let gi = i ^ (i >> 1);              // I-axis Gray code
            let gj = j ^ (j >> 1);              // Q-axis Gray code
            let label = (gi << n) | gj;         // top n bits = I, bottom n = Q
            let re = 2.0 * i as f64 - (side - 1) as f64;
            let im = 2.0 * j as f64 - (side - 1) as f64;
            constellation[label] = Complex32::new(re as f32, im as f32);
        }
    }
    normalize_constellation(&mut constellation);
    Ok(constellation)
}
```

This works because each axis index is Gray-encoded independently; stepping
by one in either I or J changes exactly one bit of the corresponding half of
the label, and the other half is unchanged. Defining property:
`popcount(label(p1) XOR label(p2)) == 1` for every pair of physical nearest
neighbours `p1, p2`. (Note: a naive "Gray-encode k then bit-split" approach
does *not* produce this property — Gray-coding must be applied per axis.)

### Rust unit test

`rust/src/modulators.rs` adds a `#[test]` that, for each
`M ∈ {16, 64, 256}`, computes the constellation, finds physical
nearest-neighbour pairs (Euclidean distance equal to the minimum spacing),
and asserts each pair's integer labels are Gray-adjacent.

### Audit (must complete before merging Track B)

The integer label → constellation point mapping is contractual for any
code that consumes `generate_qam_symbols_with_indices` or persists QAM
labels. The audit checks:

1. **Internal round-trip consumers** — `python/spectra/link/`,
   `python/spectra/receivers/coherent.py`. Expected outcome: no change
   needed (they re-read `build_qam_constellation` so the mapping is
   internally consistent).
2. **Classifier training artefacts** — `python/spectra/classifiers/`.
   Expected outcome: no persisted state references QAM integer labels
   directly; verify and document.
3. **Benchmark / dataset YAML configs** — `python/spectra/benchmarks/configs/`.
   Expected outcome: configs reference QAM by class name, not by label.
   Verify and document.
4. **Pinned IQ-snapshot tests** — `grep -r "QAM16" tests/`. A prior scan
   showed all current `tests/test_*qam*.py` and `tests/test_waveforms_*.py`
   QAM tests are structural (shape, label, bandwidth) rather than
   IQ-pinned. The audit re-confirms this; any new IQ-snapshot test would
   need its expected values regenerated.

Anything found in categories 2 or 3 escalates: it would mean the spec
underestimated blast radius, and we'd need a parking lot decision before
merging.

### Verifier updates

`examples/verification/verify_qam16.py`:

- Add `P_gray` — for each pair of physical nearest neighbours on the
  generated constellation, assert label Gray-adjacency. Citation:
  `proakis2008:§4.3.2`.
- Tighten S1 SER tolerance to match BPSK/QPSK (≤ 0.8 dB max |Δ| at full
  mode); the row-major asymmetry that motivated the looser bound is gone.
- Add `S_ber_ser` — at SNR ≥ 10 dB, `|BER − SER/log₂(M)| ≤ 5e-3`. This is
  the testable consequence of Gray coding. Citation `proakis2008:§4.3.2`.
- Delete the "Gray adjacency check omitted" inline note in the docstring.
- Regenerate `assets/verification/qam16_*.png`.

### CHANGELOG and version bump

Add a new top section to the existing `CHANGELOG.md`:

```
## [Unreleased]

### Changed (BREAKING)
- 16-QAM (and all square M-QAM ≥ 16) now uses Gray-coded labelling. The
  prior row-major labelling produced a BER↔SER mismatch of up to log₂(M)
  at moderate-to-high SNR. Datasets and classifiers trained on the prior
  mapping must be regenerated.

### Fixed
- GMSK modulation index restored to h = 0.5; was previously h_eff = 0.5/sps
  due to a zero-insertion upsample. Affects spectral occupancy and BER
  curves for `sp.GMSK`.
```

Version bumps land with PR-B (pre-1.0 minor for the breaking change):

- `pyproject.toml` line 7: `version = "0.1.0"` → `"0.2.0"`.
- `rust/Cargo.toml` line 3: `version = "0.1.0"` → `"0.2.0"`.

PR-A is non-breaking and ships at 0.1.0; the version bump rides on PR-B.

## PR sequencing

Two ordered PRs:

- **PR-A — GMSK fix.** `python/spectra/waveforms/fsk.py` (one upsample
  line), `examples/verification/verify_gmsk.py` revert,
  `tests/test_waveforms_fsk.py` new regression test,
  `assets/verification/gmsk_S1_ber.png` deleted, README finding
  rewritten. Self-contained, no downstream audit, no breaking change.

- **PR-B — QAM Gray-coding fix.** `rust/src/modulators.rs`
  `build_qam_constellation` + Rust unit test, audit findings documented in
  PR description, `examples/verification/verify_qam16.py` updates,
  `assets/verification/qam16_*.png` regenerated, README finding rewritten,
  `CHANGELOG.md` BREAKING entry, version bump. Carries the entire audit
  cost; merges after PR-A.

The GMSK fix does not block on the QAM audit, and vice versa: the QAM PR
does not block on the GMSK verifier revert. Either could land first
mechanically, but PR-A is sequenced first because it's the smaller-risk
change.

## Verification methodology

Both fixes are validated by the existing verification suite. The success
criterion for each PR is: the corresponding `verify_<waveform>.py` script
passes in both quick mode and `--full` mode with citation-grounded
tolerances and no workaround / deviation prose in the docstring.

Property checks remain the regression guard. Performance checks confirm
the fix restores theory agreement.

## Risks

- **GMSK fix** — low risk. Single-line change in pure-Python code, label
  contract unchanged, IQ-snapshot tests pinned to GMSK output need
  regenerating but the structural / shape tests are unaffected.
- **QAM fix audit** — medium risk. Spec assumes all downstream callers
  re-read `build_qam_constellation` and are therefore internally
  consistent. If the audit surfaces a persisted artefact pinning the old
  mapping, scope expands to that artefact's regeneration. Audit happens
  during plan execution; findings documented in the PR.
- **Numerical floor** — Gray-bit-split construction uses integer
  arithmetic; round-trip identity through float and back is preserved
  because the levels are exact integers before f32 conversion.
- **GFSK family** — same upsample pattern at `fsk.py:218-224` may have the
  same defect. Not fixed in this spec. Flagged as discovered work.

## Discovered work (out of scope, not blocking)

- `GFSK.generate` at `python/spectra/waveforms/fsk.py:218-224` uses
  `gaussian_taps(...)` (sum-normalised) on a zero-insertion-upsampled
  symbol train — identical pattern to the GMSK bug. Likely produces
  `h_eff = mod_index/sps` instead of `mod_index`. Tracked separately;
  needs its own verifier first.
- Square QAM orders ≥ 64 are not yet in the verification suite but the
  Gray-coding fix applies uniformly. Adding `verify_qam64.py` is mechanical
  but out of scope here.
- Reviewer tutorial notebook (the original brainstorm topic). Will be
  written against the post-fix codebase as a follow-on spec.
- A proper coherent BT=0.3 GMSK BER verifier — either a
  Laurent-decomposition detector or a Viterbi CPM decoder — to restore an
  S1 BER row that tracks `Q(√(2·Eb/N0))` within ~1 dB per Murota & Hirade
  1981. S1 was dropped from this PR because a per-bit matched filter loses
  ~26 dB on BT=0.3 GMSK (ISI spans ~3 bit intervals); building a proper
  detector is real engineering work, not a verifier patch.
