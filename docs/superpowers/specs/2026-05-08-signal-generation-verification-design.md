# Signal Generation Verification — Design

**Status:** Draft for review
**Date:** 2026-05-08
**Author:** brainstormed with Claude Code
**Topic:** Expert-defensible verification of SPECTRA waveform generation, with proof and references shipped in `examples/`.

## Motivation

SPECTRA's existing tests assert structural validity (`assert_valid_iq`, label
strings, bandwidth formula equality, deterministic seeding) but rarely
cross-check generated signals against analytical theory, published
standards, or closed-form references. The `examples/` tree is demonstrative
(plots an IQ trace, dumps a constellation), not evidentiary.

For SPECTRA to be credible to an RF / communications expert reviewer, each
core waveform should ship with explicit, citation-backed *proof* that:

1. The signal satisfies its analytical defining properties (bandwidth,
   constellation geometry, PSD shape, constant envelope, etc.).
2. The signal, when used in a link-level Monte Carlo, achieves the
   theoretical performance predicted by published equations.
3. Where a closed-form reference exists (3GPP NR sequence tables, ADS-B
   CRC-24, Barker-13 autocorrelation property, LFM matched-filter gain), the
   generated signal is **exactly equal** to that reference at the bit /
   sample / dB level.

This spec defines a small, rigorous first-cut verification suite (10
waveforms, 7 proof patterns) shipped under `examples/verification/` plus a
companion master notebook, with hard tolerances grounded in the literature
and a regression-guarding subset wired into CI.

## Scope

### In scope (first cut)

Ten waveforms, chosen for diversity across modulation classes:

| # | Waveform     | Class                       | Strongest proof                                                  |
|---|--------------|-----------------------------|------------------------------------------------------------------|
| 1 | BPSK         | Linear binary               | BER-vs-theory exact; constellation on real axis; PSD ≈ sinc²·RRC |
| 2 | QPSK         | Linear M-ary                | SER-vs-theory; Gray constellation; ACLR per 3GPP TS 38.104       |
| 3 | QAM16        | Linear high-order           | SER-vs-theory; EVM at high SNR; PAPR statistics                  |
| 4 | GMSK         | CPM                         | Constant envelope; PSD vs Laurent expansion; BT product          |
| 5 | OFDM         | Multicarrier                | Subcarrier orthogonality; CP correlation peak; PAPR statistics   |
| 6 | NR PSS       | Spec-defined sequence       | Exact sample equality with 3GPP TS 38.211 §7.4.2.2               |
| 7 | NR SSS       | Spec-defined sequence       | Exact equality with 3GPP TS 38.211 §7.4.2.3 Gold sequence        |
| 8 | LFM (chirp)  | Radar FM                    | IF = linear ramp; ambiguity knife-edge; matched-filter gain      |
| 9 | Barker-13    | Radar code                  | Autocorr peak/max-sidelobe = 13 (defining property)              |
|10 | ADS-B        | Protocol with CRC           | Exact CRC-24 byte equality (RTCA DO-260B G(x)=0x1FFF409)         |

### Out of scope (deferred)

- Other linear mods (8PSK, M-PSK ≥ 16, M-QAM ≥ 64, M-ASK).
- Other CPM (CPFSK family at non-MSK indices).
- Other 5G NR primitives (DMRS, PRACH preamble formats, PUSCH/PDSCH).
- Other radar (FMCW, NLFM, stepped-frequency, polyphase codes Frank/P1–P4,
  Costas, Zadoff-Chu).
- Other protocols (Mode S, AIS, ACARS, DME, ILS).
- Spread spectrum (DSSS, FHSS, THSS, CDMA, ChirpSS).
- Analog (AM/FM family).
- Multifunction emitters and scheduled waveforms.
- Reference-implementation cross-checks (GNU Radio, MATLAB Comm Toolbox
  fixtures, srsRAN). The first cut relies on closed-form equations and
  published standard tables only.

Once the verification *pattern* is established and the helper module is
proven, expanding to the rest of the catalog is mechanical and can be
scheduled per waveform family in follow-on plans.

## Verification methodology

Every check belongs to one of two tiers.

### Property checks (`P1, P2, …`) — deterministic, always-on

Closed-form, deterministic, and fast (< 1 s per waveform). These are exact
equalities or inequalities that follow from the waveform's mathematical
definition or from a published standard. Examples: BPSK symbols on the
real axis; NR PSS exact sample equality with 3GPP table; Barker-13
autocorrelation PSLR exactly 13; ADS-B CRC-24 byte equality.

Property checks **always run in CI**. They form the regression guard for
the underlying Rust generators: a change to a Rust function that breaks
sequence equality or a defining property fails CI within seconds.

### Performance checks (`S1, S2, …`) — statistical, gated

Monte-Carlo / sampling-bound checks: BER and SER versus AWGN theory, EVM
at fixed SNR, ACLR over long captures, PAPR percentiles. These have
sample-size-bound tolerances that loosen at smaller N. They run by default
in fast mode (~5–10 s per waveform) with relaxed tolerances, and in
publication-grade `--full` mode (~30–120 s per waveform) with tight
tolerances.

Performance checks are marked `@pytest.mark.slow` and run on demand or in
nightly / release CI, not on every push.

### Tolerances must cite

Every numeric tolerance carries a citation key. No "industry rule of thumb"
tolerances. Where the tolerance follows from a standards-mandated minimum
(e.g., ACLR ≥ 45 dB from 3GPP TS 38.104 Table 6.6.3.1-1), the citation
points to the standard. Where the tolerance follows from Monte-Carlo
variance (e.g., BER tolerance at 10⁵ vs 10⁶ symbols), the comment derives
the Wald confidence interval inline.

## Architecture

### Directory layout

```
examples/verification/
├── README.md                    # how to run, what each script proves
├── REFERENCES.md                # canonical bibliography (parsed at startup)
├── _verify_helpers.py           # shared measurement / theory / I-O primitives
├── verification_suite.ipynb     # master notebook, imports the 10 scripts
├── verify_bpsk.py
├── verify_qpsk.py
├── verify_qam16.py
├── verify_gmsk.py
├── verify_ofdm.py
├── verify_nr_pss.py
├── verify_nr_sss.py
├── verify_lfm.py
├── verify_barker13.py
└── verify_adsb.py

tests/verification/
├── __init__.py
├── test_helpers.py              # unit tests for _verify_helpers measurement primitives
├── test_verify_bpsk.py
├── test_verify_qpsk.py
├── …                            # one per script
└── test_verify_adsb.py
```

Figures land in `examples/outputs/verification/<waveform>_<test_id>.png`.

### Per-script template

Every `verify_<waveform>.py` exposes the same interface:

```python
def properties() -> ResultTable: ...
def performance(full: bool = False) -> ResultTable: ...

if __name__ == "__main__":
    args = parse_args()                 # supports --full
    p = properties()
    s = performance(full=args.full)
    print(p.render()); print(s.render())
    sys.exit(0 if (p.all_passed and s.all_passed) else 1)
```

The module-level docstring lists every check ID with one-line summary and
its citation key. Every `results.add(...)` call carries: test ID, name,
measured value, expected value, tolerance, units, and citation key.

### Helper module (`_verify_helpers.py`)

Public surface (example-local, not part of the `spectra` package API):

```python
@dataclass
class CheckResult:
    test_id: str; name: str; measured; expected; tolerance: float
    passed: bool; citation: str; units: str = ""

class ResultTable:
    def add(self, test_id, name, *, measured, expected, tol, cite, units=""): ...
    def render(self) -> str: ...           # ASCII table for terminal
    def render_html(self) -> str: ...      # rich table for notebook
    @property
    def all_passed(self) -> bool: ...

# Theoretical formulas (single source of truth, each cites its locus)
def ber_bpsk_awgn(ebn0_db) -> np.ndarray
def ser_mpsk_awgn(M, ebn0_db) -> np.ndarray
def ser_mqam_awgn(M, ebn0_db) -> np.ndarray
def psd_rrc_squared(f, Rs, alpha) -> np.ndarray
def matched_filter_gain_db(tbp) -> float

# Measurement primitives (each unit-tested with known-answer signals)
def simulate_ber_awgn(waveform, ebn0_db, n_bits, seed) -> np.ndarray
def measure_evm_rms(rx_symbols, tx_ref) -> float
def measure_acpr_db(iq, fs, channel_bw, offsets) -> dict[float, float]
def measure_obw(iq, fs, fraction=0.99) -> float
def measure_papr_db(iq, percentile=99.9) -> float
def measure_psd_shape_correlation(measured_psd, theory_psd) -> float
def autocorr_peak_to_sidelobe(seq) -> float
def measure_cp_correlation_peak(ofdm_iq, n_fft, n_cp) -> tuple[int, float]

# I-O
def parse_args() -> argparse.Namespace
def plot_theory_overlay(measured, theory, x, ...) -> None
def plot_psd_with_theory(iq, fs, theory_fn, ...) -> None
def save_verification_figure(name) -> None

# Reference loader
REFERENCES: dict[str, dict] = parse_references_md("REFERENCES.md")
def cite(key: str) -> str
```

Every `measure_*` primitive has a unit test in
`tests/verification/test_helpers.py` that feeds it a synthetic
known-answer signal and asserts the recovered value matches the
analytical answer. The verification harness must itself be trustworthy.

### Master notebook (`verification_suite.ipynb`)

A single notebook that imports the ten scripts and renders their results
end-to-end with narrative markdown.

Structure:

- Title, abstract, methodology block.
- One section per waveform: theory recap (LaTeX equations + citation key)
  → `from verify_<wf> import properties, performance` →
  `props.render_html() / perf.render_html()` → embedded figures from
  `outputs/verification/`.
- Final summary cell: 10-row × N-column pass/fail grid across all checks.

The notebook does **not** reimplement logic. Bug fixes in the `.py` files
propagate automatically. Outputs are stripped via `nbstripout` pre-commit
hook; the notebook is committed in "ready to run" state.

A top-level `FULL = False` notebook variable toggles statistical-check
sample sizes and tolerances. Default is fast (~30 s end-to-end);
`FULL = True` is publication-grade.

## Citations and `REFERENCES.md`

A single bibliography file at `examples/verification/REFERENCES.md` is the
canonical source. Citation keys (`proakis2008:eq4.3-15`,
`3gpp_38_211:§7.4.2.2.1`) referenced in code resolve to entries here at
script startup. **Unresolved keys raise an error** — there are no silently
broken citations.

Mandatory entry fields: authors / org, title (with edition), year,
publisher / URL, ISBN or DOI when available, and a `Loci used` block
naming the specific equations / sections / pages used by code.

Authoritative sources for the first cut:

| Key            | Source                                                         |
|----------------|----------------------------------------------------------------|
| proakis2008    | Proakis & Salehi, *Digital Communications*, 5th ed., 2008       |
| sklar2001      | Sklar, *Digital Communications: Fundamentals & Apps*, 2nd ed.   |
| levanon2004    | Levanon & Mozeson, *Radar Signals*, Wiley-IEEE, 2004            |
| 3gpp_38_211    | 3GPP TS 38.211 — Physical channels and modulation               |
| 3gpp_38_104    | 3GPP TS 38.104 — Base Station radio transmission and reception  |
| rtca_do260b    | RTCA DO-260B — MOPS for 1090 MHz Extended Squitter ADS-B        |
| itu_sm_328     | ITU-R Recommendation SM.328-11 — Spectra and bandwidths         |
| laurent1986    | Laurent, "Exact and approximate construction of digital phase   |
|                | modulations by superposition of amplitude modulated pulses",   |
|                | IEEE Trans. Comm., 1986 (for GMSK PSD reference)               |
| vandeBeek1997  | van de Beek et al., "ML Estimation of Time and Frequency       |
|                | Offset in OFDM Systems", IEEE Trans. Signal Proc., 1997        |
|                | (for OFDM cyclic-prefix correlation peak)                      |
| han2005        | Han & Lee, "An overview of peak-to-average power ratio         |
|                | reduction techniques for multicarrier transmission",           |
|                | IEEE Wireless Comm., 2005 (for OFDM PAPR distribution)         |

The reader's contract: open the result table, find the citation, look up
the entry in `REFERENCES.md`, find the page / equation / section, and
verify the formula matches the code. If the citation is wrong, that is a
bug.

## CI integration

| Check class       | When it runs                          | Mechanism                              |
|-------------------|---------------------------------------|----------------------------------------|
| Properties (P*)   | Every commit                          | `pytest tests/verification/`           |
| Performance (S*)  | Nightly / release / on demand         | `pytest -m slow tests/verification/`   |
| Notebook smoke    | Every commit                          | `pytest --nbmake examples/verification/verification_suite.ipynb` (FULL=False) |
| Helper primitives | Every commit                          | `pytest tests/verification/test_helpers.py` |

The `tests/verification/test_verify_<wf>.py` files are thin wrappers — they
import `properties()` and `performance()` from the script and assert
`all_passed`. No test logic is duplicated; the script is the source of
truth.

## Per-waveform proof recipes

Each of the 10 scripts implements the checks below. Property checks (`P*`)
are deterministic and always-on; performance checks (`S*`) are statistical
and slow-gated. Each row is one `results.add(...)` call carrying a
citation key.

### 1. `verify_bpsk.py`

| ID | Check | Reference |
|----|-------|-----------|
| P1 | Symbols lie on real axis: `imag(symbols)` ≈ 0 | constellation definition |
| P2 | Two unique symbols at ±1 | constellation definition |
| P3 | Bandwidth = (1+α)·Rs (RRC) within 1 % | [sklar2001:§3.5,eq3.74] |
| P4 | PSD shape correlation with squared-RRC ≥ 0.99 | [proakis2008:eq9.2-37] |
| P5 | OBW (99 %) within 5 % of theory | [itu_sm_328:§3] |
| P6 | ACLR at ±1·Rs offset ≥ 45 dB | [3gpp_38_104:T6.6.3.1-1] |
| S1 | BER vs Eb/N0 ∈ [0,10] dB, max \|Δ\| ≤ 0.3 dB | [proakis2008:eq4.3-13] |
| S2 | EVM at SNR=30 dB ≤ 1 % RMS | [3gpp_38_104:§B.2] |

### 2. `verify_qpsk.py`

| ID | Check | Reference |
|----|-------|-----------|
| P1 | 4 constellation points at ±π/4, ±3π/4 (Gray-coded) | [sklar2001:§3.5] |
| P2 | All four points equidistant from origin (\|s\|=1) | constellation |
| P3 | Bandwidth = (1+α)·Rs within 1 % | [sklar2001:§3.5,eq3.74] |
| P4 | PSD shape correlation with squared-RRC ≥ 0.99 | [proakis2008:eq9.2-37] |
| P5 | OBW (99 %) within 5 % of theory | [itu_sm_328:§3] |
| P6 | ACLR at ±1·Rs offset ≥ 45 dB | [3gpp_38_104:T6.6.3.1-1] |
| S1 | SER vs Eb/N0 ∈ [0,10] dB, max \|Δ\| ≤ 0.3 dB | [proakis2008:eq4.3-15], M=4 |
| S2 | EVM at SNR=30 dB ≤ 1 % RMS | [3gpp_38_104:§B.2] |
| S3 | PAPR (99.9 %ile) within 0.3 dB of theoretical RRC PAPR | [proakis2008:§9.2] |

### 3. `verify_qam16.py`

| ID | Check | Reference |
|----|-------|-----------|
| P1 | 16 points on rectangular grid at ±{1,3} ± j{1,3} (normalized) | [proakis2008:§4.3] |
| P2 | Average symbol energy = 10 (raw integer grid) | algebraic identity |
| P3 | Gray-coded labels: adjacent points differ by exactly 1 bit | [sklar2001:§3.5] |
| P4 | Bandwidth, P5 PSD shape, P6 OBW (as QPSK) | (as QPSK) |
| S1 | SER vs Eb/N0 ∈ [4,18] dB, max \|Δ\| ≤ 0.3 dB | [proakis2008:eq4.3-30] |
| S2 | EVM at SNR=30 dB ≤ 1 % RMS | [3gpp_38_104:§B.2] |

### 4. `verify_gmsk.py`

| ID | Check | Reference |
|----|-------|-----------|
| P1 | Constant envelope: std(\|s\|) / mean(\|s\|) ≤ 1e-3 | CPM defining property |
| P2 | Modulation index h = 0.5 (frequency separation) | [proakis2008:§4.4-3] |
| P3 | BT product matches Gaussian-filter 3-dB BW | Gaussian taps reference |
| P4 | PSD main-lobe width matches Laurent expansion ±5 % | [laurent1986] |
| P5 | OBW within 5 % of theory | [itu_sm_328:§3] |
| S1 | BER vs Eb/N0, max \|Δ\| ≤ 0.5 dB (MSK approximation) | [proakis2008:eq4.4-43] |
| S2 | EVM (after MSK demod) ≤ 2 % at SNR=30 dB | [3gpp_38_104:§B.2] |

### 5. `verify_ofdm.py`

| ID | Check | Reference |
|----|-------|-----------|
| P1 | Subcarrier orthogonality: FFT of one symbol recovers exactly the input QAM symbols (no impairments) | OFDM defining property |
| P2 | Cyclic-prefix correlation peak: argmax of `corr(x[n], x[n+N_FFT])` = N_CP | [vandeBeek1997] |
| P3 | PSD ~rectangular within signal BW; out-of-band roll-off ≥ −20 dB at one subcarrier-spacing offset | OFDM PSD |
| P4 | Average symbol energy preserved across IFFT (Parseval) | algebraic identity |
| P5 | OBW within 3 % of N_used·Δf | [itu_sm_328:§3] |
| S1 | EVM at SNR=30 dB ≤ 2 % RMS (after CP removal + FFT + ZF eq) | [3gpp_38_104:§B.2] |
| S2 | PAPR (99.9 %ile) within 1 dB of theoretical OFDM PAPR (Gaussian approx) | [han2005] |

### 6. `verify_nr_pss.py`

| ID | Check | Reference |
|----|-------|-----------|
| P1 | Exact equality with 3GPP table for NID2 ∈ {0,1,2}: 127 BPSK values | [3gpp_38_211:§7.4.2.2.1] |
| P2 | Frequency-domain placement: 127 active subcarriers centered, zeros elsewhere | [3gpp_38_211:§7.4.2.2.2] |
| P3 | Auto-correlation peak-to-sidelobe ratio ≥ specified threshold | sequence design |
| P4 | Cross-correlation between PSS for different NID2 below threshold | [3gpp_38_211] |
| P5 | After IFFT + CP, time-domain PSS length = expected sample count | [3gpp_38_211:§7.4.2.2] |

(All deterministic — there is no S-tier; PSS is a fixed sequence.)

### 7. `verify_nr_sss.py`

| ID | Check | Reference |
|----|-------|-----------|
| P1 | Exact equality with 3GPP table for sample (NID1, NID2) pairs: 127 ±1 values | [3gpp_38_211:§7.4.2.3.1] |
| P2 | Gold-sequence construction verified: m₀, m₁ shifts match formula | [3gpp_38_211:§7.4.2.3.1] |
| P3 | Cross-correlation between distinct (NID1, NID2) below threshold | sequence design |
| P4 | Frequency-domain placement matches PSS layout | [3gpp_38_211:§7.4.2.3.2] |

### 8. `verify_lfm.py`

| ID | Check | Reference |
|----|-------|-----------|
| P1 | Instantaneous frequency = f₀ + (B/T)·t — linear ramp, residual std ≤ 1e-9 of B | LFM definition |
| P2 | Total swept bandwidth equals configured B within 1 % | LFM definition |
| P3 | Matched-filter compression gain = 10·log₁₀(TBP) within 0.2 dB | [levanon2004:eq5.5] |
| P4 | Pulse-compression resolution (3-dB main-lobe width) = 0.886/B within 5 % | [levanon2004:§4.2] |
| P5 | Ambiguity function knife-edge along Doppler/delay diagonal: 3-dB ridge slope matches B/T | [levanon2004:§4.2] |
| S1 | Range-resolution after pulse compression at SNR=20 dB matches theory ±10 % | [levanon2004:§5] |

### 9. `verify_barker13.py`

| ID | Check | Reference |
|----|-------|-----------|
| P1 | Exact equality with canonical Barker-13: `[+1+1+1+1+1−1−1+1+1−1+1−1+1]` | [levanon2004:Tab.6.1] |
| P2 | Autocorrelation peak = 13, max sidelobe = 1, **PSLR exactly = 13** | [levanon2004:eq3.32] |
| P3 | Energy = 13 (each chip is ±1) | algebraic identity |
| P4 | Spectrum sinc² envelope (chip-rate sinc) — shape correlation ≥ 0.95 | rectangular pulse PSD |
| S1 | Pulse-compression detection at SNR=10 dB: peak detected at correct lag in 100 % of trials | [levanon2004:§3] |

### 10. `verify_adsb.py`

| ID | Check | Reference |
|----|-------|-----------|
| P1 | Preamble bit-pattern: 8 µs PPM preamble pulses at exact offsets 0, 1, 3.5, 4.5 µs | [rtca_do260b:§2.2.3.2.2] |
| P2 | Message length = 112 bits = 112 µs at 1 Mbps PPM | [rtca_do260b:§2.2.3.2.2] |
| P3 | CRC-24 byte-equality with G(x)=0x1FFF409, computed by independent reference impl | [rtca_do260b:§2.2.3.2.1.2] |
| P4 | PPM modulation: each bit decodes back to original payload (round-trip) | [rtca_do260b:§2.2.3.2.2] |
| P5 | Carrier frequency = 1090 MHz (or as configured), tone offset = 0 | [rtca_do260b] |
| P6 | Spectrum mask: out-of-band emissions within ITU/RTCA mask | [rtca_do260b:§2.2.4] |

(All deterministic — protocol bits-are-bits, no Monte Carlo needed.)

### Coverage summary

- **Constellation / EVM**: BPSK, QPSK, QAM16, OFDM
- **PSD / ACLR / OBW**: BPSK, QPSK, QAM16, GMSK, OFDM
- **BER / SER vs theory**: BPSK, QPSK, QAM16, GMSK
- **Constant-envelope (CPM)**: GMSK
- **Subcarrier orthogonality, CP correlation**: OFDM
- **Spec-exact sequence equality**: NR PSS, NR SSS, Barker-13, ADS-B
- **Pulse compression / matched filter**: LFM, Barker-13
- **Ambiguity function**: LFM
- **CRC byte equality**: ADS-B

Seven distinct proof patterns, ten waveforms.

## Non-goals / explicit decisions

- **No reference-implementation cross-checks in the first cut.** GNU Radio
  / MATLAB / srsRAN fixtures bit-rot and add maintenance cost. Where a
  closed-form reference exists, exact equality with that reference is
  more convincing than a noisy comparison anyway.
- **No new public `spectra` API.** The verification harness is example-local
  in `_verify_helpers.py`. Theoretical formulas and measurement primitives
  used here are not re-exported.
- **No silent checks.** Every check has a numbered ID, a citation, and
  appears in the result table. If it isn't in the table, it didn't run.
- **No looseness without justification.** A 0.5-dB tolerance carries a
  citation or an inline derivation of the Monte-Carlo confidence interval
  that justifies it.
- **No notebook logic duplication.** The notebook imports the scripts.
  Single source of truth.

## Risks / open issues

- **GMSK BER-vs-theory tolerance.** SPECTRA's GMSK uses Gaussian-filtered
  CPM; the textbook MSK BER curve is an approximation. The S1 tolerance
  for GMSK is set to 0.5 dB rather than 0.3 dB to absorb the approximation
  gap. If the measured offset systematically exceeds 0.5 dB, the right
  follow-up is to switch the reference theory to a more accurate
  Laurent-AMP-based BER expression rather than to loosen the tolerance.
- **OFDM EVM tolerance.** Without an explicit equalizer, EVM at SNR=30 dB
  for OFDM may be dominated by the fact that no impairment is present —
  the tolerance is set to 2 % to leave room for finite-FFT-length effects;
  in practice we expect EVM to be well below 1 %.
- **Notebook execution time at FULL=True.** Ten waveforms × ~60 s each is
  ~10 minutes. The notebook does not pre-render; users opting into FULL
  mode are accepting that runtime.
- **REFERENCES.md drift.** A reference moved (e.g., 3GPP version bump)
  requires updating both the entry and the loci. The parser raises on
  unknown keys but does not validate that page numbers are still correct.
  This is a documentation-discipline issue, not a tooling issue.

## Acceptance criteria

The suite is complete when:

1. All 10 `verify_<waveform>.py` scripts exist, are runnable standalone,
   and exit 0 in `--full` mode on a clean main branch.
2. `verification_suite.ipynb` executes end-to-end in `FULL=False` under
   `nbmake` with no errors.
3. `tests/verification/test_helpers.py` covers every measurement primitive
   with a known-answer test.
4. All 10 `tests/verification/test_verify_<wf>.py` wrappers pass on every
   commit (P-tier).
5. Nightly slow-tier (`pytest -m slow tests/verification/`) passes.
6. `REFERENCES.md` resolves every citation key referenced in the
   suite — startup parse succeeds with no unresolved keys.
7. `examples/README.md` lists the verification suite under a new
   "Verification" section with an entry per script.
8. An RF / communications expert can open any result table, follow the
   citation key to `REFERENCES.md`, and verify the cited formula matches
   the code.

## Follow-on work (out of scope of this plan)

- Expand the suite to the remaining waveforms (8PSK, M-QAM, FSK family,
  remaining NR primitives, FMCW / NLFM / polyphase codes, Mode S, AIS,
  ACARS, spread-spectrum family, analog AM/FM) following the same
  pattern. Each waveform family is a follow-up plan.
- Add a `--reference-impl` mode that cross-checks against a frozen GNU
  Radio / MATLAB fixture file when available (deferred per non-goals).
- Generate a rendered HTML "verification report" artifact in CI and
  publish it as a docs site page under `docs/verification/`.
