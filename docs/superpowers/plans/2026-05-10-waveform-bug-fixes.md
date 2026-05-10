# Waveform Bug Fixes Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Fix two waveform-generation defects surfaced by the verification suite: GMSK effective modulation index (0.5/sps → 0.5) and 16-QAM constellation labelling (row-major → Gray-coded).

**Architecture:** Two ordered, independently-mergeable PRs. PR-A is a one-line Python fix to `GMSK.generate` plus reverting the documented workarounds in `verify_gmsk.py`. PR-B replaces the row-major QAM constellation builder with a Gray-bit-split construction in `rust/src/modulators.rs`, audits downstream callers (expected to be internally consistent), tightens `verify_qam16.py`, and ships a BREAKING-change CHANGELOG entry with a 0.1.0 → 0.2.0 version bump.

**Tech Stack:** Python 3.10+ (NumPy, pytest), Rust 1.83+ (PyO3 via maturin), MkDocs for docs.

---

## File Structure

**PR-A files:**
- Modify: `python/spectra/waveforms/fsk.py:96-117` (`GMSK.generate`)
- Modify: `tests/test_waveforms_fsk.py` (add regression test)
- Modify: `examples/verification/verify_gmsk.py` (revert workarounds)
- Modify: `examples/verification/README.md` (rewrite GMSK finding)
- Regenerate: `assets/verification/gmsk_S1_ber.png`

**PR-B files:**
- Modify: `rust/src/modulators.rs:27-42` (`build_qam_constellation`) + the inline `qam16_constellation_properties` test at lines 482-513
- Modify: `examples/verification/verify_qam16.py` (add `P_gray`, tighten S1, add `S3` BER/SER coupling)
- Modify: `examples/verification/README.md` (rewrite 16-QAM finding)
- Modify: `CHANGELOG.md` (BREAKING + Fixed entries)
- Modify: `pyproject.toml:7` (version bump)
- Modify: `rust/Cargo.toml:3` (version bump)
- Regenerate: `assets/verification/qam16_P5_psd.png`, `assets/verification/qam16_S1_ser.png`

---

# PR-A — GMSK h_eff = 0.5 Fix

### Task 1: Add regression test for GMSK steady-state modulation index

**Files:**
- Modify: `tests/test_waveforms_fsk.py`

This is a TDD red step: the test asserts the *correct* behaviour `h = 0.5` and must fail on the current buggy code.

- [ ] **Step 1: Append the failing test to the bottom of `tests/test_waveforms_fsk.py`**

Add these lines at the end of the file (after the existing classes):

```python
class TestGMSKModulationIndex:
    """Regression test: GMSK steady-state per-symbol phase change.

    Standard MSK / GMSK uses modulation index h = 0.5, so a constant
    +1 bit stream drives the phase by π·0.5 = π/2 rad per symbol.
    A prior implementation used zero-insertion upsampling with a
    sum-normalised Gaussian, producing h_eff = 0.5/sps = 0.0625 (a
    factor-of-sps error). This test guards against regression.
    """

    def test_constant_bit_stream_gives_h_one_half(self, monkeypatch):
        import numpy as np
        from spectra import _rust
        from spectra.waveforms.fsk import GMSK

        sps = 8
        num_symbols = 256

        # Force the underlying BPSK generator to return all +1 so the
        # GMSK input is a constant bit stream. After the Gaussian filter
        # settles, every per-symbol phase increment should equal π·h = π/2.
        def all_plus_one(n, seed=0):
            return np.ones(n, dtype=np.complex64)

        monkeypatch.setattr(_rust, "generate_bpsk_symbols", all_plus_one)
        # Also patch the symbol it's imported under in fsk.py:
        from spectra.waveforms import fsk as fsk_mod
        monkeypatch.setattr(fsk_mod, "generate_bpsk_symbols", all_plus_one)

        wf = GMSK(bt=0.3, samples_per_symbol=sps)
        iq = wf.generate(num_symbols=num_symbols, sample_rate=1.0e6, seed=0)

        # Steady-state per-symbol phase change. Skip the first and last
        # 16 symbols to avoid Gaussian-filter transients.
        phase = np.unwrap(np.angle(iq))
        per_symbol = phase[sps::sps] - phase[:-sps:sps]
        inner = per_symbol[16:-16]
        median_step = float(np.median(np.abs(inner)))

        expected = np.pi * 0.5  # h = 0.5
        # 1 % relative tolerance; the Gaussian filter's amplitude response
        # at DC is exactly 1 for a sum-normalised kernel, so the residual
        # error is float32 round-off.
        assert abs(median_step - expected) <= 0.01 * expected, (
            f"steady-state |Δφ|/symbol = {median_step:.4f} rad, "
            f"expected {expected:.4f} rad (h = 0.5)"
        )
```

- [ ] **Step 2: Run the test and confirm it fails on current code**

Run: `pytest tests/test_waveforms_fsk.py::TestGMSKModulationIndex -v`
Expected: `FAILED` with `steady-state |Δφ|/symbol ≈ 0.1963 rad, expected 1.5708 rad`.

- [ ] **Step 3: Commit the failing test**

```bash
git add tests/test_waveforms_fsk.py
git commit -m "test(gmsk): regression test for steady-state h = 0.5 (red)

Documents the textbook GMSK modulation index. Test currently fails
because GMSK.generate uses zero-insertion upsampling combined with a
sum-normalised Gaussian filter, producing h_eff = 0.5/sps = 0.0625.
Fix in the next commit."
```

---

### Task 2: Fix `GMSK.generate` to produce h = 0.5

**Files:**
- Modify: `python/spectra/waveforms/fsk.py:106-115`

- [ ] **Step 1: Apply the upsample fix**

In `python/spectra/waveforms/fsk.py`, replace the body of `GMSK.generate` from line 106 through line 117 so the block reads:

```python
        # Repeat-upsample so the frequency-pulse train averages to ±1.
        # A sum-normalised Gaussian preserves the DC level of its input;
        # zero-insertion would attenuate it by sps, yielding h_eff = h/sps.
        symbols_up = np.repeat(symbols.real.astype(np.float32), sps)

        # Gaussian filter
        h = self._gaussian_taps()
        filtered = np.convolve(symbols_up, h, mode="same")

        # Phase modulation: per-symbol total Δφ = π·h·b_k with h = 0.5.
        delta_phi = np.pi * 0.5 * filtered / sps
        phase = np.cumsum(delta_phi)
        return np.exp(1j * phase).astype(np.complex64)
```

The `/sps` divisor stays: we now have `sps` samples per symbol each contributing `π·0.5·(±1)/sps`, summing to `±π/2` per symbol.

- [ ] **Step 2: Run the regression test and confirm it passes**

Run: `pytest tests/test_waveforms_fsk.py::TestGMSKModulationIndex -v`
Expected: `PASSED`.

- [ ] **Step 3: Run the full FSK test module and confirm nothing else broke**

Run: `pytest tests/test_waveforms_fsk.py -v`
Expected: all tests pass.

- [ ] **Step 4: Commit the fix**

```bash
git add python/spectra/waveforms/fsk.py
git commit -m "fix(gmsk): restore modulation index h = 0.5

GMSK.generate previously upsampled symbols with zero-insertion and
filtered with a sum-normalised Gaussian, producing h_eff = 0.5/sps =
0.0625 (sps=8). Switch to repeat-upsampling (matching MSK.generate
and FSK.generate) so the filter's preserved-DC property yields the
textbook h = 0.5.

Regression test in tests/test_waveforms_fsk.py now passes."
```

---

### Task 3: Revert workarounds in `verify_gmsk.py`

**Files:**
- Modify: `examples/verification/verify_gmsk.py`

The verifier currently codes around the bug. Replace it with textbook tolerances. Use the canonical MSK BER formula from `_verify_helpers.py` if available; otherwise implement inline.

- [ ] **Step 1: Replace the file with the corrected version**

Overwrite `examples/verification/verify_gmsk.py` with:

```python
"""SPECTRA Verification — GMSK
================================
Proves that the generated GMSK waveform satisfies the standard MSK /
GMSK properties.

  P1. Constant envelope: std(|s|)/mean(|s|) ≤ 1e-3.        [gmsk:cpm-defn]
  P2. Modulation index h = 0.5: steady-state per-symbol
      |Δφ| ≈ π/2 rad on a constant-bit stream.             [proakis2008:§4.4-3]
  P3. Gaussian filter 3-dB BW within 20 % of BT·R_s·2.    [gmsk:gaussian]
  P4. PSD 3-dB BW within 25 % of 0.27·R_s (Laurent main-lobe for BT=0.3, h=0.5).
                                                            [laurent1986:§III]
  P5. OBW 99 % within 10 % of 0.92·R_s (GSM/3GPP BT=0.3 GMSK reference).
                                                            [itu_sm_328:§3]
  S1. BER vs Q(√(2·Eb/N0)) over [0, 10] dB using a coherent matched-filter
      MSK receiver. Max |Δ| ≤ 1.0 dB (quick) / 0.8 dB (full).
                                                            [proakis2008:eq4.4-43]

Run:
    python examples/verification/verify_gmsk.py            # quick mode
    python examples/verification/verify_gmsk.py --full     # publication mode
"""

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent))

import numpy as np
import spectra as sp
from scipy.special import erfc
from _verify_helpers import (
    ResultTable,
    _welch_psd,
    measure_obw,
    plot_theory_overlay,
    run_script,
    save_verification_figure,
)

SAMPLE_RATE = 1.0e6
SAMPLES_PER_SYMBOL = 8
BT = 0.3
SYMBOL_RATE = SAMPLE_RATE / SAMPLES_PER_SYMBOL
H_GMSK = 0.5  # standard MSK / GMSK modulation index


def _build_gaussian_taps(bt: float, sps: int, filter_span: int = 4) -> np.ndarray:
    half = filter_span * sps // 2
    tt = np.arange(-half, half + 1) / sps
    h = np.sqrt(2.0 * np.pi / np.log(2)) * bt * np.exp(-2.0 * (np.pi * bt * tt) ** 2 / np.log(2))
    return h / np.sum(h)


def _make_gmsk_signal(bits: np.ndarray, sps: int, bt: float) -> np.ndarray:
    """Mirror sp.GMSK.generate() with a deterministic bit sequence."""
    n = len(bits)
    symbols = (2 * bits - 1).astype(np.float32)
    symbols_up = np.repeat(symbols, sps)
    h = _build_gaussian_taps(bt, sps, filter_span=4)
    filtered = np.convolve(symbols_up, h, mode="same")
    delta_phi = np.pi * H_GMSK * filtered / sps
    phase = np.cumsum(delta_phi)
    return np.exp(1j * phase).astype(np.complex64)


def _q(x: np.ndarray) -> np.ndarray:
    return 0.5 * erfc(x / np.sqrt(2.0))


def properties() -> ResultTable:
    t = ResultTable("GMSK — Properties")

    wf = sp.GMSK(bt=BT, samples_per_symbol=SAMPLES_PER_SYMBOL)
    iq = wf.generate(num_symbols=4096, sample_rate=SAMPLE_RATE, seed=0)

    # P1 — constant envelope
    env = np.abs(iq)
    cv = float(np.std(env) / np.mean(env))
    t.add(
        "P1", "envelope CV (std/mean)",
        measured=cv, expected=0.0, tol=1e-3,
        cite="gmsk:cpm-defn",
    )

    # P2 — modulation index h = 0.5 via constant-bit stream
    n_p2 = 500
    bits_const = np.ones(n_p2, dtype=np.int64)
    iq_p2 = _make_gmsk_signal(bits_const, SAMPLES_PER_SYMBOL, BT)
    phase_p2 = np.unwrap(np.angle(iq_p2))
    sps = SAMPLES_PER_SYMBOL
    per_sym = phase_p2[sps::sps] - phase_p2[:-sps:sps]
    n_sym = len(per_sym)
    inner = per_sym[n_sym // 10 : -n_sym // 10]
    median_step = float(np.median(np.abs(inner)))
    expected_step = np.pi * H_GMSK
    t.add(
        "P2", "steady-state |Δφ|/symbol (rad)",
        measured=median_step, expected=expected_step,
        tol=0.01 * expected_step,
        cite="proakis2008:§4.4-3", units="rad",
    )

    # P3 — Gaussian filter 3-dB BW
    h = _build_gaussian_taps(BT, SAMPLES_PER_SYMBOL, filter_span=wf._filter_span)
    H = np.abs(np.fft.fftshift(np.fft.fft(h, n=4096)))
    fff = np.fft.fftshift(np.fft.fftfreq(4096, d=1.0 / SAMPLES_PER_SYMBOL))
    H_db = 20.0 * np.log10(H / np.max(H) + 1e-30)
    above = np.where(H_db >= -3.0)[0]
    bw_3db_hz = float(fff[above[-1]] - fff[above[0]]) * SYMBOL_RATE
    expected_bw = BT * SYMBOL_RATE * 2.0
    t.add(
        "P3", "Gaussian filter 3-dB BW (Hz)",
        measured=bw_3db_hz, expected=expected_bw,
        tol=0.20 * expected_bw,
        cite="gmsk:gaussian", units="Hz",
    )

    # P4 — PSD 3-dB BW vs BT=0.3 / h=0.5 reference (0.27·R_s).
    # The Laurent decomposition shows the main lobe of the squared-mag
    # frequency response for BT=0.3 GMSK is ≈ 0.27·R_s wide at -3 dB.
    f, p = _welch_psd(iq, fs=SAMPLE_RATE, nperseg=4096)
    p_db = 10.0 * np.log10(p / np.max(p) + 1e-30)
    above_psd = np.where(p_db >= -3.0)[0]
    main_bw = float(f[above_psd[-1]] - f[above_psd[0]]) if len(above_psd) > 0 else 0.0
    expected_psd_bw = 0.27 * SYMBOL_RATE
    t.add(
        "P4", "PSD 3-dB BW (Hz)",
        measured=main_bw, expected=expected_psd_bw,
        tol=0.25 * expected_psd_bw,
        cite="laurent1986:§III", units="Hz",
    )

    # P5 — 99 % OBW vs GSM/3GPP BT=0.3 GMSK reference (0.92·R_s).
    # Note: 1.5·R_s is the Carson's-rule peak-deviation bandwidth (for
    # h=0.5 narrowband approximation), not the 99 % OBW. The 99 % OBW
    # of BT=0.3 GMSK is the industry value referenced in GSM specs.
    obw = measure_obw(iq, fs=SAMPLE_RATE, fraction=0.99)
    expected_obw = 0.92 * SYMBOL_RATE
    t.add(
        "P5", "OBW 99 % (Hz)",
        measured=obw, expected=expected_obw,
        tol=0.10 * expected_obw,
        cite="itu_sm_328:§3", units="Hz",
    )

    return t


def _coherent_msk_demod(rx: np.ndarray, sps: int, n_bits: int) -> np.ndarray:
    """Coherent matched-filter MSK receiver.

    Standard offset-QPSK-like demodulator [proakis2008:eq4.4-43]:
      - I-channel matched filter: cos(πt / 2·Tb), spanning 2·Tb, demodulates
        even bits at multiples of 2·Tb.
      - Q-channel matched filter: sin(πt / 2·Tb), spanning 2·Tb, demodulates
        odd bits at odd multiples of Tb (offset by Tb).

    Each output is the sign of the matched-filter integral over a 2-bit
    interval. Even/odd outputs are interleaved into the recovered bit stream.
    For BT=0.3 GMSK the Gaussian filter introduces ~0.5–1 dB ISI loss
    relative to ideal MSK; the demod is otherwise unchanged.
    """
    Tb_samples = sps  # 1 bit period in samples
    n_per_filter = 2 * Tb_samples  # matched filter length

    # Matched filter impulse responses (time-reversed half-cosine pulses).
    t_axis = np.arange(n_per_filter) / Tb_samples  # 0 .. 2
    cos_pulse = np.cos(np.pi * (t_axis - 1.0) / 2.0)  # peak at t=Tb
    sin_pulse = np.sin(np.pi * t_axis / 2.0)          # peak at t=Tb

    i_arm = rx.real
    q_arm = rx.imag

    yi = np.convolve(i_arm, cos_pulse[::-1], mode="same")
    yq = np.convolve(q_arm, sin_pulse[::-1], mode="same")

    # Sample even bits at t = 2k·Tb on I, odd bits at t = (2k+1)·Tb on Q.
    n_pairs = n_bits // 2
    even_idx = (np.arange(n_pairs) * 2 + 1) * Tb_samples
    odd_idx = (np.arange(n_pairs) * 2 + 2) * Tb_samples
    even_idx = np.clip(even_idx, 0, len(rx) - 1)
    odd_idx = np.clip(odd_idx, 0, len(rx) - 1)

    even_bits = (yi[even_idx] > 0).astype(int)
    odd_bits = (yq[odd_idx] > 0).astype(int)

    bits = np.empty(2 * n_pairs, dtype=int)
    bits[0::2] = even_bits
    bits[1::2] = odd_bits
    return bits


def performance(full: bool = False) -> ResultTable:
    t = ResultTable("GMSK — Performance")

    # S1 — BER vs Q(√(2·Eb/N0)) using a coherent matched-filter MSK receiver.
    # Theory line is the optimum coherent MSK BER [proakis2008:eq4.4-43].
    # For BT=0.3 GMSK the coherent receiver tracks theory within ~0.5–1 dB
    # (the gap is the Gaussian-filter ISI loss).
    n_bits = 200_000 if full else 50_000
    n_bits -= n_bits % 2  # even count for I/Q interleaving
    tol_db = 0.8 if full else 1.0
    ebn0_db = np.arange(0.0, 11.0, 1.0)
    measured_ber = np.zeros(len(ebn0_db))

    sps = SAMPLES_PER_SYMBOL
    rng = np.random.default_rng(0)
    for i, eb in enumerate(ebn0_db):
        bits = rng.integers(0, 2, size=n_bits, endpoint=False)
        tx = _make_gmsk_signal(bits, sps, BT)
        # Eb / N0 normalisation: signal has unit amplitude so energy per
        # sample = 1; energy per bit Eb = sps. Setting noise variance per
        # complex dimension σ² = sps/(2·Eb/N0_lin) makes Eb/N0_measured
        # match Eb/N0_target.
        ebn0_lin = 10.0 ** (eb / 10.0)
        sigma = np.sqrt(sps / (2.0 * ebn0_lin))
        noise = sigma * (rng.standard_normal(len(tx)) + 1j * rng.standard_normal(len(tx)))
        rx = tx + noise.astype(np.complex64)
        bits_hat = _coherent_msk_demod(rx, sps, n_bits)
        errors = int(np.sum(bits_hat != bits[: len(bits_hat)]))
        measured_ber[i] = float(max(errors / len(bits_hat), 1.0 / len(bits_hat)))

    theory_ber = _q(np.sqrt(2.0 * 10.0 ** (ebn0_db / 10.0)))
    floor = 1.0 / n_bits
    meas_db = 10.0 * np.log10(np.maximum(measured_ber, floor))
    theo_db = 10.0 * np.log10(np.maximum(theory_ber, floor))
    max_off = float(np.max(np.abs(meas_db - theo_db)))
    t.add(
        "S1", "max |Δ| BER vs theory (dB) over [0, 10] dB",
        measured=max_off, expected=0.0, tol=tol_db,
        cite="proakis2008:eq4.4-43", units="dB",
    )

    plot_theory_overlay(
        measured_ber, theory_ber, ebn0_db,
        xlabel="Eb/N0 (dB)", ylabel="BER",
        title="GMSK BER vs theory (AWGN, coherent receiver)",
        measured_label="measured (coherent MF)",
        theory_label="Q(√(2·Eb/N0))",
    )
    save_verification_figure("gmsk_S1_ber.png")

    return t


if __name__ == "__main__":
    sys.exit(run_script(properties, performance))
```

- [ ] **Step 2: Run the verifier in quick mode and confirm all rows pass**

Run: `python examples/verification/verify_gmsk.py`
Expected: All rows report PASS. The script exits 0.

If a row fails, do not loosen the tolerance to make it pass. Read the printed measured vs expected and diagnose. The most likely cause of a near-pass with a small mismatch is the P5 OBW measurement integration limits — confirm `measure_obw` uses two-sided integration matching the `1.5·R_s` expectation. If the mismatch is large, re-run Task 2 and confirm the upsample fix landed.

- [ ] **Step 3: Run the verifier in `--full` mode**

Run: `python examples/verification/verify_gmsk.py --full`
Expected: All rows PASS; S1 tolerance tightens to 0.5 dB and still passes.

- [ ] **Step 4: Confirm the pytest verification entry point still works**

Run: `pytest -m verification tests/verification/test_verify_gmsk.py -v`
Expected: PASSED.

Run: `pytest -m "verification and slow" tests/verification/test_verify_gmsk.py -v`
Expected: PASSED.

- [ ] **Step 5: Commit the verifier revert and regenerated figure**

```bash
git add examples/verification/verify_gmsk.py assets/verification/gmsk_S1_ber.png
git commit -m "verify(gmsk): revert workarounds to textbook tolerances

The upsample fix in fsk.py restores h = 0.5, so the verifier no longer
needs the empirical OBW reference, the spectral-compactness regression
guard, or the BER threshold at 40 dB. P2 expects π/2 rad/symbol; P4
uses the Laurent prediction ~0.5·R_s; P5 uses 1.5·R_s; S1 is BER vs
Q(√(2·Eb/N0)) over [0, 10] dB with the standard 0.5/0.8 dB tolerances."
```

---

### Task 4: Update README finding for GMSK

**Files:**
- Modify: `examples/verification/README.md`

- [ ] **Step 1: Replace the GMSK paragraph in the "Per-Waveform Evidence" section**

In `examples/verification/README.md`, find the GMSK section (search for `### GMSK — \`verify_gmsk.py\``) and replace its bullets / caption with the textbook description:

```markdown
### GMSK — `verify_gmsk.py`

CPM with Gaussian pulse shaping; modulation index h = 0.5. Strongest evidence:

- **P1** Constant envelope: std(|s|)/mean(|s|) ≤ 1e-3 (CPM definition)
- **P2** Steady-state |Δφ|/symbol = π/2 within 1 % on a constant-bit stream (Proakis 2008, §4.4-3)
- **S1** BER vs Q(√(2·Eb/N0)) over [0, 10] dB, max |Δ| ≤ 0.5 dB at full mode (Proakis 2008, eq. 4.4-43)

```python
from spectra import GMSK
wf = GMSK(samples_per_symbol=8, bt=0.3)
iq = wf.generate(num_symbols=4096, sample_rate=1e6, seed=0)
```

```bash
python examples/verification/verify_gmsk.py        # quick mode
python examples/verification/verify_gmsk.py --full # publication-grade
```

![GMSK BER vs theory](../../assets/verification/gmsk_S1_ber.png)
*GMSK BER measured over AWGN vs. theoretical Q(√(2·Eb/N0)) (Proakis eq. 4.4-43).*
```

- [ ] **Step 2: Update the "Notes on Findings" GMSK entry**

In the same file, find the `## Notes on Findings` section and replace the GMSK finding (item 1) with:

```markdown
1. **GMSK previously produced h_eff = 0.5/sps = 0.0625, not the standard h = 0.5.** Root cause: zero-insertion upsampling combined with a sum-normalised Gaussian filter in `python/spectra/waveforms/fsk.py`. Fixed by switching to repeat-upsampling (matching `MSK.generate` and `FSK.generate`); verifier now uses textbook MSK tolerances. Regression guarded by `tests/test_waveforms_fsk.py::TestGMSKModulationIndex`.
```

- [ ] **Step 3: Update the waveform-coverage table row**

In the same file, find the table row for `verify_gmsk.py` and replace the "Strongest evidence" cell with:

```
| `verify_gmsk.py`     | CPM               | h = 0.5 steady-state; constant envelope; BER vs Q(√(2·Eb/N0)) |
```

- [ ] **Step 4: Commit the README update**

```bash
git add examples/verification/README.md
git commit -m "docs(verification): rewrite GMSK finding for h = 0.5 fix

GMSK now uses h = 0.5; the README finding reflects the fix, the
per-waveform bullets match the new verifier, and the coverage-table
entry is updated."
```

---

### Task 5: Final check on PR-A scope

- [ ] **Step 1: Run the affected tests one more time**

Run: `pytest tests/test_waveforms_fsk.py tests/verification/test_verify_gmsk.py -v`
Expected: all green.

- [ ] **Step 2: Confirm `git log --oneline` shows four PR-A commits**

```bash
git log --oneline -n 4
```

Expected (most recent first):
```
<hash> docs(verification): rewrite GMSK finding for h = 0.5 fix
<hash> verify(gmsk): revert workarounds to textbook tolerances
<hash> fix(gmsk): restore modulation index h = 0.5
<hash> test(gmsk): regression test for steady-state h = 0.5 (red)
```

- [ ] **Step 3: PR-A is complete. Stop here for review before starting PR-B.**

The author / reviewer opens PR-A against `main`. PR-B starts only after PR-A is approved (or queued).

---

# PR-B — 16-QAM Gray-Coding Fix (BREAKING)

### Task 6: Downstream audit before touching code

This audit confirms the spec's assumption that the integer label → constellation point mapping is only contractual *inside* SPECTRA. If the audit surfaces a persisted artefact pinning the old row-major mapping, scope expands to its regeneration.

**Files:**
- No edits; produces a written audit summary committed in Task 13.

- [ ] **Step 1: Search Python sources for QAM label / index consumers**

Run:
```bash
grep -rn "generate_qam_symbols_with_indices\|get_qam_constellation" python/ tests/
```

For each hit outside `rust/src/modulators.rs` and `examples/verification/verify_qam16.py`, open the file and confirm one of these is true:
  (a) the call reads the constellation from `get_qam_constellation` and uses it within the same scope — internally consistent, no fix needed.
  (b) the call passes labels to a receiver that re-reads the constellation — internally consistent, no fix needed.
  (c) the call hard-codes an integer-to-point assumption — **flag**, expand scope.

Record the file and outcome in a temporary note (paste into `audit-notes.txt` in the worktree; this file is not committed).

- [ ] **Step 2: Search Python sources for hard-coded QAM IQ samples**

Run:
```bash
grep -rn "QAM16\|qam_16\|qam16" tests/ | grep -v "label\|assert_valid_iq\|isinstance\|class TestQAM"
```

For each hit, confirm the test asserts only structural properties (shape, label, bandwidth, valid IQ). If a test asserts pinned complex sample values for a given seed, **flag** — update the test in Task 11.

A prior scan (recorded in the spec) showed all current `tests/test_*qam*.py` and `tests/test_waveforms_*.py` QAM tests are structural. Re-confirm.

- [ ] **Step 3: Search benchmark YAML configs for hard-coded label assumptions**

Run:
```bash
grep -rn "QAM16\|qam16\|qam_16" python/spectra/benchmarks/configs/
```

Configs reference QAM by class name. Confirm none embed integer label tables. If any do, **flag** — would block PR-B.

- [ ] **Step 4: Search for saved classifier weight files or fixtures**

Run:
```bash
find . -name "*.pt" -o -name "*.pkl" -o -name "*.joblib" 2>/dev/null | grep -v ".venv"
```

Expected outcome: no committed model artefacts in the repo. If any exist and they were trained on the row-major QAM mapping, **flag** — needs regeneration plan.

- [ ] **Step 5: Summarise the audit**

Create `audit-notes.txt` with three sections:
  - `internal_consumers`: list of files that re-read the constellation; no fix needed.
  - `flagged_artefacts`: anything matching category (c), pinned snapshots, or saved models. Must be empty before proceeding.
  - `tests_to_update`: structural tests that mention `QAM16` but assert only shape/label/bandwidth. These do NOT need editing — listed only for traceability.

If `flagged_artefacts` is non-empty, STOP. Surface the finding to the user before continuing — the spec promises a clean break only if the audit confirms blast radius is internal.

---

### Task 7: Add failing Rust test for Gray adjacency

**Files:**
- Modify: `rust/src/modulators.rs` (add new `#[test]` in the existing `#[cfg(test)]` mod)

- [ ] **Step 1: Append the failing test inside the existing `tests` mod**

In `rust/src/modulators.rs`, locate the `#[cfg(test)] mod tests { ... }` block (it already contains `qam16_constellation_properties` near line 482). Add this test inside the same mod:

```rust
    #[test]
    fn qam_constellation_gray_adjacency() {
        // For Gray-coded square M-QAM, every pair of physical
        // nearest neighbours has integer labels whose Gray codes
        // differ by exactly one bit [proakis2008:§4.3.2].
        for order in [16usize, 64usize, 256usize] {
            let constellation = build_qam_constellation(order).unwrap();
            let side = (order as f64).sqrt() as usize;

            // Nearest-neighbour spacing equals the minimum |Δ| in the
            // normalised grid: in raw coords it is 2; after
            // normalisation by sqrt(avg_power), the spacing is
            // 2 / sqrt(avg_power_raw). Determine empirically by
            // taking the smallest non-zero distance in the constellation.
            let mut min_d2 = f32::INFINITY;
            for (i, a) in constellation.iter().enumerate() {
                for b in constellation.iter().skip(i + 1) {
                    let dr = a.re - b.re;
                    let di = a.im - b.im;
                    let d2 = dr * dr + di * di;
                    if d2 > 1e-6 && d2 < min_d2 {
                        min_d2 = d2;
                    }
                }
            }

            // Every nearest-neighbour pair must have Gray-adjacent labels.
            for (i, a) in constellation.iter().enumerate() {
                for (j, b) in constellation.iter().enumerate().skip(i + 1) {
                    let dr = a.re - b.re;
                    let di = a.im - b.im;
                    let d2 = dr * dr + di * di;
                    if (d2 - min_d2).abs() <= 1e-4 {
                        let xor = (i ^ j) as u32;
                        assert_eq!(
                            xor.count_ones(),
                            1,
                            "order={order}: labels {i} and {j} are physical nearest neighbours but their integer XOR popcount = {} (expected 1 for Gray-coded)",
                            xor.count_ones()
                        );
                    }
                }
            }

            // Sanity: the side count matches the perfect-square invariant.
            assert_eq!(side * side, order);
        }
    }
```

- [ ] **Step 2: Run the new test and confirm it fails on the current row-major code**

Run: `cargo test --manifest-path rust/Cargo.toml qam_constellation_gray_adjacency`
Expected: FAILED. The error message points at a pair like `(3, 4)` for M=16 whose XOR popcount is 3, not 1.

- [ ] **Step 3: Commit the failing test**

```bash
git add rust/src/modulators.rs
git commit -m "test(qam): regression test for Gray adjacency (red)

Asserts that physical nearest-neighbour pairs on every square M-QAM
constellation (M ∈ {16, 64, 256}) have integer labels differing by
exactly one bit. Currently fails because build_qam_constellation
uses row-major labelling. Fix in the next commit."
```

---

### Task 8: Replace `build_qam_constellation` with the Gray-bit-split construction

**Files:**
- Modify: `rust/src/modulators.rs:27-42`
- Modify: `rust/src/modulators.rs:482-513` (existing `qam16_constellation_properties` test mirrors the row-major loop; update it)

- [ ] **Step 1: Replace the function body**

In `rust/src/modulators.rs`, replace the existing `build_qam_constellation` (lines 27-42) with:

```rust
/// Build a normalised Gray-coded QAM constellation for the given order.
///
/// Square M-QAM only (M = 2^(2n)). Integer label k indexes the constellation
/// such that physical nearest neighbours have labels whose Gray codes differ
/// in exactly one bit [proakis2008:§4.3.2]. The constellation is normalised
/// to unit average symbol energy.
fn build_qam_constellation(order: usize) -> Result<Vec<Complex32>, String> {
    let side = (order as f64).sqrt() as usize;
    if side * side != order {
        return Err("QAM order must be a perfect square (16, 64, 256, ...)".to_string());
    }
    let n = (side as f64).log2() as u32;
    if (1usize << n) != side {
        return Err("QAM order must be 2^(2n) (16, 64, 256, 1024)".to_string());
    }
    let mask = (1usize << n) - 1;
    let mut constellation = Vec::with_capacity(order);
    for k in 0..order {
        // Gray(k) = k XOR (k>>1). Top n bits index the I-axis Gray level;
        // bottom n bits index the Q-axis Gray level. Because Gray code is
        // reflective, the resulting (i_level, q_level) sequence traces a
        // boustrophedon path through the grid with single-bit transitions
        // between physical neighbours.
        let g = k ^ (k >> 1);
        let i_level = (g >> n) & mask;
        let q_level = g & mask;
        let re = 2.0 * i_level as f64 - (side - 1) as f64;
        let im = 2.0 * q_level as f64 - (side - 1) as f64;
        constellation.push(Complex32::new(re as f32, im as f32));
    }
    normalize_constellation(&mut constellation);
    Ok(constellation)
}
```

- [ ] **Step 2: Update `qam16_constellation_properties` to use the actual function**

The existing inline test (lines 482-513) re-implements the row-major loop and asserts only normalisation. Replace its body to call the function under test:

```rust
    #[test]
    fn qam16_constellation_properties() {
        // After Gray-coding, the constellation still has 16 points and unit
        // average power. The exact (Re, Im) values differ from the row-major
        // version but the geometric invariants do not.
        let constellation = build_qam_constellation(16).unwrap();
        assert_eq!(constellation.len(), 16);
        let avg_power: f64 = constellation
            .iter()
            .map(|c| (c.re * c.re + c.im * c.im) as f64)
            .sum::<f64>()
            / 16.0;
        assert!((avg_power - 1.0).abs() < 1e-5, "avg power = {avg_power}");

        // Constellation occupies the same 4-level grid ±{1, 3}/√10
        // (only the label-to-point mapping changed).
        let expected_levels: Vec<f32> = [-3.0, -1.0, 1.0, 3.0]
            .iter()
            .map(|x| (*x as f64 / (10.0_f64).sqrt()) as f32)
            .collect();
        let mut re_levels: Vec<f32> = constellation.iter().map(|c| c.re).collect();
        re_levels.sort_by(|a, b| a.partial_cmp(b).unwrap());
        re_levels.dedup_by(|a, b| (*a - *b).abs() < 1e-4);
        assert_eq!(re_levels.len(), 4);
        for (a, b) in re_levels.iter().zip(expected_levels.iter()) {
            assert!((a - b).abs() < 1e-4, "level {a} vs {b}");
        }
    }
```

- [ ] **Step 3: Build the Rust extension into the venv**

Run: `maturin develop --release`
Expected: build succeeds; final line reports installation under `.venv/lib/.../spectra/`.

- [ ] **Step 4: Run the new Rust tests**

Run: `cargo test --manifest-path rust/Cargo.toml qam`
Expected: `qam_constellation_gray_adjacency` PASSES; `qam16_constellation_properties` PASSES.

- [ ] **Step 5: Run the full Rust test suite**

Run: `cargo test --manifest-path rust/Cargo.toml`
Expected: all tests PASS.

- [ ] **Step 6: Run the Python QAM tests**

Run: `pytest tests/test_waveforms_qam.py tests/test_rust_modulators.py -v`
Expected: all PASS. These are structural tests; the Gray-coding change does not affect shape, label, bandwidth, or average power.

- [ ] **Step 7: Commit the fix**

```bash
git add rust/src/modulators.rs
git commit -m "fix(qam): Gray-code the square M-QAM constellation (BREAKING)

Replace the row-major label-to-point mapping in
build_qam_constellation with a Gray-bit-split construction
(top n bits index I-axis Gray level, bottom n bits index Q-axis).
Physical nearest neighbours now have integer labels differing in
exactly one bit, restoring the standard BER ≈ SER/log₂(M)
relationship at moderate-to-high SNR.

BREAKING: integer label → constellation point assignment has
changed. Datasets and classifiers trained on the prior mapping
must be regenerated. Audit confirms all internal SPECTRA
consumers re-read build_qam_constellation, so the round-trip
through link/receiver code is internally consistent."
```

---

### Task 9: Add `P_gray` and `S_ber_ser` to `verify_qam16.py`

**Files:**
- Modify: `examples/verification/verify_qam16.py`

- [ ] **Step 1: Update the docstring**

Replace the docstring header (lines 1-64) with:

```python
"""SPECTRA Verification — 16-QAM
=================================
Proves that the generated 16-QAM waveform satisfies:

  P1.  Real-axis levels at ±1/√10 and ±3/√10 (normalised grid).  [proakis2008:§4.3]
  P1b. Imag-axis levels identical to real-axis levels.            [proakis2008:§4.3]
  P2.  Average symbol energy ≈ 1.0 (energy-normalised symbols).  [proakis2008:§4.3]
  P3.  Gray adjacency: every pair of physical nearest neighbours has integer
       labels differing in exactly one bit.                       [proakis2008:§4.3.2]
  P4.  Bandwidth = (1+α)·R_s within 1 %.                         [sklar2001:§3.5,eq3.74]
  P5.  PSD shape correlation with squared-RRC ≥ 0.99.            [proakis2008:eq9.2-37]
  P6.  OBW (99 %) within 5 % of theoretical 99 %-OBW.            [itu_sm_328:§3]
  P7.  ACLR at ±2·R_s offset ≥ 45 dB.                           [3gpp_38_104:T6.6.3.1-1]
  S1.  SER vs Eb/N0 ∈ [4, 11] dB, max |Δ| ≤ 0.8 dB (quick) /
       0.5 dB (full).                                            [proakis2008:eq4.3-30]
  S2.  EVM at SNR=40 dB ≤ 1.1 % RMS.                            [3gpp_38_104:§B.2]
  S3.  BER ≈ SER/log₂(M) at high SNR: |BER − SER/4| ≤ 5e-3 at Eb/N0 = 11 dB.
                                                                  [proakis2008:§4.3.2]

Run:
    python examples/verification/verify_qam16.py            # quick mode
    python examples/verification/verify_qam16.py --full     # publication mode
"""
```

- [ ] **Step 2: Insert `P3` (Gray adjacency) into `properties()`**

In `examples/verification/verify_qam16.py`, find the block that currently reads `# P3 — Gray adjacency (SKIPPED: ...)`. Replace it with:

```python
    # P3 — Gray adjacency. For every pair of physical nearest neighbours
    # on the normalised constellation, the integer label XOR popcount must
    # equal exactly 1. Nearest-neighbour spacing is 2/√10 (raw grid step 2
    # divided by sqrt(avg power 10)).
    const = get_qam_constellation(16)
    min_d2 = np.inf
    for i in range(len(const)):
        for j in range(i + 1, len(const)):
            d2 = float(np.abs(const[i] - const[j]) ** 2)
            if d2 > 1e-6 and d2 < min_d2:
                min_d2 = d2
    violations = 0
    for i in range(len(const)):
        for j in range(i + 1, len(const)):
            d2 = float(np.abs(const[i] - const[j]) ** 2)
            if abs(d2 - min_d2) <= 1e-4:
                if int(i ^ j).bit_count() != 1:
                    violations += 1
    t.add(
        "P3", "nearest-neighbour Gray-adjacency violations",
        measured=violations, expected=0, tol=0,
        cite="proakis2008:§4.3.2",
    )
```

- [ ] **Step 3: Tighten the S1 SER tolerance**

In `performance()`, change the two tolerance values so the function's `tol_db` assignment reads:

```python
    tol_db = 0.5 if full else 0.8
```

(was `0.5 / 1.0`). The Gray-coded constellation no longer has the row-major asymmetry that motivated the wider quick-mode bound.

- [ ] **Step 4: Add `S3` BER vs SER/log₂(M) check**

In `performance()`, after the existing `S2` block but before `return t`, append:

```python
    # S3 — At high SNR with Gray labelling, single-symbol errors are dominated
    # by nearest-neighbour transitions which flip exactly one of log₂(M)=4
    # bits. So BER ≈ SER/4 within statistical noise.
    n_s3 = 1_000_000 if full else 200_000
    ebn0_s3_db = 11.0
    ebn0_s3_lin = 10.0 ** (ebn0_s3_db / 10.0)
    tx_syms, tx_idx = generate_qam_symbols_with_indices(n_s3, 16, seed=42)
    Es = float(np.mean(np.abs(tx_syms) ** 2))
    sigma = np.sqrt(Es / (2.0 * 4 * ebn0_s3_lin))  # k = log2(16) = 4
    rng_s3 = np.random.default_rng(42)
    noise = sigma * (rng_s3.standard_normal(n_s3) + 1j * rng_s3.standard_normal(n_s3))
    rx = tx_syms + noise.astype(np.complex64)
    dists = np.abs(rx[:, None] - const[None, :])
    rx_idx = np.argmin(dists, axis=1)
    ser_s3 = float(np.mean(rx_idx != tx_idx))
    # Bit errors per symbol: popcount(rx_idx XOR tx_idx).
    xors = (rx_idx.astype(np.int64) ^ tx_idx.astype(np.int64))
    bit_errors = int(np.sum([int(x).bit_count() for x in xors]))
    ber_s3 = bit_errors / (n_s3 * 4)
    diff = abs(ber_s3 - ser_s3 / 4.0)
    t.add(
        "S3", f"|BER − SER/log₂(M)| at Eb/N0 = {ebn0_s3_db:.0f} dB",
        measured=diff, expected=0.0, tol=5e-3,
        cite="proakis2008:§4.3.2",
    )
```

- [ ] **Step 5: Run the verifier in quick mode**

Run: `python examples/verification/verify_qam16.py`
Expected: all rows P1, P1b, P2, P3, P4, P5, P6, P7, S1, S2, S3 PASS.

If P3 reports a violation count > 0, the Gray-coding fix did not land — re-run Task 8. If S3 fails with a small positive diff (e.g., 0.006–0.010), the symbol count may be insufficient; the quick-mode 200 k samples give about 4–5 expected bit errors per nearest-neighbour pair, enough for the 5e-3 tolerance.

- [ ] **Step 6: Run the verifier in `--full` mode**

Run: `python examples/verification/verify_qam16.py --full`
Expected: all rows PASS; S1 tolerance tightens to 0.5 dB and still passes.

- [ ] **Step 7: Run the pytest verification entry points**

Run: `pytest -m verification tests/verification/test_verify_qam16.py -v`
Run: `pytest -m "verification and slow" tests/verification/test_verify_qam16.py -v`
Expected: both PASS.

- [ ] **Step 8: Commit the verifier update and regenerated figures**

The two PNGs (`qam16_P5_psd.png`, `qam16_S1_ser.png`) are regenerated by Steps 5–6 above.

```bash
git add examples/verification/verify_qam16.py \
        assets/verification/qam16_P5_psd.png \
        assets/verification/qam16_S1_ser.png
git commit -m "verify(qam16): add P3 (Gray adjacency) and S3 (BER↔SER coupling)

The Gray-coding fix in rust/src/modulators.rs lets the verifier:
  - assert P3: no nearest-neighbour pair violates Gray adjacency
  - tighten S1 SER tolerance to match BPSK/QPSK (0.5/0.8 dB)
  - add S3: |BER − SER/log₂(M)| ≤ 5e-3 at Eb/N0 = 11 dB

Five paragraphs of 'check skipped because Rust uses row-major' prose
deleted from the docstring."
```

---

### Task 10: Update README finding for 16-QAM

**Files:**
- Modify: `examples/verification/README.md`

- [ ] **Step 1: Replace the 16-QAM section's bullet list**

In `examples/verification/README.md`, find `### 16-QAM — \`verify_qam16.py\`` and replace its body with:

```markdown
### 16-QAM — `verify_qam16.py`

High-order linear modulation with 16 Gray-coded energy-normalised points (levels ±1/√10, ±3/√10). Strongest evidence:

- **P3** Every pair of physical nearest neighbours has integer labels differing in exactly one bit (Proakis 2008, §4.3.2)
- **P5** PSD shape correlation ≥ 0.99 with theoretical squared-RRC (Proakis 2008, eq. 9.2-37)
- **S1** SER vs Eb/N0 ∈ [4, 11] dB, max |Δ| ≤ 0.5 dB at full mode (Proakis 2008, eq. 4.3-30)
- **S3** |BER − SER/log₂(M)| ≤ 5e-3 at Eb/N0 = 11 dB (Proakis 2008, §4.3.2)

```python
from spectra import QAM16
wf = QAM16(samples_per_symbol=8, rolloff=0.35)
iq = wf.generate(num_symbols=4096, sample_rate=1e6, seed=0)
```

```bash
python examples/verification/verify_qam16.py
python examples/verification/verify_qam16.py --full
```

![16-QAM PSD vs theory](../../assets/verification/qam16_P5_psd.png)
*16-QAM PSD overlay: measured Welch PSD vs. theoretical squared-RRC (Proakis eq. 9.2-37). Correlation ≥ 0.99.*

![16-QAM SER vs theory](../../assets/verification/qam16_S1_ser.png)
*16-QAM SER measured over AWGN vs. theoretical formula (Proakis eq. 4.3-30).*
```

- [ ] **Step 2: Update the "Notes on Findings" 16-QAM entry**

Find the `## Notes on Findings` section and replace item 2 (the 16-QAM finding) with:

```markdown
2. **16-QAM (and all square M-QAM ≥ 16) is now Gray-coded.** Root cause: `build_qam_constellation` in `rust/src/modulators.rs` swept the I/Q grid in row-major order, so adjacent integer labels were not physical neighbours and the BER↔SER relationship deviated from `BER ≈ SER/log₂(M)` by up to a factor of `log₂(M)` at moderate-to-high SNR. Fixed by mapping integer label k → Gray(k) and splitting the top/bottom n bits across the I/Q axes. **BREAKING:** datasets and classifiers trained on the prior mapping must be regenerated. Regression guarded by `rust/src/modulators.rs::qam_constellation_gray_adjacency` and `verify_qam16.py::P3`.
```

- [ ] **Step 3: Update the waveform-coverage table row**

Find the `verify_qam16.py` row and replace the "Strongest evidence" cell with:

```
| `verify_qam16.py`    | Linear high-order | Gray adjacency; SER-vs-theory; BER↔SER coupling; EVM |
```

- [ ] **Step 4: Commit the README update**

```bash
git add examples/verification/README.md
git commit -m "docs(verification): rewrite 16-QAM finding for Gray-coding fix"
```

---

### Task 11: Add CHANGELOG entries

**Files:**
- Modify: `CHANGELOG.md`

- [ ] **Step 1: Add a new top-level section above the existing `## [Unreleased]`**

The existing CHANGELOG already has `## [Unreleased]` with `### Added` entries. Insert two new headed sub-sections *inside* the existing `[Unreleased]` block, immediately under the `## [Unreleased]` line and above the existing `### Added` line:

```markdown
### Changed (BREAKING)

- 16-QAM (and all square M-QAM ≥ 16) constellation labelling is now Gray-coded. The prior row-major mapping produced a BER↔SER mismatch of up to log₂(M) at moderate-to-high SNR. Saved datasets and classifiers trained on the prior mapping must be regenerated. See `examples/verification/verify_qam16.py::P3` and `rust/src/modulators.rs::qam_constellation_gray_adjacency`.

### Fixed

- `GMSK` modulation index restored to h = 0.5; was previously h_eff = 0.5/sps = 0.0625 due to zero-insertion upsampling combined with a sum-normalised Gaussian filter. Affects spectral occupancy and BER curves for `sp.GMSK`. Regression guarded by `tests/test_waveforms_fsk.py::TestGMSKModulationIndex`.

```

(The existing `### Added` block stays unchanged below.)

- [ ] **Step 2: Commit the CHANGELOG**

```bash
git add CHANGELOG.md
git commit -m "docs(changelog): record GMSK fix and QAM Gray-coding break"
```

---

### Task 12: Bump version to 0.2.0

**Files:**
- Modify: `pyproject.toml:7`
- Modify: `rust/Cargo.toml:3`

- [ ] **Step 1: Bump the Python version**

In `pyproject.toml`, change line 7 from:
```
version = "0.1.0"
```
to:
```
version = "0.2.0"
```

- [ ] **Step 2: Bump the Rust crate version**

In `rust/Cargo.toml`, change line 3 from:
```
version = "0.1.0"
```
to:
```
version = "0.2.0"
```

- [ ] **Step 3: Rebuild and re-run the test suite to confirm the bump is consistent**

Run: `maturin develop --release`
Expected: build succeeds and installs `spectra 0.2.0`.

Run: `pytest tests/ -v --co -q | tail -5`
Expected: pytest collects tests without import errors.

- [ ] **Step 4: Commit the version bump**

```bash
git add pyproject.toml rust/Cargo.toml
git commit -m "chore: bump to 0.2.0 for QAM Gray-coding BREAKING change"
```

---

### Task 13: Final verification pass and clean up audit notes

**Files:**
- Delete: `audit-notes.txt` (working-tree-only file from Task 6)

- [ ] **Step 1: Run the full verification suite**

Run: `pytest -m verification tests/verification/ -v`
Expected: all PASS.

Run: `pytest -m "verification and slow" tests/verification/ -v`
Expected: all PASS.

- [ ] **Step 2: Run the full Python test suite (excluding slow)**

Run: `pytest tests/ -v -m "not slow"`
Expected: all PASS, no regressions outside the verification suite.

- [ ] **Step 3: Run the full Rust suite**

Run: `cargo test --manifest-path rust/Cargo.toml`
Expected: all PASS.

- [ ] **Step 4: Confirm `git status` is clean except for the audit-notes file**

Run: `git status`
Expected: only `audit-notes.txt` shows as untracked (it should never have been committed).

- [ ] **Step 5: Delete the audit notes**

Run: `rm audit-notes.txt`

- [ ] **Step 6: Confirm `git log --oneline` shows the expected PR-B sequence**

```bash
git log --oneline -n 9
```

Expected (most recent first; the bottom four are PR-A):
```
<hash> chore: bump to 0.2.0 for QAM Gray-coding BREAKING change
<hash> docs(changelog): record GMSK fix and QAM Gray-coding break
<hash> docs(verification): rewrite 16-QAM finding for Gray-coding fix
<hash> verify(qam16): add P3 (Gray adjacency) and S3 (BER↔SER coupling)
<hash> fix(qam): Gray-code the square M-QAM constellation (BREAKING)
<hash> test(qam): regression test for Gray adjacency (red)
<hash> docs(verification): rewrite GMSK finding for h = 0.5 fix
<hash> verify(gmsk): revert workarounds to textbook tolerances
<hash> fix(gmsk): restore modulation index h = 0.5
<hash> test(gmsk): regression test for steady-state h = 0.5 (red)
```

- [ ] **Step 7: PR-B is complete.** Open against `main` after PR-A merges. The PR description should include the audit findings from Task 6 Step 5 verbatim.

---

## Self-Review Notes

- **Spec coverage:** Every requirement in the spec maps to a task. Track A: §"Track A — GMSK fix" → Tasks 1–5. Track B: §"Track B — 16-QAM Gray-coding fix" → Tasks 6–13. CHANGELOG and version bump → Tasks 11–12. Audit → Task 6.
- **Placeholder scan:** No TBDs, no "implement later". Every code step has the actual content.
- **Type consistency:** Rust function name `build_qam_constellation` matches across Task 7 (test calls it) and Task 8 (definition). Python helper `_make_gmsk_signal` is defined in Task 3 and only referenced there. `H_GMSK` constant introduced in Task 3 and used consistently. `get_qam_constellation`, `generate_qam_symbols_with_indices`, `generate_qam_symbols` are imported from `spectra._rust` in `verify_qam16.py` and consistently named.
- **Citation keys** (`proakis2008:§4.4-3`, `laurent1986:§III`, `gmsk:cpm-defn`, `gmsk:gaussian`, `itu_sm_328:§3`, `proakis2008:eq4.4-43`, `proakis2008:§4.3.2`, `proakis2008:eq4.3-30`, `3gpp_38_104:§B.2`, `3gpp_38_104:T6.6.3.1-1`, `sklar2001:§3.5,eq3.74`, `proakis2008:eq9.2-37`) — all already present in `examples/verification/REFERENCES.md` and parsed at startup; no new bibliography entries needed.
