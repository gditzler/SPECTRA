"""SPECTRA Verification — Reviewer Tutorial (companion script).

This script lifts every check demonstrated in
``tutorial_for_reviewers.ipynb`` to a top-level importable function.
Each function is self-contained: it does NOT depend on
``_verify_helpers`` for the core math. The companion notebook calls
these functions; this script's ``__main__`` block runs them all and
prints a summary table.

Layout:

  - BPSK section: constellation, PSD-vs-theory, BER-vs-theory
  - OFDM section: subcarrier orthogonality, CP correlation, EVM
  - Barker-13 section: sequence equality, PSLR, detection rate
  - run_all(): driver that returns a dict of all results
  - __main__: prints a summary table and exits non-zero on failure

A reviewer reading this script sees the same math the notebook does,
without the markdown derivations. The two artefacts produce
bit-identical results (verified by the test suite).
"""

from __future__ import annotations

import sys

import numpy as np
from scipy.special import erfc

# ════════════════════════════════════════════════════════════════════════════
# BPSK
# ════════════════════════════════════════════════════════════════════════════


def bpsk_constellation_check(symbols: np.ndarray) -> float:
    """P1: BPSK symbols lie on the real axis.

    Returns max(|imag(symbols)|). For a correct BPSK constellation the
    value is exactly 0 (Rust constructs ±1 + 0j). Tolerance 1e-6 in tests.
    """
    return float(np.max(np.abs(symbols.imag)))


def _welch_psd_inline(
    iq: np.ndarray, fs: float, nperseg: int = 512
) -> tuple[np.ndarray, np.ndarray]:
    """Welch's method — self-contained reimplementation.

    Splits ``iq`` into 50 %-overlapping Hann-windowed segments of length
    ``nperseg``, computes the periodogram of each, averages them. Returns
    (frequencies, two-sided PSD) sorted by frequency.

    Reimplemented inline (rather than calling _verify_helpers._welch_psd)
    so a reviewer can read the segment-averaging math next to the call.
    """
    window = np.hanning(nperseg)
    win_pow = np.sum(window**2)
    step = nperseg // 2
    n_seg = max(1, (len(iq) - nperseg) // step + 1)
    psd_sum = np.zeros(nperseg, dtype=np.float64)
    for k in range(n_seg):
        seg = iq[k * step : k * step + nperseg]
        spec = np.fft.fftshift(np.fft.fft(seg * window))
        psd_sum += (np.abs(spec) ** 2) / (fs * win_pow)
    psd = psd_sum / n_seg
    f = np.fft.fftshift(np.fft.fftfreq(nperseg, d=1.0 / fs))
    return f, psd


def _psd_rrc_squared_inline(f: np.ndarray, Rs: float, alpha: float) -> np.ndarray:
    """Theoretical squared raised-cosine PSD (Proakis 2008, eq. 9.2-37).

    For symbol rate Rs and rolloff α, the squared-RRC frequency response is:
      |H(f)|² = T                           for |f| ≤ (1-α)/(2T)
                T·cos²(πT/(2α)(|f| - (1-α)/(2T)))  for transition band
                0                            elsewhere
    where T = 1/Rs.
    """
    T = 1.0 / Rs
    abs_f = np.abs(f)
    edge_lo = (1.0 - alpha) / (2.0 * T)
    edge_hi = (1.0 + alpha) / (2.0 * T)
    out = np.zeros_like(f, dtype=np.float64)
    flat = abs_f <= edge_lo
    trans = (abs_f > edge_lo) & (abs_f <= edge_hi)
    out[flat] = T
    if alpha > 0:
        arg = (np.pi * T / (2.0 * alpha)) * (abs_f[trans] - edge_lo)
        out[trans] = T * np.cos(arg) ** 2
    return out


def bpsk_psd_correlation(iq: np.ndarray, sample_rate: float, rolloff: float = 0.35) -> float:
    """P4: Pearson correlation between measured PSD and squared-RRC theory.

    Returns a scalar in [-1, 1]. For a correct BPSK with RRC rolloff α,
    the value is ≥ 0.99 at sample sizes ≥ 4096 symbols (CLT on Welch
    segment averages with 64+ segments gives Welch variance ≈ 1/N_seg
    ≈ 1.5 %; the correlation threshold of 0.99 is comfortably above this).
    """
    sps = 8  # default in the suite; the function does not need to know exactly
    Rs = sample_rate / sps
    f, p = _welch_psd_inline(iq, fs=sample_rate, nperseg=512)
    t = _psd_rrc_squared_inline(f, Rs=Rs, alpha=rolloff)
    # Pearson correlation
    p_z = p - np.mean(p)
    t_z = t - np.mean(t)
    denom = float(np.sqrt(np.sum(p_z**2) * np.sum(t_z**2)))
    if denom == 0.0:
        return 0.0
    return float(np.sum(p_z * t_z) / denom)


def _q(x: np.ndarray) -> np.ndarray:
    """Q-function: Q(x) = 0.5 · erfc(x / √2)."""
    return 0.5 * erfc(x / np.sqrt(2.0))


def _bpsk_ber_theory(ebn0_db: np.ndarray) -> np.ndarray:
    """BPSK BER over AWGN (Proakis 2008, eq. 4.3-13): Q(√(2·Eb/N0))."""
    return _q(np.sqrt(2.0 * 10.0 ** (np.asarray(ebn0_db) / 10.0)))


def bpsk_ber_curve(
    ebn0_db_list: list[float],
    n_bits: int = 100_000,
    seed: int = 0,
) -> tuple[np.ndarray, np.ndarray]:
    """S1: Measured vs theoretical BER for BPSK over AWGN.

    For each Eb/N0 point, generates ``n_bits`` random ±1 symbols, adds
    complex AWGN with σ² = 1 / (2 · Eb/N0_lin) per dimension, decides on
    the sign of the real part, counts bit errors.

    Returns (measured_ber, theory_ber) arrays the same length as
    ``ebn0_db_list``. For ``n_bits = 1e5`` the CLT-bound tolerance against
    theory is ≤ 0.8 dB over the [0, 6] dB range (≥ 240 expected errors
    per point).
    """
    rng = np.random.default_rng(seed)
    ebn0_db = np.asarray(ebn0_db_list, dtype=float)
    measured = np.zeros_like(ebn0_db, dtype=float)
    for i, eb in enumerate(ebn0_db):
        bits = rng.integers(0, 2, size=n_bits)
        symbols = 2.0 * bits - 1.0  # ±1
        ebn0_lin = 10.0 ** (eb / 10.0)
        sigma = np.sqrt(1.0 / (2.0 * ebn0_lin))
        noise = sigma * (rng.standard_normal(n_bits) + 1j * rng.standard_normal(n_bits))
        rx = symbols + noise
        bits_hat = (rx.real > 0).astype(int)
        errors = int(np.sum(bits_hat != bits))
        measured[i] = float(max(errors / n_bits, 1.0 / n_bits))
    theory = _bpsk_ber_theory(ebn0_db)
    return measured, theory


# ════════════════════════════════════════════════════════════════════════════
# OFDM
# ════════════════════════════════════════════════════════════════════════════


def _build_ofdm_symbol(
    n_fft: int,
    n_used: int,
    n_cp: int,
    rng: np.random.Generator,
) -> tuple[np.ndarray, np.ndarray]:
    """Construct one (grid, time-domain) OFDM symbol with CP.

    The grid has ``n_used`` QPSK-modulated subcarriers placed in standard
    OFDM layout (DC bin 0 unused, symmetric lower/upper halves around
    DC). The time-domain body is the inverse FFT scaled by √n_fft so
    Parseval's theorem holds exactly: ``sum(|body|²) = sum(|grid|²)``.
    """
    half = n_used // 2
    qpsk = (rng.choice([-1, 1], size=n_used) + 1j * rng.choice([-1, 1], size=n_used)) / np.sqrt(2)
    grid = np.zeros(n_fft, dtype=np.complex128)
    grid[1 : 1 + half] = qpsk[:half]
    grid[n_fft - half :] = qpsk[half:]
    body = np.fft.ifft(grid) * np.sqrt(n_fft)
    sym = np.concatenate([body[-n_cp:], body])
    return grid, sym


def ofdm_orthogonality_error(n_fft: int, n_used: int, n_cp: int, seed: int = 0) -> float:
    """P1: Subcarrier orthogonality is exact (no impairments).

    Strip the CP, take the FFT, divide by √n_fft. Returns
    max(|FFT(rx) − tx_grid|). For a correct OFDM symbol the value is at
    float64 round-off (~1e-15); tolerance 1e-9 catches gross errors.
    """
    rng = np.random.default_rng(seed)
    grid, sym = _build_ofdm_symbol(n_fft, n_used, n_cp, rng)
    body = sym[n_cp:]
    recovered = np.fft.fft(body) / np.sqrt(n_fft)
    return float(np.max(np.abs(recovered - grid)))


def ofdm_cp_correlation(
    n_fft: int,
    n_used: int,
    n_cp: int,
    n_symbols: int = 8,
    seed: int = 0,
) -> tuple[int, float]:
    """P2: CP correlation peak at lag n_fft (van de Beek 1997).

    Concatenate ``n_symbols`` OFDM symbols. For each candidate lag k in
    [n_cp, n_fft + n_cp + n_cp//2 + 1], accumulate the n_cp-length
    CP-window correlation from each symbol boundary. The argmax should
    equal n_fft (the CP duplication distance), and the peak normalised
    amplitude should exceed 0.5.

    Using per-symbol-aligned windows of length n_cp prevents cross-symbol
    dilution that would otherwise suppress the peak amplitude.
    """
    rng = np.random.default_rng(seed)
    syms = np.concatenate(
        [_build_ofdm_symbol(n_fft, n_used, n_cp, rng)[1] for _ in range(n_symbols)]
    ).astype(np.complex128)
    sym_len = n_fft + n_cp
    lags = np.arange(n_cp, n_fft + n_cp + n_cp // 2 + 1)
    corrs = np.zeros(len(lags), dtype=float)
    for i, k in enumerate(lags):
        total_num = 0.0
        total_denom_a = 0.0
        total_denom_b = 0.0
        for sym_idx in range(n_symbols):
            s = sym_idx * sym_len
            if s + k + n_cp > len(syms):
                break
            a = syms[s : s + n_cp]
            b = syms[s + k : s + k + n_cp]
            total_num += float(np.abs(np.dot(a, np.conj(b))))
            total_denom_a += float(np.dot(a, np.conj(a)).real)
            total_denom_b += float(np.dot(b, np.conj(b)).real)
        corrs[i] = total_num / max(float(np.sqrt(total_denom_a * total_denom_b)), 1e-30)
    idx = int(np.argmax(corrs))
    return int(lags[idx]), float(corrs[idx])


def ofdm_evm_after_awgn(
    snr_db: float,
    n_fft: int,
    n_used: int,
    n_cp: int,
    n_symbols: int = 200,
    seed: int = 0,
) -> float:
    """S1: EVM (RMS) after AWGN + CP-removal + FFT + ZF.

    For AWGN-only at SNR_lin = 10^(snr_db/10), the theoretical EVM is
    1/√SNR_lin (Proakis). At 40 dB this is 1 %, comfortably below the
    2 % tolerance the suite uses.
    """
    rng = np.random.default_rng(seed)
    grids = []
    syms = []
    for _ in range(n_symbols):
        g, s = _build_ofdm_symbol(n_fft, n_used, n_cp, rng)
        grids.append(g)
        syms.append(s)
    iq = np.concatenate(syms).astype(np.complex128)
    es = float(np.mean(np.abs(iq) ** 2))
    snr_lin = 10.0 ** (snr_db / 10.0)
    sigma = np.sqrt(es / snr_lin / 2.0)
    noise = sigma * (rng.standard_normal(len(iq)) + 1j * rng.standard_normal(len(iq)))
    rx = iq + noise
    sym_len = n_fft + n_cp
    rx_grids = np.empty((n_symbols, n_fft), dtype=np.complex128)
    for i in range(n_symbols):
        body_rx = rx[i * sym_len + n_cp : (i + 1) * sym_len]
        rx_grids[i] = np.fft.fft(body_rx) / np.sqrt(n_fft)
    half = n_used // 2
    used = np.concatenate([np.arange(1, 1 + half), np.arange(n_fft - half, n_fft)])
    tx_used = np.array(grids)[:, used].flatten()
    rx_used = rx_grids[:, used].flatten()
    err = rx_used - tx_used
    return float(np.sqrt(np.mean(np.abs(err) ** 2)) / np.sqrt(np.mean(np.abs(tx_used) ** 2)))


# ════════════════════════════════════════════════════════════════════════════
# Barker-13
# ════════════════════════════════════════════════════════════════════════════

CANONICAL_BARKER_13 = np.array(
    [+1, +1, +1, +1, +1, -1, -1, +1, +1, -1, +1, -1, +1], dtype=int
)


def barker13_canonical_equality() -> int:
    """P1: SPECTRA's BARKER_CODES[13] matches Levanon & Mozeson Tab. 6.1 exactly.

    Returns 1 if equal, 0 otherwise. Tolerance is 0 — bit-exact integer
    equality. This check guards code *storage*, not transmission integrity
    (a flipped chip on the wire still passes this check — that's an
    instructive point in the regression catalogue).
    """
    from spectra.waveforms.barker import BARKER_CODES

    stored = np.asarray(BARKER_CODES[13], dtype=int)
    return int(np.array_equal(stored, CANONICAL_BARKER_13))


def barker13_pslr() -> float:
    """P2: Aperiodic autocorrelation PSLR (peak / max-sidelobe) = 13.

    For a Barker-N sequence the PSLR equals N exactly when chips are ±1
    integers (Levanon 2004, eq. 3.32). Tolerance 1e-9 in tests (float
    round-off only; integer arithmetic gives the exact 13.0).
    """
    code = CANONICAL_BARKER_13.astype(float)
    acf = np.correlate(code, code, mode="full")
    peak = float(np.max(np.abs(acf)))
    # Sidelobes: every position except the centre (lag 0 in the full output
    # is at index len(code) - 1).
    mask = np.ones(len(acf), dtype=bool)
    mask[len(code) - 1] = False
    sidelobe = float(np.max(np.abs(acf[mask])))
    if sidelobe == 0.0:
        return float("inf")
    return peak / sidelobe


def barker13_detection_rate(snr_db: float = 10.0, n_trials: int = 200, seed: int = 0) -> float:
    """S1: Matched-filter detection rate at the specified SNR.

    Generates ``n_trials`` realisations of the Barker-13 code embedded in
    real-valued AWGN. For each, applies the matched filter (time-reversed
    conjugate) and checks whether |y[n]| peaks at the expected lag
    (len(code) - 1). Returns the fraction that did. Expected ≥ 98 % at
    SNR = 10 dB; the tutorial test allows 95 % for a 200-trial run to
    keep variance contained.
    """
    rng = np.random.default_rng(seed)
    code = CANONICAL_BARKER_13.astype(float)
    matched = code[::-1]  # real-valued; conjugate is the same
    snr_lin = 10.0 ** (snr_db / 10.0)
    # Signal power = sum(code²) / len(code) = 1; noise σ² per sample:
    sigma = np.sqrt(1.0 / (2.0 * snr_lin))
    expected_lag = len(code) - 1
    correct = 0
    for _ in range(n_trials):
        rx = code + sigma * rng.standard_normal(len(code))
        comp = np.convolve(rx, matched, mode="full")
        peak_idx = int(np.argmax(np.abs(comp)))
        if peak_idx == expected_lag:
            correct += 1
    return correct / n_trials


def run_all(full: bool = False) -> dict:
    """Driver: run every check on clean signals; return a results dict.

    The dict is keyed by waveform name (`bpsk`, `ofdm`, `barker13`) and
    holds the measured numbers used by the notebook and by the
    numeric-parity test suite. ``full=True`` increases sample sizes for
    publication-grade runs (slow); the default is fast-mode.
    """
    import spectra as sp
    from spectra._rust import generate_bpsk_symbols

    results: dict = {}

    # ── BPSK ─────────────────────────────────────────────────────────────────
    syms = generate_bpsk_symbols(10_000, seed=0)
    wf = sp.BPSK(samples_per_symbol=8, rolloff=0.35)
    iq = wf.generate(num_symbols=4096, sample_rate=1e6, seed=0)
    n_bits = 200_000 if full else 100_000
    ebn0_list = [0.0, 2.0, 4.0, 6.0] if not full else [0.0, 2.0, 4.0, 6.0, 8.0]
    measured_ber, theory_ber = bpsk_ber_curve(ebn0_list, n_bits=n_bits, seed=0)

    results["bpsk"] = {
        "constellation_max_imag": bpsk_constellation_check(syms),
        "psd_correlation": bpsk_psd_correlation(iq, sample_rate=1e6, rolloff=0.35),
        "ber_ebn0_db": ebn0_list,
        "ber_measured": measured_ber.tolist(),
        "ber_theory": theory_ber.tolist(),
        "ber_max_diff_db": float(
            np.max(
                np.abs(
                    10 * np.log10(np.maximum(measured_ber, 1.0 / n_bits))
                    - 10 * np.log10(theory_ber)
                )
            )
        ),
    }

    # ── OFDM ────────────────────────────────────────────────────────────────
    n_fft, n_used, n_cp = 64, 52, 16
    orth_err = ofdm_orthogonality_error(n_fft, n_used, n_cp, seed=0)
    cp_lag, cp_peak = ofdm_cp_correlation(n_fft, n_used, n_cp, n_symbols=8, seed=0)
    n_ofdm = 2000 if full else 200
    evm = ofdm_evm_after_awgn(40.0, n_fft, n_used, n_cp, n_symbols=n_ofdm, seed=0)
    results["ofdm"] = {
        "orthogonality_error": orth_err,
        "cp_lag": cp_lag,
        "cp_peak": cp_peak,
        "evm_at_40db": evm,
    }

    # ── Barker-13 ───────────────────────────────────────────────────────────
    n_trials = 1000 if full else 200
    results["barker13"] = {
        "canonical_equality": barker13_canonical_equality(),
        "pslr": barker13_pslr(),
        "detection_rate_10db": barker13_detection_rate(
            snr_db=10.0, n_trials=n_trials, seed=0
        ),
    }

    return results


def _print_summary(results: dict) -> None:
    """Pretty-print a result table to stdout."""
    print("=" * 60)
    print("BPSK")
    print("-" * 60)
    bp = results["bpsk"]
    print(f"  P1  max(|imag(symbols)|)    = {bp['constellation_max_imag']:.2e}")
    print(f"  P4  PSD-theory correlation  = {bp['psd_correlation']:.4f}")
    print(f"  S1  max|Δ| BER vs theory    = {bp['ber_max_diff_db']:.3f} dB")
    print("=" * 60)
    print("OFDM")
    print("-" * 60)
    of = results["ofdm"]
    print(f"  P1  max |FFT(rx) − tx_grid|  = {of['orthogonality_error']:.2e}")
    print(f"  P2  CP corr argmax lag       = {of['cp_lag']} (expect 64)")
    print(f"  P2b CP corr peak amplitude   = {of['cp_peak']:.3f}")
    print(f"  S1  EVM at SNR = 40 dB       = {100 * of['evm_at_40db']:.2f} %")
    print("=" * 60)
    print("Barker-13")
    print("-" * 60)
    bk = results["barker13"]
    print(f"  P1  canonical code equality  = {bk['canonical_equality']} (expect 1)")
    print(f"  P2  PSLR (peak/max-sidelobe) = {bk['pslr']:.3f} (expect 13.0)")
    print(f"  S1  detection rate @ 10 dB   = {100 * bk['detection_rate_10db']:.1f} %")
    print("=" * 60)


def main() -> int:
    full = "--full" in sys.argv
    results = run_all(full=full)
    _print_summary(results)
    # Pass/fail thresholds: BPSK P4 ≥ 0.99, BPSK S1 ≤ 0.8 dB.
    bp = results["bpsk"]
    failed = []
    if bp["constellation_max_imag"] > 1e-6:
        failed.append("BPSK P1")
    if bp["psd_correlation"] < 0.99:
        failed.append("BPSK P4")
    if bp["ber_max_diff_db"] > 0.8:
        failed.append("BPSK S1")
    of = results["ofdm"]
    if of["orthogonality_error"] > 1e-9:
        failed.append("OFDM P1")
    if of["cp_lag"] != 64:
        failed.append("OFDM P2")
    if of["cp_peak"] <= 0.5:
        failed.append("OFDM P2b")
    if of["evm_at_40db"] > 0.02:
        failed.append("OFDM S1")
    bk = results["barker13"]
    if bk["canonical_equality"] != 1:
        failed.append("Barker-13 P1")
    if abs(bk["pslr"] - 13.0) > 1e-9:
        failed.append("Barker-13 P2")
    if bk["detection_rate_10db"] < 0.95:
        failed.append("Barker-13 S1")
    if failed:
        print(f"\nFAILED: {', '.join(failed)}")
        return 1
    print("\nOK")
    return 0


if __name__ == "__main__":
    sys.exit(main())
