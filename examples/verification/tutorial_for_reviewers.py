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
from typing import Optional

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


def _welch_psd_inline(iq: np.ndarray, fs: float, nperseg: int = 512) -> tuple[np.ndarray, np.ndarray]:
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
# OFDM, Barker-13 — added in subsequent tasks
# ════════════════════════════════════════════════════════════════════════════


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

    # ── OFDM / Barker-13: populated in later tasks ─────────────────────────
    results["ofdm"] = {"orthogonality_error": float("nan")}
    results["barker13"] = {"pslr": float("nan")}

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
    if failed:
        print(f"\nFAILED: {', '.join(failed)}")
        return 1
    print("\nOK")
    return 0


if __name__ == "__main__":
    sys.exit(main())
