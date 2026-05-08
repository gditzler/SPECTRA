"""SPECTRA Verification — BPSK
=================================
Proves that the generated BPSK waveform satisfies:

  P1. Constellation: symbols lie on the real axis (imag ≈ 0).
  P2. Two unique symbols at ±1.
  P3. Bandwidth = (1+α)·R_s within 1 %.            [sklar2001:§3.5,eq3.74]
  P4. PSD shape correlation with squared-RRC ≥ 0.99. [proakis2008:eq9.2-37]
  P5. OBW (99 %) within 5 % of theoretical 99%-OBW.  [itu_sm_328:§3]
  P6. ACLR at ±2·R_s offset ≥ 45 dB.                [3gpp_38_104:T6.6.3.1-1]
  S1. BER vs Eb/N0 ∈ [0,6] dB, max |Δ| ≤ 1.0 dB.   [proakis2008:eq4.3-13]
  S2. EVM at SNR=40 dB ≤ 1 % RMS.                   [3gpp_38_104:§B.2]

Implementation notes:
  P5: Reference OBW is the theoretical 99 % occupied bandwidth computed from the
      squared-RRC spectrum, which for α=0.35 is ≈86.4 % of the Nyquist BW — NOT
      the Nyquist BW itself.  The plan's comment "OBW ≈ Nyquist BW" is inaccurate
      at the 14 % level.
  P6: The adjacent channel at ±1·R_s (125 kHz) overlaps the main channel
      (±84 kHz half-BW), making ACLR undefined.  The check uses ±2·R_s (250 kHz)
      where the guard band is clear.
  S1: With 100 k bits, BER measurements above Eb/N0=6 dB have fewer than 240
      expected errors.  Including them introduces >1 dB statistical noise.  The
      comparison is restricted to [0,6] dB where each SNR point has ≥240 errors.
  S2: Complex AWGN at SNR=30 dB yields EVM ≈ 3.16 % (= 1/√1000), which exceeds
      the 1 % threshold.  EVM ≤ 1 % requires SNR ≥ 40 dB; the check uses 40 dB.

Run:
    python examples/verification/verify_bpsk.py            # quick mode
    python examples/verification/verify_bpsk.py --full     # publication mode
"""

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent))

import numpy as np

import spectra as sp
from spectra._rust import generate_bpsk_symbols

from _verify_helpers import (
    ResultTable,
    ber_bpsk_awgn,
    measure_acpr_db,
    measure_evm_rms,
    measure_obw,
    measure_psd_shape_correlation,
    plot_psd_with_theory,
    plot_theory_overlay,
    psd_rrc_squared,
    run_script,
    save_verification_figure,
    simulate_ber_awgn,
    _welch_psd,
)


SAMPLE_RATE = 1.0e6
SAMPLES_PER_SYMBOL = 8
ROLLOFF = 0.35
SYMBOL_RATE = SAMPLE_RATE / SAMPLES_PER_SYMBOL


def _theoretical_obw_99(Rs: float, alpha: float, n_pts: int = 1_000_000) -> float:
    """Compute the theoretical 99 % occupied bandwidth from the squared-RRC PSD.

    The Nyquist bandwidth (1+α)·R_s is the spectral *extent* of the raised-cosine
    filter but is NOT equal to the 99 % OBW.  For α=0.35 the 99 % OBW is ≈86.4 %
    of the Nyquist BW.
    """
    fs = 10.0 * (1.0 + alpha) * Rs  # oversample for accurate integration
    f = np.linspace(-fs / 2, fs / 2, n_pts)
    p = psd_rrc_squared(f, Rs=Rs, alpha=alpha)
    cum = np.cumsum(p)
    cum /= cum[-1]
    lo = int(np.searchsorted(cum, 0.005))
    hi = int(np.searchsorted(cum, 0.995))
    return float(f[hi] - f[lo])


def properties() -> ResultTable:
    t = ResultTable("BPSK — Properties")

    # P1 — constellation on real axis
    syms = generate_bpsk_symbols(10_000, seed=0)
    t.add("P1", "max(|imag(symbols)|)",
          measured=float(np.max(np.abs(syms.imag))),
          expected=0.0, tol=1e-6, cite="bpsk:constellation")

    # P2 — exactly two unique symbols at ±1
    unique = np.unique(syms.real)
    t.add("P2", "unique BPSK symbol values",
          measured=tuple(np.sort(unique).tolist()),
          expected=(-1.0, 1.0), tol=1e-9, cite="bpsk:constellation")

    # P3 — analytical bandwidth = (1+α)·R_s
    wf = sp.BPSK(samples_per_symbol=SAMPLES_PER_SYMBOL, rolloff=ROLLOFF)
    expected_bw = (1.0 + ROLLOFF) * SYMBOL_RATE
    t.add("P3", "bandwidth (Hz)",
          measured=wf.bandwidth(SAMPLE_RATE),
          expected=expected_bw, tol=0.01 * expected_bw,
          cite="sklar2001:§3.5,eq3.74", units="Hz")

    # P4 — PSD shape vs theoretical squared-RRC
    # nperseg=512 gives adequate Welch averaging (64+ segments for 32 k samples).
    iq = wf.generate(num_symbols=4096, sample_rate=SAMPLE_RATE, seed=0)
    f, p = _welch_psd(iq, fs=SAMPLE_RATE, nperseg=512)
    t_psd = psd_rrc_squared(f, Rs=SYMBOL_RATE, alpha=ROLLOFF)
    corr = measure_psd_shape_correlation(p, t_psd)
    t.add("P4", "PSD–theory correlation",
          measured=corr, expected=1.0, tol=0.01,
          cite="proakis2008:eq9.2-37")
    plot_psd_with_theory(iq, fs=SAMPLE_RATE,
                         theory_fn=lambda x: psd_rrc_squared(x, Rs=SYMBOL_RATE, alpha=ROLLOFF),
                         title="BPSK PSD vs theory (squared-RRC)")
    save_verification_figure("bpsk_P4_psd.png")

    # P5 — 99 % OBW vs theoretical 99 % OBW (not vs Nyquist BW)
    obw = measure_obw(iq, fs=SAMPLE_RATE, fraction=0.99)
    obw_theory = _theoretical_obw_99(Rs=SYMBOL_RATE, alpha=ROLLOFF)
    t.add("P5", "OBW 99% (Hz)",
          measured=obw, expected=obw_theory,
          tol=0.05 * obw_theory, cite="itu_sm_328:§3", units="Hz")

    # P6 — ACLR at ±2·R_s (non-overlapping adjacent channel)
    # At ±1·R_s the adjacent channel (half-BW=84 kHz) overlaps the main channel
    # (half-BW=84 kHz, offset=125 kHz → overlap from 40 kHz to 84 kHz).
    # At ±2·R_s (250 kHz offset) there is a clear guard band.
    aclr_offset = 2.0 * SYMBOL_RATE
    acpr = measure_acpr_db(iq, fs=SAMPLE_RATE,
                           channel_bw=expected_bw, offsets=(aclr_offset,))
    aclr_db = acpr[aclr_offset]
    t.add("P6", "ACLR at ±2·Rs (dB)",
          measured=aclr_db, expected=45.0,
          tol=abs(aclr_db - 45.0) + 1e-9 if aclr_db >= 45.0 else 0.0,
          cite="3gpp_38_104:T6.6.3.1-1", units="dB")
    return t


def performance(full: bool = False) -> ResultTable:
    t = ResultTable("BPSK — Performance")
    n_bits = 1_000_000 if full else 100_000
    # Reliable Eb/N0 range: only include points with ≥240 expected errors.
    # With n_bits=100 k, theory BER at 6 dB ≈2.4e-3 → 240 errors.
    # At 7 dB, BER≈7.7e-4 → 77 errors → too noisy for a 0.3–1 dB tolerance.
    ebn0_max = 9.0 if full else 6.0
    tol_db = 0.3 if full else 1.0

    # S1 — BER vs theory
    ebn0_db = np.arange(0, ebn0_max + 1, 1.0)
    measured = simulate_ber_awgn("bpsk", ebn0_db, n_bits=n_bits, seed=0)
    theory = ber_bpsk_awgn(ebn0_db)
    measured_db = 10 * np.log10(np.maximum(measured, 1.0 / n_bits))
    theory_db = 10 * np.log10(theory)
    max_off = float(np.max(np.abs(measured_db - theory_db)))
    t.add("S1", f"max |Δ| BER vs theory (dB) over [0,{ebn0_max:.0f}] dB",
          measured=max_off, expected=0.0, tol=tol_db,
          cite="proakis2008:eq4.3-13", units="dB")
    plot_theory_overlay(measured, theory, ebn0_db,
                        xlabel="Eb/N0 (dB)", ylabel="BER",
                        title="BPSK BER vs theory (AWGN)")
    save_verification_figure("bpsk_S1_ber.png")

    # S2 — EVM at SNR=40 dB
    # At SNR=30 dB, complex AWGN gives EVM ≈ 1/√1000 ≈ 3.16 % >> 1 %.
    # EVM ≤ 1 % requires SNR ≥ 40 dB (EVM = 1/√10000 = 1.0 %).
    rng = np.random.default_rng(1)
    tx = generate_bpsk_symbols(50_000, seed=2).astype(np.complex64)
    snr_db = 40.0
    snr_lin = 10 ** (snr_db / 10)
    sigma = np.sqrt(1.0 / (2.0 * snr_lin))
    noise = sigma * (rng.standard_normal(len(tx)) + 1j * rng.standard_normal(len(tx)))
    rx = tx + noise.astype(np.complex64)
    evm = measure_evm_rms(rx_symbols=rx, tx_ref=tx)
    t.add("S2", "EVM RMS at SNR=40 dB",
          measured=evm, expected=0.0, tol=0.01,
          cite="3gpp_38_104:§B.2")
    return t


if __name__ == "__main__":
    sys.exit(run_script(properties, performance))
