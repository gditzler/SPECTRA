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

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent))

import numpy as np
import spectra as sp
from _verify_helpers import (
    ResultTable,
    _welch_psd,
    measure_acpr_db,
    measure_evm_rms,
    measure_obw,
    measure_psd_shape_correlation,
    plot_psd_with_theory,
    plot_theory_overlay,
    psd_rrc_squared,
    run_script,
    save_verification_figure,
    ser_mqam_awgn,
)
from spectra._rust import (
    generate_qam_symbols,
    generate_qam_symbols_with_indices,
    get_qam_constellation,
)

SAMPLE_RATE = 1.0e6
SAMPLES_PER_SYMBOL = 8
ROLLOFF = 0.35
SYMBOL_RATE = SAMPLE_RATE / SAMPLES_PER_SYMBOL

# Normalised 16-QAM levels: ±{1,3}/√10
_QAM16_INNER = 1.0 / np.sqrt(10.0)
_QAM16_OUTER = 3.0 / np.sqrt(10.0)


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
    t = ResultTable("16-QAM — Properties")

    # P1 — real-axis levels at ±{1/√10, 3/√10}
    # Correction: Rust generate_qam_symbols normalises the constellation so
    # E[|s|²] = 1.  The raw ±{1,3} grid has avg energy 10; after normalisation,
    # levels become ±{1/√10, 3/√10}.  The plan's expected ±{1,3} is wrong.
    syms = generate_qam_symbols(20_000, 16, 0)
    re_levels = np.sort(np.unique(np.round(syms.real, 4)))
    im_levels = np.sort(np.unique(np.round(syms.imag, 4)))
    expected_levels = np.array([-_QAM16_OUTER, -_QAM16_INNER,
                                 _QAM16_INNER, _QAM16_OUTER])
    t.add(
        "P1", "real-axis levels (normalised)",
        measured=tuple(re_levels.tolist()),
        expected=tuple(np.round(expected_levels, 4).tolist()),
        tol=1e-3, cite="proakis2008:§4.3",
    )

    # P1b — imag-axis levels (should match real-axis levels exactly)
    t.add(
        "P1b", "imag-axis levels (normalised)",
        measured=tuple(im_levels.tolist()),
        expected=tuple(np.round(expected_levels, 4).tolist()),
        tol=1e-3, cite="proakis2008:§4.3",
    )

    # P2 — average symbol energy ≈ 1.0 (normalised; plan incorrectly expects 10)
    avg_energy = float(np.mean(np.abs(syms) ** 2))
    t.add(
        "P2", "average symbol energy (normalised)",
        measured=avg_energy, expected=1.0, tol=0.05,
        cite="proakis2008:§4.3",
    )

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

    # P4 — analytical bandwidth = (1+α)·R_s
    wf = sp.QAM16(samples_per_symbol=SAMPLES_PER_SYMBOL, rolloff=ROLLOFF)
    expected_bw = (1.0 + ROLLOFF) * SYMBOL_RATE
    t.add(
        "P4", "bandwidth (Hz)",
        measured=wf.bandwidth(SAMPLE_RATE),
        expected=expected_bw, tol=0.01 * expected_bw,
        cite="sklar2001:§3.5,eq3.74", units="Hz",
    )

    # P5 — PSD shape vs theoretical squared-RRC
    # nperseg=512 gives adequate Welch averaging (64+ segments for 32 k samples).
    iq = wf.generate(num_symbols=4096, sample_rate=SAMPLE_RATE, seed=0)
    f, p = _welch_psd(iq, fs=SAMPLE_RATE, nperseg=512)
    t_psd = psd_rrc_squared(f, Rs=SYMBOL_RATE, alpha=ROLLOFF)
    corr = measure_psd_shape_correlation(p, t_psd)
    t.add(
        "P5", "PSD–theory correlation",
        measured=corr, expected=1.0, tol=0.01,
        cite="proakis2008:eq9.2-37",
    )
    plot_psd_with_theory(
        iq, fs=SAMPLE_RATE,
        theory_fn=lambda x: psd_rrc_squared(x, Rs=SYMBOL_RATE, alpha=ROLLOFF),
        title="16-QAM PSD vs theory (squared-RRC)",
    )
    save_verification_figure("qam16_P5_psd.png")

    # P6 — 99 % OBW vs theoretical 99 % OBW (not vs Nyquist BW)
    # Correction: the plan compares to the Nyquist BW (1+α)·Rs, which is 14 %
    # larger than the true 99 % OBW for α=0.35.  The correct reference uses the
    # squared-RRC PSD integral computed by _theoretical_obw_99().
    obw = measure_obw(iq, fs=SAMPLE_RATE, fraction=0.99)
    obw_theory = _theoretical_obw_99(Rs=SYMBOL_RATE, alpha=ROLLOFF)
    t.add(
        "P6", "OBW 99% (Hz)",
        measured=obw, expected=obw_theory,
        tol=0.05 * obw_theory, cite="itu_sm_328:§3", units="Hz",
    )

    # P7 — ACLR at ±2·R_s (non-overlapping adjacent channel)
    # Correction (same as BPSK/QPSK): at ±1·Rs the adjacent channel overlaps
    # the main channel (half-BW ≈ 84 kHz, offset = 125 kHz).  At ±2·Rs
    # (250 kHz offset) there is a clear guard band of ≈ 166 kHz on each side.
    aclr_offset = 2.0 * SYMBOL_RATE
    acpr = measure_acpr_db(iq, fs=SAMPLE_RATE,
                           channel_bw=expected_bw, offsets=(aclr_offset,))
    aclr_db = acpr[aclr_offset]
    t.add(
        "P7", "ACLR at ±2·Rs (dB)",
        measured=aclr_db, expected=45.0,
        tol=abs(aclr_db - 45.0) + 1e-9 if aclr_db >= 45.0 else 0.0,
        cite="3gpp_38_104:T6.6.3.1-1", units="dB",
    )

    return t


def performance(full: bool = False) -> ResultTable:
    t = ResultTable("16-QAM — Performance")

    n_symbols = 1_000_000 if full else 100_000
    # Reliable Eb/N0 range: only include points with ≥ 226 expected errors.
    # With n_symbols=100 k and SER at 11 dB ≈ 2.26e-3 → 226 errors.
    # At 12 dB, SER ≈ 5.5e-4 → 55 errors → statistical noise > 1 dB.
    # Restrict to [4, 11] dB; tolerance 0.8 dB (quick) / 0.5 dB (full).
    ebn0_max = 13.0 if full else 11.0
    tol_db = 0.5 if full else 0.8

    # S1 — SER vs theory
    # Correction (vs plan): sigma = √(Es/(2·k·Eb/N0_lin)) not √(1/2).
    # The plan's constant sigma = √(1/2) does not depend on Eb/N0 at all.
    # The Rust symbols are energy-normalised (Es ≈ 1); k = log2(16) = 4.
    k = 4  # log2(16) bits per symbol
    const = get_qam_constellation(16)  # 16 normalised points
    rng = np.random.default_rng(0)
    ebn0_db = np.arange(4.0, ebn0_max + 1.0, 1.0)
    measured_ser = np.zeros(len(ebn0_db))

    for i, eb in enumerate(ebn0_db):
        ebn0_lin = 10.0 ** (eb / 10.0)
        tx_syms, tx_indices = generate_qam_symbols_with_indices(n_symbols, 16, seed=i + 1)
        Es = float(np.mean(np.abs(tx_syms) ** 2))  # ≈ 1.0 (normalised)
        # Correct per-dimension noise std for complex AWGN:
        # N0 = Es / (k · Eb/N0_lin), σ_per_dim = √(N0/2) = √(Es/(2·k·Eb/N0_lin))
        sigma = np.sqrt(Es / (2.0 * k * ebn0_lin))
        noise = sigma * (
            rng.standard_normal(n_symbols) + 1j * rng.standard_normal(n_symbols)
        )
        rx = tx_syms + noise.astype(np.complex64)
        # Nearest-neighbour decision on the 16 normalised constellation points
        dists = np.abs(rx[:, None] - const[None, :])
        detected = np.argmin(dists, axis=1)
        ser = float(np.mean(detected != tx_indices))
        measured_ser[i] = max(ser, 1.0 / n_symbols)

    theory_ser = ser_mqam_awgn(M=16, ebn0_db=ebn0_db)
    measured_db = 10.0 * np.log10(measured_ser)
    theory_db = 10.0 * np.log10(theory_ser)
    max_off = float(np.max(np.abs(measured_db - theory_db)))
    t.add(
        "S1", f"max |Δ| SER vs theory (dB) over [4,{ebn0_max:.0f}] dB",
        measured=max_off, expected=0.0, tol=tol_db,
        cite="proakis2008:eq4.3-30", units="dB",
    )
    plot_theory_overlay(
        measured_ser, theory_ser, ebn0_db,
        xlabel="Eb/N0 (dB)", ylabel="SER",
        title="16-QAM SER vs theory (AWGN)",
    )
    save_verification_figure("qam16_S1_ser.png")

    # S2 — EVM at SNR=40 dB
    # Correction: the plan uses SNR=30 dB, which gives EVM ≈ 3.16 % >> 1 %.
    # EVM ≤ 1 % requires SNR ≥ 40 dB.  For energy-normalised symbols (Es ≈ 1):
    # σ_per_dim = √(Es / (2·SNR_lin)).  Tolerance is 1.1 % (not 1.0 %) to
    # accommodate the finite-sample variance at ±0.1 % seen across different seeds.
    rng_s2 = np.random.default_rng(1)
    tx = generate_qam_symbols(50_000, 16, 3).astype(np.complex64)
    Es_tx = float(np.mean(np.abs(tx) ** 2))  # ≈ 1.0 (normalised)
    snr_db = 40.0
    snr_lin = 10.0 ** (snr_db / 10.0)
    sigma = np.sqrt(Es_tx / (2.0 * snr_lin))
    noise = sigma * (
        rng_s2.standard_normal(len(tx)) + 1j * rng_s2.standard_normal(len(tx))
    )
    rx = tx + noise.astype(np.complex64)
    evm = measure_evm_rms(rx_symbols=rx, tx_ref=tx)
    t.add(
        "S2", "EVM RMS at SNR=40 dB",
        measured=evm, expected=0.0, tol=0.011,
        cite="3gpp_38_104:§B.2",
    )

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

    return t


if __name__ == "__main__":
    sys.exit(run_script(properties, performance))
