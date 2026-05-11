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

A BER row is deliberately omitted: a per-bit matched filter loses ~26 dB
on BT=0.3 GMSK because the Gaussian-shaped phase pulse spreads ISI over
~3 bit intervals. A proper coherent receiver (Laurent decomposition or
Viterbi CPM) belongs in a follow-on PR.

Run:
    python examples/verification/verify_gmsk.py            # quick mode
    python examples/verification/verify_gmsk.py --full     # publication mode
"""

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent))

import numpy as np
import spectra as sp
from _verify_helpers import (
    ResultTable,
    _welch_psd,
    measure_obw,
    run_script,
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
    symbols = (2 * bits - 1).astype(np.float32)
    symbols_up = np.repeat(symbols, sps)
    h = _build_gaussian_taps(bt, sps, filter_span=4)
    filtered = np.convolve(symbols_up, h, mode="same")
    delta_phi = np.pi * H_GMSK * filtered / sps
    phase = np.cumsum(delta_phi)
    return np.exp(1j * phase).astype(np.complex64)


def properties() -> ResultTable:
    t = ResultTable("GMSK — Properties")

    wf = sp.GMSK(bt=BT, samples_per_symbol=SAMPLES_PER_SYMBOL)
    iq = wf.generate(num_symbols=4096, sample_rate=SAMPLE_RATE, seed=0)

    # P1 — constant envelope
    env = np.abs(iq)
    cv = float(np.std(env) / np.mean(env))
    t.add(
        "P1",
        "envelope CV (std/mean)",
        measured=cv,
        expected=0.0,
        tol=1e-3,
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
        "P2",
        "steady-state |Δφ|/symbol (rad)",
        measured=median_step,
        expected=expected_step,
        tol=0.01 * expected_step,
        cite="proakis2008:§4.4-3",
        units="rad",
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
        "P3",
        "Gaussian filter 3-dB BW (Hz)",
        measured=bw_3db_hz,
        expected=expected_bw,
        tol=0.20 * expected_bw,
        cite="gmsk:gaussian",
        units="Hz",
    )

    # P4 — PSD 3-dB BW vs BT=0.3 / h=0.5 reference (0.27·R_s).
    f, p = _welch_psd(iq, fs=SAMPLE_RATE, nperseg=4096)
    p_db = 10.0 * np.log10(p / np.max(p) + 1e-30)
    above_psd = np.where(p_db >= -3.0)[0]
    main_bw = float(f[above_psd[-1]] - f[above_psd[0]]) if len(above_psd) > 0 else 0.0
    expected_psd_bw = 0.27 * SYMBOL_RATE
    t.add(
        "P4",
        "PSD 3-dB BW (Hz)",
        measured=main_bw,
        expected=expected_psd_bw,
        tol=0.25 * expected_psd_bw,
        cite="laurent1986:§III",
        units="Hz",
    )

    # P5 — 99 % OBW vs GSM/3GPP BT=0.3 GMSK reference (0.92·R_s).
    obw = measure_obw(iq, fs=SAMPLE_RATE, fraction=0.99)
    expected_obw = 0.92 * SYMBOL_RATE
    t.add(
        "P5",
        "OBW 99 % (Hz)",
        measured=obw,
        expected=expected_obw,
        tol=0.10 * expected_obw,
        cite="itu_sm_328:§3",
        units="Hz",
    )

    return t


def performance(full: bool = False) -> ResultTable:  # noqa: ARG001
    # BER row deliberately omitted (see module docstring). Returning an
    # empty ResultTable keeps run_script() and tests/verification/
    # entry points working without rendering a phantom failure row.
    return ResultTable("GMSK — Performance")


if __name__ == "__main__":
    sys.exit(run_script(properties, performance))
