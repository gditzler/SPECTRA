"""SPECTRA Verification — Linear Frequency Modulation (LFM, chirp)
====================================================================
  P1. Instantaneous frequency = f_0 + (B/T)·t — linear ramp.
      Residual std of IF after linear detrend ≤ 2% of B.
  P2. Total swept bandwidth = configured B within 1%.
  P3. Matched-filter compression gain = 10·log10(N_samples) within 0.5 dB.
      [levanon2004:eq5.5, corrected for oversampling — see note below]
  P4. Pulse-compression 3-dB main-lobe width = 0.886/B within 5%.
      [levanon2004:§4.2]
  P5. Chirp rate = B/T within 2%.                       [levanon2004:§4.2]
  S1. Range resolution at SNR=20 dB matches 0.886/B ±10%.  [levanon2004:§5]

Implementation notes
--------------------
API correction:
  The plan template uses ``sp.LFM(bandwidth=B, pulse_width=T)`` but SPECTRA's
  ``LFM`` class is parameterised as
  ``LFM(bandwidth_fraction=B/Fs, samples_per_pulse=int(T*Fs))``.
  Parameters are derived from the physical design values (B, T, Fs) and passed
  in the correct form throughout.

P3 — Matched-filter gain formula:
  Levanon (2004) eq 5.5 gives ``gain_dB = 10·log10(TBP)`` assuming the
  receiver samples the matched-filter output at Nyquist rate (Fs = B).  When
  the signal is oversampled (Fs > B), the pulse contains N_samples = T·Fs
  samples, not TBP = T·B samples.  For an LFM pulse with constant unit
  amplitude, the MF peak amplitude equals N_samples (coherent integration of
  N_samples unit-amplitude samples), so:
      peak_power = N_samples²
      input_power = N_samples · mean(|iq|²) = N_samples (unit amplitude)
      gain_dB = 10·log10(N_samples)
  For our design: N_samples = T·Fs = 10µs × 100 MHz = 1000, so gain = 30 dB.
  TBP = B·T = 100 → 20 dB.  The 10 dB difference is 10·log10(Fs/B).
  The check verifies 10·log10(N_samples), the correct discrete-time quantity.

P4 and S1 — 3-dB width measurement:
  Integer-sample boundary detection yields ±1 sample quantisation error.  For
  B=10 MHz and Fs=100 MHz the expected 3-dB half-width is 0.886/(B·2)=4.43
  samples; one-sample quantisation alone produces ≥10% error.  Linear
  interpolation between the two adjacent samples that bracket the half-power
  level gives sub-sample resolution and reduces the relative error to <1%.
  The 5% and 10% tolerances are therefore achievable with interpolation.

Run:
    python examples/verification/verify_lfm.py            # quick mode
    python examples/verification/verify_lfm.py --full     # publication mode
"""

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent))

import numpy as np

import spectra as sp

from _verify_helpers import (
    ResultTable,
    run_script,
    save_verification_figure,
)

# ── Physical design parameters ───────────────────────────────────────────────
SAMPLE_RATE = 100e6        # receiver sample rate [Hz]
BANDWIDTH = 10e6           # LFM sweep bandwidth [Hz]
PULSE_WIDTH = 10e-6        # pulse duration [s]

# Derived quantities
BW_FRACTION = BANDWIDTH / SAMPLE_RATE   # = 0.1  → ``bandwidth_fraction`` arg
SAMPLES_PER_PULSE = int(PULSE_WIDTH * SAMPLE_RATE)  # = 1000  → samples_per_pulse
TBP = BANDWIDTH * PULSE_WIDTH           # time-bandwidth product = 100 (unitless)
# Discrete-time MF gain: N_samples = T·Fs (see P3 note above)
N_SAMPLES = SAMPLES_PER_PULSE           # = 1000


def _interp_half_power_width(mag: np.ndarray, centre: int, sample_rate: float) -> float:
    """Return the interpolated 3-dB (half-power) main-lobe width in seconds.

    Uses linear interpolation between the two samples that bracket each
    half-power crossing to achieve sub-sample resolution.  This avoids the
    ≥10% quantisation error that arises when using integer-sample crossings
    for the narrow LFM main lobe.

    Reference: standard technique; see also Levanon (2004) §4.2 for the
    theoretical 3-dB width of 0.886/B.
    """
    half = mag[centre] / np.sqrt(2)

    # Walk left to find the crossing index
    left_i = centre
    while left_i > 0 and mag[left_i] > half:
        left_i -= 1
    # Interpolate: crossing is between left_i and left_i+1
    if left_i + 1 < len(mag) and mag[left_i] != mag[left_i + 1]:
        alpha_l = (half - mag[left_i + 1]) / (mag[left_i] - mag[left_i + 1])
        left_frac = (left_i + 1) - alpha_l
    else:
        left_frac = float(left_i)

    # Walk right to find the crossing index
    right_i = centre
    while right_i < len(mag) - 1 and mag[right_i] > half:
        right_i += 1
    # Interpolate: crossing is between right_i-1 and right_i
    if right_i > 0 and mag[right_i - 1] != mag[right_i]:
        alpha_r = (half - mag[right_i - 1]) / (mag[right_i] - mag[right_i - 1])
        right_frac = (right_i - 1) + alpha_r
    else:
        right_frac = float(right_i)

    return (right_frac - left_frac) / sample_rate


def properties() -> ResultTable:
    t = ResultTable("LFM — Properties")

    # API correction: use bandwidth_fraction and samples_per_pulse
    wf = sp.LFM(bandwidth_fraction=BW_FRACTION, samples_per_pulse=SAMPLES_PER_PULSE)
    iq = wf.generate(num_symbols=1, sample_rate=SAMPLE_RATE, seed=0)
    n = len(iq)
    tt = np.arange(n) / SAMPLE_RATE

    # ── P1 — linear instantaneous frequency ──────────────────────────────────
    # Phase unwrapping then finite difference gives IF samples; a linear fit
    # should account for almost all variance (residual std ≤ 2 % of B).
    phase = np.unwrap(np.angle(iq))
    inst_f = np.diff(phase) / (2 * np.pi) * SAMPLE_RATE
    inst_t = tt[:-1]
    coeffs = np.polyfit(inst_t, inst_f, 1)
    fit = np.polyval(coeffs, inst_t)
    residual_std = float(np.std(inst_f - fit))
    t.add(
        "P1",
        "IF residual std / B",
        measured=residual_std / BANDWIDTH,
        expected=0.0,
        tol=0.02,
        cite="lfm:definition",
    )

    # ── P2 — total swept bandwidth ────────────────────────────────────────────
    # The linear IF fit slope × pulse duration gives the swept BW.
    swept = float(coeffs[0] * (inst_t[-1] - inst_t[0]))
    t.add(
        "P2",
        "swept bandwidth (Hz)",
        measured=abs(swept),
        expected=BANDWIDTH,
        tol=0.01 * BANDWIDTH,
        cite="lfm:definition",
        units="Hz",
    )

    # ── P3 — matched-filter compression gain ─────────────────────────────────
    # Expected: 10·log10(N_samples) [NOT 10·log10(TBP)].
    # Levanon (2004) eq 5.5 states gain = 10·log10(TBP), which assumes Nyquist
    # sampling (Fs = B).  Here Fs = 100 MHz > B = 10 MHz, so N_samples = 1000
    # and TBP = 100.  The correct discrete-time gain is 10·log10(N_samples).
    # See module-level docstring for derivation.
    matched = np.conj(iq[::-1])
    comp = np.convolve(iq.astype(np.complex128), matched.astype(np.complex128), mode="full")
    peak_amp = float(np.max(np.abs(comp)))
    peak_pwr = peak_amp ** 2
    input_pwr = float(np.mean(np.abs(iq) ** 2)) * n   # total received energy
    gain_db = 10.0 * np.log10(peak_pwr / input_pwr)
    expected_db = 10.0 * np.log10(float(N_SAMPLES))   # = 30 dB for N=1000
    t.add(
        "P3",
        "matched-filter gain (dB)",
        measured=gain_db,
        expected=expected_db,
        tol=0.5,
        cite="levanon2004:eq5.5",
        units="dB",
    )

    # ── P4 — 3-dB main-lobe width ────────────────────────────────────────────
    # Theoretical: 0.886/B (sinc-like MF output of rectangular LFM pulse).
    # Integer-sample detection has ±1-sample quantisation (~10% for this TBP);
    # sub-sample linear interpolation is used instead (see _interp_half_power_width).
    mag = np.abs(comp)
    centre = int(np.argmax(mag))
    width = _interp_half_power_width(mag, centre, SAMPLE_RATE)
    expected_w = 0.886 / BANDWIDTH
    t.add(
        "P4",
        "3-dB main-lobe width (s)",
        measured=width,
        expected=expected_w,
        tol=0.05 * expected_w,
        cite="levanon2004:§4.2",
        units="s",
    )

    # ── P5 — chirp rate ───────────────────────────────────────────────────────
    # The IF slope from the poly-fit should equal B/T within 2%.
    rate = BANDWIDTH / PULSE_WIDTH
    measured_rate = float(coeffs[0])
    t.add(
        "P5",
        "chirp rate (Hz/s)",
        measured=measured_rate,
        expected=rate,
        tol=0.02 * rate,
        cite="levanon2004:§4.2",
        units="Hz/s",
    )

    # ── Plots ─────────────────────────────────────────────────────────────────
    try:
        import matplotlib

        matplotlib.use("Agg")
        import matplotlib.pyplot as plt

        fig, axes = plt.subplots(1, 2, figsize=(12, 4))

        # IF plot (P1 + P5)
        axes[0].plot(inst_t * 1e6, inst_f / 1e6, lw=0.8, label="measured IF")
        axes[0].plot(inst_t * 1e6, fit / 1e6, "k--", lw=1.2, label="linear fit")
        axes[0].set_xlabel("Time (µs)")
        axes[0].set_ylabel("Instantaneous frequency (MHz)")
        axes[0].set_title("P1/P5: LFM Instantaneous Frequency")
        axes[0].legend()
        axes[0].grid(True, alpha=0.3)

        # MF output (P3 + P4)
        t_comp = (np.arange(len(mag)) - centre) / SAMPLE_RATE * 1e6
        axes[1].plot(t_comp, 20 * np.log10(mag / mag[centre] + 1e-12), lw=0.8)
        axes[1].axhline(-3, color="red", ls="--", lw=1, label="-3 dB")
        axes[1].set_xlim([-2 / BANDWIDTH * 1e6, 2 / BANDWIDTH * 1e6])
        axes[1].set_xlabel("Time offset (µs)")
        axes[1].set_ylabel("Compressed pulse (dB, normalised)")
        axes[1].set_title("P3/P4: LFM Matched-Filter Output")
        axes[1].legend()
        axes[1].grid(True, alpha=0.3)

        plt.tight_layout()
        save_verification_figure("lfm_P1_P4.png")
        plt.close(fig)
    except Exception:
        pass  # plot generation is optional

    return t


def performance(full: bool = False) -> ResultTable:
    t = ResultTable("LFM — Performance")
    n_trials = 100 if full else 30

    # ── S1 — range resolution at SNR=20 dB ───────────────────────────────────
    # The mean 3-dB MF-output width under AWGN at SNR=20 dB should match the
    # theoretical 0.886/B within 10%.  Sub-sample linear interpolation is used
    # (same reasoning as P4) to avoid the ±1-sample quantisation bias.
    rng = np.random.default_rng(0)
    wf = sp.LFM(bandwidth_fraction=BW_FRACTION, samples_per_pulse=SAMPLES_PER_PULSE)
    iq = wf.generate(num_symbols=1, sample_rate=SAMPLE_RATE, seed=0)
    matched = np.conj(iq[::-1]).astype(np.complex128)
    iq_d = iq.astype(np.complex128)

    width_list = []
    for _ in range(n_trials):
        snr_lin = 10 ** (20.0 / 10.0)
        sigma = np.sqrt(float(np.mean(np.abs(iq) ** 2)) / (2.0 * snr_lin))
        noise = sigma * (
            rng.standard_normal(len(iq)) + 1j * rng.standard_normal(len(iq))
        )
        rx = iq_d + noise
        comp = np.convolve(rx, matched, mode="full")
        mag = np.abs(comp)
        centre = int(np.argmax(mag))
        w = _interp_half_power_width(mag, centre, SAMPLE_RATE)
        width_list.append(w)

    avg = float(np.mean(width_list))
    expected = 0.886 / BANDWIDTH
    t.add(
        "S1",
        "avg 3-dB width at SNR=20 dB (s)",
        measured=avg,
        expected=expected,
        tol=0.10 * expected,
        cite="levanon2004:§5",
        units="s",
    )

    return t


if __name__ == "__main__":
    sys.exit(run_script(properties, performance))
