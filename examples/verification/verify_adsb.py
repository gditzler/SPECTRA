"""SPECTRA Verification — ADS-B (1090ES)
=========================================
  P1. Preamble pulses at 0, 1, 3.5, 4.5 µs offsets within first 8 µs.
                                                            [rtca_do260b:§2.2.3.2.2]
  P2. Message length = 112 bits = 112 µs PPM @ 1 Mbps.       [rtca_do260b:§2.2.3.2.2]
  P3. CRC-24 round-trip residue = 0 with G(x)=0x1FFF409.     [rtca_do260b:§2.2.3.2.1.2]
  P4. PPM round-trip: decoded bits → CRC residue = 0.        [rtca_do260b:§2.2.3.2.2]

(No statistical tier — protocol bits are bits.)

Implementation notes
--------------------
Sample-rate convention:
  ADS-B uses PPM with a chip period of 0.5 µs (chip rate = 2 Mchips/s).
  ``ADSB(samples_per_chip=SPC)`` upsample each chip to ``SPC`` samples, so the
  correct sample rate is ``SPC / 0.5e-6 = 2 * SPC * 1e6`` Hz (e.g. 20 MHz for
  ``SPC=10``).  The plan template incorrectly suggests ``SPC * 1e6`` Hz (10 MHz),
  which would put preamble pulses at wrong µs offsets.  This script uses the
  correct 2 × SPC × 1 MHz formula.

CRC-24 algorithm:
  The CRC is computed bit-by-bit, MSB first, over the 88-bit information field
  (bytes 0–10 of the 14-byte frame).  The 24-bit remainder is appended as bytes
  11–13.  Running the same algorithm over the full 112 bits gives residue 0
  (standard CRC self-check property).  P3 verifies algorithm correctness on a
  synthetic frame; P4 verifies it on the Rust-generated signal.

Run:
    python examples/verification/verify_adsb.py            # quick mode
    python examples/verification/verify_adsb.py --full     # (no extra checks)
"""

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent))

import numpy as np
from _verify_helpers import ResultTable, run_script
from spectra.waveforms.aviation_maritime import ADSB

# ── Physical design parameters ────────────────────────────────────────────────
# ADS-B chip period = 0.5 µs → chip rate = 2 Mchips/s.
# With samples_per_chip=SPC the digital sample rate = 2 * SPC * 1 MHz.
SAMPLES_PER_CHIP = 10
SAMPLE_RATE = 2 * SAMPLES_PER_CHIP * 1e6  # 20 MHz — correct for 0.5 µs chips
SAMPLES_PER_US = 2 * SAMPLES_PER_CHIP  # 20 samples per microsecond

# ADS-B CRC polynomial: x^24 + x^23 + x^10 + x^3 + 1 (DO-260B §2.2.3.2.1.2)
ADSB_POLY = 0x1FFF409


def _crc24_adsb(bits) -> int:
    """CRC-24 reference implementation per RTCA DO-260B §2.2.3.2.1.2.

    Processes bits MSB-first.  Running over payload || appended_crc gives
    residue 0 (standard CRC self-check property).
    """
    poly = ADSB_POLY
    reg = 0
    for b in bits:
        reg <<= 1
        reg |= int(b)
        if reg & (1 << 24):
            reg ^= poly
    return reg & 0xFFFFFF


def properties() -> ResultTable:
    t = ResultTable("ADS-B — Properties")

    wf = ADSB(samples_per_chip=SAMPLES_PER_CHIP)
    iq = wf.generate(num_symbols=1, sample_rate=SAMPLE_RATE, seed=0)
    env = np.abs(iq)
    threshold = 0.5 * float(np.max(env))

    # ── P1 — preamble pulses in first 8 µs ───────────────────────────────────
    # DO-260B §2.2.3.2.2: the preamble consists of four 0.5 µs pulses starting
    # at 0, 1, 3.5, and 4.5 µs relative to the frame start.  We detect rising
    # edges in the envelope within the first 8 µs and check for four matches.
    # Tolerance of 0.1 µs absorbs the ±0.5-sample boundary of pulse detection.
    pulse_starts_us = []
    in_pulse = False
    for k in range(8 * SAMPLES_PER_US):
        if env[k] >= threshold and not in_pulse:
            pulse_starts_us.append(k / SAMPLES_PER_US)
            in_pulse = True
        elif env[k] < threshold:
            in_pulse = False
    expected_offsets = (0.0, 1.0, 3.5, 4.5)
    matched = sum(
        any(abs(p - e) < 0.1 for p in pulse_starts_us) for e in expected_offsets
    )
    t.add(
        "P1",
        "preamble pulses matched (of 4)",
        measured=matched,
        expected=4,
        tol=0,
        cite="rtca_do260b:§2.2.3.2.2",
    )

    # ── P2 — message duration = 112 µs (after 8 µs preamble) ─────────────────
    # 112-bit PPM payload at 1 Mbps = 112 µs.  Total frame = 120 µs.
    # IQ length = 240 chips × SPC samples, at 20 MHz → 120 µs total.
    total_us = len(iq) / SAMPLES_PER_US
    message_us = total_us - 8.0
    t.add(
        "P2",
        "message length (µs)",
        measured=message_us,
        expected=112.0,
        tol=0.5,
        cite="rtca_do260b:§2.2.3.2.2",
        units="µs",
    )

    # ── P3 — CRC-24 round-trip on a synthetic frame ───────────────────────────
    # Verify the CRC algorithm itself: compute CRC over a random 88-bit payload,
    # append the 24-bit remainder, run CRC over the full 112 bits → residue 0.
    rng = np.random.default_rng(0)
    payload = rng.integers(0, 2, size=88, endpoint=False).astype(int)
    padded = np.concatenate([payload, np.zeros(24, dtype=int)])
    crc_ref = _crc24_adsb(padded)
    crc_bits = np.array([(crc_ref >> (23 - i)) & 1 for i in range(24)])
    full_frame_bits = np.concatenate([payload, crc_bits])
    crc_residue = _crc24_adsb(full_frame_bits)
    t.add(
        "P3",
        "CRC-24 round-trip residue (synthetic)",
        measured=crc_residue,
        expected=0,
        tol=0,
        cite="rtca_do260b:§2.2.3.2.1.2",
    )

    # ── P4 — PPM round-trip: SPECTRA-generated frame → decoded bits → CRC = 0 ──
    # ADS-B PPM: each 1 µs bit period contains 2 chips.
    #   bit='1' → pulse-then-gap:  [1, 0]
    #   bit='0' → gap-then-pulse:  [0, 1]
    # Decode by comparing the mean envelope in the first and second half of each
    # bit period.  The Rust generator appends a valid CRC-24 to bits 88-111, so
    # the decoded 112-bit sequence must have CRC residue 0.
    bit_starts = (8.0 + np.arange(112)) * SAMPLES_PER_US
    decoded = np.zeros(112, dtype=int)
    half = SAMPLES_PER_US // 2
    for i, s in enumerate(bit_starts.astype(int)):
        first_half = float(np.mean(env[s : s + half]))
        second_half = float(np.mean(env[s + half : s + SAMPLES_PER_US]))
        decoded[i] = 1 if first_half > second_half else 0
    crc_decoded = _crc24_adsb(decoded)
    t.add(
        "P4",
        "PPM round-trip → CRC-24 residue = 0",
        measured=crc_decoded,
        expected=0,
        tol=0,
        cite="rtca_do260b:§2.2.3.2.2",
    )

    # ── Plots ─────────────────────────────────────────────────────────────────
    try:
        import matplotlib

        matplotlib.use("Agg")
        import matplotlib.pyplot as plt
        from _verify_helpers import save_verification_figure

        t_us = np.arange(len(iq)) / SAMPLES_PER_US
        fig, axes = plt.subplots(2, 1, figsize=(12, 6))

        # Full waveform envelope
        axes[0].plot(t_us, env, lw=0.5, color="C0")
        axes[0].axvline(8.0, color="red", ls="--", lw=0.8, label="preamble/data boundary (8 µs)")
        axes[0].set_xlabel("Time (µs)")
        axes[0].set_ylabel("|IQ|")
        axes[0].set_title("ADS-B PPM Waveform Envelope (full frame, 120 µs)")
        axes[0].legend(fontsize=8)
        axes[0].grid(True, alpha=0.3)

        # Preamble zoom (first 8 µs)
        preamble_mask = t_us < 8.0
        axes[1].plot(t_us[preamble_mask], env[preamble_mask], lw=0.8, color="C1")
        for off in expected_offsets:
            axes[1].axvline(off, color="gray", ls=":", lw=0.8)
        for p in pulse_starts_us:
            axes[1].axvline(p, color="green", ls="--", lw=1.0, alpha=0.7)
        axes[1].set_xlabel("Time (µs)")
        axes[1].set_ylabel("|IQ|")
        axes[1].set_title(
            f"P1: Preamble Pulses (detected={pulse_starts_us}, "
            f"expected={list(expected_offsets)})"
        )
        axes[1].grid(True, alpha=0.3)

        plt.tight_layout()
        save_verification_figure("adsb_P1_P4.png")
        plt.close(fig)
    except Exception:
        pass  # plot generation is optional

    return t


def performance(full: bool = False) -> ResultTable:
    return ResultTable("ADS-B — Performance (no statistical checks — protocol is bit-exact)")


if __name__ == "__main__":
    sys.exit(run_script(properties, performance))
