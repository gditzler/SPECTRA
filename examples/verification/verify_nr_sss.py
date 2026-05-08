"""SPECTRA Verification — 5G NR Secondary Synchronisation Signal (SSS)
=======================================================================
  P1. Sample-exact equality with 3GPP Gold sequence for (NID1, NID2) pairs:
      127 ±1 values from 3GPP TS 38.211 §7.4.2.3.1.
  P2. BPSK-valued (±1 only).
  P3. Max |cross-correlation| between distinct (NID1, NID2) ≤ 0.7 of auto-peak.
  P4. Sequence length = 127.

Reference: [3gpp_38_211:§7.4.2.3.1].

Implementation notes:
  P1: The plan's `_reference_sss` template specifies initial states
      [1,0,0,0,0,0,0] for both x0 and x1, matching 3GPP TS 38.211 §7.4.2.3.1.
      SPECTRA's Rust implementation (`rust/src/nr.rs:nr_sss_inner`) also uses
      [1,0,0,0,0,0,0] for both sequences.  No correction needed here — the
      plan, standard, and Rust are all in agreement.

      In contrast, the PSS (Task 15 / verify_nr_pss.py) required a correction:
      the plan's _reference_pss used [0,1,1,0,1,1,1] instead of the standard
      [1,1,1,0,1,1,0].  For SSS the plan was already correct.

  P3: The plan uses ``np.correlate(s_a, s_b, mode=\"full\").max()`` without
      ``np.abs()``, which can miss negative cross-correlation peaks.  This
      script uses ``np.abs()`` to find the true maximum absolute cross-
      correlation, which is the correct figure of merit for distinguishing
      cells.  For the chosen pair (NID1=0/NID2=0 vs NID1=50/NID2=1) the
      absolute peak is 26/127 ≈ 0.20, well under the 0.70 threshold.

Run:
    python examples/verification/verify_nr_sss.py            # quick mode
    python examples/verification/verify_nr_sss.py --full     # publication mode
"""

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent))

import numpy as np

from spectra._rust import generate_nr_sss

from _verify_helpers import ResultTable, run_script


def _reference_sss(n_id_1: int, n_id_2: int) -> np.ndarray:
    """Independent SSS Gold-sequence implementation per 3GPP TS 38.211 §7.4.2.3.1.

    d_SSS(n) = (1 - 2·x_0((n + m_0) mod 127)) · (1 - 2·x_1((n + m_1) mod 127))

    where:
        m_0 = 15·floor(NID1/112) + 5·NID2
        m_1 = NID1 mod 112

    m-sequence generators (per §7.4.2.3.1):
        x0(i+7) = (x0(i+4) + x0(i)) mod 2,  x0[0..6] = [1, 0, 0, 0, 0, 0, 0]
        x1(i+7) = (x1(i+1) + x1(i)) mod 2,  x1[0..6] = [1, 0, 0, 0, 0, 0, 0]

    Both initial states are [1,0,0,0,0,0,0] as specified in the standard.
    SPECTRA's Rust implementation (rust/src/nr.rs:nr_sss_inner) uses the same
    initial states and recurrence polynomials — no correction required.
    """
    total_len = 127 + 7  # minimum buffer to generate 127 sequence values
    x0 = np.zeros(total_len, dtype=np.int8)
    x1 = np.zeros(total_len, dtype=np.int8)
    # Standard initial conditions: x0[0..6] = x1[0..6] = [1, 0, 0, 0, 0, 0, 0]
    x0[0] = 1
    x1[0] = 1
    for i in range(total_len - 7):  # generates indices 7..133
        x0[i + 7] = (x0[i + 4] + x0[i]) % 2
        x1[i + 7] = (x1[i + 1] + x1[i]) % 2
    m0 = 15 * (n_id_1 // 112) + 5 * n_id_2
    m1 = n_id_1 % 112
    n = np.arange(127)
    # Indices are always in 0..126, safely within the 134-element buffer
    d = (1 - 2 * x0[(n + m0) % 127].astype(np.int16)) * (
        1 - 2 * x1[(n + m1) % 127].astype(np.int16)
    )
    return d.astype(np.int8)


def properties() -> ResultTable:
    t = ResultTable("NR SSS — Properties")

    # ── P1 — sample-exact for a set of (NID1, NID2) pairs ────────────────────
    # generate_nr_sss returns Complex32 (imaginary part always 0); take .real
    # to avoid a ComplexWarning when casting to int8.
    for n_id_1, n_id_2 in [(0, 0), (50, 1), (335, 2)]:
        gen = np.asarray(generate_nr_sss(n_id_1, n_id_2)).real.astype(np.int8)
        ref = _reference_sss(n_id_1, n_id_2)
        t.add(
            f"P1.{n_id_1}.{n_id_2}",
            f"sample equality (NID1={n_id_1}, NID2={n_id_2})",
            measured=int(np.array_equal(gen, ref)),
            expected=1,
            tol=0,
            cite="3gpp_38_211:§7.4.2.3.1",
        )

    # ── P2 — BPSK valuedness ─────────────────────────────────────────────────
    seq = np.asarray(generate_nr_sss(0, 0)).real.astype(np.int8)
    bpsk = bool(np.all(np.abs(seq) == 1))
    t.add(
        "P2",
        "BPSK-valued (±1 only)",
        measured=int(bpsk),
        expected=1,
        tol=0,
        cite="3gpp_38_211:§7.4.2.3.1",
    )

    # ── P3 — cross-correlation between distinct (NID1, NID2) ─────────────────
    # Compare NID1=0/NID2=0 vs NID1=50/NID2=1.  Use np.abs() to capture both
    # positive and negative cross-correlation peaks (the correct figure of merit
    # for cell discrimination).  Empirically: ratio ≈ 26/127 ≈ 0.20, well under
    # the 0.70 threshold.
    s_a = np.asarray(generate_nr_sss(0, 0)).real.astype(float)
    s_b = np.asarray(generate_nr_sss(50, 1)).real.astype(float)
    auto_peak = float(np.abs(np.correlate(s_a, s_a, mode="full")).max())
    cross_peak = float(np.abs(np.correlate(s_a, s_b, mode="full")).max())
    ratio_cross = float(cross_peak / auto_peak)
    _CROSS_THRESHOLD = 0.70
    t.add(
        "P3",
        "max |cross| / max |auto| (NID1=0,2=0 vs 50,2=1)",
        measured=ratio_cross,
        expected=_CROSS_THRESHOLD,
        tol=abs(ratio_cross - _CROSS_THRESHOLD) + 1e-9 if ratio_cross <= _CROSS_THRESHOLD else 0.0,
        cite="3gpp_38_211:§7.4.2.3.1",
    )

    # ── P4 — sequence length ─────────────────────────────────────────────────
    t.add(
        "P4",
        "SSS sequence length",
        measured=len(seq),
        expected=127,
        tol=0,
        cite="3gpp_38_211:§7.4.2.3.1",
    )

    return t


def performance(full: bool = False) -> ResultTable:
    return ResultTable("NR SSS — Performance (no statistical checks)")


if __name__ == "__main__":
    sys.exit(run_script(properties, performance))
