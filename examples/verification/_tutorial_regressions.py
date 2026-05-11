"""Tutorial-local regression helpers.

This module is example-local; do not import it from library code. Two
sections:

  Section A — post-generation IQ corruption helpers.
              Pure functions that mutate a clean IQ stream to model
              transmission-style faults (phase rotation, CP loss,
              chip flips, smearing). The generator stays correct; the
              IQ stream gets perturbed downstream.

  Section B — Buggy* waveform subclasses.
              Subclasses of sp.X that override generate() to introduce
              specific generator-side defects (wrong rolloff, omitted
              CP, flipped chip). The fault is upstream of the IQ samples.

Both sections feed the regression catalogue tables in
tutorial_for_reviewers.ipynb / .py.
"""

from __future__ import annotations

import numpy as np

# ────────────────────────────────────────────────────────────────────────────
# Section A — post-generation IQ corruption helpers
# ────────────────────────────────────────────────────────────────────────────


def rotate_phase(iq: np.ndarray, radians: float) -> np.ndarray:
    """Apply a constant phase rotation to every sample.

    Models a static phase offset between TX and RX (e.g., uncorrected
    carrier-recovery error). The constellation rotates; magnitudes and
    inter-sample phase differences are preserved.
    """
    return (iq * np.exp(1j * radians)).astype(iq.dtype)


def drop_cp_sample(iq: np.ndarray, n_fft: int, n_cp: int) -> np.ndarray:
    """Remove the first sample of the cyclic prefix of every OFDM symbol.

    Models a timing-offset bug on the receiver side. The CP correlation
    peak shifts; subsequent ZF equalisation cannot recover the loss.
    Each OFDM symbol shrinks from ``n_fft + n_cp`` to ``n_fft + n_cp − 1``.
    """
    sym_len = n_fft + n_cp
    if len(iq) % sym_len != 0:
        raise ValueError(f"IQ length {len(iq)} not divisible by OFDM symbol length {sym_len}")
    n_syms = len(iq) // sym_len
    reshaped = iq.reshape(n_syms, sym_len)
    # Drop the first sample of every symbol; keep the remaining (sym_len - 1).
    trimmed = reshaped[:, 1:]
    return trimmed.reshape(-1).astype(iq.dtype)


def flip_chip(iq: np.ndarray, samples_per_chip: int, chip_index: int) -> np.ndarray:
    """Invert the IQ samples spanning a single chip in a chip-coded waveform.

    Models a single-bit transmit error in a coded radar waveform (e.g.,
    Barker). PSLR degrades measurably even from a single chip flip.
    """
    out = iq.copy()
    start = chip_index * samples_per_chip
    stop = start + samples_per_chip
    if stop > len(out):
        raise ValueError(
            f"chip_index={chip_index} with sps={samples_per_chip} exceeds IQ length {len(iq)}"
        )
    out[start:stop] = -out[start:stop]
    return out.astype(iq.dtype)


def broaden_pulse(iq: np.ndarray, blur_kernel_len: int) -> np.ndarray:
    """Apply a uniform moving-average filter to smear the signal.

    Models a low-pass receiver front-end (or an unintended ISI source).
    The PSD shape degrades; constellation samples blur. Output length
    preserved (same-mode convolution).
    """
    if blur_kernel_len < 1:
        raise ValueError("blur_kernel_len must be ≥ 1")
    kernel = np.ones(blur_kernel_len, dtype=iq.dtype) / blur_kernel_len
    return np.convolve(iq, kernel, mode="same").astype(iq.dtype)
