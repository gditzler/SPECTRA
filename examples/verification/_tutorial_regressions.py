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
    kernel = np.ones(blur_kernel_len, dtype=np.float64) / blur_kernel_len
    return np.convolve(iq, kernel, mode="same").astype(iq.dtype)


# ────────────────────────────────────────────────────────────────────────────
# Section B — Buggy* waveform subclasses
# ────────────────────────────────────────────────────────────────────────────

import spectra as sp  # noqa: E402  (intentional: keep Section A self-contained)
from spectra.waveforms.barker import BarkerCode  # noqa: E402


class BuggyBPSK_WrongRolloff(sp.BPSK):
    """RRC rolloff bumped from 0.35 to 0.5.

    The PSD shape no longer matches the squared-RRC mask at α = 0.35.
    Constellation and BER unaffected. The PSD-correlation check should
    drop from ≥ 0.99 to ~0.74.
    """

    def __init__(self, samples_per_symbol: int = 8) -> None:
        super().__init__(samples_per_symbol=samples_per_symbol, rolloff=0.5)


class BuggyBPSK_NoRRC(sp.BPSK):
    """Pulse-shape filter omitted; symbols are emitted as flat NRZ.

    The symbol constellation at symbol-instants is unchanged (still ±1), so
    BER vs theory still passes. PSD shape collapses — correlation against
    squared-RRC drops toward 0. Demonstrates why layered checks matter.
    """

    def generate(
        self,
        num_symbols: int,
        sample_rate: float,
        seed: int | None = None,
    ) -> np.ndarray:
        from spectra._rust import generate_bpsk_symbols

        s = seed if seed is not None else 0
        symbols = generate_bpsk_symbols(num_symbols, seed=s)
        sps = self.samples_per_symbol
        # Repeat each ±1 symbol over sps samples — flat NRZ, no RRC.
        return np.repeat(symbols.astype(np.complex64), sps)


class BuggyOFDM_MissingCP(sp.OFDM):
    """Cyclic prefix not prepended.

    CP-correlation peak at lag N_FFT vanishes; EVM after ZF equalisation
    blows up in the presence of any channel. Length is shorter by
    ``cp_length`` per symbol than the clean OFDM output.
    """

    def generate(
        self,
        num_symbols: int,
        sample_rate: float,
        seed: int | None = None,
    ) -> np.ndarray:
        # Generate the clean signal, then strip the CP from each symbol.
        clean = super().generate(num_symbols=num_symbols, sample_rate=sample_rate, seed=seed)
        n_fft = self._fft_size
        n_cp = self._cp_length
        sym_len = n_fft + n_cp
        n_sym = len(clean) // sym_len
        reshaped = clean.reshape(n_sym, sym_len)
        return reshaped[:, n_cp:].reshape(-1).astype(np.complex64)


class BuggyBarker13_FlippedChip(BarkerCode):
    """Chip 7 (0-indexed) inverted in the transmitted sequence.

    The autocorrelation PSLR degrades from 13 to ~6–7 depending on which
    chip is flipped. The exact-equality P1 check (sequence vs Levanon
    Tab. 6.1) still passes because the *code definition* is not changed —
    only the *transmitted IQ* is corrupted. This is intentionally
    instructive: it shows the P1 check guards code storage, not
    transmission integrity.
    """

    def __init__(self, samples_per_chip: int = 8, chip_to_flip: int = 7) -> None:
        super().__init__(length=13, samples_per_chip=samples_per_chip)
        self._chip_to_flip = chip_to_flip

    def generate(
        self,
        num_symbols: int = 1,
        sample_rate: float = 1e6,
        seed: int | None = None,
    ) -> np.ndarray:
        clean = super().generate(num_symbols=num_symbols, sample_rate=sample_rate, seed=seed)
        sps = self._samples_per_chip
        start = self._chip_to_flip * sps
        stop = start + sps
        out = clean.copy()
        out[start:stop] = -out[start:stop]
        return out.astype(np.complex64)
