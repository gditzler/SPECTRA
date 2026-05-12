"""Pytest wrapper for examples/verification/tutorial_for_reviewers.ipynb."""

from __future__ import annotations

import importlib
import sys
from pathlib import Path

import numpy as np
import pytest
import spectra as sp
from spectra._rust import generate_bpsk_symbols
from spectra.waveforms.barker import BarkerCode

_REPO_ROOT = Path(__file__).resolve().parents[2]
_NOTEBOOK = _REPO_ROOT / "examples" / "verification" / "tutorial_for_reviewers.ipynb"

pytestmark = [pytest.mark.verification, pytest.mark.slow]


def _import_regressions():
    """Load _tutorial_regressions from examples/verification/."""
    script_dir = _REPO_ROOT / "examples" / "verification"
    if str(script_dir) not in sys.path:
        sys.path.insert(0, str(script_dir))
    return importlib.import_module("_tutorial_regressions")


def _import_tutorial():
    """Load tutorial_for_reviewers from examples/verification/."""
    script_dir = _REPO_ROOT / "examples" / "verification"
    if str(script_dir) not in sys.path:
        sys.path.insert(0, str(script_dir))
    return importlib.import_module("tutorial_for_reviewers")


@pytest.mark.skipif(not _NOTEBOOK.exists(), reason="notebook not yet created")
def test_notebook_executes():
    """Notebook must execute start-to-finish with FULL=False (default)."""
    import nbformat
    from nbclient import NotebookClient

    nb = nbformat.read(str(_NOTEBOOK), as_version=4)
    client = NotebookClient(nb, timeout=300, kernel_name="python3")
    client.execute()


def test_script_module_importable():
    """The companion script must import cleanly and expose required entry points."""
    tutorial = _import_tutorial()
    assert hasattr(tutorial, "run_all"), "must expose run_all() that returns a results dict"
    results = tutorial.run_all(full=False)
    assert isinstance(results, dict)


class TestPostIQCorruption:
    """Section A of _tutorial_regressions — post-generation corruption helpers."""

    def test_rotate_phase_preserves_magnitude(self):
        mod = _import_regressions()
        iq = (np.arange(64) + 1j * np.arange(64)).astype(np.complex64)
        rotated = mod.rotate_phase(iq, radians=0.5)
        np.testing.assert_allclose(np.abs(rotated), np.abs(iq), rtol=1e-5)
        # Phase shifted by 0.5 rad on every non-zero sample
        nonzero_mask = np.abs(iq) > 1e-6
        np.testing.assert_allclose(
            (np.angle(rotated) - np.angle(iq))[nonzero_mask],
            0.5,
            atol=1e-5,
        )

    def test_drop_cp_sample_shrinks_each_symbol_by_one(self):
        mod = _import_regressions()
        # 4 OFDM symbols of length 16 (N_FFT=12, N_CP=4)
        n_fft, n_cp = 12, 4
        sym_len = n_fft + n_cp
        iq = np.arange(4 * sym_len, dtype=np.complex64)
        out = mod.drop_cp_sample(iq, n_fft=n_fft, n_cp=n_cp)
        assert len(out) == 4 * (sym_len - 1)

    def test_flip_chip_inverts_one_chip(self):
        mod = _import_regressions()
        # 5 chips of 4 samples each, all +1
        sps = 4
        iq = np.ones(5 * sps, dtype=np.complex64)
        out = mod.flip_chip(iq, samples_per_chip=sps, chip_index=2)
        # Chip 0,1,3,4 unchanged; chip 2 inverted
        assert np.all(out[: 2 * sps] == 1.0)
        assert np.all(out[2 * sps : 3 * sps] == -1.0)
        assert np.all(out[3 * sps :] == 1.0)

    def test_broaden_pulse_returns_same_length(self):
        mod = _import_regressions()
        iq = np.random.default_rng(0).standard_normal(128).astype(np.complex64)
        out = mod.broaden_pulse(iq, blur_kernel_len=5)
        assert len(out) == len(iq)


class TestBuggySubclasses:
    """Section B of _tutorial_regressions — Buggy* waveform subclasses."""

    def test_buggy_bpsk_wrong_rolloff_differs_from_clean(self):
        mod = _import_regressions()
        clean = sp.BPSK(samples_per_symbol=8, rolloff=0.35).generate(
            num_symbols=256, sample_rate=1e6, seed=0
        )
        buggy = mod.BuggyBPSK_WrongRolloff(samples_per_symbol=8).generate(
            num_symbols=256, sample_rate=1e6, seed=0
        )
        # Same length, different content (rolloff change perturbs every sample).
        assert len(clean) == len(buggy)
        assert not np.allclose(clean, buggy)

    def test_buggy_bpsk_no_rrc_constellation_is_clean(self):
        mod = _import_regressions()
        # BuggyBPSK_NoRRC repeats each ±1 symbol sps times; no pulse shape, no noise.
        # Constellation at symbol-instants is exactly ±1+0j.
        buggy = mod.BuggyBPSK_NoRRC(samples_per_symbol=8).generate(
            num_symbols=256, sample_rate=1e6, seed=0
        )
        sps = 8
        # Sample every sps-th sample (symbol instants); should be ±1.
        symbol_samples = buggy[::sps]
        assert np.all(np.isin(symbol_samples.real.round(), [-1.0, 1.0]))
        assert np.all(np.abs(symbol_samples.imag) < 1e-3)

    def test_buggy_ofdm_missing_cp_shorter_than_clean(self):
        mod = _import_regressions()
        n_sym = 4
        clean = sp.OFDM(num_subcarriers=64, cp_length=16).generate(
            num_symbols=n_sym, sample_rate=1e6, seed=0
        )
        buggy = mod.BuggyOFDM_MissingCP(num_subcarriers=64, cp_length=16).generate(
            num_symbols=n_sym, sample_rate=1e6, seed=0
        )
        # BuggyOFDM omits the CP — shorter by n_sym * cp_length samples.
        assert len(buggy) == len(clean) - n_sym * 16

    def test_buggy_barker13_flipped_chip_differs(self):
        mod = _import_regressions()
        clean = BarkerCode(length=13, samples_per_chip=4).generate(
            num_symbols=1, sample_rate=1e6, seed=0
        )
        buggy = mod.BuggyBarker13_FlippedChip(samples_per_chip=4).generate(
            num_symbols=1, sample_rate=1e6, seed=0
        )
        # Same length; one chip-worth of samples is inverted relative to clean.
        assert len(clean) == len(buggy)
        diff = clean - buggy
        # Exactly one chip (4 samples) should differ by 2.0 in magnitude
        # (since chip is ±1 and inverted ±1 differs by ±2).
        n_diff = int(np.sum(np.abs(diff) > 0.1))
        assert n_diff == 4, f"expected one chip (4 samples) flipped, got {n_diff} samples"


class TestBPSKMeasurements:
    """Tutorial BPSK functions produce expected numeric values on clean signal."""

    def test_bpsk_constellation_check(self):
        tutorial = _import_tutorial()
        syms = generate_bpsk_symbols(10_000, seed=0)
        max_imag = tutorial.bpsk_constellation_check(syms)
        assert max_imag < 1e-6, f"BPSK symbols not on real axis: max(|imag|) = {max_imag}"

    def test_bpsk_psd_correlation_high(self):
        tutorial = _import_tutorial()
        iq = sp.BPSK(samples_per_symbol=8, rolloff=0.35).generate(
            num_symbols=4096, sample_rate=1e6, seed=0
        )
        corr = tutorial.bpsk_psd_correlation(iq, sample_rate=1e6, rolloff=0.35)
        assert corr >= 0.99, f"clean BPSK PSD correlation = {corr} < 0.99"

    def test_bpsk_ber_matches_theory(self):
        tutorial = _import_tutorial()
        n_bits = 50_000
        # Spot-check at a single SNR with a small symbol count for speed.
        measured, theory = tutorial.bpsk_ber_curve(
            ebn0_db_list=[0.0, 3.0, 6.0], n_bits=n_bits, seed=0
        )
        # Each measured BER should be within 0.8 dB of theory at these SNRs.
        meas_db = 10 * np.log10(np.maximum(measured, 1.0 / n_bits))
        theo_db = 10 * np.log10(theory)
        assert float(np.max(np.abs(meas_db - theo_db))) <= 0.8


class TestOFDMMeasurements:
    """Tutorial OFDM functions produce expected numeric values on clean signal."""

    def test_orthogonality_exact(self):
        tutorial = _import_tutorial()
        err = tutorial.ofdm_orthogonality_error(n_fft=64, n_used=52, n_cp=16, seed=0)
        assert err < 1e-9, f"orthogonality error {err} not < 1e-9"

    def test_cp_correlation_peak_at_n_fft(self):
        tutorial = _import_tutorial()
        lag, peak = tutorial.ofdm_cp_correlation(n_fft=64, n_used=52, n_cp=16, n_symbols=8, seed=0)
        assert lag == 64, f"CP peak lag = {lag}, expected 64"
        assert peak > 0.5, f"CP peak amplitude = {peak}, expected > 0.5"

    def test_ofdm_evm_at_40db(self):
        tutorial = _import_tutorial()
        evm = tutorial.ofdm_evm_after_awgn(
            snr_db=40.0, n_fft=64, n_used=52, n_cp=16, n_symbols=200, seed=0
        )
        # EVM at SNR=40 dB should be ~1 %; tolerance 2 %.
        assert evm <= 0.02, f"EVM {evm} > 0.02 at SNR=40 dB"


class TestBarker13Measurements:
    """Tutorial Barker-13 functions produce expected numeric values."""

    def test_canonical_code_equality(self):
        tutorial = _import_tutorial()
        match = tutorial.barker13_canonical_equality()
        assert match == 1, "Barker-13 code does not match Levanon Tab. 6.1"

    def test_pslr_equals_13(self):
        tutorial = _import_tutorial()
        pslr = tutorial.barker13_pslr()
        assert pslr == pytest.approx(13.0, abs=1e-9)

    def test_detection_rate_at_10db(self):
        tutorial = _import_tutorial()
        rate = tutorial.barker13_detection_rate(snr_db=10.0, n_trials=200, seed=0)
        assert rate >= 0.95, f"detection rate {rate} below 0.95 at SNR=10 dB"
