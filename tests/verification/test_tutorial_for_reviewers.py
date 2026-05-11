"""Pytest wrapper for examples/verification/tutorial_for_reviewers.ipynb."""

from pathlib import Path

import pytest

_REPO_ROOT = Path(__file__).resolve().parents[2]
_NOTEBOOK = _REPO_ROOT / "examples" / "verification" / "tutorial_for_reviewers.ipynb"

pytestmark = [pytest.mark.verification, pytest.mark.slow]


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
    import importlib
    import sys

    script_dir = _REPO_ROOT / "examples" / "verification"
    sys.path.insert(0, str(script_dir))
    try:
        tutorial = importlib.import_module("tutorial_for_reviewers")
    finally:
        sys.path.remove(str(script_dir))

    # Required top-level entry points
    assert hasattr(tutorial, "run_all"), "must expose run_all() that returns a results dict"
    results = tutorial.run_all(full=False)
    assert isinstance(results, dict)
    # Spot-check three pinned reference values that should be robust to RNG.
    assert results["bpsk"]["psd_correlation"] >= 0.99, results["bpsk"]
    assert results["ofdm"]["orthogonality_error"] <= 1e-9, results["ofdm"]
    assert results["barker13"]["pslr"] == pytest.approx(13.0, abs=1e-9), results["barker13"]


class TestPostIQCorruption:
    """Section A of _tutorial_regressions — post-generation corruption helpers."""

    def _load_module(self):
        import importlib
        import sys

        script_dir = _REPO_ROOT / "examples" / "verification"
        if str(script_dir) not in sys.path:
            sys.path.insert(0, str(script_dir))
        return importlib.import_module("_tutorial_regressions")

    def test_rotate_phase_preserves_magnitude(self):
        import numpy as np

        mod = self._load_module()
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
        import numpy as np

        mod = self._load_module()
        # 4 OFDM symbols of length 16 (N_FFT=12, N_CP=4)
        n_fft, n_cp = 12, 4
        sym_len = n_fft + n_cp
        iq = np.arange(4 * sym_len, dtype=np.complex64)
        out = mod.drop_cp_sample(iq, n_fft=n_fft, n_cp=n_cp)
        assert len(out) == 4 * (sym_len - 1)

    def test_flip_chip_inverts_one_chip(self):
        import numpy as np

        mod = self._load_module()
        # 5 chips of 4 samples each, all +1
        sps = 4
        iq = np.ones(5 * sps, dtype=np.complex64)
        out = mod.flip_chip(iq, samples_per_chip=sps, chip_index=2)
        # Chip 0,1,3,4 unchanged; chip 2 inverted
        assert np.all(out[: 2 * sps] == 1.0)
        assert np.all(out[2 * sps : 3 * sps] == -1.0)
        assert np.all(out[3 * sps :] == 1.0)

    def test_broaden_pulse_returns_same_length(self):
        import numpy as np

        mod = self._load_module()
        iq = np.random.default_rng(0).standard_normal(128).astype(np.complex64)
        out = mod.broaden_pulse(iq, blur_kernel_len=5)
        assert len(out) == len(iq)


class TestBuggySubclasses:
    """Section B of _tutorial_regressions — Buggy* waveform subclasses."""

    def _load_module(self):
        import importlib
        import sys

        script_dir = _REPO_ROOT / "examples" / "verification"
        if str(script_dir) not in sys.path:
            sys.path.insert(0, str(script_dir))
        return importlib.import_module("_tutorial_regressions")

    def test_buggy_bpsk_wrong_rolloff_differs_from_clean(self):
        import numpy as np
        import spectra as sp

        mod = self._load_module()
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
        import numpy as np

        mod = self._load_module()
        # BuggyBPSK_NoRRC skips pulse-shaping entirely. Samples should still
        # be ±1 ± tiny noise — the BPSK *symbols* are intact, only the
        # pulse-shape filter is missing. PSD will be degraded; constellation
        # at symbol-instants is unchanged.
        buggy = mod.BuggyBPSK_NoRRC(samples_per_symbol=8).generate(
            num_symbols=256, sample_rate=1e6, seed=0
        )
        sps = 8
        # Sample every sps-th sample (symbol instants); should be ±1.
        symbol_samples = buggy[::sps]
        assert np.all(np.isin(symbol_samples.real.round(), [-1.0, 1.0]))
        assert np.all(np.abs(symbol_samples.imag) < 1e-3)

    def test_buggy_ofdm_missing_cp_shorter_than_clean(self):
        import numpy as np
        import spectra as sp

        mod = self._load_module()
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
        import numpy as np
        from spectra.waveforms.barker import BarkerCode

        mod = self._load_module()
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
