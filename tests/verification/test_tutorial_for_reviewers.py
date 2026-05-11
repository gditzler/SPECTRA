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
