"""Pytest wrapper for examples/verification/verify_bpsk.py."""

import sys
from pathlib import Path

import pytest

sys.path.insert(
    0, str(Path(__file__).resolve().parents[2] / "examples" / "verification")
)

pytestmark = pytest.mark.verification


def test_bpsk_properties_pass():
    import verify_bpsk

    res = verify_bpsk.properties()
    assert res.all_passed, res.render()


@pytest.mark.slow
def test_bpsk_performance_pass():
    import verify_bpsk

    res = verify_bpsk.performance(full=False)
    assert res.all_passed, res.render()
