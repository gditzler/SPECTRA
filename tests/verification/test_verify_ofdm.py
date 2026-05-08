"""Pytest wrapper for examples/verification/verify_ofdm.py."""

import sys
from pathlib import Path

import pytest

sys.path.insert(
    0, str(Path(__file__).resolve().parents[2] / "examples" / "verification")
)

pytestmark = pytest.mark.verification


def test_ofdm_properties_pass():
    import verify_ofdm

    res = verify_ofdm.properties()
    assert res.all_passed, res.render()


@pytest.mark.slow
def test_ofdm_performance_pass():
    import verify_ofdm

    res = verify_ofdm.performance(full=False)
    assert res.all_passed, res.render()
