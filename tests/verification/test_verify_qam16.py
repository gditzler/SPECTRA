"""Pytest wrapper for examples/verification/verify_qam16.py."""

import sys
from pathlib import Path

import pytest

sys.path.insert(
    0, str(Path(__file__).resolve().parents[2] / "examples" / "verification")
)

pytestmark = pytest.mark.verification


def test_qam16_properties_pass():
    import verify_qam16

    res = verify_qam16.properties()
    assert res.all_passed, res.render()


@pytest.mark.slow
def test_qam16_performance_pass():
    import verify_qam16

    res = verify_qam16.performance(full=False)
    assert res.all_passed, res.render()
