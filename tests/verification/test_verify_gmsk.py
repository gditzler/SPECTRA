"""Pytest wrapper for examples/verification/verify_gmsk.py."""

import sys
from pathlib import Path

import pytest

sys.path.insert(0, str(Path(__file__).resolve().parents[2] / "examples" / "verification"))

pytestmark = pytest.mark.verification


def test_gmsk_properties_pass():
    import verify_gmsk

    res = verify_gmsk.properties()
    assert res.all_passed, res.render()


@pytest.mark.slow
def test_gmsk_performance_pass():
    import verify_gmsk

    res = verify_gmsk.performance(full=False)
    assert res.all_passed, res.render()
