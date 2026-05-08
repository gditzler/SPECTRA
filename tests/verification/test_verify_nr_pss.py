"""Pytest wrapper for examples/verification/verify_nr_pss.py."""

import sys
from pathlib import Path

import pytest

sys.path.insert(
    0, str(Path(__file__).resolve().parents[2] / "examples" / "verification")
)

pytestmark = pytest.mark.verification


def test_nr_pss_properties_pass():
    import verify_nr_pss

    res = verify_nr_pss.properties()
    assert res.all_passed, res.render()
