"""Known-answer tests for verification helpers."""

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[2] / "examples" / "verification"))

import numpy as np
import pytest

pytestmark = pytest.mark.verification


def test_check_result_passed_within_tolerance():
    from _verify_helpers import CheckResult

    r = CheckResult(
        test_id="P1", name="x", measured=1.0, expected=1.001,
        tolerance=0.01, passed=True, citation="dummy:eq1", units="",
    )
    assert r.passed is True
    assert r.test_id == "P1"


def test_result_table_add_and_render_ascii():
    from _verify_helpers import ResultTable

    t = ResultTable("BPSK — Properties")
    t.add("P1", "constellation imag",
          measured=0.0, expected=0.0, tol=1e-9, cite="dummy:def", units="")
    t.add("P2", "bandwidth (kHz)",
          measured=135.0, expected=135.0, tol=1.35, cite="sklar2001:§3.5,eq3.74",
          units="kHz")
    out = t.render()
    assert "BPSK — Properties" in out
    assert "P1" in out and "P2" in out
    assert "constellation imag" in out
    assert t.all_passed is True


def test_result_table_records_failure_when_outside_tolerance():
    from _verify_helpers import ResultTable

    t = ResultTable("Demo")
    t.add("P1", "x", measured=1.5, expected=1.0, tol=0.1,
          cite="dummy:eq1", units="")
    assert t.all_passed is False
    assert "[FAIL]" in t.render() or "✗" in t.render()


def test_result_table_renders_html():
    from _verify_helpers import ResultTable

    t = ResultTable("Demo")
    t.add("P1", "x", measured=1.0, expected=1.0, tol=0.01,
          cite="dummy:eq1", units="")
    html = t.render_html()
    assert "<table" in html and "</table>" in html
    assert "P1" in html
