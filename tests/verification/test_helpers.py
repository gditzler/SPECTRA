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


def test_parse_references_loads_known_keys():
    from _verify_helpers import REFERENCES

    assert "proakis2008" in REFERENCES
    assert "3gpp_38_211" in REFERENCES
    assert "rtca_do260b" in REFERENCES


def test_cite_resolves_known_locus():
    from _verify_helpers import cite

    s = cite("proakis2008:eq4.3-13")
    assert "Proakis" in s or "proakis2008" in s
    assert "eq4.3-13" in s


def test_cite_raises_on_unknown_key():
    from _verify_helpers import cite

    with pytest.raises(KeyError):
        cite("nope2099:eq1")


def test_ber_bpsk_awgn_at_known_snr():
    from _verify_helpers import ber_bpsk_awgn

    # At Eb/N0 = 0 dB:  P_b = Q(sqrt(2)) ≈ 0.0786496035
    # At Eb/N0 = 10 dB: P_b = Q(sqrt(20)) ≈ 3.872e-6
    bers = ber_bpsk_awgn(np.array([0.0, 10.0]))
    np.testing.assert_allclose(bers[0], 0.0786496035, rtol=1e-6)
    np.testing.assert_allclose(bers[1], 3.872e-6, rtol=5e-3)


def test_ser_mpsk_awgn_matches_bpsk_at_M2():
    from _verify_helpers import ber_bpsk_awgn, ser_mpsk_awgn

    ebn0_db = np.array([0.0, 5.0, 10.0])
    bpsk = ber_bpsk_awgn(ebn0_db)
    # M=2 PSK SER == BPSK BER (binary)
    mpsk2 = ser_mpsk_awgn(M=2, ebn0_db=ebn0_db)
    np.testing.assert_allclose(mpsk2, bpsk, rtol=5e-3, atol=1e-6)


def test_ser_mqam_awgn_4qam_matches_qpsk_approx():
    from _verify_helpers import ser_mqam_awgn, ser_mpsk_awgn

    # 4-QAM ≡ QPSK, SERs should be close to within 0.5 dB
    ebn0_db = np.array([6.0, 10.0])
    qam = ser_mqam_awgn(M=4, ebn0_db=ebn0_db)
    qpsk = ser_mpsk_awgn(M=4, ebn0_db=ebn0_db)
    assert np.all(np.abs(qam - qpsk) / qpsk < 0.10)


def test_psd_rrc_squared_unit_area():
    from _verify_helpers import psd_rrc_squared

    f = np.linspace(-5e6, 5e6, 4001)
    Rs = 1e6
    psd = psd_rrc_squared(f, Rs=Rs, alpha=0.35)
    # Integrated PSD over [-Rs, Rs] should be close to 1 for a unit-energy pulse
    df = f[1] - f[0]
    area = np.trapezoid(psd, dx=df)
    assert 0.85 < area < 1.15


def test_matched_filter_gain_db():
    from _verify_helpers import matched_filter_gain_db

    # TBP = 100 → gain = 20 dB
    assert matched_filter_gain_db(100.0) == pytest.approx(20.0)
    assert matched_filter_gain_db(1.0) == pytest.approx(0.0)
