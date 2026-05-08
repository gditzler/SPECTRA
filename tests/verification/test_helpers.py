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


def test_simulate_ber_awgn_bpsk_matches_theory_at_5dB():
    """At Eb/N0 = 5 dB and 100k bits, BPSK BER should be within 0.5 dB
    of theory (P_b ≈ 5.95e-3)."""
    from _verify_helpers import ber_bpsk_awgn, simulate_ber_awgn

    ebn0_db = np.array([5.0])
    measured = simulate_ber_awgn(modulation="bpsk", ebn0_db=ebn0_db,
                                 n_bits=100_000, seed=0)
    theory = ber_bpsk_awgn(ebn0_db)[0]
    measured_db = 10 * np.log10(measured[0])
    theory_db = 10 * np.log10(theory)
    assert abs(measured_db - theory_db) < 0.5


def test_measure_evm_rms_matches_snr_inverse():
    """EVM_RMS = 1 / sqrt(SNR_linear) for unit-power signal + AWGN."""
    from _verify_helpers import measure_evm_rms

    rng = np.random.default_rng(0)
    # Reference: 10000 unit-power QPSK symbols on the unit circle
    bits = rng.integers(0, 4, size=10_000)
    tx = np.exp(1j * (np.pi / 4 + bits * np.pi / 2)).astype(np.complex64)
    snr_db = 30.0
    snr_linear = 10 ** (snr_db / 10)
    sigma = np.sqrt(1.0 / (2.0 * snr_linear))
    noise = sigma * (rng.standard_normal(len(tx)) + 1j * rng.standard_normal(len(tx)))
    rx = tx + noise.astype(np.complex64)
    evm = measure_evm_rms(rx_symbols=rx, tx_ref=tx)
    expected = 1.0 / np.sqrt(snr_linear)  # 0.0316 at 30 dB
    np.testing.assert_allclose(evm, expected, rtol=0.10)


def test_measure_obw_pure_tone_is_narrow():
    """A 1 kHz tone in a 1 MHz capture should have OBW(99%) << capture BW."""
    from _verify_helpers import measure_obw

    fs = 1e6
    t = np.arange(50_000) / fs
    iq = np.exp(1j * 2 * np.pi * 1e3 * t).astype(np.complex64)
    obw = measure_obw(iq, fs=fs, fraction=0.99)
    assert obw < 0.01 * fs  # tone OBW is dominated by FFT leakage but tiny


def test_measure_papr_db_pure_tone_is_zero():
    """A constant-envelope tone has PAPR ≈ 0 dB (peak == average)."""
    from _verify_helpers import measure_papr_db

    fs = 1e6
    t = np.arange(10_000) / fs
    iq = np.exp(1j * 2 * np.pi * 1e3 * t).astype(np.complex64)
    papr = measure_papr_db(iq, percentile=99.9)
    assert -0.1 < papr < 0.1


def test_measure_psd_shape_correlation_self_is_one():
    from _verify_helpers import measure_psd_shape_correlation

    psd = np.exp(-np.linspace(-3, 3, 256) ** 2)
    c = measure_psd_shape_correlation(psd, psd)
    np.testing.assert_allclose(c, 1.0, atol=1e-12)


def test_measure_acpr_db_returns_high_for_clean_tone():
    """A pure tone confined to a narrow channel has very high ACPR."""
    from _verify_helpers import measure_acpr_db

    fs = 10e6
    t = np.arange(100_000) / fs
    iq = np.exp(1j * 2 * np.pi * 100e3 * t).astype(np.complex64)
    # Channel BW = 1 MHz, adjacent at ±1 MHz
    acpr = measure_acpr_db(iq, fs=fs, channel_bw=1e6, offsets=(1e6,))
    assert acpr[1e6] > 60.0


def test_autocorr_peak_to_sidelobe_barker_13_is_13():
    from _verify_helpers import autocorr_peak_to_sidelobe

    barker_13 = np.array([+1, +1, +1, +1, +1, -1, -1, +1, +1, -1, +1, -1, +1],
                         dtype=float)
    pslr = autocorr_peak_to_sidelobe(barker_13)
    np.testing.assert_allclose(pslr, 13.0, rtol=0, atol=1e-9)


def test_measure_cp_correlation_peak_recovers_cp_lag():
    from _verify_helpers import measure_cp_correlation_peak

    rng = np.random.default_rng(0)
    n_fft, n_cp = 64, 16
    body = (rng.standard_normal(n_fft) + 1j * rng.standard_normal(n_fft)) / np.sqrt(2)
    sym = np.concatenate([body[-n_cp:], body])
    sequence = np.tile(sym, 8).astype(np.complex64)
    lag, peak = measure_cp_correlation_peak(sequence, n_fft=n_fft, n_cp=n_cp)
    assert lag == n_fft
    assert peak > 0.5
