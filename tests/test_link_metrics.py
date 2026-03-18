"""Tests for BER, SER, and PER metric functions."""
import numpy as np
import pytest


def test_ber_perfect():
    from spectra.metrics import bit_error_rate
    tx = np.array([0, 1, 0, 1, 1, 0])
    rx = np.array([0, 1, 0, 1, 1, 0])
    assert bit_error_rate(tx, rx) == 0.0


def test_ber_all_wrong():
    from spectra.metrics import bit_error_rate
    tx = np.array([0, 0, 0, 0])
    rx = np.array([1, 1, 1, 1])
    assert bit_error_rate(tx, rx) == 1.0


def test_ber_half():
    from spectra.metrics import bit_error_rate
    tx = np.array([0, 0, 1, 1])
    rx = np.array([0, 1, 1, 0])
    assert bit_error_rate(tx, rx) == pytest.approx(0.5)


def test_ser_perfect():
    from spectra.metrics import symbol_error_rate
    tx = np.array([0, 1, 2, 3])
    rx = np.array([0, 1, 2, 3])
    assert symbol_error_rate(tx, rx) == 0.0


def test_ser_one_error():
    from spectra.metrics import symbol_error_rate
    tx = np.array([0, 1, 2, 3])
    rx = np.array([0, 1, 2, 0])
    assert symbol_error_rate(tx, rx) == pytest.approx(0.25)


def test_per_no_errors():
    from spectra.metrics import packet_error_rate
    tx = np.zeros(100, dtype=int)
    rx = np.zeros(100, dtype=int)
    assert packet_error_rate(tx, rx, packet_length=10) == 0.0


def test_per_one_packet_error():
    from spectra.metrics import packet_error_rate
    tx = np.zeros(100, dtype=int)
    rx = np.zeros(100, dtype=int)
    rx[5] = 1
    assert packet_error_rate(tx, rx, packet_length=10) == pytest.approx(0.1)


def test_per_all_packets_error():
    from spectra.metrics import packet_error_rate
    tx = np.zeros(40, dtype=int)
    rx = np.ones(40, dtype=int)
    assert packet_error_rate(tx, rx, packet_length=10) == 1.0


def test_per_truncates():
    from spectra.metrics import packet_error_rate
    tx = np.zeros(15, dtype=int)
    rx = np.zeros(15, dtype=int)
    rx[3] = 1
    assert packet_error_rate(tx, rx, packet_length=10) == 1.0
