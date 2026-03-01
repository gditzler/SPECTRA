import numpy as np
import numpy.testing as npt
import pytest


class TestGenerateQpskSymbols:
    def test_returns_complex64_ndarray(self):
        from spectra._rust import generate_qpsk_symbols
        symbols = generate_qpsk_symbols(64, seed=0)
        assert isinstance(symbols, np.ndarray)
        assert symbols.dtype == np.complex64

    def test_correct_length(self):
        from spectra._rust import generate_qpsk_symbols
        for n in [1, 10, 100, 1024]:
            symbols = generate_qpsk_symbols(n, seed=0)
            assert symbols.shape == (n,)

    def test_constellation_points_on_unit_circle(self):
        from spectra._rust import generate_qpsk_symbols
        symbols = generate_qpsk_symbols(1000, seed=0)
        magnitudes = np.abs(symbols)
        npt.assert_allclose(magnitudes, 1.0, atol=1e-6)

    def test_four_constellation_points(self):
        from spectra._rust import generate_qpsk_symbols
        symbols = generate_qpsk_symbols(10000, seed=0)
        expected_angles = [np.pi / 4, 3 * np.pi / 4, -3 * np.pi / 4, -np.pi / 4]
        for angle in expected_angles:
            point = np.exp(1j * angle)
            distances = np.abs(symbols - point)
            assert np.any(distances < 1e-5), f"Missing constellation point at angle {angle}"

    def test_deterministic_with_seed(self):
        from spectra._rust import generate_qpsk_symbols
        s1 = generate_qpsk_symbols(256, seed=42)
        s2 = generate_qpsk_symbols(256, seed=42)
        npt.assert_array_equal(s1, s2)

    def test_different_seeds_differ(self):
        from spectra._rust import generate_qpsk_symbols
        s1 = generate_qpsk_symbols(256, seed=0)
        s2 = generate_qpsk_symbols(256, seed=1)
        assert not np.array_equal(s1, s2)

    def test_zero_symbols(self):
        from spectra._rust import generate_qpsk_symbols
        symbols = generate_qpsk_symbols(0, seed=0)
        assert symbols.shape == (0,)
