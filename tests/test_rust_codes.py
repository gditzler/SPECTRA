import numpy as np
import numpy.testing as npt
import pytest


@pytest.mark.rust
class TestGenerateFrankCode:
    def test_returns_complex64_ndarray(self):
        from spectra._rust import generate_frank_code
        chips = generate_frank_code(4)
        assert isinstance(chips, np.ndarray)
        assert chips.dtype == np.complex64

    def test_correct_length(self):
        from spectra._rust import generate_frank_code
        for m in [3, 4, 5, 8]:
            chips = generate_frank_code(m)
            assert chips.shape == (m * m,)

    def test_unit_magnitude(self):
        from spectra._rust import generate_frank_code
        chips = generate_frank_code(4)
        npt.assert_allclose(np.abs(chips), 1.0, atol=1e-5)

    def test_first_row_zero_phase(self):
        from spectra._rust import generate_frank_code
        chips = generate_frank_code(4)
        # First M chips (i=0): phase = 0 for all j
        npt.assert_allclose(chips[:4].real, 1.0, atol=1e-5)
        npt.assert_allclose(chips[:4].imag, 0.0, atol=1e-5)


@pytest.mark.rust
class TestGenerateP1Code:
    def test_correct_length(self):
        from spectra._rust import generate_p1_code
        chips = generate_p1_code(4)
        assert chips.shape == (16,)

    def test_unit_magnitude(self):
        from spectra._rust import generate_p1_code
        chips = generate_p1_code(5)
        npt.assert_allclose(np.abs(chips), 1.0, atol=1e-5)


@pytest.mark.rust
class TestGenerateP2Code:
    def test_correct_length(self):
        from spectra._rust import generate_p2_code
        chips = generate_p2_code(4)
        assert chips.shape == (16,)

    def test_unit_magnitude(self):
        from spectra._rust import generate_p2_code
        chips = generate_p2_code(6)
        npt.assert_allclose(np.abs(chips), 1.0, atol=1e-5)

    def test_rejects_odd_order(self):
        from spectra._rust import generate_p2_code
        with pytest.raises(ValueError, match="even"):
            generate_p2_code(5)


@pytest.mark.rust
class TestGenerateP3Code:
    def test_correct_length(self):
        from spectra._rust import generate_p3_code
        for n in [9, 16, 25, 64]:
            chips = generate_p3_code(n)
            assert chips.shape == (n,)

    def test_unit_magnitude(self):
        from spectra._rust import generate_p3_code
        chips = generate_p3_code(25)
        npt.assert_allclose(np.abs(chips), 1.0, atol=1e-5)

    def test_first_chip_is_one(self):
        from spectra._rust import generate_p3_code
        chips = generate_p3_code(16)
        assert abs(chips[0] - 1.0) < 1e-5


@pytest.mark.rust
class TestGenerateP4Code:
    def test_correct_length(self):
        from spectra._rust import generate_p4_code
        chips = generate_p4_code(36)
        assert chips.shape == (36,)

    def test_unit_magnitude(self):
        from spectra._rust import generate_p4_code
        chips = generate_p4_code(36)
        npt.assert_allclose(np.abs(chips), 1.0, atol=1e-5)

    def test_first_chip_is_one(self):
        from spectra._rust import generate_p4_code
        chips = generate_p4_code(16)
        assert abs(chips[0] - 1.0) < 1e-5


@pytest.mark.rust
class TestGenerateCostasSequence:
    def test_correct_length(self):
        from spectra._rust import generate_costas_sequence
        seq = generate_costas_sequence(7)
        assert len(seq) == 6  # Welch: order = p - 1

    def test_is_permutation(self):
        from spectra._rust import generate_costas_sequence
        seq = generate_costas_sequence(7)
        assert sorted(seq) == list(range(1, 7))

    def test_small_prime(self):
        from spectra._rust import generate_costas_sequence
        seq = generate_costas_sequence(5)
        assert len(seq) == 4
        assert sorted(seq) == list(range(1, 5))

    def test_rejects_too_small(self):
        from spectra._rust import generate_costas_sequence
        with pytest.raises(ValueError):
            generate_costas_sequence(2)
