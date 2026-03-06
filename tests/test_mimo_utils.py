"""Tests for MIMO antenna utilities."""
import numpy as np
import pytest

from spectra.impairments.mimo_utils import (
    exponential_correlation,
    kronecker_correlation,
    steering_vector,
)


class TestSteeringVector:
    def test_unit_magnitude(self):
        a = steering_vector(8, angle_rad=np.pi / 6)
        np.testing.assert_allclose(np.abs(a), 1.0, atol=1e-12)

    def test_length(self):
        a = steering_vector(4, angle_rad=0.0)
        assert len(a) == 4

    def test_broadside_all_ones(self):
        """At broadside (angle=0), all elements have phase 0."""
        a = steering_vector(4, angle_rad=0.0)
        np.testing.assert_allclose(a, np.ones(4), atol=1e-12)

    def test_complex_output(self):
        a = steering_vector(4, angle_rad=np.pi / 4)
        assert np.iscomplexobj(a)


class TestExponentialCorrelation:
    def test_symmetric(self):
        R = exponential_correlation(4, 0.8)
        np.testing.assert_allclose(R, R.T)

    def test_positive_semidefinite(self):
        R = exponential_correlation(8, 0.9)
        eigvals = np.linalg.eigvalsh(R)
        assert np.all(eigvals >= -1e-10)

    def test_identity_when_rho_zero(self):
        R = exponential_correlation(4, 0.0)
        np.testing.assert_allclose(R, np.eye(4))

    def test_diagonal_ones(self):
        R = exponential_correlation(5, 0.7)
        np.testing.assert_allclose(np.diag(R), 1.0)


class TestKroneckerCorrelation:
    def test_shape(self):
        R_tx = exponential_correlation(2, 0.5)
        R_rx = exponential_correlation(4, 0.8)
        R = kronecker_correlation(R_tx, R_rx)
        assert R.shape == (8, 8)  # 4*2 = 8

    def test_identity_inputs(self):
        R_tx = np.eye(2)
        R_rx = np.eye(3)
        R = kronecker_correlation(R_tx, R_rx)
        np.testing.assert_allclose(R, np.eye(6))
