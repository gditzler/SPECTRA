"""Tests for shared dataset IQ utilities."""
import numpy as np
import torch
import pytest


class TestTruncatePad:
    def test_exact_length_unchanged(self):
        from spectra.datasets.iq_utils import truncate_pad

        iq = np.ones(100, dtype=np.complex64)
        result = truncate_pad(iq, 100)
        assert len(result) == 100
        np.testing.assert_array_equal(result, iq)

    def test_long_input_truncated(self):
        from spectra.datasets.iq_utils import truncate_pad

        iq = np.ones(200, dtype=np.complex64)
        result = truncate_pad(iq, 100)
        assert len(result) == 100

    def test_short_input_zero_padded(self):
        from spectra.datasets.iq_utils import truncate_pad

        iq = np.ones(50, dtype=np.complex64)
        result = truncate_pad(iq, 100)
        assert len(result) == 100
        assert result[50] == 0.0


class TestIQToTensor:
    def test_shape_and_dtype(self):
        from spectra.datasets.iq_utils import iq_to_tensor

        iq = np.array([1 + 2j, 3 + 4j], dtype=np.complex64)
        t = iq_to_tensor(iq)
        assert t.shape == (2, 2)
        assert t.dtype == torch.float32

    def test_values_correct(self):
        from spectra.datasets.iq_utils import iq_to_tensor

        iq = np.array([1 + 2j, 3 + 4j], dtype=np.complex64)
        t = iq_to_tensor(iq)
        np.testing.assert_allclose(t[0].numpy(), [1.0, 3.0])
        np.testing.assert_allclose(t[1].numpy(), [2.0, 4.0])
