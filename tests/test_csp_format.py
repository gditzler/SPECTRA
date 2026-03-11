"""Tests for shared CSP output formatting."""

import numpy as np
import pytest
import torch


@pytest.mark.csp
class TestFormatOutput:
    def test_magnitude_format(self):
        from spectra.transforms.csp_utils import format_csp_output

        data = np.array([[1 + 1j, 2 + 2j]], dtype=np.complex64)
        result = format_csp_output(data, "magnitude", db_scale=False)
        assert result.shape[0] == 1  # single channel

    def test_magnitude_db_scale(self):
        from spectra.transforms.csp_utils import format_csp_output

        data = np.array([[1 + 1j, 2 + 2j]], dtype=np.complex64)
        result = format_csp_output(data, "magnitude", db_scale=True)
        assert result.shape[0] == 1

    def test_mag_phase_format(self):
        from spectra.transforms.csp_utils import format_csp_output

        data = np.array([[1 + 1j, 2 + 2j]], dtype=np.complex64)
        result = format_csp_output(data, "mag_phase", db_scale=False)
        assert result.shape[0] == 2  # magnitude + phase channels

    def test_real_imag_format(self):
        from spectra.transforms.csp_utils import format_csp_output

        data = np.array([[1 + 1j, 2 + 2j]], dtype=np.complex64)
        result = format_csp_output(data, "real_imag", db_scale=False)
        assert result.shape[0] == 2  # real + imag channels

    def test_output_is_float32(self):
        from spectra.transforms.csp_utils import format_csp_output

        data = np.array([[1 + 1j]], dtype=np.complex64)
        for fmt in ("magnitude", "mag_phase", "real_imag"):
            result = format_csp_output(data, fmt, db_scale=False)
            assert result.dtype == torch.float32
