"""Tests for the Choi-Williams Distribution transform."""
import numpy as np
import pytest
import torch

from spectra.transforms.cwd import CWD


def _make_tone(f0: float, n: int) -> np.ndarray:
    return np.exp(1j * 2 * np.pi * f0 * np.arange(n)).astype(np.complex64)


class TestCWDShape:
    """Output shape and dtype tests."""

    def test_magnitude_shape(self):
        iq = _make_tone(0.1, 512)
        cwd = CWD(nfft=256, output_format="magnitude")
        out = cwd(iq)
        assert out.shape == (1, 512, 256)
        assert out.dtype == torch.float32

    def test_mag_phase_shape(self):
        iq = _make_tone(0.1, 512)
        cwd = CWD(nfft=128, output_format="mag_phase")
        out = cwd(iq)
        assert out.shape == (2, 512, 128)

    def test_real_imag_shape(self):
        iq = _make_tone(0.1, 512)
        cwd = CWD(nfft=128, output_format="real_imag")
        out = cwd(iq)
        assert out.shape == (2, 512, 128)

    def test_n_time_subsampling(self):
        iq = _make_tone(0.1, 1024)
        cwd = CWD(nfft=256, n_time=64, output_format="magnitude")
        out = cwd(iq)
        assert out.shape == (1, 64, 256)

    def test_default_params(self):
        iq = _make_tone(0.1, 256)
        cwd = CWD()
        out = cwd(iq)
        assert out.shape == (1, 256, 256)
        assert out.dtype == torch.float32


class TestCWDContent:
    """Signal property and content tests."""

    def test_pure_tone_concentrated_frequency(self):
        """Pure tone should have energy concentrated at one frequency."""
        iq = _make_tone(0.1, 512)
        cwd = CWD(nfft=256, output_format="magnitude")
        out = cwd(iq)
        mid_slice = out[0, 256, :]
        peak_val = torch.max(mid_slice).item()
        mean_val = torch.mean(mid_slice).item()
        assert peak_val > 5 * mean_val

    def test_db_scale_reduces_values(self):
        iq = _make_tone(0.1, 256)
        linear = CWD(nfft=64, output_format="magnitude", db_scale=False)(iq)
        db = CWD(nfft=64, output_format="magnitude", db_scale=True)(iq)
        assert not torch.allclose(linear, db)

    def test_no_nan_or_inf(self):
        iq = _make_tone(0.1, 256)
        cwd = CWD(nfft=128, output_format="magnitude")
        out = cwd(iq)
        assert not torch.any(torch.isnan(out))
        assert not torch.any(torch.isinf(out))

    def test_deterministic(self):
        iq = _make_tone(0.1, 256)
        cwd = CWD(nfft=64, sigma=1.0)
        r1 = cwd(iq)
        r2 = cwd(iq)
        assert torch.equal(r1, r2)

    def test_sigma_affects_output(self):
        """Different sigma values should produce different results."""
        iq = _make_tone(0.1, 256)
        r1 = CWD(nfft=64, sigma=0.5)(iq)
        r2 = CWD(nfft=64, sigma=5.0)(iq)
        assert not torch.allclose(r1, r2)

    def test_non_contiguous_input(self):
        """Transform should handle non-contiguous arrays."""
        iq = _make_tone(0.1, 512)
        strided = iq[::2]
        cwd = CWD(nfft=64)
        out = cwd(strided)
        assert out.shape == (1, 256, 64)
        assert not torch.any(torch.isnan(out))

    def test_db_scale_with_mag_phase(self):
        """db_scale should work with mag_phase output format."""
        iq = _make_tone(0.1, 256)
        out = CWD(nfft=64, output_format="mag_phase", db_scale=True)(iq)
        assert out.shape == (2, 256, 64)
        assert not torch.any(torch.isnan(out))
        assert not torch.any(torch.isinf(out))

    def test_all_zero_input(self):
        """All-zero input should produce no NaN or Inf."""
        iq = np.zeros(256, dtype=np.complex64)
        out = CWD(nfft=64, output_format="magnitude")(iq)
        assert not torch.any(torch.isnan(out))
        assert not torch.any(torch.isinf(out))

    def test_short_signal_shorter_than_nfft(self):
        """Signal shorter than nfft should still work."""
        iq = _make_tone(0.1, 32)
        out = CWD(nfft=128, output_format="magnitude")(iq)
        assert out.shape == (1, 32, 128)
        assert not torch.any(torch.isnan(out))

    def test_magnitude_phase_vs_real_imag_consistency(self):
        """sqrt(re^2 + im^2) should equal magnitude output."""
        iq = _make_tone(0.1, 256)
        mag_out = CWD(nfft=64, output_format="magnitude")(iq)
        ri_out = CWD(nfft=64, output_format="real_imag")(iq)
        computed_mag = torch.sqrt(ri_out[0] ** 2 + ri_out[1] ** 2)
        assert torch.allclose(mag_out[0], computed_mag, atol=1e-6)


class TestCWDValidation:
    """Input validation tests."""

    def test_invalid_output_format(self):
        with pytest.raises(ValueError, match="Unknown output_format"):
            CWD(output_format="bad_format")

    def test_invalid_sigma_zero(self):
        with pytest.raises(ValueError, match="sigma must be positive"):
            CWD(sigma=0.0)

    def test_invalid_sigma_negative(self):
        with pytest.raises(ValueError, match="sigma must be positive"):
            CWD(sigma=-1.0)

    def test_invalid_nfft_zero(self):
        with pytest.raises(ValueError, match="nfft must be positive"):
            CWD(nfft=0)

    def test_invalid_nfft_negative(self):
        with pytest.raises(ValueError, match="nfft must be positive"):
            CWD(nfft=-1)
