"""Tests for Python CSP transform wrappers."""

import numpy as np
import pytest
import torch
from spectra.transforms import CAF, PSD, SCD, SCF, Cumulants, EnergyDetector

# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------


def _make_noise(n_samples: int, seed: int = 0) -> np.ndarray:
    rng = np.random.default_rng(seed)
    return (
        (rng.standard_normal(n_samples) + 1j * rng.standard_normal(n_samples)) / np.sqrt(2)
    ).astype(np.complex64)


def _make_bpsk(n_symbols: int, sps: int = 8, seed: int = 42) -> np.ndarray:
    """Generate a simple BPSK signal at 1 sample/symbol then upsample."""
    rng = np.random.default_rng(seed)
    symbols = rng.choice([-1.0, 1.0], size=n_symbols).astype(np.float32)
    # Zero-insert upsample
    upsampled = np.zeros(n_symbols * sps, dtype=np.complex64)
    upsampled[::sps] = symbols
    # Simple rectangular pulse (no RRC for test simplicity)
    return upsampled


# ---------------------------------------------------------------------------
# SCD transform
# ---------------------------------------------------------------------------


@pytest.mark.csp
class TestSCDTransform:
    def test_output_type_and_dtype(self):
        iq = _make_noise(4096)
        scd = SCD(nfft=64, n_alpha=64, hop=16)
        result = scd(iq)
        assert isinstance(result, torch.Tensor)
        assert result.dtype == torch.float32

    def test_magnitude_shape(self):
        nfft, n_alpha = 64, 64
        iq = _make_noise(4096)
        scd = SCD(nfft=nfft, n_alpha=n_alpha, hop=16, output_format="magnitude")
        result = scd(iq)
        assert result.shape == (1, nfft, n_alpha)

    def test_mag_phase_shape(self):
        nfft, n_alpha = 64, 64
        iq = _make_noise(4096)
        scd = SCD(nfft=nfft, n_alpha=n_alpha, hop=16, output_format="mag_phase")
        result = scd(iq)
        assert result.shape == (2, nfft, n_alpha)

    def test_real_imag_shape(self):
        nfft, n_alpha = 64, 64
        iq = _make_noise(4096)
        scd = SCD(nfft=nfft, n_alpha=n_alpha, hop=16, output_format="real_imag")
        result = scd(iq)
        assert result.shape == (2, nfft, n_alpha)

    def test_db_scale(self):
        iq = _make_noise(4096)
        scd_lin = SCD(nfft=64, n_alpha=64, hop=16, db_scale=False)
        scd_db = SCD(nfft=64, n_alpha=64, hop=16, db_scale=True)
        lin = scd_lin(iq)
        db = scd_db(iq)
        # dB values should generally be negative for small magnitudes
        assert db.max() < lin.max()

    def test_invalid_method_raises(self):
        with pytest.raises(ValueError, match="Unknown method"):
            SCD(method="bogus")

    def test_invalid_output_format_raises(self):
        with pytest.raises(ValueError, match="Unknown output_format"):
            SCD(output_format="polar3d")

    def test_no_nan_inf(self):
        iq = _make_noise(4096)
        scd = SCD(nfft=64, n_alpha=64, hop=16)
        result = scd(iq)
        assert not torch.any(torch.isnan(result))
        assert not torch.any(torch.isinf(result))

    def test_deterministic(self):
        iq = _make_noise(4096, seed=99)
        scd = SCD(nfft=64, n_alpha=64, hop=16)
        r1 = scd(iq)
        r2 = scd(iq)
        torch.testing.assert_close(r1, r2)


# ---------------------------------------------------------------------------
# PSD transform
# ---------------------------------------------------------------------------


@pytest.mark.csp
class TestPSDTransform:
    def test_output_type_and_dtype(self):
        iq = _make_noise(4096)
        psd = PSD(nfft=256, overlap=128)
        result = psd(iq)
        assert isinstance(result, torch.Tensor)
        assert result.dtype == torch.float32

    def test_shape(self):
        nfft = 256
        iq = _make_noise(4096)
        psd = PSD(nfft=nfft, overlap=128)
        result = psd(iq)
        assert result.shape == (1, nfft)

    def test_nonnegative_linear(self):
        iq = _make_noise(4096)
        psd = PSD(nfft=256, overlap=128, db_scale=False)
        result = psd(iq)
        assert torch.all(result >= 0.0)

    def test_db_scale(self):
        iq = _make_noise(4096)
        psd_lin = PSD(nfft=256, overlap=128, db_scale=False)
        psd_db = PSD(nfft=256, overlap=128, db_scale=True)
        psd_lin(iq)
        db = psd_db(iq)
        # dB of values < 1 should be negative
        assert db.min() < 0.0

    def test_no_nan_inf(self):
        iq = _make_noise(4096)
        psd = PSD(nfft=256, overlap=128)
        result = psd(iq)
        assert not torch.any(torch.isnan(result))
        assert not torch.any(torch.isinf(result))


# ---------------------------------------------------------------------------
# Cumulants transform
# ---------------------------------------------------------------------------


@pytest.mark.csp
class TestCumulantsTransform:
    def test_output_type_and_dtype(self):
        iq = _make_noise(4096)
        cum = Cumulants(max_order=4)
        result = cum(iq)
        assert isinstance(result, torch.Tensor)
        assert result.dtype == torch.float32

    def test_shape_order4(self):
        iq = _make_noise(4096)
        cum = Cumulants(max_order=4)
        result = cum(iq)
        assert result.shape == (5,)

    def test_shape_order6(self):
        iq = _make_noise(4096)
        cum = Cumulants(max_order=6)
        result = cum(iq)
        assert result.shape == (9,)

    def test_invalid_order_raises(self):
        with pytest.raises(ValueError, match="max_order must be 4 or 6"):
            Cumulants(max_order=3)

    def test_nonnegative(self):
        """Magnitudes are always >= 0."""
        iq = _make_noise(4096)
        cum = Cumulants(max_order=4)
        result = cum(iq)
        assert torch.all(result >= 0.0)

    def test_gaussian_c40_near_zero(self):
        """For circular Gaussian noise, |C40| should be near zero."""
        iq = _make_noise(100_000, seed=0)
        cum = Cumulants(max_order=4)
        result = cum(iq)
        c40_mag = result[2].item()  # index 2 = |C40|
        assert c40_mag < 0.05, f"|C40| for Gaussian = {c40_mag:.4f}, expected ~0"

    def test_c21_is_signal_power(self):
        """C21 = E[|x|^2] for zero-mean signal (signal power ~1.0)."""
        iq = _make_noise(50_000, seed=1)
        cum = Cumulants(max_order=4)
        result = cum(iq)
        c21 = result[1].item()  # index 1 = |C21|
        assert abs(c21 - 1.0) < 0.1, f"|C21| = {c21}, expected ~1.0"


# ---------------------------------------------------------------------------
# Drop-in compatibility with NarrowbandDataset
# ---------------------------------------------------------------------------


@pytest.mark.csp
class TestDropInCompatibility:
    """Verify that CSP transforms work as drop-in replacements for STFT."""

    def test_scd_callable_on_complex_array(self):
        iq = _make_noise(4096)
        scd = SCD(nfft=64, n_alpha=64, hop=16)
        result = scd(iq)
        assert result.ndim == 3  # [C, Nf, Na]

    def test_psd_callable_on_complex_array(self):
        iq = _make_noise(4096)
        psd = PSD(nfft=256, overlap=128)
        result = psd(iq)
        assert result.ndim == 2  # [1, Nf]

    def test_cumulants_callable_on_complex_array(self):
        iq = _make_noise(4096)
        cum = Cumulants(max_order=4)
        result = cum(iq)
        assert result.ndim == 1  # [n_features]

    def test_accepts_float64_input(self):
        """Transforms should accept float64 complex and convert internally."""
        iq = (_make_noise(4096)).astype(np.complex128)
        scd = SCD(nfft=64, n_alpha=64, hop=16)
        result = scd(iq)
        assert result.dtype == torch.float32

    def test_accepts_non_contiguous_input(self):
        """Transforms should handle non-contiguous arrays."""
        iq = _make_noise(8192)
        iq_strided = iq[::2]  # non-contiguous
        scd = SCD(nfft=64, n_alpha=64, hop=16)
        result = scd(iq_strided)
        assert result.shape == (1, 64, 64)


# ---------------------------------------------------------------------------
# SCD with FAM method
# ---------------------------------------------------------------------------


@pytest.mark.csp
class TestSCDFamTransform:
    def test_fam_output_shape(self):
        iq = _make_noise(4096)
        scd = SCD(nfft=64, n_alpha=64, hop=16, method="fam")
        result = scd(iq)
        assert result.shape == (1, 64, 64)

    def test_fam_output_dtype(self):
        iq = _make_noise(4096)
        scd = SCD(nfft=64, n_alpha=64, hop=16, method="fam")
        result = scd(iq)
        assert result.dtype == torch.float32

    def test_fam_no_nan_inf(self):
        iq = _make_noise(4096)
        scd = SCD(nfft=64, n_alpha=64, hop=16, method="fam")
        result = scd(iq)
        assert not torch.any(torch.isnan(result))
        assert not torch.any(torch.isinf(result))

    def test_fam_mag_phase_shape(self):
        iq = _make_noise(4096)
        scd = SCD(nfft=64, n_alpha=64, hop=16, method="fam", output_format="mag_phase")
        result = scd(iq)
        assert result.shape == (2, 64, 64)


# ---------------------------------------------------------------------------
# SCF transform
# ---------------------------------------------------------------------------


@pytest.mark.csp
class TestSCFTransform:
    def test_output_shape(self):
        iq = _make_noise(4096)
        scf = SCF(nfft=64, n_alpha=64, hop=16)
        result = scf(iq)
        assert result.shape == (1, 64, 64)

    def test_output_dtype(self):
        iq = _make_noise(4096)
        scf = SCF(nfft=64, n_alpha=64, hop=16)
        result = scf(iq)
        assert result.dtype == torch.float32

    def test_no_nan_inf(self):
        iq = _make_noise(4096)
        scf = SCF(nfft=64, n_alpha=64, hop=16)
        result = scf(iq)
        assert not torch.any(torch.isnan(result))
        assert not torch.any(torch.isinf(result))

    def test_mag_phase_shape(self):
        iq = _make_noise(4096)
        scf = SCF(nfft=64, n_alpha=64, hop=16, output_format="mag_phase")
        result = scf(iq)
        assert result.shape == (2, 64, 64)

    def test_real_imag_shape(self):
        iq = _make_noise(4096)
        scf = SCF(nfft=64, n_alpha=64, hop=16, output_format="real_imag")
        result = scf(iq)
        assert result.shape == (2, 64, 64)

    def test_invalid_output_format_raises(self):
        with pytest.raises(ValueError, match="Unknown output_format"):
            SCF(output_format="bogus")


# ---------------------------------------------------------------------------
# CAF transform
# ---------------------------------------------------------------------------


@pytest.mark.csp
class TestCAFTransform:
    def test_output_shape(self):
        iq = _make_noise(4096)
        caf = CAF(n_alpha=64, max_lag=32)
        result = caf(iq)
        assert result.shape == (1, 64, 32)

    def test_output_dtype(self):
        iq = _make_noise(4096)
        caf = CAF(n_alpha=64, max_lag=32)
        result = caf(iq)
        assert result.dtype == torch.float32

    def test_no_nan_inf(self):
        iq = _make_noise(4096)
        caf = CAF(n_alpha=64, max_lag=32)
        result = caf(iq)
        assert not torch.any(torch.isnan(result))
        assert not torch.any(torch.isinf(result))

    def test_mag_phase_shape(self):
        iq = _make_noise(4096)
        caf = CAF(n_alpha=64, max_lag=32, output_format="mag_phase")
        result = caf(iq)
        assert result.shape == (2, 64, 32)

    def test_invalid_output_format_raises(self):
        with pytest.raises(ValueError, match="Unknown output_format"):
            CAF(output_format="bogus")


# ---------------------------------------------------------------------------
# EnergyDetector transform
# ---------------------------------------------------------------------------


@pytest.mark.csp
class TestEnergyDetectorTransform:
    def test_output_shape(self):
        iq = _make_noise(4096)
        det = EnergyDetector(nfft=256, overlap=128)
        result = det(iq)
        assert result.shape == (1, 256)

    def test_output_dtype(self):
        iq = _make_noise(4096)
        det = EnergyDetector(nfft=256, overlap=128)
        result = det(iq)
        assert result.dtype == torch.float32

    def test_binary_output(self):
        """Output should be 0 or 1."""
        iq = _make_noise(4096)
        det = EnergyDetector(nfft=256, overlap=128)
        result = det(iq)
        unique = torch.unique(result)
        for val in unique:
            assert val.item() in (0.0, 1.0)

    def test_tone_detects_peak(self):
        """A pure tone should be detected above noise."""
        rng = np.random.default_rng(0)
        n = 16384
        t = np.arange(n)
        tone = np.exp(1j * 2 * np.pi * 0.2 * t).astype(np.complex64)
        noise = ((rng.standard_normal(n) + 1j * rng.standard_normal(n)) / np.sqrt(2) * 0.01).astype(
            np.complex64
        )
        iq = tone + noise
        det = EnergyDetector(nfft=256, overlap=128, threshold_db=6.0)
        result = det(iq)
        # At least one bin should be detected
        assert result.sum().item() > 0
        # But not all bins (noise bins should not be detected)
        assert result.sum().item() < 256
