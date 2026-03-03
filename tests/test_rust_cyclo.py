"""Tests for Rust-backed cyclostationary signal processing primitives."""

import numpy as np
import numpy.testing as npt
import pytest

from spectra._rust import (
    channelize,
    compute_caf,
    compute_cumulants,
    compute_psd_welch,
    compute_scd_fam,
    compute_scd_ssca,
)


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------


def _make_tone(freq: float, n_samples: int, fs: float = 1.0) -> np.ndarray:
    """Generate a complex tone at *freq* Hz sampled at *fs*."""
    t = np.arange(n_samples) / fs
    return np.exp(1j * 2 * np.pi * freq * t).astype(np.complex64)


def _make_noise(n_samples: int, seed: int = 0) -> np.ndarray:
    rng = np.random.default_rng(seed)
    return (
        (rng.standard_normal(n_samples) + 1j * rng.standard_normal(n_samples)) / np.sqrt(2)
    ).astype(np.complex64)


# ---------------------------------------------------------------------------
# compute_scd_ssca
# ---------------------------------------------------------------------------


@pytest.mark.rust
class TestComputeScdSsca:
    def test_output_shape(self):
        iq = _make_noise(4096)
        scd = compute_scd_ssca(iq, nfft=64, n_alpha=64, hop=16)
        assert scd.shape == (64, 64)

    def test_output_dtype(self):
        iq = _make_noise(4096)
        scd = compute_scd_ssca(iq, nfft=64, n_alpha=64, hop=16)
        assert scd.dtype == np.complex64

    def test_no_nan_inf(self):
        iq = _make_noise(4096)
        scd = compute_scd_ssca(iq, nfft=64, n_alpha=64, hop=16)
        assert not np.any(np.isnan(scd))
        assert not np.any(np.isinf(scd))

    def test_deterministic(self):
        iq = _make_noise(4096, seed=42)
        s1 = compute_scd_ssca(iq, nfft=64, n_alpha=64, hop=16)
        s2 = compute_scd_ssca(iq, nfft=64, n_alpha=64, hop=16)
        npt.assert_array_equal(s1, s2)

    def test_short_signal_returns_zeros(self):
        iq = _make_noise(10)
        scd = compute_scd_ssca(iq, nfft=64, n_alpha=64, hop=16)
        assert scd.shape == (64, 64)
        npt.assert_array_equal(scd, np.zeros((64, 64), dtype=np.complex64))

    def test_tone_peak_at_alpha_zero(self):
        """A pure tone has all energy at alpha=0 (the PSD slice)."""
        iq = _make_tone(freq=0.1, n_samples=8192, fs=1.0)
        nfft, n_alpha = 128, 128
        scd = compute_scd_ssca(iq, nfft=nfft, n_alpha=n_alpha, hop=32)
        mag = np.abs(scd)
        # alpha=0 is at the centre column
        alpha_zero_col = n_alpha // 2
        psd_slice = mag[:, alpha_zero_col]
        # The peak energy should be in the alpha=0 column
        assert psd_slice.max() > 0.0
        # Energy in alpha=0 should dominate over other columns
        off_alpha = np.delete(mag, alpha_zero_col, axis=1)
        assert psd_slice.sum() > off_alpha.mean(axis=1).sum()

    def test_configurable_sizes(self):
        iq = _make_noise(4096)
        for nfft, na in [(32, 64), (128, 32), (64, 128)]:
            scd = compute_scd_ssca(iq, nfft=nfft, n_alpha=na, hop=nfft // 4)
            assert scd.shape == (nfft, na)


# ---------------------------------------------------------------------------
# compute_psd_welch
# ---------------------------------------------------------------------------


@pytest.mark.rust
class TestComputePsdWelch:
    def test_output_shape(self):
        iq = _make_noise(4096)
        psd = compute_psd_welch(iq, nfft=256, overlap=128)
        assert psd.shape == (256,)

    def test_output_dtype(self):
        iq = _make_noise(4096)
        psd = compute_psd_welch(iq, nfft=256, overlap=128)
        assert psd.dtype == np.float32

    def test_no_nan_inf(self):
        iq = _make_noise(4096)
        psd = compute_psd_welch(iq, nfft=256, overlap=128)
        assert not np.any(np.isnan(psd))
        assert not np.any(np.isinf(psd))

    def test_nonnegative(self):
        iq = _make_noise(4096)
        psd = compute_psd_welch(iq, nfft=256, overlap=128)
        assert np.all(psd >= 0.0)

    def test_deterministic(self):
        iq = _make_noise(4096, seed=7)
        p1 = compute_psd_welch(iq, nfft=256, overlap=128)
        p2 = compute_psd_welch(iq, nfft=256, overlap=128)
        npt.assert_array_equal(p1, p2)

    def test_white_noise_approximately_flat(self):
        """White noise PSD should be roughly flat."""
        iq = _make_noise(65536, seed=99)
        psd = compute_psd_welch(iq, nfft=256, overlap=128)
        # Coefficient of variation should be small for many averages
        cv = psd.std() / psd.mean()
        assert cv < 0.5, f"PSD not flat enough (CV = {cv:.3f})"

    def test_tone_peak(self):
        """A pure tone should produce a clear PSD peak."""
        nfft = 256
        fs = 1.0
        f0 = 0.2  # normalised frequency
        iq = _make_tone(freq=f0, n_samples=16384, fs=fs)
        psd = compute_psd_welch(iq, nfft=nfft, overlap=nfft // 2)
        peak_bin = np.argmax(psd)
        # DC-centred: bin 0 corresponds to -fs/2, bin nfft//2 to DC
        expected_bin = int(nfft // 2 + f0 * nfft)
        assert abs(peak_bin - expected_bin) <= 2, (
            f"PSD peak at bin {peak_bin}, expected ~{expected_bin}"
        )


# ---------------------------------------------------------------------------
# channelize
# ---------------------------------------------------------------------------


@pytest.mark.rust
class TestChannelize:
    def test_output_shape(self):
        iq = _make_noise(1024)
        frames = channelize(iq, nfft=64, hop=32)
        expected_n_frames = (1024 - 64) // 32 + 1
        assert frames.shape == (expected_n_frames, 64)

    def test_output_dtype(self):
        iq = _make_noise(1024)
        frames = channelize(iq, nfft=64, hop=32)
        assert frames.dtype == np.complex64

    def test_short_signal(self):
        iq = _make_noise(10)
        frames = channelize(iq, nfft=64, hop=32)
        assert frames.shape[0] == 0


# ---------------------------------------------------------------------------
# compute_cumulants
# ---------------------------------------------------------------------------


@pytest.mark.rust
class TestComputeCumulants:
    def test_output_shape_order4(self):
        iq = _make_noise(4096)
        c = compute_cumulants(iq, max_order=4)
        assert c.shape == (5,)

    def test_output_shape_order6(self):
        iq = _make_noise(4096)
        c = compute_cumulants(iq, max_order=6)
        assert c.shape == (9,)

    def test_output_dtype(self):
        iq = _make_noise(4096)
        c = compute_cumulants(iq, max_order=4)
        assert c.dtype == np.complex64

    def test_no_nan_inf(self):
        iq = _make_noise(4096)
        c = compute_cumulants(iq, max_order=6)
        assert not np.any(np.isnan(c))
        assert not np.any(np.isinf(c))

    def test_deterministic(self):
        iq = _make_noise(4096, seed=11)
        c1 = compute_cumulants(iq, max_order=4)
        c2 = compute_cumulants(iq, max_order=4)
        npt.assert_array_equal(c1, c2)

    def test_gaussian_c40_near_zero(self):
        """For circular Gaussian noise, C40 ≈ 0."""
        iq = _make_noise(100_000, seed=0)
        c = compute_cumulants(iq, max_order=4)
        c40_mag = np.abs(c[2])  # index 2 = C40
        assert c40_mag < 0.05, f"|C40| for Gaussian = {c40_mag:.4f}, expected ~0"

    def test_c21_is_signal_power(self):
        """C21 = E[|x|^2] for zero-mean signal (signal power)."""
        iq = _make_noise(50_000, seed=1)
        c = compute_cumulants(iq, max_order=4)
        c21 = c[1]  # index 1 = C21
        # Signal power should be ~1.0 (unit-variance noise)
        assert abs(c21.real - 1.0) < 0.1, f"C21 = {c21}, expected ~1.0"
        assert abs(c21.imag) < 0.05, f"C21 should be real, imag = {c21.imag}"

    def test_empty_signal(self):
        iq = np.array([], dtype=np.complex64)
        c = compute_cumulants(iq, max_order=4)
        assert c.shape == (5,)
        npt.assert_array_equal(c, np.zeros(5, dtype=np.complex64))


# ---------------------------------------------------------------------------
# compute_scd_fam
# ---------------------------------------------------------------------------


@pytest.mark.rust
class TestComputeScdFam:
    def test_output_shape(self):
        iq = _make_noise(4096)
        scd = compute_scd_fam(iq, nfft_chan=64, nfft_fft=64, hop=16)
        assert scd.shape == (64, 64)

    def test_output_dtype(self):
        iq = _make_noise(4096)
        scd = compute_scd_fam(iq, nfft_chan=64, nfft_fft=64, hop=16)
        assert scd.dtype == np.complex64

    def test_no_nan_inf(self):
        iq = _make_noise(4096)
        scd = compute_scd_fam(iq, nfft_chan=64, nfft_fft=64, hop=16)
        assert not np.any(np.isnan(scd))
        assert not np.any(np.isinf(scd))

    def test_deterministic(self):
        iq = _make_noise(4096, seed=42)
        s1 = compute_scd_fam(iq, nfft_chan=64, nfft_fft=64, hop=16)
        s2 = compute_scd_fam(iq, nfft_chan=64, nfft_fft=64, hop=16)
        npt.assert_array_equal(s1, s2)

    def test_short_signal_returns_zeros(self):
        iq = _make_noise(10)
        scd = compute_scd_fam(iq, nfft_chan=64, nfft_fft=64, hop=16)
        assert scd.shape == (64, 64)
        npt.assert_array_equal(scd, np.zeros((64, 64), dtype=np.complex64))

    def test_configurable_sizes(self):
        iq = _make_noise(4096)
        for nc, nf in [(32, 64), (128, 32), (64, 128)]:
            scd = compute_scd_fam(iq, nfft_chan=nc, nfft_fft=nf, hop=nc // 4)
            assert scd.shape == (nc, nf)


# ---------------------------------------------------------------------------
# compute_caf
# ---------------------------------------------------------------------------


@pytest.mark.rust
class TestComputeCaf:
    def test_output_shape(self):
        iq = _make_noise(4096)
        caf = compute_caf(iq, n_alpha=64, max_lag=32)
        assert caf.shape == (64, 32)

    def test_output_dtype(self):
        iq = _make_noise(4096)
        caf = compute_caf(iq, n_alpha=64, max_lag=32)
        assert caf.dtype == np.complex64

    def test_no_nan_inf(self):
        iq = _make_noise(4096)
        caf = compute_caf(iq, n_alpha=64, max_lag=32)
        assert not np.any(np.isnan(caf))
        assert not np.any(np.isinf(caf))

    def test_deterministic(self):
        iq = _make_noise(4096, seed=55)
        c1 = compute_caf(iq, n_alpha=64, max_lag=32)
        c2 = compute_caf(iq, n_alpha=64, max_lag=32)
        npt.assert_array_equal(c1, c2)

    def test_empty_signal(self):
        iq = np.array([], dtype=np.complex64)
        caf = compute_caf(iq, n_alpha=64, max_lag=32)
        assert caf.shape == (64, 32)
        npt.assert_array_equal(caf, np.zeros((64, 32), dtype=np.complex64))

    def test_alpha_zero_is_autocorrelation(self):
        """At alpha=0 (centre row), the CAF should equal the autocorrelation."""
        iq = _make_noise(8192, seed=7)
        n_alpha = 64
        caf = compute_caf(iq, n_alpha=n_alpha, max_lag=16)
        # Alpha = 0 is the centre row
        alpha_zero_row = n_alpha // 2
        r0 = caf[alpha_zero_row, 0]
        # At lag 0, R(0) = E[|x|^2] ≈ 1.0 for unit-variance noise
        assert abs(r0.real - 1.0) < 0.1, f"R(0) = {r0}, expected ~1.0"
