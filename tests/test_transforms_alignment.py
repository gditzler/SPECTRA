import numpy as np
import numpy.testing as npt
import pytest
from spectra.scene.signal_desc import SignalDescription
from spectra.transforms.alignment import (
    AGCNormalize,
    BandpassAlign,
    ClipNormalize,
    DCRemove,
    NoiseFloorMatch,
    NoiseProfileTransfer,
    PowerNormalize,
    ReceiverEQ,
    Resample,
    SpectralWhitening,
)

pytest.importorskip("scipy")


@pytest.fixture
def sample_iq():
    """Complex64 IQ signal with known properties."""
    rng = np.random.default_rng(42)
    n = 4096
    t = np.arange(n) / 1e6
    signal = np.exp(1j * 2 * np.pi * 50_000 * t).astype(np.complex64)
    noise = 0.1 * (rng.standard_normal(n) + 1j * rng.standard_normal(n)).astype(np.complex64)
    return signal + noise


@pytest.fixture
def sample_desc():
    """Minimal SignalDescription for testing."""
    return SignalDescription(
        t_start=0.0,
        t_stop=0.004096,
        f_low=-50_000.0,
        f_high=50_000.0,
        label="test",
        snr=20.0,
    )


@pytest.fixture
def sample_rate():
    return 1e6


# --- DCRemove ---


class TestDCRemove:
    def test_output_mean_near_zero(self, sample_iq, sample_desc):
        t = DCRemove()
        iq_out, desc_out = t(sample_iq, sample_desc)
        assert abs(np.mean(iq_out)) < 1e-6

    def test_preserves_dtype(self, sample_iq, sample_desc):
        t = DCRemove()
        iq_out, _ = t(sample_iq, sample_desc)
        assert iq_out.dtype == np.complex64

    def test_preserves_length(self, sample_iq, sample_desc):
        t = DCRemove()
        iq_out, _ = t(sample_iq, sample_desc)
        assert len(iq_out) == len(sample_iq)

    def test_desc_unchanged(self, sample_iq, sample_desc):
        t = DCRemove()
        _, desc_out = t(sample_iq, sample_desc)
        assert desc_out.f_low == sample_desc.f_low
        assert desc_out.f_high == sample_desc.f_high

    def test_with_dc_offset(self, sample_desc):
        iq = np.ones(1024, dtype=np.complex64) * (5.0 + 3.0j)
        t = DCRemove()
        iq_out, _ = t(iq, sample_desc)
        assert abs(np.mean(iq_out)) < 1e-6


# --- ClipNormalize ---


class TestClipNormalize:
    def test_output_bounded(self, sample_iq, sample_desc):
        t = ClipNormalize(clip_sigma=3.0)
        iq_out, _ = t(sample_iq, sample_desc)
        assert np.max(np.abs(iq_out.real)) <= 1.0 + 1e-6
        assert np.max(np.abs(iq_out.imag)) <= 1.0 + 1e-6

    def test_preserves_dtype(self, sample_iq, sample_desc):
        t = ClipNormalize()
        iq_out, _ = t(sample_iq, sample_desc)
        assert iq_out.dtype == np.complex64

    def test_preserves_length(self, sample_iq, sample_desc):
        t = ClipNormalize()
        iq_out, _ = t(sample_iq, sample_desc)
        assert len(iq_out) == len(sample_iq)

    def test_outliers_clipped(self, sample_desc):
        iq = np.zeros(1000, dtype=np.complex64)
        iq[500] = 100.0 + 100.0j  # extreme outlier
        t = ClipNormalize(clip_sigma=2.0)
        iq_out, _ = t(iq, sample_desc)
        assert np.abs(iq_out[500]) < np.abs(iq[500])

    def test_custom_clip_sigma(self, sample_iq, sample_desc):
        t1 = ClipNormalize(clip_sigma=1.0)
        t2 = ClipNormalize(clip_sigma=5.0)
        iq1, _ = t1(sample_iq, sample_desc)
        iq2, _ = t2(sample_iq, sample_desc)
        assert not np.array_equal(iq1, iq2)


# --- PowerNormalize ---


class TestPowerNormalize:
    def test_output_rms_matches_target(self, sample_iq, sample_desc):
        target = -20.0
        t = PowerNormalize(target_power_dbfs=target)
        iq_out, _ = t(sample_iq, sample_desc)
        rms = np.sqrt(np.mean(np.abs(iq_out) ** 2))
        rms_db = 20.0 * np.log10(rms)
        assert abs(rms_db - target) < 0.1

    def test_preserves_dtype(self, sample_iq, sample_desc):
        t = PowerNormalize()
        iq_out, _ = t(sample_iq, sample_desc)
        assert iq_out.dtype == np.complex64

    def test_preserves_length(self, sample_iq, sample_desc):
        t = PowerNormalize()
        iq_out, _ = t(sample_iq, sample_desc)
        assert len(iq_out) == len(sample_iq)

    def test_zero_power_unchanged(self, sample_desc):
        iq = np.zeros(1024, dtype=np.complex64)
        t = PowerNormalize(target_power_dbfs=-20.0)
        iq_out, _ = t(iq, sample_desc)
        npt.assert_array_equal(iq_out, iq)

    def test_different_targets_differ(self, sample_iq, sample_desc):
        t1 = PowerNormalize(target_power_dbfs=-10.0)
        t2 = PowerNormalize(target_power_dbfs=-30.0)
        iq1, _ = t1(sample_iq, sample_desc)
        iq2, _ = t2(sample_iq, sample_desc)
        rms1 = np.sqrt(np.mean(np.abs(iq1) ** 2))
        rms2 = np.sqrt(np.mean(np.abs(iq2) ** 2))
        assert rms1 > rms2


# --- AGCNormalize ---


class TestAGCNormalize:
    def test_rms_mode_unit_power(self, sample_iq, sample_desc):
        t = AGCNormalize(method="rms", target_level=1.0)
        iq_out, _ = t(sample_iq, sample_desc)
        rms = np.sqrt(np.mean(np.abs(iq_out) ** 2))
        assert abs(rms - 1.0) < 1e-5

    def test_peak_mode_bounded(self, sample_iq, sample_desc):
        t = AGCNormalize(method="peak", target_level=1.0)
        iq_out, _ = t(sample_iq, sample_desc)
        assert np.max(np.abs(iq_out)) <= 1.0 + 1e-6

    def test_preserves_dtype(self, sample_iq, sample_desc):
        t = AGCNormalize()
        iq_out, _ = t(sample_iq, sample_desc)
        assert iq_out.dtype == np.complex64

    def test_preserves_length(self, sample_iq, sample_desc):
        t = AGCNormalize()
        iq_out, _ = t(sample_iq, sample_desc)
        assert len(iq_out) == len(sample_iq)

    def test_zero_amplitude_safe(self, sample_desc):
        iq = np.zeros(1024, dtype=np.complex64)
        t = AGCNormalize(method="rms")
        iq_out, _ = t(iq, sample_desc)
        npt.assert_array_equal(iq_out, iq)

    def test_invalid_method_raises(self, sample_iq, sample_desc):
        with pytest.raises(ValueError):
            AGCNormalize(method="invalid")

    def test_custom_target_level(self, sample_iq, sample_desc):
        t = AGCNormalize(method="rms", target_level=0.5)
        iq_out, _ = t(sample_iq, sample_desc)
        rms = np.sqrt(np.mean(np.abs(iq_out) ** 2))
        assert abs(rms - 0.5) < 1e-5


# --- Resample ---


class TestResample:
    def test_upsample_length(self, sample_iq, sample_desc):
        t = Resample(target_sample_rate=2e6)
        iq_out, _ = t(sample_iq, sample_desc, sample_rate=1e6)
        expected_len = len(sample_iq) * 2
        assert abs(len(iq_out) - expected_len) <= 1

    def test_downsample_length(self, sample_iq, sample_desc):
        t = Resample(target_sample_rate=500_000)
        iq_out, _ = t(sample_iq, sample_desc, sample_rate=1e6)
        expected_len = len(sample_iq) // 2
        assert abs(len(iq_out) - expected_len) <= 1

    def test_preserves_dtype(self, sample_iq, sample_desc):
        t = Resample(target_sample_rate=2e6)
        iq_out, _ = t(sample_iq, sample_desc, sample_rate=1e6)
        assert iq_out.dtype == np.complex64

    def test_same_rate_unchanged(self, sample_iq, sample_desc):
        t = Resample(target_sample_rate=1e6)
        iq_out, _ = t(sample_iq, sample_desc, sample_rate=1e6)
        npt.assert_array_equal(iq_out, sample_iq)

    def test_round_trip(self, sample_desc):
        rng = np.random.default_rng(99)
        iq = (rng.standard_normal(1024) + 1j * rng.standard_normal(1024)).astype(np.complex64)
        up = Resample(target_sample_rate=2e6)
        down = Resample(target_sample_rate=1e6)
        iq_up, _ = up(iq, sample_desc, sample_rate=1e6)
        iq_back, _ = down(iq_up, sample_desc, sample_rate=2e6)
        min_len = min(len(iq), len(iq_back))
        corr = np.abs(np.corrcoef(np.abs(iq[:min_len]), np.abs(iq_back[:min_len]))[0, 1])
        assert corr > 0.9

    def test_missing_sample_rate_raises(self, sample_iq, sample_desc):
        t = Resample(target_sample_rate=2e6)
        with pytest.raises(ValueError, match="sample_rate"):
            t(sample_iq, sample_desc)


# --- SpectralWhitening ---


class TestSpectralWhitening:
    def test_psd_flatter_after(self, sample_iq, sample_desc):
        t = SpectralWhitening(smoothing_window=64)
        iq_out, _ = t(sample_iq, sample_desc)
        psd_before = np.abs(np.fft.fft(sample_iq)) ** 2
        psd_after = np.abs(np.fft.fft(iq_out)) ** 2
        assert np.var(psd_after) < np.var(psd_before)

    def test_preserves_dtype(self, sample_iq, sample_desc):
        t = SpectralWhitening()
        iq_out, _ = t(sample_iq, sample_desc)
        assert iq_out.dtype == np.complex64

    def test_preserves_length(self, sample_iq, sample_desc):
        t = SpectralWhitening()
        iq_out, _ = t(sample_iq, sample_desc)
        assert len(iq_out) == len(sample_iq)

    def test_energy_preserved(self, sample_iq, sample_desc):
        t = SpectralWhitening(smoothing_window=32)
        iq_out, _ = t(sample_iq, sample_desc)
        power_before = np.mean(np.abs(sample_iq) ** 2)
        power_after = np.mean(np.abs(iq_out) ** 2)
        ratio = power_after / power_before
        assert 0.1 < ratio < 10.0


# --- NoiseFloorMatch ---


class TestNoiseFloorMatch:
    def test_preserves_dtype(self, sample_iq, sample_desc):
        t = NoiseFloorMatch(target_noise_floor_db=-40.0)
        iq_out, _ = t(sample_iq, sample_desc)
        assert iq_out.dtype == np.complex64

    def test_preserves_length(self, sample_iq, sample_desc):
        t = NoiseFloorMatch()
        iq_out, _ = t(sample_iq, sample_desc)
        assert len(iq_out) == len(sample_iq)

    def test_different_targets_scale_differently(self, sample_iq, sample_desc):
        t1 = NoiseFloorMatch(target_noise_floor_db=-30.0)
        t2 = NoiseFloorMatch(target_noise_floor_db=-50.0)
        iq1, _ = t1(sample_iq, sample_desc)
        iq2, _ = t2(sample_iq, sample_desc)
        power1 = np.mean(np.abs(iq1) ** 2)
        power2 = np.mean(np.abs(iq2) ** 2)
        assert power1 > power2

    def test_estimation_methods(self, sample_iq, sample_desc):
        t1 = NoiseFloorMatch(target_noise_floor_db=-40.0, estimation_method="median")
        t2 = NoiseFloorMatch(target_noise_floor_db=-40.0, estimation_method="minimum")
        iq1, _ = t1(sample_iq, sample_desc)
        iq2, _ = t2(sample_iq, sample_desc)
        assert not np.array_equal(iq1, iq2)


# --- BandpassAlign ---


class TestBandpassAlign:
    def test_preserves_dtype(self, sample_iq, sample_desc):
        t = BandpassAlign(center_freq=0.0, bandwidth=0.5)
        iq_out, _ = t(sample_iq, sample_desc, sample_rate=1e6)
        assert iq_out.dtype == np.complex64

    def test_preserves_length(self, sample_iq, sample_desc):
        t = BandpassAlign(center_freq=0.0, bandwidth=0.5)
        iq_out, _ = t(sample_iq, sample_desc, sample_rate=1e6)
        assert len(iq_out) == len(sample_iq)

    def test_out_of_band_suppressed(self, sample_desc):
        n = 4096
        t_arr = np.arange(n) / 1e6
        iq = np.exp(1j * 2 * np.pi * 250_000 * t_arr).astype(np.complex64)
        t = BandpassAlign(center_freq=0.0, bandwidth=0.1)
        iq_out, _ = t(iq, sample_desc, sample_rate=1e6)
        power_before = np.mean(np.abs(iq) ** 2)
        power_after = np.mean(np.abs(iq_out) ** 2)
        assert power_after < power_before * 0.5

    def test_updates_desc_freq_bounds(self, sample_iq, sample_desc):
        t = BandpassAlign(center_freq=0.0, bandwidth=0.5)
        _, desc_out = t(sample_iq, sample_desc, sample_rate=1e6)
        assert desc_out.f_low == -250_000.0
        assert desc_out.f_high == 250_000.0


# --- Stubs ---


class TestNoiseProfileTransfer:
    def test_raises_not_implemented(self, sample_iq, sample_desc):
        t = NoiseProfileTransfer(noise_source=np.zeros(100, dtype=np.complex64))
        with pytest.raises(NotImplementedError):
            t(sample_iq, sample_desc)


class TestReceiverEQ:
    def test_raises_not_implemented(self, sample_iq, sample_desc):
        t = ReceiverEQ(reference_psd=np.ones(256))
        with pytest.raises(NotImplementedError):
            t(sample_iq, sample_desc)


# --- Compose Integration ---


class TestComposeIntegration:
    def test_chain_produces_valid_iq(self, sample_iq, sample_desc):
        from spectra.impairments import Compose

        chain = Compose(
            [
                DCRemove(),
                AGCNormalize(method="rms"),
                SpectralWhitening(smoothing_window=32),
            ]
        )
        iq_out, desc_out = chain(sample_iq, sample_desc, sample_rate=1e6)
        assert isinstance(iq_out, np.ndarray)
        assert iq_out.dtype == np.complex64
        assert len(iq_out) == len(sample_iq)
        assert not np.any(np.isnan(iq_out))
        assert not np.any(np.isinf(iq_out))

    def test_chain_all_statistical(self, sample_iq, sample_desc):
        from spectra.impairments import Compose

        chain = Compose(
            [
                DCRemove(),
                ClipNormalize(clip_sigma=3.0),
                PowerNormalize(target_power_dbfs=-20.0),
            ]
        )
        iq_out, _ = chain(sample_iq, sample_desc, sample_rate=1e6)
        rms = np.sqrt(np.mean(np.abs(iq_out) ** 2))
        rms_db = 20.0 * np.log10(rms)
        assert abs(rms_db - (-20.0)) < 0.1

    def test_all_deterministic(self, sample_iq, sample_desc):
        from spectra.impairments import Compose

        chain = Compose(
            [
                DCRemove(),
                AGCNormalize(),
                SpectralWhitening(),
            ]
        )
        iq1, _ = chain(sample_iq.copy(), sample_desc, sample_rate=1e6)
        iq2, _ = chain(sample_iq.copy(), sample_desc, sample_rate=1e6)
        npt.assert_array_equal(iq1, iq2)
