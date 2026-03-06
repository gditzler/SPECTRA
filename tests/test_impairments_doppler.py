import numpy as np
import numpy.testing as npt
import pytest

from spectra.scene.signal_desc import SignalDescription


def _make_desc(f_center=100e6):
    bw = 200e3
    return SignalDescription(0.0, 0.001, f_center - bw / 2, f_center + bw / 2, "QPSK", 20.0)


def test_doppler_exported_from_package():
    from spectra.impairments import DopplerShift
    assert DopplerShift is not None


class TestDopplerShift:
    # --- Construction ---

    def test_requires_at_least_one_param(self):
        from spectra.impairments.doppler import DopplerShift
        with pytest.raises(ValueError, match="fd_hz"):
            DopplerShift()

    def test_construct_with_fd_hz(self):
        from spectra.impairments.doppler import DopplerShift
        d = DopplerShift(fd_hz=1000.0)
        assert d is not None

    def test_construct_with_max_fd_hz(self):
        from spectra.impairments.doppler import DopplerShift
        d = DopplerShift(max_fd_hz=5000.0)
        assert d is not None

    def test_construct_with_physical_params(self):
        from spectra.impairments.doppler import DopplerShift
        # 30 m/s (highway speed), head-on approach, 2.4 GHz carrier
        d = DopplerShift(speed_mps=30.0, carrier_hz=2.4e9, angle_deg=0.0)
        assert d is not None

    def test_physical_params_require_speed_and_carrier(self):
        from spectra.impairments.doppler import DopplerShift
        with pytest.raises(ValueError):
            DopplerShift(speed_mps=30.0)  # missing carrier_hz

    def test_invalid_profile_raises(self):
        from spectra.impairments.doppler import DopplerShift
        with pytest.raises(ValueError, match="profile"):
            DopplerShift(fd_hz=100.0, profile="random_walk")

    # --- Requires sample_rate ---

    def test_requires_sample_rate(self):
        from spectra.impairments.doppler import DopplerShift
        iq = np.ones(512, dtype=np.complex64)
        desc = _make_desc()
        with pytest.raises(ValueError, match="sample_rate"):
            DopplerShift(fd_hz=1000.0)(iq, desc)

    # --- Output shape and dtype ---

    def test_output_shape_and_dtype(self, sample_rate):
        from spectra.impairments.doppler import DopplerShift
        iq = np.ones(1024, dtype=np.complex64)
        desc = _make_desc()
        result, _ = DopplerShift(fd_hz=1000.0)(iq, desc, sample_rate=sample_rate)
        assert result.shape == iq.shape
        assert result.dtype == np.complex64

    # --- No NaNs or Infs ---

    def test_no_nans_or_infs_constant(self, sample_rate):
        from spectra.impairments.doppler import DopplerShift
        iq = np.ones(2048, dtype=np.complex64)
        desc = _make_desc()
        result, _ = DopplerShift(fd_hz=5000.0)(iq, desc, sample_rate=sample_rate)
        assert not np.any(np.isnan(result))
        assert not np.any(np.isinf(result))

    def test_no_nans_or_infs_linear(self, sample_rate):
        from spectra.impairments.doppler import DopplerShift
        iq = np.ones(2048, dtype=np.complex64)
        desc = _make_desc()
        result, _ = DopplerShift(fd_hz=5000.0, profile="linear")(
            iq, desc, sample_rate=sample_rate
        )
        assert not np.any(np.isnan(result))
        assert not np.any(np.isinf(result))

    # --- Constant Doppler: phase ramp correctness ---

    def test_constant_doppler_is_frequency_shift(self, sample_rate):
        """
        Constant Doppler on a pure tone at 0 Hz should produce a tone at fd Hz.
        Verify by measuring the instantaneous frequency.
        """
        from spectra.impairments.doppler import DopplerShift
        n = 4096
        fd = 1000.0
        iq = np.ones(n, dtype=np.complex64)
        desc = _make_desc()
        result, _ = DopplerShift(fd_hz=fd)(iq, desc, sample_rate=sample_rate)
        # Instantaneous phase should increase by 2*pi*fd/fs per sample
        phase = np.unwrap(np.angle(result))
        inst_freq = np.diff(phase) / (2 * np.pi) * sample_rate
        npt.assert_allclose(inst_freq, fd, atol=1.0)

    # --- Linear profile: zero net shift (symmetric flyby) ---

    def test_linear_doppler_zero_net_phase(self, sample_rate):
        """Linear profile: fd goes +fd to -fd, net phase change ~ 0."""
        from spectra.impairments.doppler import DopplerShift
        n = 4096
        fd = 1000.0
        iq = np.ones(n, dtype=np.complex64)
        desc = _make_desc()
        result, _ = DopplerShift(fd_hz=fd, profile="linear")(
            iq, desc, sample_rate=sample_rate
        )
        # Total phase accumulated = integral of fd(t) from 0 to T
        # fd(t) = fd*(1 - 2t/T), integral = fd*T*(1 - 1) = 0
        phase = np.unwrap(np.angle(result))
        total_phase = phase[-1] - phase[0]
        npt.assert_allclose(total_phase, 0.0, atol=0.1)

    # --- SignalDescription update ---

    def test_constant_desc_f_center_updated(self, sample_rate):
        """Constant Doppler: f_low and f_high shift by fd."""
        from spectra.impairments.doppler import DopplerShift
        iq = np.ones(1024, dtype=np.complex64)
        f_center = 100e6
        desc = _make_desc(f_center=f_center)
        fd = 2000.0
        _, new_desc = DopplerShift(fd_hz=fd)(iq, desc, sample_rate=sample_rate)
        npt.assert_allclose(new_desc.f_low, desc.f_low + fd, atol=1.0)
        npt.assert_allclose(new_desc.f_high, desc.f_high + fd, atol=1.0)

    def test_linear_desc_unchanged(self, sample_rate):
        """Linear flyby: f_center net shift is zero, desc unchanged."""
        from spectra.impairments.doppler import DopplerShift
        iq = np.ones(1024, dtype=np.complex64)
        desc = _make_desc()
        _, new_desc = DopplerShift(fd_hz=2000.0, profile="linear")(
            iq, desc, sample_rate=sample_rate
        )
        assert new_desc.f_low == desc.f_low
        assert new_desc.f_high == desc.f_high

    # --- Physical parameter construction computes correct fd ---

    def test_physical_params_head_on(self, sample_rate):
        """Head-on approach at 30 m/s at 1 GHz ~ 100 Hz Doppler."""
        from spectra.impairments.doppler import DopplerShift
        # f_d = v/c * f_c = 30/3e8 * 1e9 = 100 Hz
        d = DopplerShift(speed_mps=30.0, carrier_hz=1e9, angle_deg=0.0)
        iq = np.ones(4096, dtype=np.complex64)
        desc = _make_desc()
        result, new_desc = d(iq, desc, sample_rate=sample_rate)
        npt.assert_allclose(new_desc.f_low, desc.f_low + 100.0, atol=1.0)

    def test_physical_params_perpendicular_no_shift(self, sample_rate):
        """90-degree angle: radial velocity = 0, no Doppler shift."""
        from spectra.impairments.doppler import DopplerShift
        d = DopplerShift(speed_mps=100.0, carrier_hz=2.4e9, angle_deg=90.0)
        iq = np.ones(1024, dtype=np.complex64)
        desc = _make_desc()
        result, new_desc = d(iq, desc, sample_rate=sample_rate)
        npt.assert_allclose(result, iq, atol=1e-4)
        assert new_desc.f_low == desc.f_low

    # --- Randomized max_fd_hz produces variation ---

    def test_max_fd_hz_randomizes(self, sample_rate):
        from spectra.impairments.doppler import DopplerShift
        d = DopplerShift(max_fd_hz=5000.0)
        iq = np.ones(1024, dtype=np.complex64)
        desc = _make_desc()
        results = [d(iq.copy(), desc, sample_rate=sample_rate)[0] for _ in range(10)]
        diffs = [np.max(np.abs(results[i] - results[i + 1])) for i in range(9)]
        assert not all(d < 1e-6 for d in diffs)

    # --- Power preservation ---

    def test_power_preserved(self, sample_rate):
        """Doppler shift is a phase rotation: power must be preserved."""
        from spectra.impairments.doppler import DopplerShift
        iq = (np.random.randn(2048) + 1j * np.random.randn(2048)).astype(np.complex64)
        desc = _make_desc()
        for profile in ("constant", "linear"):
            result, _ = DopplerShift(fd_hz=3000.0, profile=profile)(
                iq, desc, sample_rate=sample_rate
            )
            npt.assert_allclose(
                np.mean(np.abs(result) ** 2),
                np.mean(np.abs(iq) ** 2),
                rtol=1e-4,
            )
