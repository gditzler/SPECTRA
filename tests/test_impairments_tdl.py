import numpy as np
import pytest
from spectra.impairments.tdl_channel import TDLChannel
from spectra.scene.signal_desc import SignalDescription


@pytest.fixture
def desc():
    return SignalDescription(
        t_start=0.0,
        t_stop=1.0,
        f_low=-500e3,
        f_high=500e3,
        label="test",
        snr=20.0,
    )


@pytest.fixture
def tone_iq():
    t = np.linspace(0, 1, 1024, endpoint=False)
    return np.exp(2j * np.pi * 100 * t).astype(np.complex64)


class TestTDLChannel:
    def test_output_shape(self, tone_iq, desc):
        ch = TDLChannel(profile="TDL-A")
        out, _ = ch(tone_iq, desc, sample_rate=1e6)
        assert out.shape == tone_iq.shape
        assert out.dtype == np.complex64

    def test_different_profiles(self, tone_iq, desc):
        results = []
        for profile in ["TDL-A", "TDL-B", "TDL-C"]:
            np.random.seed(42)
            ch = TDLChannel(profile=profile)
            out, _ = ch(tone_iq, desc, sample_rate=1e6)
            results.append(out)
        # Different profiles should produce different outputs
        assert not np.array_equal(results[0], results[1])

    def test_los_profiles(self, tone_iq, desc):
        for profile in ["TDL-D", "TDL-E"]:
            ch = TDLChannel(profile=profile)
            out, _ = ch(tone_iq, desc, sample_rate=1e6)
            assert out.shape == tone_iq.shape

    def test_itu_profiles(self, tone_iq, desc):
        for profile in ["PedestrianA", "PedestrianB", "VehicularA", "VehicularB"]:
            ch = TDLChannel(profile=profile)
            out, _ = ch(tone_iq, desc, sample_rate=1e6)
            assert out.shape == tone_iq.shape

    def test_custom_profile(self, tone_iq, desc):
        ch = TDLChannel.custom(
            delays_ns=[0, 50, 100],
            powers_db=[0, -3, -6],
            doppler_hz=10.0,
        )
        out, _ = ch(tone_iq, desc, sample_rate=1e6)
        assert out.shape == tone_iq.shape

    def test_custom_los_profile(self, tone_iq, desc):
        ch = TDLChannel.custom(
            delays_ns=[0, 100],
            powers_db=[0, -10],
            doppler_hz=5.0,
            k_factor_db=10.0,
        )
        out, _ = ch(tone_iq, desc, sample_rate=1e6)
        assert out.shape == tone_iq.shape

    def test_invalid_profile_raises(self):
        with pytest.raises(ValueError, match="Unknown profile"):
            TDLChannel(profile="INVALID")

    def test_power_approximately_preserved(self, tone_iq, desc):
        ch = TDLChannel(profile="TDL-A")
        out, _ = ch(tone_iq, desc, sample_rate=1e6)
        in_power = np.mean(np.abs(tone_iq) ** 2)
        out_power = np.mean(np.abs(out) ** 2)
        # Within 10 dB (fading can cause significant power variation)
        assert out_power > in_power * 0.01
        assert out_power < in_power * 100
