import numpy as np
import pytest
from spectra.scene.signal_desc import SignalDescription


@pytest.fixture
def sample_iq():
    rng = np.random.default_rng(42)
    return (rng.standard_normal(1024) + 1j * rng.standard_normal(1024)).astype(np.complex64)


@pytest.fixture
def sample_desc():
    return SignalDescription(
        t_start=0.0, t_stop=1.0, f_low=-1000.0, f_high=1000.0, label="test", snr=10.0
    )


class TestIQImbalance:
    def test_shape_preserved(self, sample_iq, sample_desc):
        from spectra.impairments import IQImbalance

        result, _ = IQImbalance()(sample_iq, sample_desc)
        assert len(result) == len(sample_iq)
        assert result.dtype == np.complex64


class TestRayleighFading:
    def test_shape_preserved(self, sample_iq, sample_desc):
        from spectra.impairments import RayleighFading

        result, _ = RayleighFading()(sample_iq, sample_desc)
        assert len(result) == len(sample_iq)
        assert result.dtype == np.complex64


class TestRicianFading:
    def test_shape_preserved(self, sample_iq, sample_desc):
        from spectra.impairments import RicianFading

        result, _ = RicianFading()(sample_iq, sample_desc)
        assert len(result) == len(sample_iq)
        assert result.dtype == np.complex64


class TestPhaseNoise:
    def test_shape_preserved(self, sample_iq, sample_desc):
        from spectra.impairments import PhaseNoise

        result, _ = PhaseNoise()(sample_iq, sample_desc)
        assert len(result) == len(sample_iq)
        assert result.dtype == np.complex64


class TestFrequencyDrift:
    def test_shape_preserved(self, sample_iq, sample_desc):
        from spectra.impairments import FrequencyDrift

        result, _ = FrequencyDrift()(sample_iq, sample_desc, sample_rate=1e6)
        assert len(result) == len(sample_iq)
        assert result.dtype == np.complex64


class TestQuantization:
    def test_shape_preserved(self, sample_iq, sample_desc):
        from spectra.impairments import Quantization

        result, _ = Quantization(num_bits=8)(sample_iq, sample_desc)
        assert len(result) == len(sample_iq)
        assert result.dtype == np.complex64

    def test_reduced_levels(self, sample_iq, sample_desc):
        from spectra.impairments import Quantization

        result, _ = Quantization(num_bits=2)(sample_iq, sample_desc)
        # Very few bits should create many duplicate values
        unique_real = len(np.unique(np.round(result.real, 6)))
        assert unique_real <= 2**2 + 1


class TestSpectralInversion:
    def test_conjugate(self, sample_iq, sample_desc):
        from spectra.impairments import SpectralInversion

        result, new_desc = SpectralInversion()(sample_iq, sample_desc)
        np.testing.assert_array_equal(result, sample_iq.conj().astype(np.complex64))
        assert new_desc.f_low == -sample_desc.f_high
        assert new_desc.f_high == -sample_desc.f_low


class TestColoredNoise:
    def test_shape_preserved(self, sample_iq, sample_desc):
        from spectra.impairments import ColoredNoise

        result, _ = ColoredNoise(snr=10.0, color="pink")(sample_iq, sample_desc)
        assert len(result) == len(sample_iq)


class TestPassbandRipple:
    def test_shape_preserved(self, sample_iq, sample_desc):
        from spectra.impairments import PassbandRipple

        result, _ = PassbandRipple()(sample_iq, sample_desc)
        assert len(result) == len(sample_iq)


class TestAdjacentChannelInterference:
    def test_shape_preserved(self, sample_iq, sample_desc):
        from spectra.impairments import AdjacentChannelInterference

        result, _ = AdjacentChannelInterference()(sample_iq, sample_desc, sample_rate=1e6)
        assert len(result) == len(sample_iq)


class TestIntermodulationProducts:
    def test_shape_preserved(self, sample_iq, sample_desc):
        from spectra.impairments import IntermodulationProducts

        result, _ = IntermodulationProducts()(sample_iq, sample_desc)
        assert len(result) == len(sample_iq)


class TestComposeWithNewImpairments:
    def test_compose_chain(self, sample_iq, sample_desc):
        from spectra.impairments import Compose, IQImbalance, PhaseNoise

        chain = Compose([IQImbalance(), PhaseNoise()])
        result, _ = chain(sample_iq, sample_desc)
        assert len(result) == len(sample_iq)
        assert result.dtype == np.complex64
