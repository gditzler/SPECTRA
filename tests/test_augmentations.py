import numpy as np
import numpy.testing as npt
import pytest


@pytest.fixture
def sample_iq():
    rng = np.random.default_rng(42)
    return (rng.standard_normal(1024) + 1j * rng.standard_normal(1024)).astype(np.complex64)


class TestCutOut:
    def test_shape_preserved(self, sample_iq):
        from spectra.transforms import CutOut

        result = CutOut()(sample_iq, rng=np.random.default_rng(0))
        assert len(result) == len(sample_iq)

    def test_has_zeros(self, sample_iq):
        from spectra.transforms import CutOut

        result = CutOut(max_length_fraction=0.5)(sample_iq, rng=np.random.default_rng(0))
        assert np.any(result == 0)


class TestTimeReversal:
    def test_reversed(self, sample_iq):
        from spectra.transforms import TimeReversal

        result = TimeReversal()(sample_iq)
        npt.assert_array_equal(result, sample_iq[::-1])


class TestChannelSwap:
    def test_conjugate(self, sample_iq):
        from spectra.transforms import ChannelSwap

        result = ChannelSwap()(sample_iq)
        npt.assert_array_equal(result, sample_iq.conj())


class TestPatchShuffle:
    def test_shape_preserved(self, sample_iq):
        from spectra.transforms import PatchShuffle

        result = PatchShuffle()(sample_iq, rng=np.random.default_rng(0))
        assert len(result) == len(sample_iq)


class TestRandomDropSamples:
    def test_shape_preserved(self, sample_iq):
        from spectra.transforms import RandomDropSamples

        result = RandomDropSamples(drop_rate=0.1)(sample_iq, rng=np.random.default_rng(0))
        assert len(result) == len(sample_iq)


class TestAddSlope:
    def test_shape_preserved(self, sample_iq):
        from spectra.transforms import AddSlope

        result = AddSlope()(sample_iq, rng=np.random.default_rng(0))
        assert len(result) == len(sample_iq)


class TestRandomMagRescale:
    def test_shape_preserved(self, sample_iq):
        from spectra.transforms import RandomMagRescale

        result = RandomMagRescale()(sample_iq, rng=np.random.default_rng(0))
        assert len(result) == len(sample_iq)


class TestAGC:
    def test_target_power(self, sample_iq):
        from spectra.transforms import AGC

        result = AGC(target_power=1.0)(sample_iq)
        power = np.mean(np.abs(result) ** 2)
        assert power == pytest.approx(1.0, abs=0.01)
