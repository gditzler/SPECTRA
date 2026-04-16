import numpy as np
import numpy.testing as npt

from conftest import make_signal_description


class TestRayleighFading:
    def test_modifies_signal(self, sample_rate):
        from spectra.impairments.fading import RayleighFading

        iq = np.ones(2048, dtype=np.complex64)
        desc = make_signal_description()
        result, _ = RayleighFading(num_taps=8, doppler_spread=0.01)(
            iq, desc, sample_rate=sample_rate
        )
        assert not np.allclose(result, iq, atol=0.1)

    def test_output_shape_and_dtype(self, sample_rate):
        from spectra.impairments.fading import RayleighFading

        iq = np.ones(1024, dtype=np.complex64)
        desc = make_signal_description()
        result, _ = RayleighFading(num_taps=8, doppler_spread=0.01)(
            iq, desc, sample_rate=sample_rate
        )
        assert result.shape == iq.shape
        assert result.dtype == np.complex64

    def test_no_nans_or_infs(self, sample_rate):
        from spectra.impairments.fading import RayleighFading

        iq = np.ones(2048, dtype=np.complex64)
        desc = make_signal_description()
        result, _ = RayleighFading(num_taps=8, doppler_spread=0.05)(
            iq, desc, sample_rate=sample_rate
        )
        assert not np.any(np.isnan(result))
        assert not np.any(np.isinf(result))

    def test_desc_unchanged(self, sample_rate):
        from spectra.impairments.fading import RayleighFading

        iq = np.ones(512, dtype=np.complex64)
        desc = make_signal_description()
        _, new_desc = RayleighFading(num_taps=8, doppler_spread=0.01)(
            iq, desc, sample_rate=sample_rate
        )
        assert new_desc.f_low == desc.f_low

    def test_different_taps_produce_different_results(self, sample_rate):
        from spectra.impairments.fading import RayleighFading

        iq = np.ones(1024, dtype=np.complex64)
        desc = make_signal_description()
        results = [
            RayleighFading(num_taps=8, doppler_spread=0.01)(
                iq.copy(), desc, sample_rate=sample_rate
            )[0]
            for _ in range(10)
        ]
        # Random taps should produce different fading patterns
        diffs = [np.max(np.abs(results[i] - results[i + 1])) for i in range(9)]
        assert not all(d < 1e-6 for d in diffs)


class TestRicianFading:
    def test_high_k_preserves_signal(self, sample_rate):
        from spectra.impairments.fading import RicianFading

        iq = np.ones(2048, dtype=np.complex64)
        desc = make_signal_description()
        result, _ = RicianFading(k_factor=40.0, num_taps=8)(iq, desc, sample_rate=sample_rate)
        # Very high K = mostly LOS, signal should be close to original
        # Exclude edges where convolution artifacts can appear
        npt.assert_allclose(np.abs(result[8:-8]), 1.0, atol=0.3)

    def test_output_shape_and_dtype(self, sample_rate):
        from spectra.impairments.fading import RicianFading

        iq = np.ones(1024, dtype=np.complex64)
        desc = make_signal_description()
        result, _ = RicianFading(k_factor=10.0, num_taps=8)(iq, desc, sample_rate=sample_rate)
        assert result.shape == iq.shape
        assert result.dtype == np.complex64

    def test_no_nans_or_infs(self, sample_rate):
        from spectra.impairments.fading import RicianFading

        iq = np.ones(2048, dtype=np.complex64)
        desc = make_signal_description()
        result, _ = RicianFading(k_factor=3.0, num_taps=8)(iq, desc, sample_rate=sample_rate)
        assert not np.any(np.isnan(result))
        assert not np.any(np.isinf(result))

    def test_low_k_approaches_rayleigh(self, sample_rate):
        from spectra.impairments.fading import RicianFading

        iq = np.ones(2048, dtype=np.complex64)
        desc = make_signal_description()
        result, _ = RicianFading(k_factor=0.01, num_taps=8)(iq, desc, sample_rate=sample_rate)
        # Low K should cause significant fading
        assert not np.allclose(np.abs(result), 1.0, atol=0.1)

    def test_default_params(self):
        from spectra.impairments.fading import RicianFading

        rf = RicianFading()
        assert rf._k == 4.0
        assert rf._num_taps == 8
