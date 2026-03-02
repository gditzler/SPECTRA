import numpy as np
import numpy.testing as npt
import pytest


class TestCostasCodeWaveform:
    def test_generate_returns_valid_iq(self, assert_valid_iq, sample_rate):
        from spectra.waveforms.costas import CostasCode
        waveform = CostasCode()
        iq = waveform.generate(num_symbols=2, sample_rate=sample_rate)
        assert_valid_iq(iq)

    def test_output_length(self, sample_rate):
        from spectra.waveforms.costas import CostasCode
        prime, sph = 7, 64
        waveform = CostasCode(prime=prime, samples_per_hop=sph)
        n_hops = prime - 1  # Welch order
        iq = waveform.generate(num_symbols=3, sample_rate=sample_rate)
        assert len(iq) == 3 * n_hops * sph

    def test_label(self):
        from spectra.waveforms.costas import CostasCode
        assert CostasCode().label == "Costas"

    def test_unit_magnitude(self, sample_rate):
        from spectra.waveforms.costas import CostasCode
        waveform = CostasCode()
        iq = waveform.generate(num_symbols=1, sample_rate=sample_rate)
        npt.assert_allclose(np.abs(iq), 1.0, atol=1e-5)

    def test_bandwidth(self, sample_rate):
        from spectra.waveforms.costas import CostasCode
        prime, sph = 7, 64
        waveform = CostasCode(prime=prime, samples_per_hop=sph)
        n_hops = prime - 1
        delta_f = sample_rate / sph
        expected_bw = n_hops * delta_f
        assert waveform.bandwidth(sample_rate) == pytest.approx(expected_bw)

    def test_samples_per_symbol_attribute(self):
        from spectra.waveforms.costas import CostasCode
        waveform = CostasCode(prime=7, samples_per_hop=64)
        assert waveform.samples_per_symbol == 6 * 64

    def test_different_primes(self, assert_valid_iq, sample_rate):
        from spectra.waveforms.costas import CostasCode
        for p in [5, 7, 11, 13]:
            waveform = CostasCode(prime=p)
            iq = waveform.generate(num_symbols=1, sample_rate=sample_rate)
            assert_valid_iq(iq)
