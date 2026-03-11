import numpy as np
import numpy.testing as npt


class TestApplyRrcFilter:
    def test_returns_complex64_ndarray(self):
        from spectra._rust import apply_rrc_filter, generate_qpsk_symbols

        symbols = generate_qpsk_symbols(64, seed=0)
        filtered = apply_rrc_filter(symbols, rolloff=0.35, span=10, sps=8)
        assert isinstance(filtered, np.ndarray)
        assert filtered.dtype == np.complex64

    def test_output_is_upsampled(self):
        from spectra._rust import apply_rrc_filter, generate_qpsk_symbols

        symbols = generate_qpsk_symbols(64, seed=0)
        sps = 8
        span = 10
        filtered = apply_rrc_filter(symbols, rolloff=0.35, span=span, sps=sps)
        expected_len = len(symbols) * sps + span * sps
        assert len(filtered) == expected_len

    def test_no_nan_or_inf(self):
        from spectra._rust import apply_rrc_filter, generate_qpsk_symbols

        symbols = generate_qpsk_symbols(128, seed=0)
        for rolloff in [0.0, 0.25, 0.35, 0.5, 1.0]:
            filtered = apply_rrc_filter(symbols, rolloff=rolloff, span=6, sps=4)
            assert not np.any(np.isnan(filtered)), f"NaN with rolloff={rolloff}"
            assert not np.any(np.isinf(filtered)), f"Inf with rolloff={rolloff}"

    def test_energy_preservation(self):
        from spectra._rust import apply_rrc_filter, generate_qpsk_symbols

        symbols = generate_qpsk_symbols(256, seed=0)
        sps = 4
        filtered = apply_rrc_filter(symbols, rolloff=0.35, span=10, sps=sps)
        input_energy = np.sum(np.abs(symbols) ** 2)
        output_energy = np.sum(np.abs(filtered) ** 2)
        ratio = output_energy / input_energy
        assert 0.1 < ratio < 10.0, f"Energy ratio {ratio} outside acceptable range"

    def test_deterministic(self):
        from spectra._rust import apply_rrc_filter, generate_qpsk_symbols

        symbols = generate_qpsk_symbols(64, seed=0)
        f1 = apply_rrc_filter(symbols, rolloff=0.35, span=10, sps=8)
        f2 = apply_rrc_filter(symbols, rolloff=0.35, span=10, sps=8)
        npt.assert_array_equal(f1, f2)
