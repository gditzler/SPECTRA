import numpy as np
import pytest


@pytest.mark.rust
class TestRRCTapCache:
    def test_cached_taps_match_original(self):
        """Cached taps must equal taps from apply_rrc_filter path."""
        from spectra._rust import rrc_taps_py
        from spectra.utils.rrc_cache import cached_rrc_taps
        direct = np.array(rrc_taps_py(0.35, 10, 8))
        cached = cached_rrc_taps(0.35, 10, 8)
        np.testing.assert_array_equal(direct, cached)

    def test_cache_returns_same_object(self):
        """Second call with same params should return cached array."""
        from spectra.utils.rrc_cache import cached_rrc_taps
        a = cached_rrc_taps(0.35, 10, 8)
        b = cached_rrc_taps(0.35, 10, 8)
        assert a is b  # Same object from cache

    def test_different_params_produce_different_taps(self):
        from spectra.utils.rrc_cache import cached_rrc_taps
        a = cached_rrc_taps(0.25, 10, 8)
        b = cached_rrc_taps(0.50, 10, 8)
        assert not np.array_equal(a, b)

    def test_taps_correct_length(self):
        from spectra.utils.rrc_cache import cached_rrc_taps
        taps = cached_rrc_taps(0.35, 10, 8)
        assert len(taps) == 10 * 8 + 1  # span * sps + 1

    def test_waveform_generate_unchanged(self):
        """QPSK.generate() must produce identical output after caching change."""
        from spectra.waveforms.psk import QPSK
        w = QPSK()
        iq1 = w.generate(128, 1e6, seed=42)
        iq2 = w.generate(128, 1e6, seed=42)
        np.testing.assert_array_equal(iq1, iq2)
