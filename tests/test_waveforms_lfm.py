import numpy as np
import numpy.testing as npt
import pytest


class TestLFMWaveform:
    def test_generate_returns_valid_iq(self, assert_valid_iq, sample_rate):
        from spectra.waveforms.lfm import LFM

        waveform = LFM()
        iq = waveform.generate(num_symbols=4, sample_rate=sample_rate)
        assert_valid_iq(iq)

    def test_output_length(self, sample_rate):
        from spectra.waveforms.lfm import LFM

        spp = 256
        waveform = LFM(samples_per_pulse=spp)
        iq = waveform.generate(num_symbols=3, sample_rate=sample_rate)
        assert len(iq) == 3 * spp

    def test_label(self):
        from spectra.waveforms.lfm import LFM

        assert LFM().label == "LFM"

    def test_bandwidth(self, sample_rate):
        from spectra.waveforms.lfm import LFM

        waveform = LFM(bandwidth_fraction=0.4)
        assert waveform.bandwidth(sample_rate) == pytest.approx(0.4 * sample_rate)

    def test_unit_magnitude(self, sample_rate):
        from spectra.waveforms.lfm import LFM

        waveform = LFM()
        iq = waveform.generate(num_symbols=2, sample_rate=sample_rate)
        npt.assert_allclose(np.abs(iq), 1.0, atol=1e-5)

    def test_samples_per_symbol_attribute(self):
        from spectra.waveforms.lfm import LFM

        waveform = LFM(samples_per_pulse=512)
        assert waveform.samples_per_symbol == 512

    def test_deterministic(self, sample_rate):
        from spectra.waveforms.lfm import LFM

        waveform = LFM()
        iq1 = waveform.generate(num_symbols=4, sample_rate=sample_rate, seed=42)
        iq2 = waveform.generate(num_symbols=4, sample_rate=sample_rate, seed=42)
        npt.assert_array_equal(iq1, iq2)

    def test_direction_default_is_up_regression(self, sample_rate):
        """Default direction must be byte-identical to legacy up-chirp output."""
        from spectra.waveforms.lfm import LFM

        legacy = LFM(samples_per_pulse=128)
        explicit_up = LFM(samples_per_pulse=128, direction="up")
        a = legacy.generate(num_symbols=3, sample_rate=sample_rate, seed=7)
        b = explicit_up.generate(num_symbols=3, sample_rate=sample_rate, seed=7)
        np.testing.assert_array_equal(a, b)

    def test_down_chirp_is_frequency_reversed(self, sample_rate):
        """Down-chirp equals the conjugate of the up-chirp (mirrored instantaneous frequency)."""
        from spectra.waveforms.lfm import LFM

        up = LFM(samples_per_pulse=256, direction="up").generate(1, sample_rate, seed=1)
        down = LFM(samples_per_pulse=256, direction="down").generate(1, sample_rate, seed=1)
        np.testing.assert_allclose(down, np.conj(up), rtol=1e-5, atol=1e-5)

    def test_down_chirp_has_negative_slope(self, sample_rate):
        """Instantaneous frequency of a down-chirp decreases over the pulse."""
        from spectra.waveforms.lfm import LFM

        down = LFM(samples_per_pulse=256, direction="down").generate(1, sample_rate, seed=1)
        inst_phase = np.unwrap(np.angle(down))
        inst_freq = np.diff(inst_phase)
        # Later half sweeps lower in frequency than the earlier half.
        assert inst_freq[-1] < inst_freq[0]

    def test_random_direction_is_deterministic(self, sample_rate):
        """Same seed reproduces the same randomized direction/output."""
        from spectra.waveforms.lfm import LFM

        wf = LFM(samples_per_pulse=128, direction="random")
        a = wf.generate(num_symbols=2, sample_rate=sample_rate, seed=42)
        b = wf.generate(num_symbols=2, sample_rate=sample_rate, seed=42)
        np.testing.assert_array_equal(a, b)

    def test_random_direction_matches_up_or_down(self, sample_rate):
        """A randomized call is exactly one of the two deterministic directions."""
        from spectra.waveforms.lfm import LFM

        rnd = LFM(samples_per_pulse=128, direction="random").generate(1, sample_rate, seed=3)
        up = LFM(samples_per_pulse=128, direction="up").generate(1, sample_rate, seed=3)
        down = LFM(samples_per_pulse=128, direction="down").generate(1, sample_rate, seed=3)
        matches_up = np.array_equal(rnd, up)
        matches_down = np.array_equal(rnd, down)
        assert matches_up or matches_down

    def test_random_direction_varies_across_seeds(self, sample_rate):
        """Across many seeds, the randomized mode produces both directions."""
        from spectra.waveforms.lfm import LFM

        up_wf = LFM(samples_per_pulse=64, direction="up")
        rnd_wf = LFM(samples_per_pulse=64, direction="random")
        saw_up = saw_down = False
        for s in range(20):
            # The pulse content is seed-independent (no data bits), so any fixed
            # seed's up-chirp is the reference for "is this call an up-chirp?".
            up_ref = up_wf.generate(1, sample_rate, seed=s)
            is_up = np.array_equal(rnd_wf.generate(1, sample_rate, seed=s), up_ref)
            saw_up |= is_up
            saw_down |= not is_up
        assert saw_up and saw_down

    def test_invalid_direction_raises(self):
        """Unknown direction is rejected at construction time."""
        from spectra.waveforms.lfm import LFM

        with pytest.raises(ValueError):
            LFM(direction="sideways")

    def test_label_unaffected_by_direction(self):
        from spectra.waveforms.lfm import LFM

        assert LFM(direction="down").label == "LFM"
        assert LFM(direction="random").label == "LFM"
