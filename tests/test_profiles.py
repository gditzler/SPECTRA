"""Tests for the emitter-profile registry."""

import numpy as np
import pytest

import spectra as sp
from spectra.profiles import (
    Choice,
    EmitterProfile,
    Fixed,
    LogUniform,
    ProfileNotRepresentable,
    Uniform,
)


class TestParamSpec:
    def test_fixed(self):
        rng = np.random.default_rng(0)
        assert Fixed(42).sample(rng) == 42

    def test_choice_draws_from_options(self):
        rng = np.random.default_rng(0)
        draws = {Choice([1, 2, 3]).sample(rng) for _ in range(50)}
        assert draws <= {1, 2, 3} and len(draws) > 1

    def test_uniform_bounds(self):
        rng = np.random.default_rng(0)
        for _ in range(50):
            v = Uniform(2.0, 3.0).sample(rng)
            assert 2.0 <= v <= 3.0

    def test_loguniform_bounds(self):
        rng = np.random.default_rng(0)
        for _ in range(50):
            v = LogUniform(1e3, 1e6).sample(rng)
            assert 1e3 <= v <= 1e6

    def test_invalid_bounds_raise(self):
        with pytest.raises(ValueError):
            Uniform(3.0, 2.0)
        with pytest.raises(ValueError):
            LogUniform(0.0, 1.0)

    def test_deterministic_under_seed(self):
        a = Uniform(0, 1).sample(np.random.default_rng(7))
        b = Uniform(0, 1).sample(np.random.default_rng(7))
        assert a == b


class TestEmitterProfile:
    def _qpsk_profile(self, rate_spec):
        return EmitterProfile(
            name="test-qpsk",
            label="TESTQPSK",
            waveform_cls=sp.QPSK,
            params={"symbol_rate": rate_spec, "rolloff": Fixed(0.35)},
            reference="test",
        )

    def test_sample_constructs_waveform(self):
        prof = self._qpsk_profile(Fixed(250e3))
        wf = prof.sample(np.random.default_rng(0), sample_rate=10e6)
        assert isinstance(wf, sp.QPSK)
        assert wf.bandwidth(10e6) == pytest.approx(250e3 * 1.35)

    def test_sample_deterministic(self):
        prof = self._qpsk_profile(Uniform(100e3, 1e6))
        a = prof.sample(np.random.default_rng(3), 10e6)
        b = prof.sample(np.random.default_rng(3), 10e6)
        assert a.symbol_rate == b.symbol_rate

    def test_redraws_until_representable(self):
        # Range straddles fs: draws with bandwidth > fs or symbol rate above
        # fs/2 are rejected, but representable draws exist, so sample() must
        # succeed AND the result must be generable.
        prof = self._qpsk_profile(Uniform(1e6, 50e6))
        wf = prof.sample(np.random.default_rng(1), sample_rate=10e6)
        assert wf.bandwidth(10e6) <= 10e6
        iq = wf.generate(num_symbols=16, sample_rate=10e6, seed=0)
        assert len(iq) > 0

    def test_unrepresentable_raises(self):
        prof = self._qpsk_profile(Fixed(50e6))
        with pytest.raises(ProfileNotRepresentable, match="test-qpsk"):
            prof.sample(np.random.default_rng(0), sample_rate=10e6)


from spectra import profiles as _profiles  # noqa: F401  (imported for clarity; profiles.get used below is from a later task and not needed here)


class TestComposerProfiles:
    def _cfg(self, pool):
        return sp.SceneConfig(
            capture_duration=1e-3, capture_bandwidth=10e6, sample_rate=10e6,
            num_signals=2, signal_pool=pool, snr_range=(10, 20),
        )

    def _tetra_profile(self):
        from spectra.profiles import EmitterProfile, Fixed

        return EmitterProfile(
            name="test-tetra", label="TETRA", waveform_cls=sp.QPSK,
            params={"symbol_rate": Fixed(18000.0), "rolloff": Fixed(0.35)},
            reference="test",
        )

    def _ais_profile(self):
        from spectra.profiles import EmitterProfile, Fixed

        return EmitterProfile(
            name="test-ais", label="AIS", waveform_cls=sp.QPSK,
            params={"symbol_rate": Fixed(9600.0)},
            reference="test",
        )

    def test_profile_in_pool(self):
        pool = [self._tetra_profile(), sp.QPSK()]
        iq, descs = sp.Composer(self._cfg(pool)).generate(seed=5)
        assert len(descs) == 2
        assert set(d.label for d in descs) <= {"TETRA", "QPSK"}

    def test_profile_label_used(self):
        pool = [self._ais_profile()]
        _, descs = sp.Composer(self._cfg(pool)).generate(seed=5)
        assert all(d.label == "AIS" for d in descs)

    def test_deterministic(self):
        pool = [self._tetra_profile(), self._ais_profile()]
        a, da = sp.Composer(self._cfg(pool)).generate(seed=9)
        b, db = sp.Composer(self._cfg(pool)).generate(seed=9)
        np.testing.assert_array_equal(a, b)
        assert [(d.label, d.f_low, d.f_high) for d in da] == [
            (d.label, d.f_low, d.f_high) for d in db
        ]

    def test_box_width_matches_sampled_bandwidth(self):
        pool = [self._tetra_profile()]
        _, descs = sp.Composer(self._cfg(pool)).generate(seed=1)
        for d in descs:
            assert (d.f_high - d.f_low) == pytest.approx(18000.0 * 1.35, rel=1e-6)
