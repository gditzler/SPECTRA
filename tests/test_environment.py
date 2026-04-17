"""Tests for Environment, Emitter, ReceiverConfig, and LinkParams."""

import math
import os
import tempfile
from pathlib import Path

import pytest
from spectra.environment.core import Emitter, Environment, LinkParams, ReceiverConfig
from spectra.environment.position import Position
from spectra.environment.propagation import FreeSpacePathLoss, LogDistancePL
from spectra.waveforms import QPSK

yaml = pytest.importorskip("yaml")

SPEED_OF_LIGHT = 299_792_458.0
BOLTZMANN_K = 1.380649e-23  # J/K
# Exact thermal noise floor: 10*log10(k_B * T) + 30 dBm/Hz at T=290K
BOLTZMANN_DBM_HZ = 10 * math.log10(BOLTZMANN_K * 290.0) + 30


@pytest.fixture
def simple_env():
    """Single QPSK emitter at 1 km, free-space."""
    return Environment(
        propagation=FreeSpacePathLoss(),
        emitters=[
            Emitter(
                waveform=QPSK(samples_per_symbol=8),
                position=Position(1000.0, 0.0),
                power_dbm=30.0,
                freq_hz=2.4e9,
            ),
        ],
        receiver=ReceiverConfig(
            position=Position(0.0, 0.0),
            noise_figure_db=6.0,
            bandwidth_hz=1e6,
        ),
    )


class TestEmitter:
    def test_defaults(self):
        e = Emitter(
            waveform=QPSK(samples_per_symbol=8),
            position=Position(0, 0),
            power_dbm=30.0,
            freq_hz=2.4e9,
        )
        assert e.velocity_mps is None
        assert e.antenna_gain_dbi == 0.0

    def test_with_velocity(self):
        e = Emitter(
            waveform=QPSK(samples_per_symbol=8),
            position=Position(0, 0),
            power_dbm=30.0,
            freq_hz=2.4e9,
            velocity_mps=(30.0, 0.0),
        )
        assert e.velocity_mps == (30.0, 0.0)


class TestReceiverConfig:
    def test_defaults(self):
        r = ReceiverConfig(position=Position(0, 0))
        assert r.noise_figure_db == 6.0
        assert r.bandwidth_hz == 1e6
        assert r.antenna_gain_dbi == 0.0
        assert r.temperature_k == 290.0


class TestLinkParams:
    def test_fields(self):
        lp = LinkParams(
            emitter_index=0,
            snr_db=15.0,
            path_loss_db=100.0,
            received_power_dbm=-70.0,
            delay_s=3.3e-6,
            doppler_hz=0.0,
            distance_m=1000.0,
            fading_suggestion=None,
        )
        assert lp.snr_db == 15.0
        assert lp.fading_suggestion is None

    def test_mutable_for_override(self):
        lp = LinkParams(
            emitter_index=0,
            snr_db=15.0,
            path_loss_db=100.0,
            received_power_dbm=-70.0,
            delay_s=3.3e-6,
            doppler_hz=0.0,
            distance_m=1000.0,
            fading_suggestion=None,
        )
        lp.snr_db = 20.0
        assert lp.snr_db == 20.0


class TestEnvironmentCompute:
    def test_returns_list_of_link_params(self, simple_env):
        result = simple_env.compute()
        assert isinstance(result, list)
        assert len(result) == 1
        assert isinstance(result[0], LinkParams)

    def test_distance_computed(self, simple_env):
        result = simple_env.compute()
        assert math.isclose(result[0].distance_m, 1000.0)

    def test_delay_from_distance(self, simple_env):
        result = simple_env.compute()
        expected_delay = 1000.0 / SPEED_OF_LIGHT
        assert math.isclose(result[0].delay_s, expected_delay, rel_tol=1e-6)

    def test_snr_link_budget(self, simple_env):
        result = simple_env.compute()
        fspl = FreeSpacePathLoss()
        pl = fspl(1000.0, 2.4e9).path_loss_db
        rx_power = 30.0 + 0.0 + 0.0 - pl
        noise_floor = BOLTZMANN_DBM_HZ + 10 * math.log10(1e6) + 6.0  # NF=6 dB
        expected_snr = rx_power - noise_floor
        assert math.isclose(result[0].snr_db, expected_snr, rel_tol=1e-4)

    def test_no_doppler_without_velocity(self, simple_env):
        result = simple_env.compute()
        assert result[0].doppler_hz == 0.0

    def test_doppler_with_velocity(self):
        env = Environment(
            propagation=FreeSpacePathLoss(),
            emitters=[
                Emitter(
                    waveform=QPSK(samples_per_symbol=8),
                    position=Position(1000.0, 0.0),
                    power_dbm=30.0,
                    freq_hz=2.4e9,
                    velocity_mps=(-30.0, 0.0),
                ),
            ],
            receiver=ReceiverConfig(position=Position(0.0, 0.0)),
        )
        result = env.compute()
        expected = (30.0 / SPEED_OF_LIGHT) * 2.4e9
        assert math.isclose(result[0].doppler_hz, expected, rel_tol=1e-4)

    def test_doppler_perpendicular_velocity_zero(self):
        env = Environment(
            propagation=FreeSpacePathLoss(),
            emitters=[
                Emitter(
                    waveform=QPSK(samples_per_symbol=8),
                    position=Position(1000.0, 0.0),
                    power_dbm=30.0,
                    freq_hz=2.4e9,
                    velocity_mps=(0.0, 30.0),
                ),
            ],
            receiver=ReceiverConfig(position=Position(0.0, 0.0)),
        )
        result = env.compute()
        assert abs(result[0].doppler_hz) < 0.1

    def test_multiple_emitters(self):
        env = Environment(
            propagation=FreeSpacePathLoss(),
            emitters=[
                Emitter(
                    waveform=QPSK(samples_per_symbol=8),
                    position=Position(100.0, 0.0),
                    power_dbm=30.0,
                    freq_hz=2.4e9,
                ),
                Emitter(
                    waveform=QPSK(samples_per_symbol=8),
                    position=Position(500.0, 0.0),
                    power_dbm=30.0,
                    freq_hz=2.4e9,
                ),
            ],
            receiver=ReceiverConfig(position=Position(0.0, 0.0)),
        )
        result = env.compute()
        assert len(result) == 2
        assert result[0].emitter_index == 0
        assert result[1].emitter_index == 1
        assert result[0].snr_db > result[1].snr_db

    def test_antenna_gain_increases_snr(self):
        env_no_gain = Environment(
            propagation=FreeSpacePathLoss(),
            emitters=[
                Emitter(
                    waveform=QPSK(samples_per_symbol=8),
                    position=Position(1000.0, 0.0),
                    power_dbm=30.0,
                    freq_hz=2.4e9,
                    antenna_gain_dbi=0.0,
                )
            ],
            receiver=ReceiverConfig(position=Position(0.0, 0.0), antenna_gain_dbi=0.0),
        )
        env_with_gain = Environment(
            propagation=FreeSpacePathLoss(),
            emitters=[
                Emitter(
                    waveform=QPSK(samples_per_symbol=8),
                    position=Position(1000.0, 0.0),
                    power_dbm=30.0,
                    freq_hz=2.4e9,
                    antenna_gain_dbi=10.0,
                )
            ],
            receiver=ReceiverConfig(position=Position(0.0, 0.0), antenna_gain_dbi=5.0),
        )
        r1 = env_no_gain.compute()
        r2 = env_with_gain.compute()
        assert math.isclose(r2[0].snr_db - r1[0].snr_db, 15.0, rel_tol=1e-4)

    def test_deterministic_with_seed(self):
        env = Environment(
            propagation=LogDistancePL(n=3.5, sigma_db=8.0),
            emitters=[
                Emitter(
                    waveform=QPSK(samples_per_symbol=8),
                    position=Position(500.0, 0.0),
                    power_dbm=30.0,
                    freq_hz=2.4e9,
                )
            ],
            receiver=ReceiverConfig(position=Position(0.0, 0.0)),
        )
        r1 = env.compute(seed=42)
        r2 = env.compute(seed=42)
        assert r1[0].snr_db == r2[0].snr_db

    def test_emitter_index_preserved(self):
        env = Environment(
            propagation=FreeSpacePathLoss(),
            emitters=[
                Emitter(
                    waveform=QPSK(samples_per_symbol=8),
                    position=Position(100.0, 0.0),
                    power_dbm=20.0,
                    freq_hz=1e9,
                ),
                Emitter(
                    waveform=QPSK(samples_per_symbol=8),
                    position=Position(200.0, 0.0),
                    power_dbm=40.0,
                    freq_hz=2e9,
                ),
            ],
            receiver=ReceiverConfig(position=Position(0.0, 0.0)),
        )
        result = env.compute()
        assert result[0].emitter_index == 0
        assert result[1].emitter_index == 1


class TestEnvironmentYAML:
    def test_to_yaml_creates_file(self, simple_env, tmp_path):
        path = str(tmp_path / "env.yaml")
        simple_env.to_yaml(path)
        assert os.path.exists(path)

    def test_round_trip(self, tmp_path):
        env = Environment(
            propagation=LogDistancePL(n=3.5, sigma_db=8.0),
            emitters=[
                Emitter(
                    waveform=QPSK(samples_per_symbol=8),
                    position=Position(500.0, 200.0),
                    power_dbm=30.0,
                    freq_hz=2.4e9,
                ),
            ],
            receiver=ReceiverConfig(
                position=Position(0.0, 0.0),
                noise_figure_db=6.0,
                bandwidth_hz=1e6,
            ),
        )
        path = str(tmp_path / "env.yaml")
        env.to_yaml(path)
        loaded = Environment.from_yaml(path)

        orig = env.compute(seed=42)
        reloaded = loaded.compute(seed=42)
        assert len(orig) == len(reloaded)
        assert math.isclose(orig[0].snr_db, reloaded[0].snr_db, rel_tol=1e-6)
        assert math.isclose(orig[0].distance_m, reloaded[0].distance_m, rel_tol=1e-6)

    def test_round_trip_free_space(self, tmp_path):
        env = Environment(
            propagation=FreeSpacePathLoss(),
            emitters=[
                Emitter(
                    waveform=QPSK(samples_per_symbol=8),
                    position=Position(1000.0, 0.0),
                    power_dbm=20.0,
                    freq_hz=1e9,
                ),
            ],
            receiver=ReceiverConfig(position=Position(0.0, 0.0)),
        )
        path = str(tmp_path / "env.yaml")
        env.to_yaml(path)
        loaded = Environment.from_yaml(path)
        orig = env.compute()
        reloaded = loaded.compute()
        assert math.isclose(orig[0].snr_db, reloaded[0].snr_db, rel_tol=1e-6)

    def test_round_trip_cost231(self, tmp_path):
        from spectra.environment.propagation import COST231HataPL

        env = Environment(
            propagation=COST231HataPL(h_bs_m=50, h_ms_m=2.0, environment="suburban"),
            emitters=[
                Emitter(
                    waveform=QPSK(samples_per_symbol=8),
                    position=Position(2000.0, 0.0),
                    power_dbm=40.0,
                    freq_hz=1800e6,
                ),
            ],
            receiver=ReceiverConfig(position=Position(0.0, 0.0)),
        )
        path = str(tmp_path / "env.yaml")
        env.to_yaml(path)
        loaded = Environment.from_yaml(path)
        orig = env.compute()
        reloaded = loaded.compute()
        assert math.isclose(orig[0].snr_db, reloaded[0].snr_db, rel_tol=1e-6)


def test_link_params_defaults_for_new_fields():
    from spectra.environment.core import LinkParams
    lp = LinkParams(
        emitter_index=0,
        snr_db=10.0,
        path_loss_db=100.0,
        received_power_dbm=-70.0,
        delay_s=1e-6,
        doppler_hz=0.0,
        distance_m=100.0,
        fading_suggestion=None,
    )
    assert lp.shadow_fading_db == 0.0
    assert lp.rms_delay_spread_s is None
    assert lp.k_factor_db is None
    assert lp.angular_spread_deg is None


def test_environment_compute_populates_multipath_from_uma():
    from spectra.environment import Emitter, Environment, Position, ReceiverConfig
    from spectra.environment.propagation import GPP38901UMa
    from spectra.waveforms import QPSK

    env = Environment(
        propagation=GPP38901UMa(h_bs_m=25.0, h_ut_m=1.5, los_mode="force_los"),
        emitters=[
            Emitter(
                waveform=QPSK(samples_per_symbol=8),
                position=Position(100.0, 0.0),
                power_dbm=30.0,
                freq_hz=3.5e9,
            )
        ],
        receiver=ReceiverConfig(position=Position(0.0, 0.0)),
    )
    lp = env.compute(seed=0)[0]
    assert lp.rms_delay_spread_s is not None
    assert lp.k_factor_db is not None
    assert lp.angular_spread_deg is not None


def _basic_env(propagation):
    from spectra.environment import Emitter, Environment, Position, ReceiverConfig
    from spectra.waveforms import QPSK
    return Environment(
        propagation=propagation,
        emitters=[
            Emitter(
                waveform=QPSK(samples_per_symbol=8),
                position=Position(500.0, 0.0),
                power_dbm=30.0,
                freq_hz=3.5e9,
            )
        ],
        receiver=ReceiverConfig(position=Position(0.0, 0.0)),
    )


@pytest.mark.parametrize(
    "prop_factory",
    [
        lambda: __import__(
            "spectra.environment.propagation", fromlist=["ITU_R_P525"]
        ).ITU_R_P525(include_gaseous=True),
        lambda: __import__(
            "spectra.environment.propagation", fromlist=["OkumuraHataPL"]
        ).OkumuraHataPL(h_bs_m=50.0, h_ms_m=1.5, environment="urban_small_medium"),
        lambda: __import__(
            "spectra.environment.propagation", fromlist=["GPP38901UMa"]
        ).GPP38901UMa(h_bs_m=25.0, h_ut_m=1.5),
        lambda: __import__(
            "spectra.environment.propagation", fromlist=["GPP38901UMi"]
        ).GPP38901UMi(h_bs_m=10.0, h_ut_m=1.5),
        lambda: __import__(
            "spectra.environment.propagation", fromlist=["GPP38901RMa"]
        ).GPP38901RMa(h_bs_m=35.0, h_ut_m=1.5),
        lambda: __import__(
            "spectra.environment.propagation", fromlist=["GPP38901InH"]
        ).GPP38901InH(h_bs_m=3.0, h_ut_m=1.0, variant="mixed_office"),
        lambda: __import__(
            "spectra.environment.propagation", fromlist=["ITU_R_P1411"]
        ).ITU_R_P1411(environment="urban_high_rise"),
    ],
)
def test_yaml_roundtrip_for_new_propagation_models(prop_factory):
    from spectra.environment import Environment
    env = _basic_env(prop_factory())
    with tempfile.TemporaryDirectory() as td:
        path = Path(td) / "env.yaml"
        env.to_yaml(str(path))
        env2 = Environment.from_yaml(str(path))
        assert type(env2.propagation) is type(env.propagation)
        # Key attributes match
        for k in vars(env.propagation):
            if k.startswith("_"):
                continue
            assert getattr(env2.propagation, k) == getattr(env.propagation, k), k
