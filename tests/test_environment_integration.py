"""Tests for environment-to-impairment integration."""

import pytest
from spectra.environment.core import Emitter, Environment, LinkParams, ReceiverConfig
from spectra.environment.integration import link_params_to_impairments
from spectra.environment.position import Position
from spectra.environment.propagation import FreeSpacePathLoss
from spectra.impairments import AWGN, Compose, DopplerShift, RayleighFading, RicianFading
from spectra.impairments.base import Transform
from spectra.scene import SignalDescription
from spectra.waveforms import QPSK


@pytest.fixture
def signal_description():
    return SignalDescription(
        t_start=0.0,
        t_stop=0.001,
        f_low=-5e3,
        f_high=5e3,
        label="QPSK",
        snr=20.0,
    )


class TestLinkParamsToImpairments:
    def test_returns_list_of_transforms(self):
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
        result = link_params_to_impairments(lp)
        assert isinstance(result, list)
        assert all(isinstance(t, Transform) for t in result)

    def test_awgn_always_present(self):
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
        result = link_params_to_impairments(lp)
        awgn_list = [t for t in result if isinstance(t, AWGN)]
        assert len(awgn_list) == 1

    def test_no_doppler_when_zero(self):
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
        result = link_params_to_impairments(lp)
        doppler_list = [t for t in result if isinstance(t, DopplerShift)]
        assert len(doppler_list) == 0

    def test_doppler_included_when_nonzero(self):
        lp = LinkParams(
            emitter_index=0,
            snr_db=15.0,
            path_loss_db=100.0,
            received_power_dbm=-70.0,
            delay_s=3.3e-6,
            doppler_hz=240.0,
            distance_m=1000.0,
            fading_suggestion=None,
        )
        result = link_params_to_impairments(lp)
        doppler_list = [t for t in result if isinstance(t, DopplerShift)]
        assert len(doppler_list) == 1

    def test_rician_fading_from_suggestion(self):
        lp = LinkParams(
            emitter_index=0,
            snr_db=15.0,
            path_loss_db=100.0,
            received_power_dbm=-70.0,
            delay_s=3.3e-6,
            doppler_hz=0.0,
            distance_m=1000.0,
            fading_suggestion="rician_k6",
        )
        result = link_params_to_impairments(lp)
        rician_list = [t for t in result if isinstance(t, RicianFading)]
        assert len(rician_list) == 1

    def test_rayleigh_fading_from_suggestion(self):
        lp = LinkParams(
            emitter_index=0,
            snr_db=15.0,
            path_loss_db=100.0,
            received_power_dbm=-70.0,
            delay_s=3.3e-6,
            doppler_hz=0.0,
            distance_m=1000.0,
            fading_suggestion="rayleigh",
        )
        result = link_params_to_impairments(lp)
        rayleigh_list = [t for t in result if isinstance(t, RayleighFading)]
        assert len(rayleigh_list) == 1

    def test_no_fading_when_none(self):
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
        result = link_params_to_impairments(lp)
        fading_list = [t for t in result if isinstance(t, (RayleighFading, RicianFading))]
        assert len(fading_list) == 0

    def test_impairment_order_doppler_fading_awgn(self):
        lp = LinkParams(
            emitter_index=0,
            snr_db=15.0,
            path_loss_db=100.0,
            received_power_dbm=-70.0,
            delay_s=3.3e-6,
            doppler_hz=240.0,
            distance_m=1000.0,
            fading_suggestion="rayleigh",
        )
        result = link_params_to_impairments(lp)
        assert len(result) == 3
        assert isinstance(result[0], DopplerShift)
        assert isinstance(result[1], RayleighFading)
        assert isinstance(result[2], AWGN)


class TestEndToEnd:
    def test_environment_to_impairments_apply(self, signal_description, assert_valid_iq):
        env = Environment(
            propagation=FreeSpacePathLoss(),
            emitters=[
                Emitter(
                    waveform=QPSK(samples_per_symbol=8),
                    position=Position(500.0, 0.0),
                    power_dbm=30.0,
                    freq_hz=2.4e9,
                )
            ],
            receiver=ReceiverConfig(
                position=Position(0.0, 0.0), noise_figure_db=6.0, bandwidth_hz=1e6
            ),
        )
        params = env.compute()[0]
        impairments = link_params_to_impairments(params)
        iq = env.emitters[0].waveform.generate(num_symbols=128, sample_rate=1e6, seed=42)
        desc = signal_description
        for t in impairments:
            iq, desc = t(iq, desc, sample_rate=1e6)
        assert_valid_iq(iq)

    def test_compose_wrapping(self, signal_description, assert_valid_iq):
        lp = LinkParams(
            emitter_index=0,
            snr_db=20.0,
            path_loss_db=80.0,
            received_power_dbm=-50.0,
            delay_s=1e-6,
            doppler_hz=0.0,
            distance_m=300.0,
            fading_suggestion=None,
        )
        chain = Compose(link_params_to_impairments(lp))
        waveform = QPSK(samples_per_symbol=8)
        iq = waveform.generate(num_symbols=128, sample_rate=1e6, seed=42)
        iq, desc = chain(iq, signal_description, sample_rate=1e6)
        assert_valid_iq(iq)

    def test_override_snr_before_conversion(self, signal_description, assert_valid_iq):
        env = Environment(
            propagation=FreeSpacePathLoss(),
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
        params = env.compute()[0]
        params.snr_db = 5.0
        impairments = link_params_to_impairments(params)
        iq = env.emitters[0].waveform.generate(num_symbols=128, sample_rate=1e6, seed=42)
        for t in impairments:
            iq, desc = t(iq, signal_description, sample_rate=1e6)
        assert_valid_iq(iq)
