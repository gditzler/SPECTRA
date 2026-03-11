import numpy as np
import numpy.testing as npt
import pytest


class TestPulsedRadar:
    def test_generate_returns_valid_iq(self, assert_valid_iq, sample_rate):
        from spectra.waveforms.radar import PulsedRadar

        waveform = PulsedRadar()
        iq = waveform.generate(num_symbols=1, sample_rate=sample_rate)
        assert_valid_iq(iq)

    def test_output_length(self, sample_rate):
        from spectra.waveforms.radar import PulsedRadar

        pri = 512
        num_pulses = 4
        waveform = PulsedRadar(pri_samples=pri, num_pulses=num_pulses)
        iq = waveform.generate(num_symbols=2, sample_rate=sample_rate)
        assert len(iq) == pri * num_pulses * 2

    def test_label(self):
        from spectra.waveforms.radar import PulsedRadar

        assert PulsedRadar().label == "PulsedRadar"

    def test_bandwidth(self, sample_rate):
        from spectra.waveforms.radar import PulsedRadar

        pw = 64
        waveform = PulsedRadar(pulse_width_samples=pw)
        assert waveform.bandwidth(sample_rate) == pytest.approx(
            sample_rate / pw
        )

    def test_deterministic(self, sample_rate):
        from spectra.waveforms.radar import PulsedRadar

        waveform = PulsedRadar()
        iq1 = waveform.generate(num_symbols=1, sample_rate=sample_rate, seed=42)
        iq2 = waveform.generate(num_symbols=1, sample_rate=sample_rate, seed=42)
        npt.assert_array_equal(iq1, iq2)

    def test_duty_cycle(self, sample_rate):
        from spectra.waveforms.radar import PulsedRadar

        pw = 32
        pri = 256
        num_pulses = 8
        waveform = PulsedRadar(
            pulse_width_samples=pw, pri_samples=pri, num_pulses=num_pulses
        )
        iq = waveform.generate(num_symbols=1, sample_rate=sample_rate)
        nonzero_frac = np.count_nonzero(iq) / len(iq)
        expected_duty = pw / pri
        assert nonzero_frac == pytest.approx(expected_duty, abs=0.01)

    def test_pri_stagger(self, sample_rate):
        from spectra.waveforms.radar import PulsedRadar

        stagger = [0, 4, -4]
        waveform = PulsedRadar(
            pulse_width_samples=16,
            pri_samples=256,
            num_pulses=3,
            pri_stagger=stagger,
        )
        iq = waveform.generate(num_symbols=1, sample_rate=sample_rate)
        assert_valid_iq = lambda x: (  # noqa: E731
            isinstance(x, np.ndarray) and x.dtype == np.complex64
        )
        assert assert_valid_iq(iq)

    def test_jitter(self, sample_rate):
        from spectra.waveforms.radar import PulsedRadar

        waveform = PulsedRadar(pri_jitter_fraction=0.1)
        iq = waveform.generate(num_symbols=1, sample_rate=sample_rate, seed=42)
        assert len(iq) > 0

    @pytest.mark.parametrize("shape", ["rect", "hamming", "hann"])
    def test_pulse_shapes(self, sample_rate, shape):
        from spectra.waveforms.radar import PulsedRadar

        waveform = PulsedRadar(pulse_shape=shape)
        iq = waveform.generate(num_symbols=1, sample_rate=sample_rate)
        assert len(iq) > 0
        assert iq.dtype == np.complex64

    def test_weather_preset(self, assert_valid_iq, sample_rate):
        from spectra.waveforms.radar import PulsedRadar

        waveform = PulsedRadar.weather()
        iq = waveform.generate(num_symbols=1, sample_rate=sample_rate)
        assert_valid_iq(iq)

    def test_marine_nav_preset(self, assert_valid_iq, sample_rate):
        from spectra.waveforms.radar import PulsedRadar

        waveform = PulsedRadar.marine_nav()
        iq = waveform.generate(num_symbols=1, sample_rate=sample_rate)
        assert_valid_iq(iq)


class TestBarkerCodedPulse:
    def test_generate_returns_valid_iq(self, assert_valid_iq, sample_rate):
        from spectra.waveforms.radar import BarkerCodedPulse

        waveform = BarkerCodedPulse()
        iq = waveform.generate(num_symbols=1, sample_rate=sample_rate)
        assert_valid_iq(iq)

    def test_output_length(self, sample_rate):
        from spectra.waveforms.radar import BarkerCodedPulse

        pri = 1024
        num_pulses = 4
        waveform = BarkerCodedPulse(pri_samples=pri, num_pulses=num_pulses)
        iq = waveform.generate(num_symbols=2, sample_rate=sample_rate)
        assert len(iq) == pri * num_pulses * 2

    def test_label(self):
        from spectra.waveforms.radar import BarkerCodedPulse

        assert BarkerCodedPulse().label == "BarkerCodedPulse"

    def test_bandwidth(self, sample_rate):
        from spectra.waveforms.radar import BarkerCodedPulse

        spc = 8
        waveform = BarkerCodedPulse(samples_per_chip=spc)
        assert waveform.bandwidth(sample_rate) == pytest.approx(sample_rate / spc)

    def test_invalid_barker_length(self):
        from spectra.waveforms.radar import BarkerCodedPulse

        with pytest.raises(ValueError):
            BarkerCodedPulse(barker_length=6)

    @pytest.mark.parametrize("length", [2, 3, 4, 5, 7, 11, 13])
    def test_barker_lengths(self, assert_valid_iq, sample_rate, length):
        from spectra.waveforms.radar import BarkerCodedPulse

        waveform = BarkerCodedPulse(barker_length=length)
        iq = waveform.generate(num_symbols=1, sample_rate=sample_rate)
        assert_valid_iq(iq)


class TestPolyphaseCodedPulse:
    def test_generate_returns_valid_iq(self, assert_valid_iq, sample_rate):
        from spectra.waveforms.radar import PolyphaseCodedPulse

        waveform = PolyphaseCodedPulse()
        iq = waveform.generate(num_symbols=1, sample_rate=sample_rate)
        assert_valid_iq(iq)

    def test_label(self):
        from spectra.waveforms.radar import PolyphaseCodedPulse

        assert PolyphaseCodedPulse().label == "PolyphaseCodedPulse"

    def test_bandwidth(self, sample_rate):
        from spectra.waveforms.radar import PolyphaseCodedPulse

        spc = 8
        waveform = PolyphaseCodedPulse(samples_per_chip=spc)
        assert waveform.bandwidth(sample_rate) == pytest.approx(sample_rate / spc)

    @pytest.mark.parametrize("code_type", ["frank", "p1", "p3", "p4"])
    def test_code_types(self, assert_valid_iq, sample_rate, code_type):
        from spectra.waveforms.radar import PolyphaseCodedPulse

        waveform = PolyphaseCodedPulse(code_type=code_type, code_order=4)
        iq = waveform.generate(num_symbols=1, sample_rate=sample_rate)
        assert_valid_iq(iq)

    def test_p2_even_order(self, assert_valid_iq, sample_rate):
        from spectra.waveforms.radar import PolyphaseCodedPulse

        waveform = PolyphaseCodedPulse(code_type="p2", code_order=4)
        iq = waveform.generate(num_symbols=1, sample_rate=sample_rate)
        assert_valid_iq(iq)

    def test_invalid_code_type(self):
        from spectra.waveforms.radar import PolyphaseCodedPulse

        with pytest.raises(ValueError):
            PolyphaseCodedPulse(code_type="invalid")


class TestFMCW:
    def test_generate_returns_valid_iq(self, assert_valid_iq, sample_rate):
        from spectra.waveforms.radar import FMCW

        waveform = FMCW()
        iq = waveform.generate(num_symbols=1, sample_rate=sample_rate)
        assert_valid_iq(iq)

    def test_output_length(self, sample_rate):
        from spectra.waveforms.radar import FMCW

        sweep_samples = 256
        idle_samples = 64
        num_sweeps = 4
        waveform = FMCW(
            sweep_samples=sweep_samples,
            idle_samples=idle_samples,
            num_sweeps=num_sweeps,
        )
        iq = waveform.generate(num_symbols=2, sample_rate=sample_rate)
        expected = (sweep_samples + idle_samples) * num_sweeps * 2
        assert len(iq) == expected

    def test_label(self):
        from spectra.waveforms.radar import FMCW

        assert FMCW().label == "FMCW"

    def test_bandwidth(self, sample_rate):
        from spectra.waveforms.radar import FMCW

        bw_frac = 0.4
        waveform = FMCW(sweep_bandwidth_fraction=bw_frac)
        assert waveform.bandwidth(sample_rate) == pytest.approx(
            bw_frac * sample_rate
        )

    def test_deterministic(self, sample_rate):
        from spectra.waveforms.radar import FMCW

        waveform = FMCW()
        iq1 = waveform.generate(num_symbols=1, sample_rate=sample_rate, seed=42)
        iq2 = waveform.generate(num_symbols=1, sample_rate=sample_rate, seed=42)
        npt.assert_array_equal(iq1, iq2)

    @pytest.mark.parametrize("sweep_type", ["sawtooth", "triangle"])
    def test_sweep_types(self, assert_valid_iq, sample_rate, sweep_type):
        from spectra.waveforms.radar import FMCW

        waveform = FMCW(sweep_type=sweep_type)
        iq = waveform.generate(num_symbols=1, sample_rate=sample_rate)
        assert_valid_iq(iq)

    def test_automotive_preset(self, assert_valid_iq, sample_rate):
        from spectra.waveforms.radar import FMCW

        waveform = FMCW.automotive()
        iq = waveform.generate(num_symbols=1, sample_rate=sample_rate)
        assert_valid_iq(iq)

    def test_weather_preset(self, assert_valid_iq, sample_rate):
        from spectra.waveforms.radar import FMCW

        waveform = FMCW.weather()
        iq = waveform.generate(num_symbols=1, sample_rate=sample_rate)
        assert_valid_iq(iq)

    def test_marine_nav_preset(self, assert_valid_iq, sample_rate):
        from spectra.waveforms.radar import FMCW

        waveform = FMCW.marine_nav()
        iq = waveform.generate(num_symbols=1, sample_rate=sample_rate)
        assert_valid_iq(iq)

    def test_atc_preset(self, assert_valid_iq, sample_rate):
        from spectra.waveforms.radar import FMCW

        waveform = FMCW.atc()
        iq = waveform.generate(num_symbols=1, sample_rate=sample_rate)
        assert_valid_iq(iq)


class TestSteppedFrequency:
    def test_generate_returns_valid_iq(self, assert_valid_iq, sample_rate):
        from spectra.waveforms.radar import SteppedFrequency

        waveform = SteppedFrequency()
        iq = waveform.generate(num_symbols=1, sample_rate=sample_rate)
        assert_valid_iq(iq)

    def test_output_length(self, sample_rate):
        from spectra.waveforms.radar import SteppedFrequency

        num_steps = 8
        sps = 64
        num_bursts = 4
        waveform = SteppedFrequency(
            num_steps=num_steps,
            samples_per_step=sps,
            num_bursts=num_bursts,
        )
        iq = waveform.generate(num_symbols=2, sample_rate=sample_rate)
        assert len(iq) == num_steps * sps * num_bursts * 2

    def test_label(self):
        from spectra.waveforms.radar import SteppedFrequency

        assert SteppedFrequency().label == "SteppedFrequency"

    def test_bandwidth(self, sample_rate):
        from spectra.waveforms.radar import SteppedFrequency

        num_steps = 8
        freq_step_frac = 0.05
        waveform = SteppedFrequency(
            num_steps=num_steps, freq_step_fraction=freq_step_frac
        )
        expected_bw = num_steps * freq_step_frac * sample_rate
        assert waveform.bandwidth(sample_rate) == pytest.approx(expected_bw)

    def test_deterministic(self, sample_rate):
        from spectra.waveforms.radar import SteppedFrequency

        waveform = SteppedFrequency()
        iq1 = waveform.generate(num_symbols=1, sample_rate=sample_rate, seed=42)
        iq2 = waveform.generate(num_symbols=1, sample_rate=sample_rate, seed=42)
        npt.assert_array_equal(iq1, iq2)


class TestPulseDoppler:
    def test_generate_returns_valid_iq(self, assert_valid_iq, sample_rate):
        from spectra.waveforms.radar import PulseDoppler

        waveform = PulseDoppler()
        iq = waveform.generate(num_symbols=1, sample_rate=sample_rate)
        assert_valid_iq(iq)

    def test_label(self):
        from spectra.waveforms.radar import PulseDoppler

        assert PulseDoppler().label == "PulseDoppler"

    def test_bandwidth(self, sample_rate):
        from spectra.waveforms.radar import PulseDoppler

        pw = 32
        waveform = PulseDoppler(pulse_width_samples=pw)
        assert waveform.bandwidth(sample_rate) == pytest.approx(sample_rate / pw)

    @pytest.mark.parametrize("prf_mode", ["low", "medium", "high"])
    def test_prf_modes(self, assert_valid_iq, sample_rate, prf_mode):
        from spectra.waveforms.radar import PulseDoppler

        waveform = PulseDoppler(prf_mode=prf_mode)
        iq = waveform.generate(num_symbols=1, sample_rate=sample_rate)
        assert_valid_iq(iq)

    def test_invalid_prf_mode(self):
        from spectra.waveforms.radar import PulseDoppler

        with pytest.raises(ValueError):
            PulseDoppler(prf_mode="invalid")

    def test_atc_preset(self, assert_valid_iq, sample_rate):
        from spectra.waveforms.radar import PulseDoppler

        waveform = PulseDoppler.atc()
        iq = waveform.generate(num_symbols=1, sample_rate=sample_rate)
        assert_valid_iq(iq)

    def test_deterministic(self, sample_rate):
        from spectra.waveforms.radar import PulseDoppler

        waveform = PulseDoppler()
        iq1 = waveform.generate(num_symbols=1, sample_rate=sample_rate, seed=42)
        iq2 = waveform.generate(num_symbols=1, sample_rate=sample_rate, seed=42)
        npt.assert_array_equal(iq1, iq2)


class TestNonlinearFM:
    def test_generate_returns_valid_iq(self, assert_valid_iq, sample_rate):
        from spectra.waveforms.radar import NonlinearFM

        waveform = NonlinearFM()
        iq = waveform.generate(num_symbols=1, sample_rate=sample_rate)
        assert_valid_iq(iq)

    def test_output_length(self, sample_rate):
        from spectra.waveforms.radar import NonlinearFM

        ns = 256
        waveform = NonlinearFM(num_samples=ns)
        iq = waveform.generate(num_symbols=3, sample_rate=sample_rate)
        assert len(iq) == ns * 3

    def test_label(self):
        from spectra.waveforms.radar import NonlinearFM

        assert NonlinearFM().label == "NonlinearFM"

    def test_bandwidth(self, sample_rate):
        from spectra.waveforms.radar import NonlinearFM

        bw_frac = 0.4
        waveform = NonlinearFM(bandwidth_fraction=bw_frac)
        assert waveform.bandwidth(sample_rate) == pytest.approx(
            bw_frac * sample_rate
        )

    def test_deterministic(self, sample_rate):
        from spectra.waveforms.radar import NonlinearFM

        waveform = NonlinearFM()
        iq1 = waveform.generate(num_symbols=1, sample_rate=sample_rate, seed=42)
        iq2 = waveform.generate(num_symbols=1, sample_rate=sample_rate, seed=42)
        npt.assert_array_equal(iq1, iq2)

    @pytest.mark.parametrize("sweep_type", ["tandem_hooked", "s_curve"])
    def test_sweep_types(self, assert_valid_iq, sample_rate, sweep_type):
        from spectra.waveforms.radar import NonlinearFM

        waveform = NonlinearFM(sweep_type=sweep_type)
        iq = waveform.generate(num_symbols=1, sample_rate=sample_rate)
        assert_valid_iq(iq)

    def test_invalid_sweep_type(self):
        from spectra.waveforms.radar import NonlinearFM

        with pytest.raises(ValueError):
            NonlinearFM(sweep_type="invalid")

    def test_unit_magnitude(self, sample_rate):
        from spectra.waveforms.radar import NonlinearFM

        waveform = NonlinearFM()
        iq = waveform.generate(num_symbols=1, sample_rate=sample_rate)
        npt.assert_allclose(np.abs(iq), 1.0, atol=1e-4)
