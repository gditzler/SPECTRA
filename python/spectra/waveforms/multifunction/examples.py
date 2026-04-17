"""Reference multi-function emitter factory functions."""

from __future__ import annotations

import numpy as np  # noqa: F401 — used in _agile_factory via rng (np.random.Generator)

from spectra.waveforms.multifunction.schedule import StaticSchedule, StochasticSchedule
from spectra.waveforms.multifunction.scheduled_waveform import ScheduledWaveform
from spectra.waveforms.multifunction.segment import ModeSpec, SegmentSpec
from spectra.waveforms.psk import QPSK
from spectra.waveforms.radar import PulsedRadar


def multifunction_search_track_radar() -> ScheduledWaveform:
    """Canonical multi-function radar alternating search and track dwells.

    Search dwell: long CPI (16 pulses of 2048-sample PRI, 256-sample Hamming
    pulse), representative of a wide-area surveillance scan.
    Track dwell: shorter CPI (8 pulses of 512-sample PRI, 64-sample rect
    pulse), offset by +50 kHz, +3 dB — representative of a dedicated track
    beam pointed at a designated target.

    Returns:
        A :class:`ScheduledWaveform` that alternates search/track forever.
    """
    search = SegmentSpec(
        waveform=PulsedRadar(
            pulse_width_samples=256,
            pri_samples=2048,
            num_pulses=16,
            pulse_shape="hamming",
        ),
        duration_samples=2048 * 4,  # 8_192 samples — one CPI for test / real use
        mode="search",
    )
    track = SegmentSpec(
        waveform=PulsedRadar(
            pulse_width_samples=64,
            pri_samples=512,
            num_pulses=8,
            pulse_shape="rect",
        ),
        duration_samples=512 * 4,  # 2_048 samples
        mode="track",
        power_offset_db=3.0,
        freq_offset_hz=50e3,
    )
    schedule = StaticSchedule(segments=[search, track, search, track], loop=True)
    return ScheduledWaveform(schedule=schedule, label="MFR_SearchTrack")


def _low_prf_factory(rng: np.random.Generator) -> PulsedRadar:
    return PulsedRadar(pulse_width_samples=128, pri_samples=4096, num_pulses=8)


def _medium_prf_factory(rng: np.random.Generator) -> PulsedRadar:
    return PulsedRadar(pulse_width_samples=64, pri_samples=1024, num_pulses=16)


def _high_prf_factory(rng: np.random.Generator) -> PulsedRadar:
    return PulsedRadar(pulse_width_samples=16, pri_samples=256, num_pulses=32)


def multi_prf_pulse_doppler_radar() -> ScheduledWaveform:
    """Multi-PRF pulse-Doppler radar using Markov-model dwell sequencing.

    Three modes (``low_prf``, ``medium_prf``, ``high_prf``) differ by PRI
    and CPI length. Transitions favor mode diversity (self-loop probability
    0.2; others 0.4 each). Dwell duration is uniform in ``(2048, 16384)``
    samples.

    Returns:
        A :class:`ScheduledWaveform` wrapping a :class:`StochasticSchedule`.
    """
    modes = {
        "low_prf": ModeSpec(
            waveform_factory=_low_prf_factory,
            duration_samples=(2048, 16384),
        ),
        "medium_prf": ModeSpec(
            waveform_factory=_medium_prf_factory,
            duration_samples=(2048, 16384),
        ),
        "high_prf": ModeSpec(
            waveform_factory=_high_prf_factory,
            duration_samples=(2048, 16384),
        ),
    }
    transitions = {
        "low_prf": {"low_prf": 0.2, "medium_prf": 0.4, "high_prf": 0.4},
        "medium_prf": {"low_prf": 0.4, "medium_prf": 0.2, "high_prf": 0.4},
        "high_prf": {"low_prf": 0.4, "medium_prf": 0.4, "high_prf": 0.2},
    }
    schedule = StochasticSchedule(
        modes=modes,
        transitions=transitions,
        initial_mode={"low_prf": 1.0 / 3, "medium_prf": 1.0 / 3, "high_prf": 1.0 / 3},
    )
    return ScheduledWaveform(schedule=schedule, label="PulseDopplerMultiPRF")


_PRI_CHOICES = (512, 768, 1024, 1536)


def _agile_factory(rng: np.random.Generator) -> PulsedRadar:
    pri = int(rng.choice(_PRI_CHOICES))
    return PulsedRadar(pulse_width_samples=64, pri_samples=pri, num_pulses=8)


def frequency_agile_stepped_pri_radar() -> ScheduledWaveform:
    """Frequency-agile, PRI-stepped pulsed radar.

    Single mode ``"agile"``; each dwell instantiates a fresh
    :class:`PulsedRadar` with PRI drawn from ``{512, 768, 1024, 1536}`` and
    carrier offset drawn uniformly in ``(-200 kHz, +200 kHz)``.

    Returns:
        A :class:`ScheduledWaveform` wrapping a :class:`StochasticSchedule`
        with a single self-looping mode.
    """
    modes = {
        "agile": ModeSpec(
            waveform_factory=_agile_factory,
            duration_samples=(2048, 8192),
            freq_offset_hz=(-200e3, 200e3),
        ),
    }
    schedule = StochasticSchedule(
        modes=modes,
        transitions={"agile": {"agile": 1.0}},
        initial_mode="agile",
    )
    return ScheduledWaveform(schedule=schedule, label="FreqAgileRadar")


def radcom_emitter() -> ScheduledWaveform:
    """Joint radar-communications emitter alternating radar and comms dwells.

    Radar dwells use a short-pulse :class:`PulsedRadar`; comms bursts use
    :class:`QPSK` with RRC pulse shaping. Both share the emitter's carrier.

    Returns:
        A :class:`ScheduledWaveform` with a static radar/comms interleave.
    """
    radar = SegmentSpec(
        waveform=PulsedRadar(
            pulse_width_samples=64,
            pri_samples=512,
            num_pulses=8,
        ),
        duration_samples=512 * 8,  # 4_096
        mode="radar",
    )
    comms = SegmentSpec(
        waveform=QPSK(samples_per_symbol=8, rolloff=0.35),
        duration_samples=2048,
        mode="comms",
    )
    schedule = StaticSchedule(
        segments=[radar, comms, radar, radar, comms, radar, comms],
        loop=True,
    )
    return ScheduledWaveform(schedule=schedule, label="RadCom")
