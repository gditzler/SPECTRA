"""Dataclasses for describing segments of a multi-function emitter's timeline."""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Callable, Union

import numpy as np

from spectra.waveforms.base import Waveform


@dataclass
class SegmentSpec:
    """One segment of a scheduled emitter timeline.

    Used directly by :class:`StaticSchedule` and produced dynamically by
    :class:`StochasticSchedule` and :class:`CognitiveSchedule`.

    Attributes:
        waveform: The sub-waveform to play during this segment.
        duration_samples: Exact duration of the segment in samples.
        mode: User-facing label for this segment's mode (e.g., ``"search"``).
        power_offset_db: Baseband amplitude delta applied as
            ``iq * 10 ** (power_offset_db / 20)``. Default ``0.0``.
        freq_offset_hz: Baseband frequency shift (relative to the emitter's
            carrier) applied to the segment IQ. Default ``0.0``.
        gain_offset_db: Recorded in the segment's SignalDescription metadata;
            **not** applied to IQ (handled downstream if needed). Default ``0.0``.
        metadata: Free-form dict copied into each segment's SignalDescription.
    """

    waveform: Waveform
    duration_samples: int
    mode: str
    power_offset_db: float = 0.0
    freq_offset_hz: float = 0.0
    gain_offset_db: float = 0.0
    metadata: dict = field(default_factory=dict)


@dataclass
class ModeSpec:
    """Describes one Markov mode in a :class:`StochasticSchedule`.

    Attributes:
        waveform_factory: Callable ``(rng) -> Waveform`` producing a fresh
            sub-waveform per dwell. ``rng`` is a ``numpy.random.Generator``
            owned by the schedule and advanced across dwells.
        duration_samples: Either a fixed int, a ``(min, max)`` tuple
            (uniform-int range, inclusive), or a callable ``(rng) -> int``.
        power_offset_db: Scalar or ``(min, max)`` tuple (uniform float range).
        freq_offset_hz: Scalar or ``(min, max)`` tuple (uniform float range).
        metadata: Free-form dict copied into each resulting SegmentSpec.
    """

    waveform_factory: Callable[[np.random.Generator], Waveform]
    duration_samples: Union[int, tuple[int, int], Callable[[np.random.Generator], int]]
    power_offset_db: Union[float, tuple[float, float]] = 0.0
    freq_offset_hz: Union[float, tuple[float, float]] = 0.0
    metadata: dict = field(default_factory=dict)
