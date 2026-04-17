"""Schedule strategies for multi-function emitters."""

from __future__ import annotations

from abc import ABC, abstractmethod
from typing import Any, Callable, Iterator, Optional, Union

import numpy as np

from spectra.waveforms.multifunction.segment import ModeSpec, SegmentSpec


class Schedule(ABC):
    """Abstract base for anything that yields a stream of :class:`SegmentSpec`.

    Subclasses implement ``iter_segments(total_samples, sample_rate, seed)``
    which yields ``SegmentSpec`` instances until the caller has accumulated
    enough samples. The schedule does not need to track cumulative samples
    itself — the caller (typically :class:`ScheduledWaveform`) stops iterating
    once cumulative duration reaches ``total_samples``.
    """

    @abstractmethod
    def iter_segments(
        self,
        total_samples: int,
        sample_rate: float,
        seed: int,
    ) -> Iterator[SegmentSpec]:
        """Yield SegmentSpecs lazily; implementations must be generator functions."""
        if False:  # pragma: no cover
            yield  # marks this as an abstract generator function


class StaticSchedule(Schedule):
    """Deterministic timeline of segments, optionally looped.

    Args:
        segments: Ordered list of :class:`SegmentSpec` instances. Each must
            have ``duration_samples >= 1`` so that iteration can terminate.
        loop: If ``True``, wraps to the start when the list is exhausted. If
            ``False``, stops when the list is exhausted (the caller will
            zero-pad any remaining samples). Default ``True``.

    Raises:
        ValueError: If any segment has ``duration_samples <= 0`` (would hang
            the iterator when ``loop=True``).
    """

    def __init__(self, segments: list[SegmentSpec], loop: bool = True):
        for seg in segments:
            if seg.duration_samples <= 0:
                raise ValueError(
                    f"SegmentSpec has non-positive duration_samples={seg.duration_samples}; "
                    "must be >= 1"
                )
        self._segments = list(segments)
        self._loop = loop

    def iter_segments(
        self, total_samples: int, sample_rate: float, seed: int
    ) -> Iterator[SegmentSpec]:
        if not self._segments:
            return
        cumulative = 0
        idx = 0
        while cumulative < total_samples:
            seg = self._segments[idx]
            yield seg
            cumulative += seg.duration_samples
            idx += 1
            if idx >= len(self._segments):
                if self._loop:
                    idx = 0
                else:
                    return


_ROW_SUM_TOL = 1e-6


def _draw_int_range(
    val: Union[int, tuple[int, int], Callable[[np.random.Generator], int]],
    rng: np.random.Generator,
) -> int:
    """Resolve an int-or-range-or-callable into a concrete int."""
    if callable(val):
        return int(val(rng))
    if isinstance(val, tuple):
        lo, hi = val
        return int(rng.integers(int(lo), int(hi) + 1))
    return int(val)


def _draw_float_range(val: Union[float, tuple[float, float]], rng: np.random.Generator) -> float:
    """Resolve a float-or-range into a concrete float."""
    if isinstance(val, tuple):
        lo, hi = val
        return float(rng.uniform(float(lo), float(hi)))
    return float(val)


class StochasticSchedule(Schedule):
    """Markov-modelled schedule with per-mode parameter distributions.

    Args:
        modes: Mapping of mode name to :class:`ModeSpec`.
        transitions: Markov transition matrix as ``{from_mode: {to_mode: prob}}``.
            Every row must sum to 1.0 (± 1e-6) and reference only known modes.
        initial_mode: Either a single mode name (fixed initial state) or a
            distribution ``{mode_name: prob}`` (drawn at ``iter_segments`` start).

    Raises:
        ValueError: If any mode name is unknown, any row doesn't sum to 1,
            ``initial_mode`` references an unknown mode, or a known mode has
            no transition row.
    """

    def __init__(
        self,
        modes: dict[str, ModeSpec],
        transitions: dict[str, dict[str, float]],
        initial_mode: Union[str, dict[str, float]],
    ):
        self._modes = dict(modes)
        self._transitions = {k: dict(v) for k, v in transitions.items()}
        self._initial_mode = initial_mode if isinstance(initial_mode, str) else dict(initial_mode)
        self._validate()

    def _validate(self):
        names = set(self._modes)
        for from_mode, row in self._transitions.items():
            if from_mode not in names:
                raise ValueError(f"transitions from unknown mode {from_mode!r}")
            for to_mode in row:
                if to_mode not in names:
                    raise ValueError(f"transitions to unknown mode {to_mode!r}")
            total = sum(row.values())
            if abs(total - 1.0) > _ROW_SUM_TOL:
                raise ValueError(f"transitions[{from_mode!r}] row sum is {total}, expected 1.0")
        if isinstance(self._initial_mode, str):
            if self._initial_mode not in names:
                raise ValueError(f"initial_mode {self._initial_mode!r} is unknown")
        else:
            for name in self._initial_mode:
                if name not in names:
                    raise ValueError(f"initial_mode {name!r} is unknown")
            total = sum(self._initial_mode.values())
            if abs(total - 1.0) > _ROW_SUM_TOL:
                raise ValueError(f"initial_mode distribution sum is {total}, expected 1.0")
        for name in names:
            if name not in self._transitions:
                raise ValueError(f"mode {name!r} has no transition row")
        # Match StaticSchedule's guard: reject non-positive duration_samples on
        # scalar and tuple forms. (Callable durations are user-owned — we can't
        # introspect them at construction time, so they rely on runtime trust.)
        for mode_name, spec in self._modes.items():
            dur = spec.duration_samples
            if isinstance(dur, int):
                if dur <= 0:
                    raise ValueError(f"mode {mode_name!r} has non-positive duration_samples={dur}")
            elif isinstance(dur, tuple):
                lo, hi = dur
                if lo <= 0 or hi < lo:
                    raise ValueError(f"mode {mode_name!r} has invalid duration range ({lo}, {hi})")

    def _draw_mode(self, distribution: dict[str, float], rng: np.random.Generator) -> str:
        names = list(distribution.keys())
        probs = np.array([distribution[n] for n in names], dtype=np.float64)
        probs = probs / probs.sum()
        idx = int(rng.choice(len(names), p=probs))
        return names[idx]

    def iter_segments(
        self, total_samples: int, sample_rate: float, seed: int
    ) -> Iterator[SegmentSpec]:
        rng = np.random.default_rng(int(seed))

        if isinstance(self._initial_mode, str):
            current = self._initial_mode
        else:
            current = self._draw_mode(self._initial_mode, rng)

        cumulative = 0
        while cumulative < total_samples:
            mode_spec = self._modes[current]
            dur = _draw_int_range(mode_spec.duration_samples, rng)
            power = _draw_float_range(mode_spec.power_offset_db, rng)
            freq = _draw_float_range(mode_spec.freq_offset_hz, rng)
            waveform = mode_spec.waveform_factory(rng)

            yield SegmentSpec(
                waveform=waveform,
                duration_samples=dur,
                mode=current,
                power_offset_db=power,
                freq_offset_hz=freq,
                metadata=dict(mode_spec.metadata),
            )
            cumulative += dur
            current = self._draw_mode(self._transitions[current], rng)


class CognitiveSchedule(Schedule):
    """Abstract base for closed-loop / event-driven schedules.

    Subclasses implement :meth:`next_segment`, which is called repeatedly with
    the history of emitted segments and an opaque ``state`` object. Returning
    ``None`` ends the schedule (the caller zero-pads any remaining samples).

    Actual wiring to scene/target state (the reason to have this ABC at all)
    is deferred to a future spec. This version ships with no concrete
    subclass; the abstract contract lets us validate the seam in tests and
    commit to the interface shape.
    """

    @abstractmethod
    def next_segment(
        self,
        history: list[SegmentSpec],
        state: Any,
    ) -> Optional[SegmentSpec]:
        """Return the next segment, or ``None`` to end the schedule."""
        ...

    def iter_segments(
        self, total_samples: int, sample_rate: float, seed: int
    ) -> Iterator[SegmentSpec]:
        history: list[SegmentSpec] = []
        state: Any = None  # subclasses may use this; unused in v1
        cumulative = 0
        while cumulative < total_samples:
            nxt = self.next_segment(history, state)
            if nxt is None:
                return
            yield nxt
            history.append(nxt)
            cumulative += int(nxt.duration_samples)
