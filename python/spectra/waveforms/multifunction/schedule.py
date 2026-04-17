"""Schedule strategies for multi-function emitters."""

from __future__ import annotations

from abc import ABC, abstractmethod
from typing import Iterator

from spectra.waveforms.multifunction.segment import SegmentSpec


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
