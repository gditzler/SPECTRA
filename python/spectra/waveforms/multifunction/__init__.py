"""Multi-function emitter subpackage — schedule-driven multi-mode waveforms."""

from __future__ import annotations

import numpy as np

from spectra.scene.signal_desc import SignalDescription
from spectra.waveforms.multifunction.schedule import (
    CognitiveSchedule,
    Schedule,
    StaticSchedule,
    StochasticSchedule,
)
from spectra.waveforms.multifunction.scheduled_waveform import ScheduledWaveform
from spectra.waveforms.multifunction.segment import ModeSpec, SegmentSpec


def segments_to_mode_mask(
    segments: list[SignalDescription],
    total_samples: int,
    sample_rate: float,
    mode_to_index: dict[str, int],
    fill_index: int = -1,
) -> np.ndarray:
    """Return a per-sample int array with the active mode index per sample.

    Samples not covered by any segment receive ``fill_index``.

    Args:
        segments: Per-segment SignalDescription list (from
            :meth:`ScheduledWaveform.generate_with_segments`).
        total_samples: Length of the output mask array.
        sample_rate: Sample rate in Hz (to convert segment times to indices).
        mode_to_index: Mapping of mode string to int index.
        fill_index: Value for samples not covered by any segment.

    Returns:
        int64 array of shape ``(total_samples,)``.

    Raises:
        KeyError: If a segment's ``mode`` is not in ``mode_to_index``.
    """
    mask = np.full(total_samples, fill_index, dtype=np.int64)
    for seg in segments:
        if seg.mode is None:
            continue
        idx = mode_to_index[seg.mode]  # KeyError if unknown — intentional
        start = max(0, int(round(seg.t_start * sample_rate)))
        stop = min(total_samples, int(round(seg.t_stop * sample_rate)))
        if stop > start:
            mask[start:stop] = idx
    return mask


__all__ = [
    "CognitiveSchedule",
    "ModeSpec",
    "Schedule",
    "ScheduledWaveform",
    "SegmentSpec",
    "StaticSchedule",
    "StochasticSchedule",
    "segments_to_mode_mask",
]
