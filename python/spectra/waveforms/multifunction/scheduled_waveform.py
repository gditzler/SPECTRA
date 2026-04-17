"""ScheduledWaveform: a Waveform that plays a schedule of sub-waveforms."""

from __future__ import annotations

import math
from typing import Optional

import numpy as np

from spectra.scene.signal_desc import SignalDescription
from spectra.utils.dsp import frequency_shift
from spectra.waveforms.base import Waveform
from spectra.waveforms.multifunction.schedule import Schedule

_PREVIEW_DURATION_SEC = 0.1
_FALLBACK_BANDWIDTH_FRACTION = 1.0 / 16.0  # used when preview yields nothing


class ScheduledWaveform(Waveform):
    """A Waveform that iterates a :class:`Schedule` and concatenates child IQ.

    Per-segment ``power_offset_db`` and ``freq_offset_hz`` are applied to the
    child IQ at baseband before concatenation. The emitter carrier, antenna
    gain, and receiver-side link budget are handled downstream by
    :class:`spectra.environment.Emitter` and :class:`spectra.environment.Environment`
    exactly as for any other waveform.

    Args:
        schedule: The schedule driving segment emission.
        label: User-facing classification label (returned by ``label``).
            Default ``"multifunction"``.
        default_bandwidth_hz: If set, overrides the automatic bandwidth
            heuristic. Recommended when the heuristic underestimates (e.g.,
            rare wide-band modes in a stochastic schedule).
    """

    samples_per_symbol: int = 1  # so num_symbols == num_samples in callers

    def __init__(
        self,
        schedule: Schedule,
        label: str = "multifunction",
        default_bandwidth_hz: Optional[float] = None,
    ):
        self._schedule = schedule
        self._label = label
        self._default_bandwidth_hz = default_bandwidth_hz

    @property
    def label(self) -> str:
        return self._label

    def generate(
        self,
        num_symbols: int,
        sample_rate: float,
        seed: Optional[int] = None,
    ) -> np.ndarray:
        iq, _ = self.generate_with_segments(num_symbols, sample_rate, seed)
        return iq

    def generate_with_segments(
        self,
        num_samples: int,
        sample_rate: float,
        seed: Optional[int] = None,
    ) -> tuple[np.ndarray, list[SignalDescription]]:
        """Generate IQ plus a list of per-segment :class:`SignalDescription`.

        Args:
            num_samples: Total output sample count.
            sample_rate: Sample rate in Hz.
            seed: Master seed. Same seed -> byte-identical output.

        Returns:
            ``(iq, segments)`` where ``iq`` is a complex64 array of exactly
            ``num_samples`` samples, and ``segments`` is a list of
            :class:`SignalDescription` entries, one per emitted segment.
        """
        # Single-RNG seeding model (mirror the spec exactly):
        # one draw from the parent rng seeds the schedule; each iteration of
        # the schedule loop then draws one child seed from the same rng.
        # Do not split this into independent streams — later reviewers should
        # preserve the interleaved-draws invariant so inserting or removing a
        # segment only shifts later child seeds, not earlier ones.
        if seed is None:
            seed = int(np.random.default_rng().integers(0, 2**32))
        rng = np.random.default_rng(seed)
        seed_for_schedule = int(rng.integers(0, 2**32))

        out = np.zeros(num_samples, dtype=np.complex64)
        segments: list[SignalDescription] = []
        cursor = 0

        for spec in self._schedule.iter_segments(
            total_samples=num_samples,
            sample_rate=sample_rate,
            seed=seed_for_schedule,
        ):
            if cursor >= num_samples:
                break
            remaining = num_samples - cursor
            dur = min(int(spec.duration_samples), remaining)
            if dur <= 0:
                continue

            child_sps = max(1, int(getattr(spec.waveform, "samples_per_symbol", 1)))
            child_num_symbols = max(1, math.ceil(dur / child_sps))
            child_seed = int(rng.integers(0, 2**32))
            child_iq = spec.waveform.generate(
                num_symbols=child_num_symbols,
                sample_rate=sample_rate,
                seed=child_seed,
            )
            child_iq = np.asarray(child_iq, dtype=np.complex64)

            # Truncate or zero-pad to exactly `dur` samples.
            if len(child_iq) >= dur:
                child_iq = child_iq[:dur]
            else:
                pad = np.zeros(dur - len(child_iq), dtype=np.complex64)
                child_iq = np.concatenate([child_iq, pad])

            # Apply per-segment power offset (amplitude).
            if spec.power_offset_db != 0.0:
                child_iq = (child_iq * (10.0 ** (spec.power_offset_db / 20.0))).astype(np.complex64)

            # Apply per-segment frequency offset at baseband.
            if spec.freq_offset_hz != 0.0:
                child_iq = frequency_shift(child_iq, spec.freq_offset_hz, sample_rate)

            out[cursor : cursor + dur] = child_iq

            child_bw = float(spec.waveform.bandwidth(sample_rate))
            segments.append(
                SignalDescription(
                    t_start=cursor / sample_rate,
                    t_stop=(cursor + dur) / sample_rate,
                    f_low=spec.freq_offset_hz - child_bw / 2.0,
                    f_high=spec.freq_offset_hz + child_bw / 2.0,
                    label=spec.waveform.label,
                    snr=0.0,
                    modulation_params={
                        **dict(spec.metadata),
                        "gain_offset_db": float(spec.gain_offset_db),
                        "power_offset_db": float(spec.power_offset_db),
                        "freq_offset_hz": float(spec.freq_offset_hz),
                    },
                    mode=spec.mode,
                )
            )
            cursor += dur

        return out, segments

    def bandwidth(self, sample_rate: float) -> float:
        """Return the multi-function emitter's aggregate bandwidth envelope.

        If ``default_bandwidth_hz`` was provided to the constructor, that is
        returned. Otherwise a preview run of the schedule over
        ~``_PREVIEW_DURATION_SEC`` seconds (seeded with ``seed=0``) is
        inspected and the maximum of ``child.bandwidth(sample_rate) +
        2*|freq_offset_hz|`` is returned. A fixed-seed preview can miss rare
        wide-bandwidth modes in a stochastic schedule — pass
        ``default_bandwidth_hz`` to the constructor when that matters.
        """
        if self._default_bandwidth_hz is not None:
            return float(self._default_bandwidth_hz)

        total_samples = int(sample_rate * _PREVIEW_DURATION_SEC)
        max_bw = 0.0
        for spec in self._schedule.iter_segments(
            total_samples=total_samples, sample_rate=sample_rate, seed=0
        ):
            child_bw = float(spec.waveform.bandwidth(sample_rate))
            envelope = child_bw + 2.0 * abs(float(spec.freq_offset_hz))
            if envelope > max_bw:
                max_bw = envelope
        if max_bw == 0.0:
            return float(sample_rate * _FALLBACK_BANDWIDTH_FRACTION)
        return max_bw
