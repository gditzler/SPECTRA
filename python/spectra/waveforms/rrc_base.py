"""Shared base class for RRC-filtered waveforms (PSK, QAM, ASK)."""

from abc import abstractmethod
from typing import Optional

import numpy as np

from spectra._rust import apply_rrc_filter_with_taps
from spectra.utils.rrc_cache import cached_rrc_taps
from spectra.waveforms.base import Waveform
from spectra.waveforms.physical import resample_to_rate, resolve_symbol_rate


class _RRCWaveformBase(Waveform):
    """Base class for waveforms that pulse-shape symbols through an RRC filter.

    Generates discrete symbols, upsamples by ``samples_per_symbol``, then
    convolves with a Root-Raised-Cosine (RRC) filter. The matched RRC at the
    receiver yields a Raised-Cosine response with zero ISI at symbol boundaries.

    Subclasses must define ``label`` and implement ``_generate_symbols()``.

    Args:
        rolloff: RRC excess bandwidth factor in [0, 1]. Higher values widen
            the spectrum but reduce ISI sensitivity. Default 0.35.
        filter_span: Filter half-length in symbols (filter has
            ``2 * filter_span * samples_per_symbol + 1`` taps). Default 10.
        samples_per_symbol: Upsampling factor (samples per symbol). Default 8.
            Mutually exclusive with ``symbol_rate``.
        symbol_rate: Physical symbol rate in baud. When set, the
            samples-per-symbol value is derived from the sample rate at
            ``generate()`` time and ``bandwidth()`` becomes
            ``symbol_rate * (1 + rolloff)`` independent of sample rate.

    Note:
        Bandwidth = ``symbol_rate * (1 + rolloff)``
        where ``symbol_rate = sample_rate / samples_per_symbol`` in the
        legacy (sample-domain) parameterization.
    """

    def __init__(
        self,
        rolloff: float = 0.35,
        filter_span: int = 10,
        samples_per_symbol: Optional[int] = None,
        symbol_rate: Optional[float] = None,
    ):
        if symbol_rate is not None and samples_per_symbol is not None:
            raise ValueError(
                "symbol_rate and samples_per_symbol are mutually exclusive; "
                "pass one or the other"
            )
        self.rolloff = rolloff
        self.filter_span = filter_span
        self.symbol_rate = symbol_rate
        # Legacy attribute stays an int for external consumers; it is unused
        # (and meaningless) when symbol_rate is set.
        self.samples_per_symbol = 8 if samples_per_symbol is None else samples_per_symbol

    def _resolved_sps(self, sample_rate: float):
        """Return (sps, up, down) for this waveform at ``sample_rate``."""
        if self.symbol_rate is None:
            return self.samples_per_symbol, 1, 1
        return resolve_symbol_rate(sample_rate, self.symbol_rate)

    def bandwidth(self, sample_rate: float) -> float:
        if self.symbol_rate is not None:
            return self.symbol_rate * (1.0 + self.rolloff)
        symbol_rate = sample_rate / self.samples_per_symbol
        return symbol_rate * (1.0 + self.rolloff)

    def num_symbols_for(self, num_samples: int, sample_rate: float) -> int:
        if self.symbol_rate is None:
            return int(num_samples // self.samples_per_symbol)
        return int(num_samples * self.symbol_rate / sample_rate)

    @abstractmethod
    def _generate_symbols(self, num_symbols: int, seed: int) -> np.ndarray:
        """Return complex64 symbol array of length num_symbols."""
        ...

    def generate(
        self, num_symbols: int, sample_rate: float, seed: Optional[int] = None
    ) -> np.ndarray:
        if self.symbol_rate is not None and self.bandwidth(sample_rate) > sample_rate:
            raise ValueError(
                f"{self.label} bandwidth {self.bandwidth(sample_rate):g} Hz exceeds "
                f"sample_rate {sample_rate:g} Hz"
            )
        sps, up, down = self._resolved_sps(sample_rate)
        s = seed if seed is not None else np.random.randint(0, 2**32)
        symbols = self._generate_symbols(num_symbols, s)
        taps = cached_rrc_taps(self.rolloff, self.filter_span, sps)
        filtered = apply_rrc_filter_with_taps(symbols, taps, sps)
        if self.symbol_rate is None:
            # Legacy path: return the filter output unchanged (byte-identical).
            return filtered
        # Physical path: trim the RRC transient tail to an exact whole number
        # of symbols so the sample count reflects the requested duration, then
        # rational-resample to hit the exact physical symbol rate.
        filtered = filtered[: num_symbols * sps]
        return resample_to_rate(filtered, up, down)
