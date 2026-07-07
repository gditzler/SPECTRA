from typing import Optional

import numpy as np

from spectra.waveforms.base import Waveform

BARKER_CODES = {
    2: [+1, -1],
    3: [+1, +1, -1],
    4: [+1, +1, -1, +1],
    5: [+1, +1, +1, -1, +1],
    7: [+1, +1, +1, -1, -1, +1, -1],
    11: [+1, +1, +1, -1, -1, -1, +1, -1, -1, +1, -1],
    13: [+1, +1, +1, +1, +1, -1, -1, +1, +1, -1, +1, -1, +1],
}


class BarkerCode(Waveform):
    """Barker code radar waveform with optimal sidelobe properties."""

    def __init__(
        self,
        length: int = 13,
        samples_per_chip: Optional[int] = None,
        chip_rate: Optional[float] = None,
    ):
        if length not in BARKER_CODES:
            valid = sorted(BARKER_CODES.keys())
            raise ValueError(f"Barker code length must be one of {valid}, got {length}")
        if chip_rate is not None and samples_per_chip is not None:
            raise ValueError("chip_rate and samples_per_chip are mutually exclusive")
        self._length = length
        self._code = np.array(BARKER_CODES[length], dtype=np.float32)
        self._chip_rate = chip_rate
        self._samples_per_chip = 8 if samples_per_chip is None else samples_per_chip
        self.samples_per_symbol = length * self._samples_per_chip

    def _resolved_spc(self, sample_rate: float) -> int:
        if self._chip_rate is None:
            return self._samples_per_chip
        from spectra.waveforms.physical import resolve_symbol_rate

        spc, _, _ = resolve_symbol_rate(sample_rate, self._chip_rate)
        return spc

    def generate(
        self,
        num_symbols: int,
        sample_rate: float,
        seed: Optional[int] = None,
    ) -> np.ndarray:
        spc = self._resolved_spc(sample_rate)
        # Upsample code chips
        chips_up = np.repeat(self._code, spc)
        # Tile for num_symbols repetitions
        signal = np.tile(chips_up, num_symbols)
        return (signal + 0j).astype(np.complex64)

    def bandwidth(self, sample_rate: float) -> float:
        if self._chip_rate is not None:
            return self._chip_rate
        return sample_rate / self._samples_per_chip

    def num_symbols_for(self, num_samples: int, sample_rate: float) -> int:
        return max(1, int(num_samples // (self._length * self._resolved_spc(sample_rate))))

    @property
    def label(self) -> str:
        return "Barker"
