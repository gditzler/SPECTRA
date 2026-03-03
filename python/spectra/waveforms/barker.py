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

    def __init__(self, length: int = 13, samples_per_chip: int = 8):
        if length not in BARKER_CODES:
            valid = sorted(BARKER_CODES.keys())
            raise ValueError(
                f"Barker code length must be one of {valid}, got {length}"
            )
        self._length = length
        self._code = np.array(BARKER_CODES[length], dtype=np.float32)
        self._samples_per_chip = samples_per_chip
        self.samples_per_symbol = length * samples_per_chip

    def generate(
        self,
        num_symbols: int,
        sample_rate: float,
        seed: Optional[int] = None,
    ) -> np.ndarray:
        # Upsample code chips
        chips_up = np.repeat(self._code, self._samples_per_chip)
        # Tile for num_symbols repetitions
        signal = np.tile(chips_up, num_symbols)
        return (signal + 0j).astype(np.complex64)

    def bandwidth(self, sample_rate: float) -> float:
        return sample_rate / self._samples_per_chip

    @property
    def label(self) -> str:
        return "Barker"
