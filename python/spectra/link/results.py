"""LinkResults dataclass for BER/SER/PER simulation results."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Optional

import numpy as np


@dataclass
class LinkResults:
    """Results from a link-level simulation sweep."""

    eb_n0_db: np.ndarray
    ber: np.ndarray
    ser: np.ndarray
    per: np.ndarray
    num_bits: int
    num_symbols: int
    packet_length: int
    waveform_label: str

    def theoretical_ber(self) -> Optional[np.ndarray]:
        """Return closed-form AWGN BER for BPSK, else None.

        BPSK BER = 0.5 * erfc(sqrt(Eb/N0)).
        """
        if self.waveform_label.upper() != "BPSK":
            return None
        import math

        eb_n0_lin = 10.0 ** (self.eb_n0_db / 10.0)
        return np.array([0.5 * math.erfc(math.sqrt(x)) for x in eb_n0_lin])
