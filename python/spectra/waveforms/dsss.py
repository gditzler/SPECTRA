"""Direct-Sequence Spread Spectrum (DSSS) waveforms."""

from typing import Optional

import numpy as np

from spectra.waveforms.base import Waveform


def _msequence(order: int) -> np.ndarray:
    """Generate a maximal-length sequence (m-sequence) of length 2^order - 1.

    Uses standard polynomial taps for orders 2-10.  Returns a bipolar
    sequence of {-1, +1} values.
    """
    # Feedback polynomial taps (standard LFSR polynomials)
    taps = {
        2: [2, 1],
        3: [3, 1],
        4: [4, 1],
        5: [5, 2],
        6: [6, 1],
        7: [7, 1],
        8: [8, 6, 5, 4],
        9: [9, 4],
        10: [10, 3],
    }
    if order not in taps:
        raise ValueError(f"m-sequence order must be in {list(taps.keys())}, got {order}")

    length = (1 << order) - 1
    register = [1] * order
    sequence = np.empty(length, dtype=np.float32)

    for i in range(length):
        sequence[i] = register[-1]
        feedback = 0
        for tap in taps[order]:
            feedback ^= register[tap - 1]
        register = [feedback] + register[:-1]

    # Convert {0, 1} to {-1, +1}
    return 2.0 * sequence - 1.0


class DSSS_BPSK(Waveform):
    """Direct-Sequence Spread Spectrum with BPSK modulation.

    Generates a DSSS-BPSK signal by spreading random BPSK data symbols
    with a maximal-length PN sequence and upsampling to the desired
    samples-per-chip rate.

    This matches the test signal used in Li et al. (IEEE SPL 2015) for
    validating the S3CA algorithm.

    Parameters
    ----------
    processing_gain : int
        Length of the spreading code (PN sequence length).  Must be 2^n - 1
        for n in {2..10}.  Default 31 (order-5 m-sequence).
    samples_per_chip : int
        Number of samples per chip.  With normalized sample rate fs=1, the
        chip rate is 1/samples_per_chip.  Default 4 (chip_rate = 0.25).
    """

    def __init__(
        self,
        processing_gain: int = 31,
        samples_per_chip: int = 4,
    ):
        self.processing_gain = processing_gain
        self.samples_per_chip = samples_per_chip

        # Determine m-sequence order from processing_gain
        order = int(np.round(np.log2(processing_gain + 1)))
        if (1 << order) - 1 != processing_gain:
            raise ValueError(
                f"processing_gain must be 2^n - 1, got {processing_gain}"
            )
        self._pn_code = _msequence(order)

    @property
    def label(self) -> str:
        return "DSSS-BPSK"

    def bandwidth(self, sample_rate: float) -> float:
        chip_rate = sample_rate / self.samples_per_chip
        return chip_rate

    def generate(
        self,
        num_symbols: int,
        sample_rate: float,
        seed: Optional[int] = None,
    ) -> np.ndarray:
        """Generate DSSS-BPSK IQ samples.

        Parameters
        ----------
        num_symbols : int
            Number of data symbols (bits).  Total chips = num_symbols *
            processing_gain, total samples = chips * samples_per_chip.
        sample_rate : float
            Sample rate (used only for API compatibility; the signal is
            generated at ``samples_per_chip`` samples per chip regardless).
        seed : int, optional
            Random seed for reproducible data symbols.

        Returns
        -------
        np.ndarray
            Complex64 baseband IQ samples.
        """
        rng = np.random.RandomState(
            seed if seed is not None else np.random.randint(0, 2**32)
        )

        # BPSK data symbols: {-1, +1}
        data = 2 * rng.randint(0, 2, size=num_symbols).astype(np.float32) - 1.0

        # Spread: repeat each symbol and multiply by PN code
        chips = np.repeat(data, self.processing_gain) * np.tile(
            self._pn_code, num_symbols
        )

        # Upsample to samples_per_chip via sample-and-hold (rectangular pulse)
        samples = np.repeat(chips, self.samples_per_chip)

        return samples.astype(np.complex64)
