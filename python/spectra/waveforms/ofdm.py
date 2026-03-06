from typing import List, Optional, Tuple

import numpy as np

from spectra.waveforms.base import Waveform


def _qam_constellation(order: int) -> np.ndarray:
    """Generate a square QAM constellation of given order."""
    k = int(np.sqrt(order))
    points = np.array(
        [complex(2 * i - k + 1, 2 * q - k + 1) for i in range(k) for q in range(k)],
        dtype=np.complex128,
    )
    return points / np.sqrt(np.mean(np.abs(points) ** 2))


_CONSTELLATIONS = {
    "BPSK": np.array([-1.0 + 0j, 1.0 + 0j]),
    "QPSK": np.exp(1j * np.pi / 4 * np.array([1, 3, 5, 7])),
    "QAM16": _qam_constellation(16),
    "QAM64": _qam_constellation(64),
}


class OFDM(Waveform):
    """OFDM waveform with configurable subcarriers and cyclic prefix.

    Uses QPSK subcarrier modulation. Active subcarriers are placed
    symmetrically around DC in the frequency domain.
    """

    def __init__(
        self,
        num_subcarriers: int = 64,
        fft_size: int = 256,
        cp_length: int = 16,
        modulation: str = "QPSK",
        guard_bands: Tuple[int, int] = (0, 0),
        dc_null: bool = False,
        pilot_indices: Optional[List[int]] = None,
        pilot_value: complex = 1 + 0j,
    ):
        self._num_subcarriers = num_subcarriers
        self._fft_size = fft_size
        self._cp_length = cp_length
        self._modulation = modulation
        self._guard_bands = guard_bands
        self._dc_null = dc_null
        self._pilot_indices = pilot_indices
        self._pilot_value = pilot_value
        self.samples_per_symbol = fft_size + cp_length

    def _get_constellation(self) -> np.ndarray:
        if self._modulation not in _CONSTELLATIONS:
            raise ValueError(
                f"Unsupported modulation: {self._modulation}. "
                f"Choose from {list(_CONSTELLATIONS.keys())}"
            )
        return _CONSTELLATIONS[self._modulation]

    def _build_active_mask(self) -> np.ndarray:
        """Return boolean mask over num_subcarriers indicating active data carriers."""
        mask = np.ones(self._num_subcarriers, dtype=bool)
        # Guard bands: lower guard removes first N, upper guard removes last N
        lower, upper = self._guard_bands
        if lower > 0:
            mask[:lower] = False
        if upper > 0:
            mask[self._num_subcarriers - upper :] = False
        # DC null: the subcarrier at index num_subcarriers//2 maps to DC
        if self._dc_null:
            dc_idx = self._num_subcarriers // 2
            if dc_idx < self._num_subcarriers:
                mask[dc_idx] = False
        # Pilot indices
        if self._pilot_indices:
            for idx in self._pilot_indices:
                if 0 <= idx < self._num_subcarriers:
                    mask[idx] = False
        return mask

    def generate(
        self,
        num_symbols: int,
        sample_rate: float,
        seed: Optional[int] = None,
    ) -> np.ndarray:
        rng = np.random.default_rng(seed)
        constellation = self._get_constellation()
        active_mask = self._build_active_mask()
        num_active = int(np.sum(active_mask))
        segments = []

        for _ in range(num_symbols):
            # Generate data symbols for active subcarriers
            subcarrier_symbols = np.zeros(self._num_subcarriers, dtype=np.complex128)

            # Data on active subcarriers
            if num_active > 0:
                data_indices = rng.integers(0, len(constellation), size=num_active)
                subcarrier_symbols[active_mask] = constellation[data_indices]

            # Place pilots
            if self._pilot_indices:
                for idx in self._pilot_indices:
                    if 0 <= idx < self._num_subcarriers:
                        subcarrier_symbols[idx] = self._pilot_value

            # Place subcarriers symmetrically around DC in FFT bins
            fft_bins = np.zeros(self._fft_size, dtype=np.complex128)
            half = self._num_subcarriers // 2
            # Positive frequencies: bins 1..half
            fft_bins[1 : half + 1] = subcarrier_symbols[:half]
            # Negative frequencies: bins fft_size-half..fft_size-1
            fft_bins[self._fft_size - half :] = subcarrier_symbols[half:]

            # IFFT to time domain
            time_signal = np.fft.ifft(fft_bins)

            # Add cyclic prefix
            if self._cp_length > 0:
                cp = time_signal[-self._cp_length :]
                ofdm_symbol = np.concatenate([cp, time_signal])
            else:
                ofdm_symbol = time_signal

            segments.append(ofdm_symbol)

        iq = np.concatenate(segments).astype(np.complex64)
        return iq

    def bandwidth(self, sample_rate: float) -> float:
        active = self._num_subcarriers - self._guard_bands[0] - self._guard_bands[1]
        if self._dc_null:
            active -= 1
        if self._pilot_indices:
            active -= len(self._pilot_indices)
        return max(active, 1) * sample_rate / self._fft_size

    @property
    def label(self) -> str:
        return "OFDM"


class SCFDMA(OFDM):
    """SC-FDMA (DFT-spread OFDM) waveform.

    Adds M-point DFT precoding before IFFT for lower PAPR.
    """

    def generate(
        self,
        num_symbols: int,
        sample_rate: float,
        seed: Optional[int] = None,
    ) -> np.ndarray:
        rng = np.random.default_rng(seed)
        constellation = self._get_constellation()
        active_mask = self._build_active_mask()
        num_active = int(np.sum(active_mask))
        segments = []

        for _ in range(num_symbols):
            subcarrier_symbols = np.zeros(self._num_subcarriers, dtype=np.complex128)

            # Data on active subcarriers
            if num_active > 0:
                data_indices = rng.integers(0, len(constellation), size=num_active)
                data_syms = constellation[data_indices]
                # DFT precoding: spread data symbols across frequency
                data_syms = np.fft.fft(data_syms)
                subcarrier_symbols[active_mask] = data_syms

            # Place pilots
            if self._pilot_indices:
                for idx in self._pilot_indices:
                    if 0 <= idx < self._num_subcarriers:
                        subcarrier_symbols[idx] = self._pilot_value

            # Place subcarriers symmetrically around DC in FFT bins
            fft_bins = np.zeros(self._fft_size, dtype=np.complex128)
            half = self._num_subcarriers // 2
            fft_bins[1 : half + 1] = subcarrier_symbols[:half]
            fft_bins[self._fft_size - half :] = subcarrier_symbols[half:]

            # IFFT to time domain
            time_signal = np.fft.ifft(fft_bins)

            # Add cyclic prefix
            if self._cp_length > 0:
                cp = time_signal[-self._cp_length :]
                ofdm_symbol = np.concatenate([cp, time_signal])
            else:
                ofdm_symbol = time_signal

            segments.append(ofdm_symbol)

        iq = np.concatenate(segments).astype(np.complex64)
        return iq

    @property
    def label(self) -> str:
        return "SC-FDMA"


class OFDM72(OFDM):
    def __init__(self, cp_length: int = 16):
        super().__init__(num_subcarriers=72, fft_size=128, cp_length=cp_length)

    @property
    def label(self) -> str:
        return "OFDM-72"


class OFDM128(OFDM):
    def __init__(self, cp_length: int = 16):
        super().__init__(num_subcarriers=128, fft_size=256, cp_length=cp_length)

    @property
    def label(self) -> str:
        return "OFDM-128"


class OFDM180(OFDM):
    def __init__(self, cp_length: int = 32):
        super().__init__(num_subcarriers=180, fft_size=256, cp_length=cp_length)

    @property
    def label(self) -> str:
        return "OFDM-180"


class OFDM256(OFDM):
    def __init__(self, cp_length: int = 32):
        super().__init__(num_subcarriers=256, fft_size=512, cp_length=cp_length)

    @property
    def label(self) -> str:
        return "OFDM-256"


class OFDM300(OFDM):
    def __init__(self, cp_length: int = 32):
        super().__init__(num_subcarriers=300, fft_size=512, cp_length=cp_length)

    @property
    def label(self) -> str:
        return "OFDM-300"


class OFDM512(OFDM):
    def __init__(self, cp_length: int = 64):
        super().__init__(num_subcarriers=512, fft_size=1024, cp_length=cp_length)

    @property
    def label(self) -> str:
        return "OFDM-512"


class OFDM600(OFDM):
    def __init__(self, cp_length: int = 64):
        super().__init__(num_subcarriers=600, fft_size=1024, cp_length=cp_length)

    @property
    def label(self) -> str:
        return "OFDM-600"


class OFDM900(OFDM):
    def __init__(self, cp_length: int = 128):
        super().__init__(num_subcarriers=900, fft_size=1024, cp_length=cp_length)

    @property
    def label(self) -> str:
        return "OFDM-900"


class OFDM1200(OFDM):
    def __init__(self, cp_length: int = 128):
        super().__init__(num_subcarriers=1200, fft_size=2048, cp_length=cp_length)

    @property
    def label(self) -> str:
        return "OFDM-1200"


class OFDM2048(OFDM):
    def __init__(self, cp_length: int = 256):
        super().__init__(num_subcarriers=2048, fft_size=4096, cp_length=cp_length)

    @property
    def label(self) -> str:
        return "OFDM-2048"
