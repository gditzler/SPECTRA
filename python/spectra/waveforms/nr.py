"""5G NR waveform generators.

Implements NR_OFDM, NR_SSB, NR_PDSCH, NR_PUSCH, and NR_PRACH waveforms
following 3GPP TS 38.211 structure. Uses Rust-backed OFDM symbol generation
for performance.
"""

from typing import Optional

import numpy as np

from spectra._rust import (
    generate_nr_dmrs,
    generate_nr_ofdm_symbol,
    generate_nr_pss,
    generate_nr_sss,
)
from spectra.waveforms.base import Waveform

# ---------------------------------------------------------------------------
# Numerology table: mu -> (SCS_kHz, symbols_per_slot, slots_per_subframe)
# ---------------------------------------------------------------------------
_NUMEROLOGY = {
    0: (15, 14, 1),
    1: (30, 14, 2),
    2: (60, 14, 4),
    3: (120, 14, 8),
    4: (240, 14, 16),
}


# Subcarrier modulation constellations (unit-power normalized)
def _qam_constellation(order: int) -> np.ndarray:
    k = int(np.sqrt(order))
    pts = np.array(
        [complex(2 * i - k + 1, 2 * q - k + 1) for i in range(k) for q in range(k)],
        dtype=np.complex128,
    )
    return pts / np.sqrt(np.mean(np.abs(pts) ** 2))


_CONSTELLATIONS = {
    "qpsk": np.exp(1j * np.pi / 4 * np.array([1, 3, 5, 7])),
    "qam16": _qam_constellation(16),
    "qam64": _qam_constellation(64),
    "qam256": _qam_constellation(256),
}


def _nr_cp_lengths(fft_size: int, mu: int):
    """Return (cp_normal, cp_first) for a given FFT size and numerology."""
    cp_normal = fft_size * 144 // 2048
    cp_first = cp_normal + fft_size * 16 // 2048
    return cp_normal, cp_first


def _map_symbols(rng, constellation, count):
    """Generate random constellation symbols."""
    idx = rng.integers(0, len(constellation), size=count)
    return constellation[idx]


def _ofdm_modulate_symbol(fft_bins, fft_size, cp_length):
    """OFDM-modulate a single symbol via the Rust backend."""
    data = np.ascontiguousarray(fft_bins[:fft_size], dtype=np.complex64)
    return np.asarray(generate_nr_ofdm_symbol(fft_size, cp_length, data))


# ===================================================================
# NR_OFDM
# ===================================================================


class NR_OFDM(Waveform):
    """5G NR OFDM waveform with configurable numerology and resource blocks.

    Parameters
    ----------
    numerology : int
        Numerology index mu (0-4). Default 1 (30 kHz SCS).
    num_resource_blocks : int
        Number of resource blocks. Default 25.
    fft_size : int
        FFT size for OFDM modulation. Default 512.
    modulation : str
        Subcarrier modulation: "qpsk", "qam16", "qam64". Default "qpsk".
    """

    def __init__(
        self,
        numerology: int = 1,
        num_resource_blocks: int = 25,
        fft_size: int = 512,
        modulation: str = "qpsk",
    ):
        if numerology not in _NUMEROLOGY:
            raise ValueError(f"numerology must be 0-4, got {numerology}")
        mod_key = modulation.lower()
        if mod_key not in _CONSTELLATIONS:
            raise ValueError(
                f"Unsupported modulation: {modulation}. Choose from {list(_CONSTELLATIONS.keys())}"
            )
        self._numerology = numerology
        self._num_rbs = num_resource_blocks
        self._fft_size = fft_size
        self._modulation = mod_key
        self._scs_khz, self._symbols_per_slot, _ = _NUMEROLOGY[numerology]
        self._num_subcarriers = num_resource_blocks * 12
        cp_normal, cp_first = _nr_cp_lengths(fft_size, numerology)
        self._cp_normal = cp_normal
        self._cp_first = cp_first
        # Average samples per OFDM symbol (slot)
        self.samples_per_symbol = (fft_size + cp_first) + (self._symbols_per_slot - 1) * (
            fft_size + cp_normal
        )

    def generate(
        self,
        num_symbols: int,
        sample_rate: float,
        seed: Optional[int] = None,
    ) -> np.ndarray:
        s = seed if seed is not None else np.random.randint(0, 2**32)
        rng = np.random.default_rng(s)
        constellation = _CONSTELLATIONS[self._modulation]
        segments = []

        for slot_idx in range(num_symbols):
            for sym_idx in range(self._symbols_per_slot):
                # First symbol of each half-subframe gets extended CP
                is_first = sym_idx == 0 or sym_idx == self._symbols_per_slot // 2
                cp_len = self._cp_first if is_first else self._cp_normal

                # Build frequency-domain bins
                fft_bins = np.zeros(self._fft_size, dtype=np.complex64)
                data = _map_symbols(rng, constellation, self._num_subcarriers)

                # Place subcarriers symmetrically around DC
                half = self._num_subcarriers // 2
                fft_bins[1 : half + 1] = data[:half]
                fft_bins[self._fft_size - half :] = data[half:]

                sym = _ofdm_modulate_symbol(fft_bins, self._fft_size, cp_len)
                segments.append(sym)

        return np.concatenate(segments).astype(np.complex64)

    def bandwidth(self, sample_rate: float) -> float:
        scs_hz = self._scs_khz * 1000
        return self._num_rbs * 12 * scs_hz

    @property
    def label(self) -> str:
        return "NR_OFDM"

    @classmethod
    def fr1_20mhz(cls) -> "NR_OFDM":
        """Preset for FR1 20 MHz channel (mu=1, 51 RBs, 1024-pt FFT)."""
        return cls(numerology=1, num_resource_blocks=51, fft_size=1024)


# ===================================================================
# NR_SSB
# ===================================================================


class NR_SSB(Waveform):
    """5G NR Synchronization Signal Block (SSB).

    Builds a 240-subcarrier x 4-symbol grid with PSS, SSS, and PBCH data.

    Parameters
    ----------
    n_id_1 : int
        Physical cell identity group (0-335).
    n_id_2 : int
        Physical cell identity sector (0-2).
    numerology : int
        Numerology index mu. Default 1.
    """

    def __init__(
        self,
        n_id_1: int = 0,
        n_id_2: int = 0,
        numerology: int = 1,
    ):
        if n_id_1 < 0 or n_id_1 > 335:
            raise ValueError(f"n_id_1 must be 0..335, got {n_id_1}")
        if n_id_2 < 0 or n_id_2 > 2:
            raise ValueError(f"n_id_2 must be 0..2, got {n_id_2}")
        if numerology not in _NUMEROLOGY:
            raise ValueError(f"numerology must be 0-4, got {numerology}")
        self._n_id_1 = n_id_1
        self._n_id_2 = n_id_2
        self._numerology = numerology
        self._fft_size = 512  # SSB uses 240 subcarriers within a larger FFT
        self._scs_khz = _NUMEROLOGY[numerology][0]
        cp_normal, cp_first = _nr_cp_lengths(self._fft_size, numerology)
        self._cp_normal = cp_normal
        self._cp_first = cp_first
        self.samples_per_symbol = 4 * (self._fft_size + cp_normal)

    def generate(
        self,
        num_symbols: int,
        sample_rate: float,
        seed: Optional[int] = None,
    ) -> np.ndarray:
        s = seed if seed is not None else np.random.randint(0, 2**32)
        rng = np.random.default_rng(s)

        # Get PSS and SSS sequences from Rust
        pss = np.asarray(generate_nr_pss(self._n_id_2))
        sss = np.asarray(generate_nr_sss(self._n_id_1, self._n_id_2))

        qpsk = _CONSTELLATIONS["qpsk"]
        segments = []

        for _ in range(num_symbols):
            for sym_idx in range(4):
                fft_bins = np.zeros(self._fft_size, dtype=np.complex64)

                # SSB occupies 240 subcarriers (20 RBs)
                ssb_grid = np.zeros(240, dtype=np.complex64)

                if sym_idx == 0:
                    # PSS in subcarriers 56-182 (127 subcarriers)
                    ssb_grid[56:183] = pss.astype(np.complex64)
                    # Fill remaining with QPSK (PBCH candidates)
                    ssb_grid[:56] = _map_symbols(rng, qpsk, 56).astype(np.complex64)
                    ssb_grid[183:] = _map_symbols(rng, qpsk, 57).astype(np.complex64)
                elif sym_idx == 2:
                    # SSS in subcarriers 56-182 (127 subcarriers)
                    ssb_grid[56:183] = sss.astype(np.complex64)
                    ssb_grid[:56] = _map_symbols(rng, qpsk, 56).astype(np.complex64)
                    ssb_grid[183:] = _map_symbols(rng, qpsk, 57).astype(np.complex64)
                else:
                    # Symbols 1 and 3: PBCH data (QPSK)
                    ssb_grid[:] = _map_symbols(rng, qpsk, 240).astype(np.complex64)

                # Map 240 subcarriers symmetrically into FFT bins
                half = 120
                fft_bins[1 : half + 1] = ssb_grid[:half]
                fft_bins[self._fft_size - half :] = ssb_grid[half:]

                cp_len = self._cp_first if sym_idx == 0 else self._cp_normal
                sym = _ofdm_modulate_symbol(fft_bins, self._fft_size, cp_len)
                segments.append(sym)

        return np.concatenate(segments).astype(np.complex64)

    def bandwidth(self, sample_rate: float) -> float:
        scs_hz = self._scs_khz * 1000
        return 240 * scs_hz  # SSB is always 240 subcarriers = 20 RBs

    @property
    def label(self) -> str:
        return "NR_SSB"

    @classmethod
    def n78(cls) -> "NR_SSB":
        """Preset for n78 band (mu=1, 30 kHz SCS)."""
        return cls(n_id_1=0, n_id_2=0, numerology=1)


# ===================================================================
# NR_PDSCH
# ===================================================================


class NR_PDSCH(Waveform):
    """5G NR Physical Downlink Shared Channel.

    Parameters
    ----------
    numerology : int
        Numerology index mu (0-4). Default 1.
    num_resource_blocks : int
        Number of resource blocks. Default 25.
    modulation : str
        Subcarrier modulation. Default "qpsk".
    num_dmrs_symbols : int
        Number of DMRS symbols per slot. Default 2.
    """

    def __init__(
        self,
        numerology: int = 1,
        num_resource_blocks: int = 25,
        modulation: str = "qpsk",
        num_dmrs_symbols: int = 2,
    ):
        if numerology not in _NUMEROLOGY:
            raise ValueError(f"numerology must be 0-4, got {numerology}")
        mod_key = modulation.lower()
        if mod_key not in _CONSTELLATIONS:
            raise ValueError(f"Unsupported modulation: {modulation}")
        self._numerology = numerology
        self._num_rbs = num_resource_blocks
        self._fft_size = max(256, 2 ** int(np.ceil(np.log2(num_resource_blocks * 12 + 1))))
        self._modulation = mod_key
        self._num_dmrs_symbols = num_dmrs_symbols
        self._scs_khz, self._symbols_per_slot, _ = _NUMEROLOGY[numerology]
        self._num_subcarriers = num_resource_blocks * 12
        cp_normal, cp_first = _nr_cp_lengths(self._fft_size, numerology)
        self._cp_normal = cp_normal
        self._cp_first = cp_first
        # DMRS symbol indices (simplified: symbols 2 and 11 for type A)
        self._dmrs_symbol_indices = list(range(2, 2 + num_dmrs_symbols))
        self.samples_per_symbol = (self._fft_size + cp_first) + (self._symbols_per_slot - 1) * (
            self._fft_size + cp_normal
        )

    def generate(
        self,
        num_symbols: int,
        sample_rate: float,
        seed: Optional[int] = None,
    ) -> np.ndarray:
        s = seed if seed is not None else np.random.randint(0, 2**32)
        rng = np.random.default_rng(s)
        constellation = _CONSTELLATIONS[self._modulation]
        n_id = s % 1008  # simplified cell ID derivation
        segments = []

        for slot_idx in range(num_symbols):
            for sym_idx in range(self._symbols_per_slot):
                is_first = sym_idx == 0 or sym_idx == self._symbols_per_slot // 2
                cp_len = self._cp_first if is_first else self._cp_normal

                fft_bins = np.zeros(self._fft_size, dtype=np.complex64)

                if sym_idx in self._dmrs_symbol_indices:
                    # DMRS symbol: use Rust DMRS generator
                    dmrs = np.asarray(generate_nr_dmrs(self._num_rbs, n_id, slot_idx, sym_idx, s))
                    # Place DMRS on every other subcarrier (type 1 mapping)
                    subcarrier_data = np.zeros(self._num_subcarriers, dtype=np.complex64)
                    # DMRS on even subcarriers, data on odd
                    dmrs_indices = np.arange(0, self._num_subcarriers, 2)
                    data_indices = np.arange(1, self._num_subcarriers, 2)
                    dmrs_len = min(len(dmrs), len(dmrs_indices))
                    subcarrier_data[dmrs_indices[:dmrs_len]] = dmrs[:dmrs_len].astype(np.complex64)
                    data_syms = _map_symbols(rng, constellation, len(data_indices))
                    subcarrier_data[data_indices] = data_syms.astype(np.complex64)
                else:
                    # Data symbol
                    subcarrier_data = _map_symbols(
                        rng, constellation, self._num_subcarriers
                    ).astype(np.complex64)

                half = self._num_subcarriers // 2
                fft_bins[1 : half + 1] = subcarrier_data[:half]
                fft_bins[self._fft_size - half :] = subcarrier_data[half:]

                sym = _ofdm_modulate_symbol(fft_bins, self._fft_size, cp_len)
                segments.append(sym)

        return np.concatenate(segments).astype(np.complex64)

    def bandwidth(self, sample_rate: float) -> float:
        scs_hz = self._scs_khz * 1000
        return self._num_rbs * 12 * scs_hz

    @property
    def label(self) -> str:
        return "NR_PDSCH"


# ===================================================================
# NR_PUSCH
# ===================================================================


class NR_PUSCH(Waveform):
    """5G NR Physical Uplink Shared Channel.

    Parameters
    ----------
    numerology : int
        Numerology index mu (0-4). Default 1.
    num_resource_blocks : int
        Number of resource blocks. Default 25.
    modulation : str
        Subcarrier modulation. Default "qpsk".
    transform_precoding : bool
        If True, apply DFT precoding (SC-FDMA / DFT-s-OFDM). Default False.
    """

    def __init__(
        self,
        numerology: int = 1,
        num_resource_blocks: int = 25,
        modulation: str = "qpsk",
        transform_precoding: bool = False,
    ):
        if numerology not in _NUMEROLOGY:
            raise ValueError(f"numerology must be 0-4, got {numerology}")
        mod_key = modulation.lower()
        if mod_key not in _CONSTELLATIONS:
            raise ValueError(f"Unsupported modulation: {modulation}")
        self._numerology = numerology
        self._num_rbs = num_resource_blocks
        self._fft_size = max(256, 2 ** int(np.ceil(np.log2(num_resource_blocks * 12 + 1))))
        self._modulation = mod_key
        self._transform_precoding = transform_precoding
        self._scs_khz, self._symbols_per_slot, _ = _NUMEROLOGY[numerology]
        self._num_subcarriers = num_resource_blocks * 12
        cp_normal, cp_first = _nr_cp_lengths(self._fft_size, numerology)
        self._cp_normal = cp_normal
        self._cp_first = cp_first
        # DMRS at symbol 2 for uplink
        self._dmrs_symbol_indices = [2]
        self.samples_per_symbol = (self._fft_size + cp_first) + (self._symbols_per_slot - 1) * (
            self._fft_size + cp_normal
        )

    def generate(
        self,
        num_symbols: int,
        sample_rate: float,
        seed: Optional[int] = None,
    ) -> np.ndarray:
        s = seed if seed is not None else np.random.randint(0, 2**32)
        rng = np.random.default_rng(s)
        constellation = _CONSTELLATIONS[self._modulation]
        n_id = s % 1008
        segments = []

        for slot_idx in range(num_symbols):
            for sym_idx in range(self._symbols_per_slot):
                is_first = sym_idx == 0 or sym_idx == self._symbols_per_slot // 2
                cp_len = self._cp_first if is_first else self._cp_normal

                fft_bins = np.zeros(self._fft_size, dtype=np.complex64)

                if sym_idx in self._dmrs_symbol_indices:
                    dmrs = np.asarray(generate_nr_dmrs(self._num_rbs, n_id, slot_idx, sym_idx, s))
                    subcarrier_data = np.zeros(self._num_subcarriers, dtype=np.complex64)
                    dmrs_indices = np.arange(0, self._num_subcarriers, 2)
                    data_indices = np.arange(1, self._num_subcarriers, 2)
                    dmrs_len = min(len(dmrs), len(dmrs_indices))
                    subcarrier_data[dmrs_indices[:dmrs_len]] = dmrs[:dmrs_len].astype(np.complex64)
                    data_syms = _map_symbols(rng, constellation, len(data_indices))
                    subcarrier_data[data_indices] = data_syms.astype(np.complex64)
                else:
                    subcarrier_data = _map_symbols(
                        rng, constellation, self._num_subcarriers
                    ).astype(np.complex64)

                if self._transform_precoding:
                    # DFT precoding (SC-FDMA): apply DFT before IFFT, normalize power
                    subcarrier_data = (
                        np.fft.fft(subcarrier_data) / np.sqrt(self._num_subcarriers)
                    ).astype(np.complex64)

                half = self._num_subcarriers // 2
                fft_bins[1 : half + 1] = subcarrier_data[:half]
                fft_bins[self._fft_size - half :] = subcarrier_data[half:]

                sym = _ofdm_modulate_symbol(fft_bins, self._fft_size, cp_len)
                segments.append(sym)

        return np.concatenate(segments).astype(np.complex64)

    def bandwidth(self, sample_rate: float) -> float:
        scs_hz = self._scs_khz * 1000
        return self._num_rbs * 12 * scs_hz

    @property
    def label(self) -> str:
        return "NR_PUSCH"


# ===================================================================
# NR_PRACH
# ===================================================================


class NR_PRACH(Waveform):
    """5G NR Physical Random Access Channel.

    Uses Zadoff-Chu sequences for preamble generation.

    Parameters
    ----------
    preamble_format : int
        PRACH preamble format (0-3 for long, 4+ for short). Default 0.
    root_index : int
        Zadoff-Chu root sequence index. Default 1.
    cyclic_shift : int
        Cyclic shift applied to the ZC sequence. Default 0.
    """

    # Sequence lengths per format group
    _LONG_SEQ_LEN = 839
    _SHORT_SEQ_LEN = 139

    def __init__(
        self,
        preamble_format: int = 0,
        root_index: int = 1,
        cyclic_shift: int = 0,
    ):
        self._preamble_format = preamble_format
        self._root_index = root_index
        self._cyclic_shift = cyclic_shift

        if preamble_format <= 3:
            self._seq_len = self._LONG_SEQ_LEN
            self._scs_khz = 1.25  # Long formats use 1.25 kHz SCS
        else:
            self._seq_len = self._SHORT_SEQ_LEN
            self._scs_khz = 15.0  # Short formats use 15 or 30 kHz

        # Validate root index
        if root_index < 1 or root_index >= self._seq_len:
            raise ValueError(f"root_index must be in [1, {self._seq_len - 1}], got {root_index}")

        # FFT size for PRACH OFDM modulation
        self._fft_size = 2 ** int(np.ceil(np.log2(self._seq_len + 1)))
        self._cp_length = self._fft_size // 8  # Simplified CP
        self.samples_per_symbol = self._fft_size + self._cp_length

    def _generate_zc_sequence(self) -> np.ndarray:
        """Generate Zadoff-Chu sequence with cyclic shift."""
        n = np.arange(self._seq_len)
        u = self._root_index
        N = self._seq_len
        # ZC sequence for odd N_zc
        phase = -np.pi * u * n * (n + 1) / N
        zc = np.exp(1j * phase).astype(np.complex64)

        # Apply cyclic shift
        if self._cyclic_shift > 0:
            zc = np.roll(zc, -self._cyclic_shift)

        return zc

    def generate(
        self,
        num_symbols: int,
        sample_rate: float,
        seed: Optional[int] = None,
    ) -> np.ndarray:
        zc = self._generate_zc_sequence()
        segments = []

        for _ in range(num_symbols):
            # Place ZC sequence into FFT bins
            fft_bins = np.zeros(self._fft_size, dtype=np.complex64)
            fft_bins[1 : self._seq_len + 1] = zc

            sym = _ofdm_modulate_symbol(fft_bins, self._fft_size, self._cp_length)
            segments.append(sym)

        return np.concatenate(segments).astype(np.complex64)

    def bandwidth(self, sample_rate: float) -> float:
        return self._seq_len * self._scs_khz * 1000

    @property
    def label(self) -> str:
        return "NR_PRACH"

    @classmethod
    def format_0(cls) -> "NR_PRACH":
        """Preset for PRACH preamble format 0 (long sequence)."""
        return cls(preamble_format=0, root_index=1, cyclic_shift=0)
