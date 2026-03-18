"""Coherent receiver with matched filter and nearest-neighbor slicer."""

from __future__ import annotations

from typing import Tuple

import numpy as np

from spectra._rust import (
    apply_rrc_filter_with_taps,
    get_bpsk_constellation,
    get_qpsk_constellation,
    get_psk_constellation,
    get_qam_constellation,
    get_ask_constellation,
)
from spectra.receivers.base import Receiver
from spectra.waveforms.base import Waveform
from spectra.utils.rrc_cache import cached_rrc_taps


def constellation_to_bits(indices: np.ndarray, constellation_size: int) -> np.ndarray:
    """Convert symbol indices to bits using natural binary mapping.

    Args:
        indices: Symbol indices, shape ``(N,)``, dtype uint32.
        constellation_size: Number of constellation points M.

    Returns:
        Flat bit array, shape ``(N * bits_per_symbol,)``, dtype uint8.
    """
    bits_per_symbol = int(np.log2(constellation_size))
    n = len(indices)
    bits = np.zeros(n * bits_per_symbol, dtype=np.uint8)
    for b in range(bits_per_symbol):
        bits[b::bits_per_symbol] = (indices >> (bits_per_symbol - 1 - b)) & 1
    return bits


def _get_constellation(waveform: Waveform) -> np.ndarray:
    """Get the reference constellation for a waveform from Rust."""
    label = waveform.label.upper()
    order = getattr(waveform, "_order", None)

    if label == "BPSK":
        return np.array(get_bpsk_constellation(), dtype=np.complex64)
    elif label == "QPSK":
        return np.array(get_qpsk_constellation(), dtype=np.complex64)
    elif label == "OOK" or "ASK" in label:
        m = order if order else 2
        return np.array(get_ask_constellation(m), dtype=np.complex64)
    elif "QAM" in label:
        m = order if order else 16
        return np.array(get_qam_constellation(m), dtype=np.complex64)
    elif "PSK" in label:
        m = order if order else 8
        return np.array(get_psk_constellation(m), dtype=np.complex64)
    else:
        raise ValueError(f"Unsupported waveform for CoherentReceiver: {waveform.label}")


class CoherentReceiver(Receiver):
    """Perfect-synchronization coherent receiver.

    Pipeline: RRC matched filter -> downsample -> nearest-neighbor slicer -> bit demap.

    Args:
        waveform: Transmit waveform (PSK, square QAM, or ASK).
    """

    def __init__(self, waveform: Waveform) -> None:
        self.waveform = waveform
        self.samples_per_symbol = waveform.samples_per_symbol
        self.rolloff = getattr(waveform, "rolloff", 0.35)
        self.filter_span = getattr(waveform, "filter_span", 10)
        self.constellation = _get_constellation(waveform)
        self.constellation_size = len(self.constellation)

    def demodulate(self, received_iq: np.ndarray) -> Tuple[np.ndarray, np.ndarray]:
        """Demodulate received IQ through matched filter + slicer.

        Returns:
            ``(symbol_indices, bits)`` -- uint32 and uint8 arrays.
        """
        # 1. RRC matched filter
        taps = cached_rrc_taps(self.rolloff, self.filter_span, self.samples_per_symbol)
        filtered = np.array(
            apply_rrc_filter_with_taps(
                np.asarray(received_iq, dtype=np.complex64), taps, 1
            )
        )

        # 2. Downsample to symbol rate (account for filter group delay)
        delay = self.filter_span * self.samples_per_symbol
        symbols = filtered[delay::self.samples_per_symbol]

        # 3. Nearest-neighbor constellation slicer
        const = self.constellation
        diffs = symbols[:, np.newaxis] - const[np.newaxis, :]
        distances = np.abs(diffs) ** 2
        rx_indices = np.argmin(distances, axis=1).astype(np.uint32)

        # 4. Bit demapper
        rx_bits = constellation_to_bits(rx_indices, self.constellation_size)

        return rx_indices, rx_bits
