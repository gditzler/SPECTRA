"""LinkSimulator: BER/SER/PER vs. Eb/N0 simulation."""

from __future__ import annotations

from typing import List, Optional

import numpy as np

from spectra._rust import (
    apply_rrc_filter_with_taps,
    generate_ask_symbols_with_indices,
    generate_bpsk_symbols_with_indices,
    generate_psk_symbols_with_indices,
    generate_qam_symbols_with_indices,
    generate_qpsk_symbols_with_indices,
)
from spectra.link.results import LinkResults
from spectra.metrics import bit_error_rate, packet_error_rate, symbol_error_rate
from spectra.receivers.base import Decoder, PassthroughDecoder
from spectra.receivers.coherent import CoherentReceiver, constellation_to_bits
from spectra.scene.signal_desc import SignalDescription
from spectra.utils.rrc_cache import cached_rrc_taps
from spectra.waveforms.base import Waveform


def _generate_with_indices(waveform: Waveform, num_symbols: int, seed: int):
    """Call the appropriate Rust _with_indices generator for a waveform."""
    label = waveform.label.upper()
    order = getattr(waveform, "_order", None)

    if label == "BPSK":
        return generate_bpsk_symbols_with_indices(num_symbols, seed)
    elif label == "QPSK":
        return generate_qpsk_symbols_with_indices(num_symbols, seed)
    elif label == "OOK" or "ASK" in label:
        m = order if order else 2
        return generate_ask_symbols_with_indices(num_symbols, m, seed)
    elif "QAM" in label:
        m = order if order else 16
        return generate_qam_symbols_with_indices(num_symbols, m, seed)
    elif "PSK" in label:
        m = order if order else 8
        return generate_psk_symbols_with_indices(num_symbols, m, seed)
    else:
        raise ValueError(f"Unsupported waveform for LinkSimulator: {waveform.label}")


class LinkSimulator:
    """Link-level simulator for BER/SER/PER vs. Eb/N0 curves.

    Args:
        waveform: Modulation waveform (PSK, square QAM, or ASK).
        channel: Optional impairments applied AFTER noise injection.
        decoder: FEC decoder. Default: PassthroughDecoder.
        num_symbols: Symbols per Eb/N0 point.
        packet_length: Bits per packet for PER.
        seed: Base seed for reproducibility.
    """

    def __init__(
        self,
        waveform: Waveform,
        channel: Optional[List] = None,
        decoder: Optional[Decoder] = None,
        num_symbols: int = 10000,
        packet_length: int = 1000,
        seed: int = 0,
    ) -> None:
        self.waveform = waveform
        self.channel = channel or []
        self.decoder = decoder or PassthroughDecoder()
        self.num_symbols = num_symbols
        self.packet_length = packet_length
        self.seed = seed
        self.receiver = CoherentReceiver(waveform)

    def run(self, eb_n0_points: np.ndarray) -> LinkResults:
        """Sweep Eb/N0 values and return BER/SER/PER curves."""
        eb_n0_points = np.asarray(eb_n0_points, dtype=float)
        n_points = len(eb_n0_points)

        ber_arr = np.zeros(n_points)
        ser_arr = np.zeros(n_points)
        per_arr = np.zeros(n_points)

        sps = self.waveform.samples_per_symbol
        rolloff = getattr(self.waveform, "rolloff", 0.35)
        filter_span = getattr(self.waveform, "filter_span", 10)
        constellation_size = self.receiver.constellation_size
        bits_per_symbol = int(np.log2(constellation_size))

        # Transmit (same for all Eb/N0 points)
        symbols, tx_indices = _generate_with_indices(
            self.waveform, self.num_symbols, self.seed
        )
        symbols = np.asarray(symbols, dtype=np.complex64)
        tx_indices = np.asarray(tx_indices, dtype=np.uint32)

        # RRC pulse shaping
        taps = cached_rrc_taps(rolloff, filter_span, sps)
        tx_iq = np.array(apply_rrc_filter_with_taps(symbols, taps, sps))

        # Trim symbols affected by filter transients
        trim = filter_span
        valid_tx_indices = tx_indices[trim:-trim] if trim > 0 else tx_indices
        valid_tx_bits = constellation_to_bits(valid_tx_indices, constellation_size)

        for i, eb_n0_db in enumerate(eb_n0_points):
            # Noise injection (analytical, per-point seeded RNG)
            rng = np.random.default_rng((self.seed, i))
            signal_power = np.mean(np.abs(tx_iq) ** 2)
            eb = signal_power * sps / bits_per_symbol
            eb_n0_lin = 10.0 ** (eb_n0_db / 10.0)
            n0 = eb / eb_n0_lin
            noise_std = np.sqrt(n0 / 2.0)
            noise = noise_std * (
                rng.standard_normal(tx_iq.shape)
                + 1j * rng.standard_normal(tx_iq.shape)
            ).astype(np.complex64)
            rx_iq = tx_iq + noise

            # Optional channel impairments
            if self.channel:
                bw = self.waveform.bandwidth(1.0)
                desc = SignalDescription(
                    t_start=0.0,
                    t_stop=len(rx_iq) / sps,
                    f_low=-bw / 2, f_high=bw / 2,
                    label=self.waveform.label,
                    snr=float(eb_n0_db),
                )
                for transform in self.channel:
                    rx_iq, desc = transform(rx_iq, desc)

            # Demodulate
            rx_indices, rx_bits = self.receiver.demodulate(rx_iq)

            # Trim to valid region (align with TX)
            valid_rx_indices = rx_indices[trim : trim + len(valid_tx_indices)]
            valid_rx_bits = constellation_to_bits(valid_rx_indices, constellation_size)

            # Decode
            decoded_bits = self.decoder.decode(valid_rx_bits)

            # Score
            min_len_bits = min(len(valid_tx_bits), len(decoded_bits))
            min_len_sym = min(len(valid_tx_indices), len(valid_rx_indices))

            ber_arr[i] = bit_error_rate(
                valid_tx_bits[:min_len_bits], decoded_bits[:min_len_bits]
            )
            ser_arr[i] = symbol_error_rate(
                valid_tx_indices[:min_len_sym], valid_rx_indices[:min_len_sym]
            )
            per_arr[i] = packet_error_rate(
                valid_tx_bits[:min_len_bits], decoded_bits[:min_len_bits],
                self.packet_length,
            )

        return LinkResults(
            eb_n0_db=eb_n0_points,
            ber=ber_arr,
            ser=ser_arr,
            per=per_arr,
            num_bits=len(valid_tx_bits),
            num_symbols=len(valid_tx_indices),
            packet_length=self.packet_length,
            waveform_label=self.waveform.label,
        )
