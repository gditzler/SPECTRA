"""Aviation and maritime protocol waveforms.

Includes ADS-B, Mode S, AIS, ACARS, DME, and ILS Localizer waveform
generators for legacy aviation and maritime signal simulation.
"""

from typing import Optional

import numpy as np

from spectra._rust import (
    generate_acars_frame,
    generate_adsb_frame,
    generate_ais_frame,
    generate_mode_s_frame,
)
from spectra.waveforms.base import Waveform


def _bytes_to_bits(data: np.ndarray) -> np.ndarray:
    """Convert a byte array to a bit array (MSB first)."""
    bits = np.unpackbits(data)
    return bits


class ADSB(Waveform):
    """ADS-B (1090 MHz Extended Squitter) waveform.

    Generates PPM-encoded ADS-B signals with a standard 8 us preamble
    followed by 112-bit data. Each bit period is 1 us with PPM encoding:
    '1' = [1, 0], '0' = [0, 1] per half-bit (0.5 us chip).

    Args:
        samples_per_chip: Samples per 0.5 us chip period.
    """

    def __init__(self, samples_per_chip: int = 10):
        self.samples_per_chip = samples_per_chip

    def generate(
        self,
        num_symbols: int,
        sample_rate: float,
        seed: Optional[int] = None,
    ) -> np.ndarray:
        s = seed if seed is not None else np.random.randint(0, 2**32)
        spc = self.samples_per_chip

        # Generate num_symbols frames and concatenate
        frames = []
        for i in range(num_symbols):
            frame_seed = (s + i) & 0xFFFFFFFF
            frame_bytes = np.array(generate_adsb_frame(seed=frame_seed), dtype=np.uint8)
            bits = _bytes_to_bits(frame_bytes)[:112]  # 112 data bits

            # Preamble: 8 us = 16 chips (each chip 0.5 us)
            # Pulses at 0, 1, 3.5, 4.5 us -> chip positions 0, 2, 7, 9
            preamble_chips = np.zeros(16, dtype=np.float32)
            preamble_chips[0] = 1.0
            preamble_chips[2] = 1.0
            preamble_chips[7] = 1.0
            preamble_chips[9] = 1.0

            # Data: PPM encoding, 2 chips per bit
            # '1' -> [1, 0], '0' -> [0, 1]
            data_chips = np.zeros(112 * 2, dtype=np.float32)
            for b_idx in range(112):
                if bits[b_idx] == 1:
                    data_chips[b_idx * 2] = 1.0
                else:
                    data_chips[b_idx * 2 + 1] = 1.0

            # Concatenate preamble and data
            all_chips = np.concatenate([preamble_chips, data_chips])

            # Upsample to samples_per_chip
            signal = np.repeat(all_chips, spc)
            frames.append(signal)

        iq = np.concatenate(frames)
        return iq.astype(np.complex64)

    def bandwidth(self, sample_rate: float) -> float:
        return sample_rate / self.samples_per_chip

    @property
    def label(self) -> str:
        return "ADSB"


class ModeS(Waveform):
    """Mode S Interrogation/Reply waveform.

    Similar structure to ADS-B with configurable message length (56 or 112 bits)
    and PPM data encoding.

    Args:
        message_length: Number of data bits (56 or 112).
        samples_per_chip: Samples per 0.5 us chip period.
    """

    def __init__(self, message_length: int = 112, samples_per_chip: int = 10):
        if message_length not in (56, 112):
            raise ValueError("message_length must be 56 or 112")
        self._message_length = message_length
        self.samples_per_chip = samples_per_chip

    def generate(
        self,
        num_symbols: int,
        sample_rate: float,
        seed: Optional[int] = None,
    ) -> np.ndarray:
        s = seed if seed is not None else np.random.randint(0, 2**32)
        spc = self.samples_per_chip
        msg_len = self._message_length

        frames = []
        for i in range(num_symbols):
            frame_seed = (s + i) & 0xFFFFFFFF
            frame_bytes = np.array(
                generate_mode_s_frame(message_length=msg_len, seed=frame_seed),
                dtype=np.uint8,
            )
            bits = _bytes_to_bits(frame_bytes)[:msg_len]

            # Preamble: 8 us = 16 chips, pulses at 0, 2, 7, 9
            preamble_chips = np.zeros(16, dtype=np.float32)
            preamble_chips[0] = 1.0
            preamble_chips[2] = 1.0
            preamble_chips[7] = 1.0
            preamble_chips[9] = 1.0

            # Data: PPM encoding, 2 chips per bit
            data_chips = np.zeros(msg_len * 2, dtype=np.float32)
            for b_idx in range(msg_len):
                if bits[b_idx] == 1:
                    data_chips[b_idx * 2] = 1.0
                else:
                    data_chips[b_idx * 2 + 1] = 1.0

            all_chips = np.concatenate([preamble_chips, data_chips])
            signal = np.repeat(all_chips, spc)
            frames.append(signal)

        iq = np.concatenate(frames)
        return iq.astype(np.complex64)

    def bandwidth(self, sample_rate: float) -> float:
        return sample_rate / self.samples_per_chip

    @property
    def label(self) -> str:
        return "ModeS"


class AIS(Waveform):
    """Automatic Identification System (AIS) waveform.

    GMSK modulation with BT=0.4 at 9600 baud, used for maritime vessel
    identification and tracking.

    Args:
        samples_per_symbol: Samples per symbol period.
        bt: Gaussian filter bandwidth-time product.
        filter_span: Gaussian filter span in symbols.
    """

    def __init__(
        self,
        samples_per_symbol: int = 8,
        bt: float = 0.4,
        filter_span: int = 4,
    ):
        self.samples_per_symbol = samples_per_symbol
        self._bt = bt
        self._filter_span = filter_span

    def _gaussian_taps(self) -> np.ndarray:
        sps = self.samples_per_symbol
        half = self._filter_span * sps // 2
        t = np.arange(-half, half + 1) / sps
        bt = self._bt
        h = np.sqrt(2.0 * np.pi / np.log(2)) * bt * np.exp(-2.0 * (np.pi * bt * t) ** 2 / np.log(2))
        return h / np.sum(h)

    def generate(
        self,
        num_symbols: int,
        sample_rate: float,
        seed: Optional[int] = None,
    ) -> np.ndarray:
        s = seed if seed is not None else np.random.randint(0, 2**32)
        sps = self.samples_per_symbol

        frames = []
        for i in range(num_symbols):
            frame_seed = (s + i) & 0xFFFFFFFF
            frame_bytes = np.array(generate_ais_frame(seed=frame_seed), dtype=np.uint8)
            bits = _bytes_to_bits(frame_bytes)

            # NRZ encoding: 0 -> -1, 1 -> +1
            nrz = 2.0 * bits.astype(np.float32) - 1.0

            # Upsample with zero-insertion
            symbols_up = np.zeros(len(nrz) * sps, dtype=np.float32)
            symbols_up[::sps] = nrz

            # Gaussian filter
            h = self._gaussian_taps()
            filtered = np.convolve(symbols_up, h, mode="same")

            # Phase modulation (MSK: h=0.5)
            delta_phi = np.pi * 0.5 * filtered / sps
            phase = np.cumsum(delta_phi)
            signal = np.exp(1j * phase).astype(np.complex64)
            frames.append(signal)

        return np.concatenate(frames)

    def bandwidth(self, sample_rate: float) -> float:
        symbol_rate = sample_rate / self.samples_per_symbol
        return symbol_rate * (1.0 + self._bt)

    @property
    def label(self) -> str:
        return "AIS"


class ACARS(Waveform):
    """Aircraft Communications Addressing and Reporting System (ACARS) waveform.

    AM-MSK modulation at 2400 baud for air-ground data communications.

    Args:
        samples_per_symbol: Samples per symbol period.
    """

    def __init__(self, samples_per_symbol: int = 8):
        self.samples_per_symbol = samples_per_symbol

    def generate(
        self,
        num_symbols: int,
        sample_rate: float,
        seed: Optional[int] = None,
    ) -> np.ndarray:
        s = seed if seed is not None else np.random.randint(0, 2**32)
        sps = self.samples_per_symbol

        frames = []
        for i in range(num_symbols):
            frame_seed = (s + i) & 0xFFFFFFFF
            frame_bytes = np.array(generate_acars_frame(seed=frame_seed), dtype=np.uint8)
            bits = _bytes_to_bits(frame_bytes)

            # NRZ: 0 -> -1, 1 -> +1
            nrz = 2.0 * bits.astype(np.float32) - 1.0

            # Upsample
            freq_up = np.repeat(nrz, sps)

            # MSK: continuous-phase FSK with modulation index 0.5
            delta_phi = np.pi * 0.5 * freq_up / sps
            phase = np.cumsum(delta_phi)
            signal = np.exp(1j * phase).astype(np.complex64)
            frames.append(signal)

        return np.concatenate(frames)

    def bandwidth(self, sample_rate: float) -> float:
        symbol_rate = sample_rate / self.samples_per_symbol
        return symbol_rate * 1.5

    @property
    def label(self) -> str:
        return "ACARS"


class DME(Waveform):
    """Distance Measuring Equipment (DME) waveform.

    Generates Gaussian pulse pairs with configurable pulse width and spacing.
    Pure Python implementation (no Rust frame generator needed).

    Args:
        pulse_width_us: Gaussian pulse width in microseconds.
        pulse_spacing_us: Spacing between pulse pairs in microseconds.
        samples_per_us: Number of samples per microsecond.
    """

    def __init__(
        self,
        pulse_width_us: float = 3.5,
        pulse_spacing_us: float = 12.0,
        samples_per_us: int = 10,
    ):
        self._pulse_width_us = pulse_width_us
        self._pulse_spacing_us = pulse_spacing_us
        self._samples_per_us = samples_per_us

    def generate(
        self,
        num_symbols: int,
        sample_rate: float,
        seed: Optional[int] = None,
    ) -> np.ndarray:
        s = seed if seed is not None else np.random.randint(0, 2**32)
        rng = np.random.default_rng(s)

        spu = self._samples_per_us
        pw = self._pulse_width_us
        ps = self._pulse_spacing_us

        # Gaussian pulse parameters
        sigma = pw / (2.0 * np.sqrt(2.0 * np.log(2)))  # FWHM to sigma
        pulse_half_len = int(3.0 * sigma * spu)
        t_pulse = np.arange(-pulse_half_len, pulse_half_len + 1) / spu
        pulse = np.exp(-0.5 * (t_pulse / sigma) ** 2).astype(np.float32)

        spacing_samples = int(ps * spu)

        # Generate num_symbols pulse pairs with random inter-pair spacing
        segments = []
        for _ in range(num_symbols):
            # First pulse
            pair = np.zeros(spacing_samples + len(pulse), dtype=np.float32)
            pair[: len(pulse)] = pulse
            # Second pulse at pulse_spacing_us offset
            start = spacing_samples
            end = start + len(pulse)
            if end <= len(pair):
                pair[start:end] += pulse
            else:
                pair[start:] += pulse[: len(pair) - start]

            segments.append(pair)

            # Random inter-pair gap (20-100 us)
            gap_us = 20.0 + rng.random() * 80.0
            gap_samples = int(gap_us * spu)
            segments.append(np.zeros(gap_samples, dtype=np.float32))

        iq = np.concatenate(segments)
        return iq.astype(np.complex64)

    def bandwidth(self, sample_rate: float) -> float:
        # DME bandwidth approximation: ~1 MHz for standard pulse
        sigma = self._pulse_width_us / (2.0 * np.sqrt(2.0 * np.log(2)))
        # Gaussian pulse bandwidth ~ 1/(2*pi*sigma) in MHz, convert to Hz
        bw_mhz = 1.0 / (2.0 * np.pi * sigma)
        return min(bw_mhz * 1e6, sample_rate)

    @property
    def label(self) -> str:
        return "DME"


class ILS_Localizer(Waveform):
    """Instrument Landing System (ILS) Localizer waveform.

    AM carrier with 90 Hz and 150 Hz modulation tones. The Difference in
    Depth of Modulation (DDM) indicates lateral deviation from the runway
    centerline.

    Args:
        modulation_depth: Depth of AM modulation (0 to 1).
        samples_per_symbol: Samples per symbol (determines signal length).
    """

    def __init__(
        self,
        modulation_depth: float = 0.2,
        samples_per_symbol: int = 256,
    ):
        self._modulation_depth = modulation_depth
        self.samples_per_symbol = samples_per_symbol

    def generate(
        self,
        num_symbols: int,
        sample_rate: float,
        seed: Optional[int] = None,
    ) -> np.ndarray:
        s = seed if seed is not None else np.random.randint(0, 2**32)
        rng = np.random.default_rng(s)

        total_samples = num_symbols * self.samples_per_symbol
        t = np.arange(total_samples) / sample_rate

        # DDM varies slightly with random offset for realism
        ddm_offset = rng.uniform(-0.05, 0.05)
        m = self._modulation_depth

        # 90 Hz and 150 Hz modulation tones
        tone_90 = np.sin(2.0 * np.pi * 90.0 * t)
        tone_150 = np.sin(2.0 * np.pi * 150.0 * t)

        # AM modulated carrier
        # m90 and m150 differ by DDM to simulate off-centerline
        m90 = m * (1.0 + ddm_offset)
        m150 = m * (1.0 - ddm_offset)
        envelope = 1.0 + m90 * tone_90 + m150 * tone_150

        # Carrier (baseband representation)
        signal = envelope.astype(np.float32)
        return (signal + 0j).astype(np.complex64)

    def bandwidth(self, sample_rate: float) -> float:
        # ILS localizer is very narrowband: tones at 90 and 150 Hz
        return min(300.0, sample_rate)

    @property
    def label(self) -> str:
        return "ILS_Localizer"
