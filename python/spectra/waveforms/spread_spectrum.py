"""Spread Spectrum waveform generators.

Provides DSSS (QPSK), FHSS, CDMA (forward/reverse link), and THSS waveforms.
"""

from typing import List, Optional

import numpy as np

from spectra._rust import (
    generate_bpsk_symbols,
    generate_gold_code,
    generate_kasami_code,
    generate_qpsk_symbols,
    generate_walsh_hadamard,
)
from spectra.waveforms.base import Waveform
from spectra.waveforms.dsss import _msequence


class DSSS_QPSK(Waveform):
    """Direct-Sequence Spread Spectrum with QPSK modulation.

    Generates a DSSS-QPSK signal by spreading random QPSK data symbols
    with a spreading code (m-sequence, Gold, or Kasami) and upsampling
    via sample-and-hold.

    Parameters
    ----------
    code_type : str
        Type of spreading code: "msequence", "gold", or "kasami".
    code_order : int
        Order of the spreading code (5-10). Code length is 2^order - 1.
    samples_per_chip : int
        Number of samples per chip. Default 4.
    code_index : int
        Index for Gold preferred pair or Kasami shift. Default 0.
    """

    def __init__(
        self,
        code_type: str = "msequence",
        code_order: int = 5,
        samples_per_chip: int = 4,
        code_index: int = 0,
    ):
        if code_type not in ("msequence", "gold", "kasami"):
            raise ValueError(
                f"code_type must be 'msequence', 'gold', or 'kasami', got '{code_type}'"
            )
        if code_type == "kasami" and code_order % 2 != 0:
            raise ValueError("Kasami codes require even code_order")
        self.code_type = code_type
        self.code_order = code_order
        self.samples_per_chip = samples_per_chip
        self.code_index = code_index
        self._code_len = (1 << code_order) - 1

        # Pre-generate the spreading code
        if code_type == "msequence":
            self._code = _msequence(code_order)
        elif code_type == "gold":
            self._code = np.array(generate_gold_code(code_order, code_index), dtype=np.float32)
        else:  # kasami
            self._code = np.array(generate_kasami_code(code_order, code_index), dtype=np.float32)

    @property
    def label(self) -> str:
        return "DSSS_QPSK"

    def bandwidth(self, sample_rate: float) -> float:
        chip_rate = sample_rate / self.samples_per_chip
        return chip_rate

    def generate(
        self,
        num_symbols: int,
        sample_rate: float,
        seed: Optional[int] = None,
    ) -> np.ndarray:
        s = seed if seed is not None else np.random.randint(0, 2**32)

        # Generate QPSK data symbols via Rust
        qpsk_symbols = np.array(generate_qpsk_symbols(num_symbols, int(s)), dtype=np.complex64)

        # Spread: repeat each QPSK symbol by code length, multiply by tiled code
        chips_i = np.repeat(qpsk_symbols.real, self._code_len) * np.tile(
            self._code, num_symbols
        )
        chips_q = np.repeat(qpsk_symbols.imag, self._code_len) * np.tile(
            self._code, num_symbols
        )
        chips = chips_i + 1j * chips_q

        # Upsample via sample-and-hold
        samples = np.repeat(chips, self.samples_per_chip)

        return samples.astype(np.complex64)


class FHSS(Waveform):
    """Frequency Hopping Spread Spectrum.

    Generates an FHSS signal where the carrier frequency hops across
    channels according to a hopping pattern.

    Parameters
    ----------
    num_channels : int
        Number of frequency channels to hop across. Default 8.
    hop_pattern : str
        Hopping pattern: "random", "linear", or "costas". Default "random".
    dwell_samples : int
        Number of samples per dwell (hop). Default 64.
    modulation : str
        Per-hop modulation: "bpsk", "qpsk", or "fsk". Default "bpsk".
    """

    def __init__(
        self,
        num_channels: int = 8,
        hop_pattern: str = "random",
        dwell_samples: int = 64,
        modulation: str = "bpsk",
    ):
        if hop_pattern not in ("random", "linear", "costas"):
            raise ValueError(
                f"hop_pattern must be 'random', 'linear', or 'costas', got '{hop_pattern}'"
            )
        if modulation not in ("bpsk", "qpsk", "fsk"):
            raise ValueError(
                f"modulation must be 'bpsk', 'qpsk', or 'fsk', got '{modulation}'"
            )
        self.num_channels = num_channels
        self.hop_pattern = hop_pattern
        self.dwell_samples = dwell_samples
        self.modulation = modulation

    @property
    def label(self) -> str:
        return "FHSS"

    def bandwidth(self, sample_rate: float) -> float:
        return sample_rate * 0.8  # approximate total hopping bandwidth

    def generate(
        self,
        num_symbols: int,
        sample_rate: float,
        seed: Optional[int] = None,
    ) -> np.ndarray:
        s = seed if seed is not None else np.random.randint(0, 2**32)
        rng = np.random.default_rng(int(s))

        num_hops = num_symbols

        # Generate hop pattern
        if self.hop_pattern == "random":
            channels = rng.integers(0, self.num_channels, size=num_hops)
        elif self.hop_pattern == "linear":
            channels = np.arange(num_hops) % self.num_channels
        else:  # costas
            from spectra._rust import generate_costas_sequence

            # Find a suitable prime >= num_channels + 1
            prime = self.num_channels + 1
            while not self._is_prime(prime):
                prime += 1
            costas_seq = generate_costas_sequence(prime)
            # Map to channel indices (0-based, mod num_channels)
            pattern = [(v - 1) % self.num_channels for v in costas_seq]
            channels = np.array(
                [pattern[i % len(pattern)] for i in range(num_hops)]
            )

        # Channel spacing: divide bandwidth evenly
        channel_spacing = sample_rate / self.num_channels
        center_offset = (self.num_channels - 1) / 2.0

        segments = []
        for hop_idx in range(num_hops):
            ch = channels[hop_idx]
            freq_offset = (ch - center_offset) * channel_spacing

            # Generate modulated baseband signal for this dwell
            hop_seed = int(rng.integers(0, 2**31))
            if self.modulation == "bpsk":
                sym = np.array(
                    generate_bpsk_symbols(self.dwell_samples, hop_seed),
                    dtype=np.complex64,
                )
            elif self.modulation == "qpsk":
                sym = np.array(
                    generate_qpsk_symbols(self.dwell_samples, hop_seed),
                    dtype=np.complex64,
                )
            else:  # fsk
                # Simple 2-FSK
                bits = 2 * rng.integers(0, 2, size=self.dwell_samples).astype(np.float32) - 1.0
                t = np.arange(self.dwell_samples) / sample_rate
                fsk_dev = channel_spacing * 0.25
                phase = 2.0 * np.pi * np.cumsum(bits * fsk_dev) / sample_rate
                sym = np.exp(1j * phase).astype(np.complex64)

            # Frequency-shift to hop channel
            t = np.arange(self.dwell_samples) / sample_rate
            shift = np.exp(1j * 2.0 * np.pi * freq_offset * t).astype(np.complex64)
            segments.append(sym * shift)

        return np.concatenate(segments).astype(np.complex64)

    @staticmethod
    def _is_prime(n: int) -> bool:
        if n < 2:
            return False
        if n < 4:
            return True
        if n % 2 == 0 or n % 3 == 0:
            return False
        i = 5
        while i * i <= n:
            if n % i == 0 or n % (i + 2) == 0:
                return False
            i += 6
        return True


class CDMA_Forward(Waveform):
    """CDMA Forward Link (downlink) waveform.

    Generates a CDMA forward link signal using Walsh-Hadamard channelization
    with PN scrambling. Multiple users are code-multiplexed and summed.

    Parameters
    ----------
    num_users : int
        Number of users (channels). Default 4.
    spreading_factor : int
        Walsh code length (must be power of 2). Default 64.
    user_powers : list of float, optional
        Per-user power levels (linear). If None, all users have equal power.
    """

    def __init__(
        self,
        num_users: int = 4,
        spreading_factor: int = 64,
        user_powers: Optional[List[float]] = None,
    ):
        # Validate spreading_factor is power of 2
        sf_order = int(np.log2(spreading_factor))
        if (1 << sf_order) != spreading_factor:
            raise ValueError(
                f"spreading_factor must be a power of 2, got {spreading_factor}"
            )
        if num_users > spreading_factor:
            raise ValueError(
                f"num_users ({num_users}) must be <= spreading_factor ({spreading_factor})"
            )
        self.num_users = num_users
        self.spreading_factor = spreading_factor
        self._sf_order = sf_order
        self.user_powers = user_powers or [1.0] * num_users

    @property
    def label(self) -> str:
        return "CDMA_Forward"

    def bandwidth(self, sample_rate: float) -> float:
        # Chip rate equals sample rate (1 sample per chip)
        return float(sample_rate)

    def generate(
        self,
        num_symbols: int,
        sample_rate: float,
        seed: Optional[int] = None,
    ) -> np.ndarray:
        s = seed if seed is not None else np.random.randint(0, 2**32)
        rng = np.random.default_rng(int(s))

        total_chips = num_symbols * self.spreading_factor

        # Generate PN scrambling sequence
        pn_seed = int(rng.integers(0, 2**31))
        pn_rng = np.random.default_rng(pn_seed)
        pn_code = 2.0 * pn_rng.integers(0, 2, size=total_chips).astype(np.float32) - 1.0

        # Pre-fetch Walsh codes
        walsh_codes = []
        for u in range(self.num_users):
            wc = np.array(
                generate_walsh_hadamard(self._sf_order, u), dtype=np.float32
            )
            walsh_codes.append(wc)

        composite = np.zeros(total_chips, dtype=np.float32)

        for u in range(self.num_users):
            user_seed = int(rng.integers(0, 2**31))
            # BPSK data for this user
            data = np.array(
                generate_bpsk_symbols(num_symbols, user_seed), dtype=np.complex64
            ).real

            # Spread with Walsh code
            spread = np.repeat(data, self.spreading_factor) * np.tile(
                walsh_codes[u], num_symbols
            )

            # Scramble with PN
            scrambled = spread * pn_code

            # Scale by user power
            power = self.user_powers[u]
            composite += np.sqrt(power) * scrambled

        # Normalize
        composite = composite / np.sqrt(self.num_users)

        return composite.astype(np.complex64)


class CDMA_Reverse(Waveform):
    """CDMA Reverse Link (uplink) waveform.

    Generates a CDMA reverse link signal where each user has a unique
    PN offset for spreading. Users are summed together.

    Parameters
    ----------
    num_users : int
        Number of users. Default 4.
    spreading_factor : int
        PN code length per symbol. Default 64.
    user_powers : list of float, optional
        Per-user power levels (linear). If None, all users have equal power.
    """

    def __init__(
        self,
        num_users: int = 4,
        spreading_factor: int = 64,
        user_powers: Optional[List[float]] = None,
    ):
        self.num_users = num_users
        self.spreading_factor = spreading_factor
        self.user_powers = user_powers or [1.0] * num_users

    @property
    def label(self) -> str:
        return "CDMA_Reverse"

    def bandwidth(self, sample_rate: float) -> float:
        return float(sample_rate)

    def generate(
        self,
        num_symbols: int,
        sample_rate: float,
        seed: Optional[int] = None,
    ) -> np.ndarray:
        s = seed if seed is not None else np.random.randint(0, 2**32)
        rng = np.random.default_rng(int(s))

        total_chips = num_symbols * self.spreading_factor

        # Generate a long PN sequence (we'll use offsets for each user)
        pn_seed = int(rng.integers(0, 2**31))
        pn_rng = np.random.default_rng(pn_seed)
        # Generate enough PN for all offsets
        pn_long = 2.0 * pn_rng.integers(
            0, 2, size=total_chips + self.num_users * self.spreading_factor
        ).astype(np.float32) - 1.0

        composite = np.zeros(total_chips, dtype=np.float32)

        for u in range(self.num_users):
            user_seed = int(rng.integers(0, 2**31))
            # BPSK data for this user
            data = np.array(
                generate_bpsk_symbols(num_symbols, user_seed), dtype=np.complex64
            ).real

            # Each user gets a different PN offset
            offset = u * self.spreading_factor
            pn_user = pn_long[offset : offset + total_chips]

            # Spread with offset PN code
            spread = np.repeat(data, self.spreading_factor) * pn_user

            # Scale by user power
            power = self.user_powers[u]
            composite += np.sqrt(power) * spread

        # Normalize
        composite = composite / np.sqrt(self.num_users)

        return composite.astype(np.complex64)


class THSS(Waveform):
    """Time Hopping Spread Spectrum.

    Generates a THSS signal where pulses are placed in pseudorandom
    time slots within frames.

    Parameters
    ----------
    num_frames : int
        Number of frames. Default 32.
    slots_per_frame : int
        Number of time slots per frame. Default 8.
    pulse_samples : int
        Number of samples per pulse. Default 16.
    pulse_shape : str
        Pulse shape: "gaussian" or "rect". Default "gaussian".
    """

    def __init__(
        self,
        num_frames: int = 32,
        slots_per_frame: int = 8,
        pulse_samples: int = 16,
        pulse_shape: str = "gaussian",
    ):
        if pulse_shape not in ("gaussian", "rect"):
            raise ValueError(
                f"pulse_shape must be 'gaussian' or 'rect', got '{pulse_shape}'"
            )
        self.num_frames = num_frames
        self.slots_per_frame = slots_per_frame
        self.pulse_samples = pulse_samples
        self.pulse_shape = pulse_shape
        self._frame_samples = slots_per_frame * pulse_samples

    @property
    def label(self) -> str:
        return "THSS"

    def bandwidth(self, sample_rate: float) -> float:
        # Bandwidth is approximately the pulse bandwidth
        return sample_rate / self.pulse_samples

    def generate(
        self,
        num_symbols: int,
        sample_rate: float,
        seed: Optional[int] = None,
    ) -> np.ndarray:
        s = seed if seed is not None else np.random.randint(0, 2**32)
        rng = np.random.default_rng(int(s))

        n_frames = num_symbols * self.num_frames

        # Generate pulse shape
        if self.pulse_shape == "gaussian":
            t = np.linspace(-3, 3, self.pulse_samples)
            pulse = np.exp(-0.5 * t**2).astype(np.float32)
        else:  # rect
            pulse = np.ones(self.pulse_samples, dtype=np.float32)

        # Build the signal frame by frame
        total_samples = n_frames * self._frame_samples
        signal = np.zeros(total_samples, dtype=np.complex64)

        # Random BPSK modulation per frame
        data = 2 * rng.integers(0, 2, size=n_frames).astype(np.float32) - 1.0

        # Random slot selection per frame
        slots = rng.integers(0, self.slots_per_frame, size=n_frames)

        for f in range(n_frames):
            slot = slots[f]
            start = f * self._frame_samples + slot * self.pulse_samples
            end = start + self.pulse_samples
            signal[start:end] = data[f] * pulse

        return signal.astype(np.complex64)
