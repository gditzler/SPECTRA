"""Radar waveform generators for SPECTRA.

Includes pulsed, coded, FMCW, stepped-frequency, pulse-Doppler,
and nonlinear FM radar waveforms backed by Rust primitives.
"""

from typing import List, Optional

import numpy as np

from spectra._rust import (
    generate_fmcw_sweep,
    generate_frank_code,
    generate_nlfm_sweep,
    generate_p1_code,
    generate_p2_code,
    generate_p3_code,
    generate_p4_code,
    generate_pulse_train,
    generate_stepped_frequency,
)
from spectra.waveforms.base import Waveform

BARKER_CODES = {
    2: [1, -1],
    3: [1, 1, -1],
    4: [1, 1, -1, 1],
    5: [1, 1, 1, -1, 1],
    7: [1, 1, 1, -1, -1, 1, -1],
    11: [1, 1, 1, -1, -1, -1, 1, -1, -1, 1, -1],
    13: [1, 1, 1, 1, 1, -1, -1, 1, 1, -1, 1, -1, 1],
}


def _make_pulse_shape(pulse_width_samples: int, shape: str) -> np.ndarray:
    """Create a pulse envelope of the given shape."""
    if shape == "rect":
        return np.ones(pulse_width_samples, dtype=np.complex64)
    elif shape == "hamming":
        window = np.hamming(pulse_width_samples).astype(np.float32)
        return (window + 0j).astype(np.complex64)
    elif shape == "hann":
        window = np.hanning(pulse_width_samples).astype(np.float32)
        return (window + 0j).astype(np.complex64)
    else:
        raise ValueError(f"Unknown pulse_shape '{shape}'. Use 'rect', 'hamming', or 'hann'.")


class PulsedRadar(Waveform):
    """Simple pulsed radar waveform with configurable pulse shape and PRI.

    Args:
        pulse_width_samples: Number of samples per pulse.
        pri_samples: Pulse repetition interval in samples.
        num_pulses: Number of pulses per symbol.
        pulse_shape: Pulse envelope shape ("rect", "hamming", or "hann").
        pri_stagger: Optional list of per-pulse timing offsets in samples.
        pri_jitter_fraction: Fractional jitter applied to PRI (0.0 = no jitter).
    """

    def __init__(
        self,
        pulse_width_samples: int = 64,
        pri_samples: int = 512,
        num_pulses: int = 16,
        pulse_shape: str = "rect",
        pri_stagger: Optional[List[int]] = None,
        pri_jitter_fraction: float = 0.0,
    ):
        self._pulse_width_samples = pulse_width_samples
        self._pri_samples = pri_samples
        self._num_pulses = num_pulses
        self._pulse_shape = pulse_shape
        self._pri_stagger = pri_stagger
        self._pri_jitter_fraction = pri_jitter_fraction
        self.samples_per_symbol = pri_samples * num_pulses

    def generate(
        self,
        num_symbols: int,
        sample_rate: float,
        seed: Optional[int] = None,
    ) -> np.ndarray:
        pulse = _make_pulse_shape(self._pulse_width_samples, self._pulse_shape)

        if self._pri_stagger is not None:
            stagger = np.array(self._pri_stagger, dtype=np.int64)
        elif self._pri_jitter_fraction > 0.0:
            s = seed if seed is not None else np.random.randint(0, 2**32)
            rng = np.random.default_rng(s)
            max_jitter = int(self._pri_samples * self._pri_jitter_fraction)
            stagger = rng.integers(
                -max_jitter, max_jitter + 1, size=self._num_pulses, dtype=np.int64
            )
        else:
            stagger = np.array([], dtype=np.int64)

        bursts = []
        for _ in range(num_symbols):
            train = generate_pulse_train(pulse, self._pri_samples, self._num_pulses, stagger)
            bursts.append(train)

        return np.concatenate(bursts) if bursts else np.array([], dtype=np.complex64)

    def bandwidth(self, sample_rate: float) -> float:
        return sample_rate / self._pulse_width_samples

    @property
    def label(self) -> str:
        return "PulsedRadar"

    @classmethod
    def weather(cls) -> "PulsedRadar":
        """Long-pulse, low-PRF weather radar preset."""
        return cls(
            pulse_width_samples=256,
            pri_samples=4096,
            num_pulses=8,
            pulse_shape="rect",
        )

    @classmethod
    def marine_nav(cls) -> "PulsedRadar":
        """Short-pulse, high-PRF marine navigation radar preset."""
        return cls(
            pulse_width_samples=16,
            pri_samples=128,
            num_pulses=32,
            pulse_shape="rect",
        )


class BarkerCodedPulse(Waveform):
    """Barker-coded pulsed radar waveform.

    Args:
        barker_length: Barker code length (2, 3, 4, 5, 7, 11, or 13).
        samples_per_chip: Upsampling factor per chip.
        pri_samples: Pulse repetition interval in samples.
        num_pulses: Number of pulses per symbol.
    """

    def __init__(
        self,
        barker_length: int = 13,
        samples_per_chip: int = 8,
        pri_samples: int = 1024,
        num_pulses: int = 16,
    ):
        if barker_length not in BARKER_CODES:
            valid = sorted(BARKER_CODES.keys())
            raise ValueError(f"Barker code length must be one of {valid}, got {barker_length}")
        self._barker_length = barker_length
        self._samples_per_chip = samples_per_chip
        self._pri_samples = pri_samples
        self._num_pulses = num_pulses
        self._code = np.array(BARKER_CODES[barker_length], dtype=np.float32)
        self.samples_per_symbol = pri_samples * num_pulses

    def generate(
        self,
        num_symbols: int,
        sample_rate: float,
        seed: Optional[int] = None,
    ) -> np.ndarray:
        # Create Barker-coded pulse: upsample chips
        chips_up = np.repeat(self._code, self._samples_per_chip)
        pulse = (chips_up + 0j).astype(np.complex64)
        stagger = np.array([], dtype=np.int64)

        bursts = []
        for _ in range(num_symbols):
            train = generate_pulse_train(pulse, self._pri_samples, self._num_pulses, stagger)
            bursts.append(train)

        return np.concatenate(bursts) if bursts else np.array([], dtype=np.complex64)

    def bandwidth(self, sample_rate: float) -> float:
        return sample_rate / self._samples_per_chip

    @property
    def label(self) -> str:
        return "BarkerCodedPulse"


class PolyphaseCodedPulse(Waveform):
    """Polyphase-coded pulsed radar waveform.

    Uses Rust-generated polyphase codes (Frank, P1-P4) as intra-pulse
    modulation within a pulse train.

    Args:
        code_type: Polyphase code type ("frank", "p1", "p2", "p3", or "p4").
        code_order: Code order (for Frank/P1/P2) or code length (for P3/P4).
        samples_per_chip: Upsampling factor per chip.
        pri_samples: Pulse repetition interval in samples.
        num_pulses: Number of pulses per symbol.
    """

    _CODE_GENERATORS = {
        "frank": lambda order: generate_frank_code(order),
        "p1": lambda order: generate_p1_code(order),
        "p2": lambda order: generate_p2_code(order),
        "p3": lambda order: generate_p3_code(order),
        "p4": lambda order: generate_p4_code(order),
    }

    def __init__(
        self,
        code_type: str = "frank",
        code_order: int = 4,
        samples_per_chip: int = 8,
        pri_samples: int = 1024,
        num_pulses: int = 16,
    ):
        if code_type not in self._CODE_GENERATORS:
            raise ValueError(
                f"Unknown code_type '{code_type}'. Use one of {list(self._CODE_GENERATORS.keys())}."
            )
        self._code_type = code_type
        self._code_order = code_order
        self._samples_per_chip = samples_per_chip
        self._pri_samples = pri_samples
        self._num_pulses = num_pulses
        self.samples_per_symbol = pri_samples * num_pulses

    def generate(
        self,
        num_symbols: int,
        sample_rate: float,
        seed: Optional[int] = None,
    ) -> np.ndarray:
        chips = self._CODE_GENERATORS[self._code_type](self._code_order)
        chips_up = np.repeat(chips, self._samples_per_chip)
        pulse = chips_up.astype(np.complex64)
        stagger = np.array([], dtype=np.int64)

        bursts = []
        for _ in range(num_symbols):
            train = generate_pulse_train(pulse, self._pri_samples, self._num_pulses, stagger)
            bursts.append(train)

        return np.concatenate(bursts) if bursts else np.array([], dtype=np.complex64)

    def bandwidth(self, sample_rate: float) -> float:
        return sample_rate / self._samples_per_chip

    @property
    def label(self) -> str:
        return "PolyphaseCodedPulse"


class FMCW(Waveform):
    """Frequency-Modulated Continuous Wave radar waveform.

    Args:
        sweep_bandwidth_fraction: Sweep bandwidth as fraction of sample rate.
        sweep_samples: Number of samples per sweep.
        idle_samples: Number of zero samples between sweeps.
        num_sweeps: Number of sweeps per symbol.
        sweep_type: Sweep type ("sawtooth" or "triangle").
    """

    def __init__(
        self,
        sweep_bandwidth_fraction: float = 0.5,
        sweep_samples: int = 256,
        idle_samples: int = 64,
        num_sweeps: int = 16,
        sweep_type: str = "sawtooth",
    ):
        self._sweep_bandwidth_fraction = sweep_bandwidth_fraction
        self._sweep_samples = sweep_samples
        self._idle_samples = idle_samples
        self._num_sweeps = num_sweeps
        self._sweep_type = sweep_type
        self.samples_per_symbol = (sweep_samples + idle_samples) * num_sweeps

    def generate(
        self,
        num_symbols: int,
        sample_rate: float,
        seed: Optional[int] = None,
    ) -> np.ndarray:
        bw = sample_rate * self._sweep_bandwidth_fraction
        sweep = generate_fmcw_sweep(self._sweep_samples, bw, sample_rate, self._sweep_type)
        idle = np.zeros(self._idle_samples, dtype=np.complex64)

        # One symbol = num_sweeps repetitions of (sweep + idle)
        one_symbol_parts = []
        for _ in range(self._num_sweeps):
            one_symbol_parts.append(sweep)
            if self._idle_samples > 0:
                one_symbol_parts.append(idle)
        one_symbol = np.concatenate(one_symbol_parts)

        return np.tile(one_symbol, num_symbols)

    def bandwidth(self, sample_rate: float) -> float:
        return sample_rate * self._sweep_bandwidth_fraction

    @property
    def label(self) -> str:
        return "FMCW"

    @classmethod
    def automotive(cls) -> "FMCW":
        """Automotive radar preset (wide bandwidth, sawtooth)."""
        return cls(
            sweep_bandwidth_fraction=0.8,
            sweep_samples=512,
            idle_samples=32,
            num_sweeps=16,
            sweep_type="sawtooth",
        )

    @classmethod
    def weather(cls) -> "FMCW":
        """Weather radar FMCW preset."""
        return cls(
            sweep_bandwidth_fraction=0.3,
            sweep_samples=256,
            idle_samples=128,
            num_sweeps=8,
            sweep_type="triangle",
        )

    @classmethod
    def marine_nav(cls) -> "FMCW":
        """Marine navigation FMCW preset."""
        return cls(
            sweep_bandwidth_fraction=0.4,
            sweep_samples=128,
            idle_samples=32,
            num_sweeps=16,
            sweep_type="sawtooth",
        )

    @classmethod
    def atc(cls) -> "FMCW":
        """Air traffic control FMCW preset."""
        return cls(
            sweep_bandwidth_fraction=0.5,
            sweep_samples=256,
            idle_samples=64,
            num_sweeps=12,
            sweep_type="triangle",
        )


class SteppedFrequency(Waveform):
    """Stepped-frequency radar waveform.

    Generates CW tones at equally spaced frequencies with phase
    continuity between steps.

    Args:
        num_steps: Number of frequency steps per burst.
        samples_per_step: Number of samples at each frequency step.
        freq_step_fraction: Frequency step as fraction of sample rate.
        num_bursts: Number of burst repetitions per symbol.
    """

    def __init__(
        self,
        num_steps: int = 8,
        samples_per_step: int = 64,
        freq_step_fraction: float = 0.05,
        num_bursts: int = 4,
    ):
        self._num_steps = num_steps
        self._samples_per_step = samples_per_step
        self._freq_step_fraction = freq_step_fraction
        self._num_bursts = num_bursts
        self.samples_per_symbol = num_steps * samples_per_step * num_bursts

    def generate(
        self,
        num_symbols: int,
        sample_rate: float,
        seed: Optional[int] = None,
    ) -> np.ndarray:
        freq_step = sample_rate * self._freq_step_fraction
        one_burst = generate_stepped_frequency(
            self._num_steps, self._samples_per_step, freq_step, sample_rate
        )
        # Tile for bursts and symbols
        one_symbol = np.tile(one_burst, self._num_bursts)
        return np.tile(one_symbol, num_symbols)

    def bandwidth(self, sample_rate: float) -> float:
        return self._num_steps * self._freq_step_fraction * sample_rate

    @property
    def label(self) -> str:
        return "SteppedFrequency"


class PulseDoppler(Waveform):
    """Pulse-Doppler radar waveform.

    Generates coherent pulse trains organized into Coherent Processing
    Intervals (CPIs).

    Args:
        prf_mode: PRF mode ("low", "medium", or "high").
        num_pulses_per_cpi: Number of pulses per CPI.
        pulse_width_samples: Number of samples per pulse.
        num_cpis: Number of CPIs per symbol.
    """

    _PRI_MULTIPLIERS = {
        "low": 32,  # Long PRI (low PRF)
        "medium": 16,  # Medium PRI
        "high": 8,  # Short PRI (high PRF)
    }

    def __init__(
        self,
        prf_mode: str = "medium",
        num_pulses_per_cpi: int = 32,
        pulse_width_samples: int = 32,
        num_cpis: int = 4,
    ):
        if prf_mode not in self._PRI_MULTIPLIERS:
            raise ValueError(
                f"Unknown prf_mode '{prf_mode}'. Use one of {list(self._PRI_MULTIPLIERS.keys())}."
            )
        self._prf_mode = prf_mode
        self._num_pulses_per_cpi = num_pulses_per_cpi
        self._pulse_width_samples = pulse_width_samples
        self._num_cpis = num_cpis
        self._pri_samples = pulse_width_samples * self._PRI_MULTIPLIERS[prf_mode]
        self.samples_per_symbol = self._pri_samples * num_pulses_per_cpi * num_cpis

    def generate(
        self,
        num_symbols: int,
        sample_rate: float,
        seed: Optional[int] = None,
    ) -> np.ndarray:
        pulse = np.ones(self._pulse_width_samples, dtype=np.complex64)
        stagger = np.array([], dtype=np.int64)

        cpis = []
        for _ in range(num_symbols):
            for _ in range(self._num_cpis):
                train = generate_pulse_train(
                    pulse,
                    self._pri_samples,
                    self._num_pulses_per_cpi,
                    stagger,
                )
                cpis.append(train)

        return np.concatenate(cpis) if cpis else np.array([], dtype=np.complex64)

    def bandwidth(self, sample_rate: float) -> float:
        return sample_rate / self._pulse_width_samples

    @property
    def label(self) -> str:
        return "PulseDoppler"

    @classmethod
    def atc(cls) -> "PulseDoppler":
        """Air traffic control pulse-Doppler preset."""
        return cls(
            prf_mode="medium",
            num_pulses_per_cpi=64,
            pulse_width_samples=16,
            num_cpis=2,
        )


class NonlinearFM(Waveform):
    """Nonlinear frequency modulation radar waveform.

    Produces FM sweeps with nonlinear frequency-vs-time profiles for
    improved sidelobe performance without amplitude weighting.

    Args:
        sweep_type: NLFM sweep type ("tandem_hooked" or "s_curve").
        bandwidth_fraction: Sweep bandwidth as fraction of sample rate.
        num_samples: Number of samples per sweep.
    """

    def __init__(
        self,
        sweep_type: str = "s_curve",
        bandwidth_fraction: float = 0.5,
        num_samples: int = 256,
    ):
        if sweep_type not in ("tandem_hooked", "s_curve"):
            raise ValueError(
                f"Unknown sweep_type '{sweep_type}'. Use 'tandem_hooked' or 's_curve'."
            )
        self._sweep_type = sweep_type
        self._bandwidth_fraction = bandwidth_fraction
        self._num_samples = num_samples
        self.samples_per_symbol = num_samples

    def generate(
        self,
        num_symbols: int,
        sample_rate: float,
        seed: Optional[int] = None,
    ) -> np.ndarray:
        bw = sample_rate * self._bandwidth_fraction
        one_sweep = generate_nlfm_sweep(self._num_samples, sample_rate, bw, self._sweep_type)
        return np.tile(one_sweep, num_symbols)

    def bandwidth(self, sample_rate: float) -> float:
        return sample_rate * self._bandwidth_fraction

    @property
    def label(self) -> str:
        return "NonlinearFM"
