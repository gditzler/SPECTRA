import math
import warnings
from typing import Optional

import numpy as np

from spectra._rust import gaussian_taps, generate_bpsk_symbols, generate_fsk_symbols
from spectra.waveforms.base import Waveform


def _warn_if_aliased(label: str, order: int, mod_index: float, samples_per_symbol: int) -> None:
    """Warn when the Carson-rule band edge extends beyond Nyquist.

    CPFSK levels are ±1, ±3, …, ±(order-1), so the outermost tone sits at
    (order-1)·mod_index/(2·sps) of the sample rate and the Carson edge adds one
    symbol rate on each side. The signal aliases when
    ((order-1)·mod_index + 2) > sps.
    """
    carson = (order - 1) * mod_index + 2
    if carson > samples_per_symbol:
        warnings.warn(
            f"{label}: order={order}, mod_index={mod_index}, "
            f"samples_per_symbol={samples_per_symbol} places the Carson bandwidth "
            f"({carson:.2f}× the symbol rate) beyond the sampling bandwidth; the "
            f"generated signal will alias. Increase samples_per_symbol to at least "
            f"{math.ceil(carson)} or reduce mod_index.",
            UserWarning,
            stacklevel=3,
        )


class FSK(Waveform):
    """Continuous-phase FSK (CPFSK) waveform."""

    def __init__(
        self,
        order: int = 2,
        mod_index: float = 1.0,
        samples_per_symbol: int = 8,
    ):
        self._order = order
        self._mod_index = mod_index
        self.samples_per_symbol = samples_per_symbol

    def generate(
        self,
        num_symbols: int,
        sample_rate: float,
        seed: Optional[int] = None,
    ) -> np.ndarray:
        _warn_if_aliased(self.label, self._order, self._mod_index, self.samples_per_symbol)
        s = seed if seed is not None else np.random.randint(0, 2**32)
        freq_symbols = generate_fsk_symbols(num_symbols, self._order, seed=s)
        # Upsample: repeat each symbol sps times
        freq_up = np.repeat(freq_symbols, self.samples_per_symbol)
        # Phase increments per sample
        delta_phi = np.pi * self._mod_index * freq_up / self.samples_per_symbol
        phase = np.cumsum(delta_phi)
        return np.exp(1j * phase).astype(np.complex64)

    def bandwidth(self, sample_rate: float) -> float:
        symbol_rate = sample_rate / self.samples_per_symbol
        return symbol_rate * ((self._order - 1) * self._mod_index + 2)

    @property
    def label(self) -> str:
        return "FSK"


class MSK(Waveform):
    """Minimum Shift Keying — binary CPFSK with modulation index 0.5."""

    def __init__(self, samples_per_symbol: int = 8):
        self.samples_per_symbol = samples_per_symbol

    def generate(
        self,
        num_symbols: int,
        sample_rate: float,
        seed: Optional[int] = None,
    ) -> np.ndarray:
        s = seed if seed is not None else np.random.randint(0, 2**32)
        # BPSK symbols are +1/-1 on real axis
        symbols = generate_bpsk_symbols(num_symbols, seed=s)
        freq_up = np.repeat(symbols.real, self.samples_per_symbol)
        delta_phi = np.pi * 0.5 * freq_up / self.samples_per_symbol
        phase = np.cumsum(delta_phi)
        return np.exp(1j * phase).astype(np.complex64)

    def bandwidth(self, sample_rate: float) -> float:
        symbol_rate = sample_rate / self.samples_per_symbol
        return symbol_rate * 1.5  # Carson's rule: (h + 1) * R_s, h=0.5

    @property
    def label(self) -> str:
        return "MSK"


class GMSK(Waveform):
    """Gaussian Minimum Shift Keying — MSK with Gaussian pulse shaping."""

    def __init__(
        self,
        bt: float = 0.3,
        filter_span: int = 4,
        samples_per_symbol: int = 8,
    ):
        self._bt = bt
        self._filter_span = filter_span
        self.samples_per_symbol = samples_per_symbol

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
        symbols = generate_bpsk_symbols(num_symbols, seed=s)
        sps = self.samples_per_symbol

        # Repeat-upsample so the frequency-pulse train averages to ±1.
        # A sum-normalised Gaussian preserves the DC level of its input;
        # zero-insertion would attenuate it by sps, yielding h_eff = h/sps.
        symbols_up = np.repeat(symbols.real.astype(np.float32), sps)

        # Gaussian filter
        h = self._gaussian_taps()
        filtered = np.convolve(symbols_up, h, mode="same")

        # Phase modulation: per-symbol total Δφ = π·h·b_k with h = 0.5.
        delta_phi = np.pi * 0.5 * filtered / sps
        phase = np.cumsum(delta_phi)
        return np.exp(1j * phase).astype(np.complex64)

    def bandwidth(self, sample_rate: float) -> float:
        symbol_rate = sample_rate / self.samples_per_symbol
        return symbol_rate * 1.5

    @property
    def label(self) -> str:
        return "GMSK"


# --- Named FSK subclasses ---


class FSK4(FSK):
    def __init__(self, mod_index: float = 1.0, samples_per_symbol: int = 8):
        super().__init__(order=4, mod_index=mod_index, samples_per_symbol=samples_per_symbol)

    @property
    def label(self) -> str:
        return "4FSK"


class FSK8(FSK):
    # Default sps=16 keeps the Carson bandwidth (9× the symbol rate for
    # mod_index=1) inside the sampling bandwidth; sps=8 would alias.
    def __init__(self, mod_index: float = 1.0, samples_per_symbol: int = 16):
        super().__init__(order=8, mod_index=mod_index, samples_per_symbol=samples_per_symbol)

    @property
    def label(self) -> str:
        return "8FSK"


class FSK16(FSK):
    # Default sps=32 keeps the Carson bandwidth (17× the symbol rate for
    # mod_index=1) inside the sampling bandwidth; sps=8 put the outermost
    # tones at ±0.94·fs — heavily aliased.
    def __init__(self, mod_index: float = 1.0, samples_per_symbol: int = 32):
        super().__init__(order=16, mod_index=mod_index, samples_per_symbol=samples_per_symbol)

    @property
    def label(self) -> str:
        return "16FSK"


# --- Named MSK subclasses ---


class MSK4(FSK):
    """4-ary FSK with modulation index 0.5."""

    def __init__(self, samples_per_symbol: int = 8):
        super().__init__(order=4, mod_index=0.5, samples_per_symbol=samples_per_symbol)

    @property
    def label(self) -> str:
        return "4MSK"


class MSK8(FSK):
    """8-ary FSK with modulation index 0.5."""

    def __init__(self, samples_per_symbol: int = 8):
        super().__init__(order=8, mod_index=0.5, samples_per_symbol=samples_per_symbol)

    @property
    def label(self) -> str:
        return "8MSK"


# --- GFSK (Gaussian FSK) ---


class GFSK(Waveform):
    """Gaussian Frequency Shift Keying — arbitrary order with Gaussian pulse shaping."""

    def __init__(
        self,
        order: int = 2,
        mod_index: float = 1.0,
        bt: float = 0.3,
        filter_span: int = 4,
        samples_per_symbol: int = 8,
    ):
        self._order = order
        self._mod_index = mod_index
        self._bt = bt
        self._filter_span = filter_span
        self.samples_per_symbol = samples_per_symbol

    def generate(
        self,
        num_symbols: int,
        sample_rate: float,
        seed: Optional[int] = None,
    ) -> np.ndarray:
        _warn_if_aliased(self.label, self._order, self._mod_index, self.samples_per_symbol)
        s = seed if seed is not None else np.random.randint(0, 2**32)
        freq_symbols = generate_fsk_symbols(num_symbols, self._order, seed=s)
        sps = self.samples_per_symbol

        # Repeat-upsample so the frequency-pulse train averages to the
        # symbol level. A sum-normalised Gaussian preserves the DC level
        # of its input; zero-insertion would attenuate it by sps,
        # yielding h_eff = h/sps.
        symbols_up = np.repeat(freq_symbols.astype(np.float32), sps)

        # Gaussian filter (from Rust)
        h = np.array(gaussian_taps(self._bt, self._filter_span, sps))
        filtered = np.convolve(symbols_up, h, mode="same")

        # Phase modulation
        delta_phi = np.pi * self._mod_index * filtered / sps
        phase = np.cumsum(delta_phi)
        return np.exp(1j * phase).astype(np.complex64)

    def bandwidth(self, sample_rate: float) -> float:
        symbol_rate = sample_rate / self.samples_per_symbol
        return symbol_rate * ((self._order - 1) * self._mod_index + 2)

    @property
    def label(self) -> str:
        return "GFSK"


class GFSK4(GFSK):
    def __init__(self, bt: float = 0.3, filter_span: int = 4, samples_per_symbol: int = 8):
        super().__init__(
            order=4,
            bt=bt,
            filter_span=filter_span,
            samples_per_symbol=samples_per_symbol,
        )

    @property
    def label(self) -> str:
        return "4GFSK"


class GFSK8(GFSK):
    # sps=16 for the same Nyquist-fit reason as FSK8.
    def __init__(self, bt: float = 0.3, filter_span: int = 4, samples_per_symbol: int = 16):
        super().__init__(
            order=8,
            bt=bt,
            filter_span=filter_span,
            samples_per_symbol=samples_per_symbol,
        )

    @property
    def label(self) -> str:
        return "8GFSK"


class GFSK16(GFSK):
    # sps=32 for the same Nyquist-fit reason as FSK16.
    def __init__(self, bt: float = 0.3, filter_span: int = 4, samples_per_symbol: int = 32):
        super().__init__(
            order=16,
            bt=bt,
            filter_span=filter_span,
            samples_per_symbol=samples_per_symbol,
        )

    @property
    def label(self) -> str:
        return "16GFSK"


# --- GMSK variants (GFSK with mod_index=0.5) ---


class GMSK4(GFSK):
    """4-ary GMSK (GFSK with mod_index=0.5)."""

    def __init__(self, bt: float = 0.3, filter_span: int = 4, samples_per_symbol: int = 8):
        super().__init__(
            order=4,
            mod_index=0.5,
            bt=bt,
            filter_span=filter_span,
            samples_per_symbol=samples_per_symbol,
        )

    @property
    def label(self) -> str:
        return "4GMSK"


class GMSK8(GFSK):
    """8-ary GMSK (GFSK with mod_index=0.5)."""

    def __init__(self, bt: float = 0.3, filter_span: int = 4, samples_per_symbol: int = 8):
        super().__init__(
            order=8,
            mod_index=0.5,
            bt=bt,
            filter_span=filter_span,
            samples_per_symbol=samples_per_symbol,
        )

    @property
    def label(self) -> str:
        return "8GMSK"
