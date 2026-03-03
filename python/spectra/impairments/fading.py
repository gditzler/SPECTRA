from typing import Optional, Tuple

import numpy as np

from spectra.impairments.base import Transform
from spectra.scene.signal_desc import SignalDescription


def _generate_rayleigh_taps(n: int, max_doppler: float, sample_rate: float) -> np.ndarray:
    """Generate Rayleigh fading channel taps using filtered Gaussian noise."""
    # Complex white Gaussian noise
    noise = (np.random.randn(n) + 1j * np.random.randn(n)) / np.sqrt(2)
    # Doppler filter in frequency domain
    freqs = np.fft.fftfreq(n, d=1.0 / sample_rate)
    # Classic Jakes Doppler spectrum: 1/sqrt(1 - (f/f_d)^2) for |f| < f_d
    doppler_filter = np.zeros(n)
    mask = np.abs(freqs) < max_doppler
    doppler_filter[mask] = 1.0 / np.sqrt(
        1.0 - (freqs[mask] / max_doppler) ** 2 + 1e-10
    )
    # Apply filter in frequency domain
    taps_freq = np.fft.fft(noise) * doppler_filter
    taps = np.fft.ifft(taps_freq)
    # Normalize to unit average power
    taps = taps / (np.sqrt(np.mean(np.abs(taps) ** 2)) + 1e-10)
    return taps.astype(np.complex64)


class RayleighFading(Transform):
    def __init__(
        self,
        max_doppler: Optional[float] = None,
        max_doppler_range: Optional[Tuple[float, float]] = None,
    ):
        if max_doppler is None and max_doppler_range is None:
            raise ValueError("Must provide either max_doppler or max_doppler_range")
        self.max_doppler = max_doppler
        self.max_doppler_range = max_doppler_range

    def __call__(
        self, iq: np.ndarray, desc: SignalDescription, **kwargs
    ) -> Tuple[np.ndarray, SignalDescription]:
        sample_rate = kwargs.get("sample_rate")
        if sample_rate is None:
            raise ValueError("RayleighFading requires sample_rate kwarg")

        if self.max_doppler_range is not None:
            fd = np.random.uniform(*self.max_doppler_range)
        else:
            fd = self.max_doppler

        taps = _generate_rayleigh_taps(len(iq), fd, sample_rate)
        return (iq * taps).astype(np.complex64), desc


class RicianFading(Transform):
    def __init__(
        self,
        k_factor: Optional[float] = None,
        max_doppler: Optional[float] = None,
        k_factor_range: Optional[Tuple[float, float]] = None,
        max_doppler_range: Optional[Tuple[float, float]] = None,
    ):
        has_k = k_factor is not None or k_factor_range is not None
        has_fd = max_doppler is not None or max_doppler_range is not None
        if not has_k:
            raise ValueError("Must provide either k_factor or k_factor_range")
        if not has_fd:
            raise ValueError("Must provide either max_doppler or max_doppler_range")
        self.k_factor = k_factor
        self.max_doppler = max_doppler
        self.k_factor_range = k_factor_range
        self.max_doppler_range = max_doppler_range

    def __call__(
        self, iq: np.ndarray, desc: SignalDescription, **kwargs
    ) -> Tuple[np.ndarray, SignalDescription]:
        sample_rate = kwargs.get("sample_rate")
        if sample_rate is None:
            raise ValueError("RicianFading requires sample_rate kwarg")

        if self.k_factor_range is not None:
            k_db = np.random.uniform(*self.k_factor_range)
        else:
            k_db = self.k_factor
        if self.max_doppler_range is not None:
            fd = np.random.uniform(*self.max_doppler_range)
        else:
            fd = self.max_doppler

        k_lin = 10.0 ** (k_db / 10.0)
        # LOS component weight and scatter weight
        los_weight = np.sqrt(k_lin / (k_lin + 1.0))
        scatter_weight = np.sqrt(1.0 / (k_lin + 1.0))

        scatter = _generate_rayleigh_taps(len(iq), fd, sample_rate)
        taps = los_weight + scatter_weight * scatter
        return (iq * taps).astype(np.complex64), desc
