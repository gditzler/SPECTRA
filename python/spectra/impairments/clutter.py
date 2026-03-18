"""Radar clutter model with terrain-typed presets.

:class:`RadarClutter` generates Doppler-colored complex Gaussian clutter
on a 2-D slow-time / fast-time matrix. It is a standalone callable (not a
:class:`~spectra.impairments.base.Transform` subclass) because radar clutter
operates on ``(num_pulses, num_range_bins)`` matrices rather than 1-D IQ.

Usage::

    clutter = RadarClutter.sea(sample_rate=1e6, sea_state=3)
    received = clutter(pulse_matrix, rng)
"""

from __future__ import annotations

from typing import Optional, Tuple

import numpy as np


_GROUND_PRESETS = {
    "rural":          (20.0,     5.0),
    "urban":          (30.0,     2.0),
    "forest":         (25.0,     8.0),
    "desert":         (15.0,     3.0),
}

_SEA_PRESETS = {
    1: (10.0, 15.0),
    2: (15.0, 25.0),
    3: (20.0, 40.0),
    4: (25.0, 60.0),
    5: (30.0, 80.0),
    6: (35.0, 100.0),
}


class RadarClutter:
    """Radar clutter generator with Doppler-coloured noise.

    Args:
        cnr: Clutter-to-noise ratio in dB (relative to unit thermal noise).
        doppler_spread: Clutter Doppler spectral width in Hz.
        sample_rate: Receiver sample rate in Hz.
        doppler_center: Center Doppler frequency in Hz. Default 0.
        range_extent: ``(start, stop)`` range bin indices. Default all bins.
        spectral_shape: ``"gaussian"`` or ``"exponential"``.
    """

    def __init__(
        self,
        cnr: float,
        doppler_spread: float,
        sample_rate: float,
        doppler_center: float = 0.0,
        range_extent: Optional[Tuple[int, int]] = None,
        spectral_shape: str = "gaussian",
    ) -> None:
        self.cnr = cnr
        self.doppler_spread = doppler_spread
        self.sample_rate = sample_rate
        self.doppler_center = doppler_center
        self.range_extent = range_extent
        self.spectral_shape = spectral_shape

    def __call__(
        self, pulse_matrix: np.ndarray, rng: np.random.Generator
    ) -> np.ndarray:
        """Add clutter to a ``(num_pulses, num_range_bins)`` complex matrix."""
        num_pulses, num_range_bins = pulse_matrix.shape
        r_start = 0
        r_stop = num_range_bins
        if self.range_extent is not None:
            r_start, r_stop = self.range_extent

        n_bins = r_stop - r_start

        white = np.sqrt(0.5) * (
            rng.standard_normal((num_pulses, n_bins))
            + 1j * rng.standard_normal((num_pulses, n_bins))
        )

        freqs = np.fft.fftfreq(num_pulses, d=1.0)
        norm_center = self.doppler_center / self.sample_rate
        norm_spread = self.doppler_spread / self.sample_rate

        if self.spectral_shape == "gaussian":
            if norm_spread > 0:
                psd = np.exp(-0.5 * ((freqs - norm_center) / norm_spread) ** 2)
            else:
                psd = np.ones(num_pulses)
        elif self.spectral_shape == "exponential":
            if norm_spread > 0:
                psd = np.exp(-np.abs(freqs - norm_center) / norm_spread)
            else:
                psd = np.ones(num_pulses)
        else:
            raise ValueError(
                f"spectral_shape must be 'gaussian' or 'exponential', got {self.spectral_shape!r}"
            )

        psd = psd / (np.sum(psd) + 1e-30) * num_pulses

        H = np.sqrt(psd)
        white_fft = np.fft.fft(white, axis=0)
        shaped_fft = white_fft * H[:, np.newaxis]
        clutter = np.fft.ifft(shaped_fft, axis=0)

        cnr_linear = 10.0 ** (self.cnr / 10.0)
        current_power = np.mean(np.abs(clutter) ** 2)
        if current_power > 0:
            clutter = clutter * np.sqrt(cnr_linear / current_power)

        out = pulse_matrix.copy()
        out[:, r_start:r_stop] = out[:, r_start:r_stop] + clutter
        return out

    @classmethod
    def ground(cls, sample_rate: float, terrain: str = "rural", **overrides) -> RadarClutter:
        """Return a :class:`RadarClutter` configured for ground clutter.

        Args:
            sample_rate: Receiver sample rate in Hz.
            terrain: One of ``"rural"``, ``"urban"``, ``"forest"``, ``"desert"``.
            **overrides: Keyword arguments forwarded to :class:`RadarClutter`.
        """
        if terrain not in _GROUND_PRESETS:
            raise ValueError(
                f"terrain must be one of {list(_GROUND_PRESETS)}, got {terrain!r}"
            )
        cnr, spread = _GROUND_PRESETS[terrain]
        defaults = dict(
            cnr=cnr,
            doppler_spread=spread,
            sample_rate=sample_rate,
            doppler_center=0.0,
            spectral_shape="gaussian",
        )
        defaults.update(overrides)
        return cls(**defaults)

    @classmethod
    def sea(cls, sample_rate: float, sea_state: int = 3, **overrides) -> RadarClutter:
        """Return a :class:`RadarClutter` configured for sea clutter.

        Args:
            sample_rate: Receiver sample rate in Hz.
            sea_state: Beaufort sea state 1–6.
            **overrides: Keyword arguments forwarded to :class:`RadarClutter`.
        """
        if sea_state not in _SEA_PRESETS:
            raise ValueError(f"sea_state must be 1-6, got {sea_state}")
        cnr, spread = _SEA_PRESETS[sea_state]
        defaults = dict(
            cnr=cnr,
            doppler_spread=spread,
            sample_rate=sample_rate,
            doppler_center=0.0,
            spectral_shape="exponential",
        )
        defaults.update(overrides)
        return cls(**defaults)

    @classmethod
    def weather(
        cls, sample_rate: float, rain_rate_mmhr: float = 10.0, **overrides
    ) -> RadarClutter:
        """Return a :class:`RadarClutter` configured for weather (rain) clutter.

        Args:
            sample_rate: Receiver sample rate in Hz.
            rain_rate_mmhr: Rain rate in mm/hr.
            **overrides: Keyword arguments forwarded to :class:`RadarClutter`.
        """
        cnr = 10.0 * np.log10(max(rain_rate_mmhr, 0.1)) + 10.0
        spread = 20.0 * np.sqrt(rain_rate_mmhr)
        doppler_center = 30.0
        defaults = dict(
            cnr=cnr,
            doppler_spread=spread,
            sample_rate=sample_rate,
            doppler_center=doppler_center,
            spectral_shape="gaussian",
        )
        defaults.update(overrides)
        return cls(**defaults)
