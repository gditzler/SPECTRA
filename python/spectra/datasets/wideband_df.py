# python/spectra/datasets/wideband_df.py
"""WidebandDirectionFindingDataset: joint wideband spectrum + DoA dataset."""

from __future__ import annotations

import warnings
from dataclasses import dataclass, field
from typing import Callable, List, Optional, Tuple, Union

import numpy as np
import torch

from spectra.arrays.array import AntennaArray
from spectra.datasets._base import BaseIQDataset
from spectra.datasets.iq_utils import truncate_pad
from spectra.scene.signal_desc import SignalDescription
from spectra.waveforms.base import Waveform


@dataclass
class WidebandDFTarget:
    """Ground-truth labels for a wideband direction-finding item.

    Attributes:
        azimuths: Source azimuth angles in radians, shape ``(num_signals,)``.
        elevations: Source elevation angles in radians, shape ``(num_signals,)``.
        center_freqs: Per-source center frequencies in Hz relative to DC,
            shape ``(num_signals,)``. Negative values are below DC.
        snrs: Per-source SNR in dB, shape ``(num_signals,)``.
        num_signals: Number of active signals.
        labels: Modulation label string per source.
        signal_descs: Full :class:`~spectra.scene.signal_desc.SignalDescription`
            per source.
    """

    azimuths: np.ndarray
    elevations: np.ndarray
    center_freqs: np.ndarray
    snrs: np.ndarray
    num_signals: int
    labels: List[str]
    signal_descs: List[SignalDescription] = field(default_factory=list)


class WidebandDirectionFindingDataset(BaseIQDataset[Tuple[torch.Tensor, WidebandDFTarget]]):
    """On-the-fly wideband direction-finding dataset.

    Generates multi-antenna wideband IQ captures with ``num_signals`` co-channel
    sources.  Each source occupies a distinct sub-band (separated by at least
    ``min_freq_separation`` Hz) and arrives from a distinct spatial direction
    (separated by at least ``min_angular_separation`` radians when specified).

    The received wideband signal at element ``n`` is::

        x_n[t] = sum_k a_n(az_k, el_k, f_k) * s_k[t] * exp(j*2*pi*f_k*t/fs) + w_n[t]

    where ``a_n(az, el, f)`` is the frequency-dependent element response and
    ``s_k[t]`` is the baseband signal of source ``k``.

    **Frequency-dependent steering:** Positions are stored in wavelengths at
    ``array.reference_frequency``. For source ``k`` at frequency ``f_k``, the
    phase shifts are rescaled by ``(reference_frequency + f_k) / reference_frequency``.

    **Output:** ``(Tensor[N_elements, 2, num_snapshots], WidebandDFTarget)``

    .. note::
        Use a custom ``collate_fn`` with DataLoader::

            def collate_fn(batch):
                return torch.stack([x for x, _ in batch]), [t for _, t in batch]

    Args:
        array: :class:`~spectra.arrays.array.AntennaArray` geometry.
        signal_pool: Waveforms to draw from for each source.
        num_signals: Fixed source count (int) or ``(min, max)`` range.
        num_snapshots: IQ samples per antenna element.
        sample_rate: Wideband receiver sample rate in Hz.
        capture_bandwidth: Usable bandwidth in Hz. Center frequencies are
            sampled uniformly from ``(-capture_bandwidth/2, +capture_bandwidth/2)``.
        snr_range: ``(min_db, max_db)`` per-source SNR.
        azimuth_range: ``(min_rad, max_rad)`` azimuth range.
        elevation_range: ``(min_rad, max_rad)`` elevation range. Default (0, 0).
        min_freq_separation: Minimum Hz between source center frequencies.
            If ``None``, no constraint is applied.
        min_angular_separation: Minimum radians between source angles.
            If ``None``, no constraint is applied.
        transform: Optional callable on the output tensor.
        num_samples: Dataset size.
        seed: Base integer seed.
    """

    def __init__(
        self,
        array: AntennaArray,
        signal_pool: List[Waveform],
        num_signals: Union[int, Tuple[int, int]],
        num_snapshots: int,
        sample_rate: float,
        capture_bandwidth: float,
        snr_range: Tuple[float, float],
        azimuth_range: Tuple[float, float] = (0.0, 2 * np.pi),
        elevation_range: Tuple[float, float] = (0.0, 0.0),
        min_freq_separation: Optional[float] = None,
        min_angular_separation: Optional[float] = None,
        transform: Optional[Callable] = None,
        num_samples: int = 10000,
        seed: int = 0,
    ):
        if num_snapshots <= 0:
            raise ValueError(f"num_snapshots must be positive, got {num_snapshots}")
        self.array = array
        self.signal_pool = signal_pool
        self.num_signals = num_signals
        self.num_snapshots = num_snapshots
        self.sample_rate = sample_rate
        self.capture_bandwidth = capture_bandwidth
        self.snr_range = snr_range
        self.azimuth_range = azimuth_range
        self.elevation_range = elevation_range
        self.min_freq_separation = min_freq_separation
        self.min_angular_separation = min_angular_separation
        self.transform = transform
        self.num_samples = num_samples
        self.seed = seed

    def __len__(self) -> int:
        return self.num_samples

    def __getitem__(self, index: int) -> Tuple[torch.Tensor, WidebandDFTarget]:
        rng = np.random.default_rng(seed=(self.seed, index))

        # Number of sources
        if isinstance(self.num_signals, tuple):
            n_src = int(rng.integers(self.num_signals[0], self.num_signals[1] + 1))
        else:
            n_src = int(self.num_signals)

        # Sample center frequencies within capture bandwidth
        center_freqs = self._sample_freqs(rng, n_src)

        # Sample azimuths/elevations
        azimuths, elevations = self._sample_angles(rng, n_src)

        # Sample SNRs
        snrs_db = rng.uniform(self.snr_range[0], self.snr_range[1], size=n_src)

        # Generate baseband signals and apply frequency shift
        source_iq = []
        labels = []
        signal_descs = []

        for k in range(n_src):
            wf_idx = int(rng.integers(0, len(self.signal_pool)))
            waveform = self.signal_pool[wf_idx]
            sps = getattr(waveform, "samples_per_symbol", 8)
            num_symbols = self.num_snapshots // sps + 1
            sig_seed = int(rng.integers(0, 2**32))

            # Generate baseband IQ
            iq = waveform.generate(
                num_symbols=num_symbols,
                sample_rate=self.sample_rate,
                seed=sig_seed,
            )
            iq = truncate_pad(iq, self.num_snapshots)

            # Frequency-shift to center_freqs[k]
            t = np.arange(self.num_snapshots) / self.sample_rate
            iq = iq * np.exp(1j * 2 * np.pi * center_freqs[k] * t)

            bw = waveform.bandwidth(self.sample_rate)
            desc = SignalDescription(
                t_start=0.0,
                t_stop=self.num_snapshots / self.sample_rate,
                f_low=center_freqs[k] - bw / 2,
                f_high=center_freqs[k] + bw / 2,
                label=waveform.label,
                snr=float(snrs_db[k]),
                modulation_params={
                    "doa": {
                        "azimuth_rad": float(azimuths[k]),
                        "elevation_rad": float(elevations[k]),
                    },
                    "center_freq_hz": float(center_freqs[k]),
                },
            )
            source_iq.append(iq)
            labels.append(waveform.label)
            signal_descs.append(desc)

        # Spatial mixing with frequency-dependent steering vectors
        X = self._wideband_spatial_mix(
            source_iq, azimuths, elevations, center_freqs, snrs_db, rng
        )

        # Convert (N, T) complex → (N, 2, T) float32
        tensor = torch.from_numpy(
            np.stack([X.real, X.imag], axis=1).astype(np.float32)
        )

        if self.transform is not None:
            tensor = self.transform(tensor)

        target = WidebandDFTarget(
            azimuths=azimuths,
            elevations=elevations,
            center_freqs=center_freqs,
            snrs=snrs_db,
            num_signals=n_src,
            labels=labels,
            signal_descs=signal_descs,
        )
        return tensor, target

    # ── private helpers ────────────────────────────────────────────────────────

    def _sample_freqs(self, rng: np.random.Generator, n_src: int) -> np.ndarray:
        """Sample center frequencies within ±capture_bandwidth/2."""
        f_min = -self.capture_bandwidth / 2.0
        f_max = self.capture_bandwidth / 2.0

        if self.min_freq_separation is None or n_src == 1:
            return rng.uniform(f_min, f_max, size=n_src)

        max_attempts = 1000
        for _ in range(max_attempts):
            freqs = rng.uniform(f_min, f_max, size=n_src)
            freqs_sorted = np.sort(freqs)
            if np.all(np.diff(freqs_sorted) >= self.min_freq_separation):
                return freqs
        warnings.warn(
            f"Could not satisfy min_freq_separation={self.min_freq_separation:.0f} Hz "
            f"for {n_src} signals after {max_attempts} attempts.",
            UserWarning,
            stacklevel=2,
        )
        return freqs

    def _sample_angles(
        self, rng: np.random.Generator, n_src: int
    ) -> Tuple[np.ndarray, np.ndarray]:
        """Sample azimuth/elevation angles with optional separation constraint."""
        az_min, az_max = self.azimuth_range
        el_min, el_max = self.elevation_range

        if self.min_angular_separation is None or n_src == 1:
            return rng.uniform(az_min, az_max, n_src), rng.uniform(el_min, el_max, n_src)

        max_attempts = 1000
        for _ in range(max_attempts):
            azimuths = rng.uniform(az_min, az_max, n_src)
            elevations = rng.uniform(el_min, el_max, n_src)
            ok = True
            for i in range(n_src):
                for j in range(i + 1, n_src):
                    cos_sep = (
                        np.sin(elevations[i]) * np.sin(elevations[j])
                        + np.cos(elevations[i]) * np.cos(elevations[j])
                        * np.cos(azimuths[i] - azimuths[j])
                    )
                    sep = float(np.arccos(np.clip(cos_sep, -1.0, 1.0)))
                    if sep < self.min_angular_separation:
                        ok = False
                        break
                if not ok:
                    break
            if ok:
                return azimuths, elevations

        warnings.warn(
            f"Could not satisfy min_angular_separation for {n_src} sources.",
            UserWarning,
            stacklevel=2,
        )
        return azimuths, elevations

    def _wideband_steering_vector(
        self, azimuth: float, elevation: float, center_freq: float
    ) -> np.ndarray:
        """Compute frequency-scaled steering vector for a single direction.

        Scales the stored element positions (in wavelengths at reference_frequency)
        to wavelengths at ``reference_frequency + center_freq``, then computes
        phase shifts.

        Returns:
            Complex array, shape ``(N_elements,)``.
        """
        freq_scale = (
            (self.array.reference_frequency + center_freq)
            / self.array.reference_frequency
        )
        x = self.array.positions[:, 0] * freq_scale
        y = self.array.positions[:, 1] * freq_scale
        cos_el = np.cos(elevation)
        cos_az = np.cos(azimuth)
        sin_az = np.sin(azimuth)
        phase_arg = x * cos_el * cos_az + y * cos_el * sin_az
        phase = np.exp(1j * 2 * np.pi * phase_arg)

        # Apply per-element patterns
        pattern = np.zeros(self.array.num_elements, dtype=complex)
        az_arr = np.array([azimuth])
        el_arr = np.array([elevation])
        for i, elem in enumerate(self.array.elements):
            pattern[i] = elem.pattern(az_arr, el_arr)[0]

        return pattern * phase

    def _wideband_spatial_mix(
        self,
        source_iq: List[np.ndarray],
        azimuths: np.ndarray,
        elevations: np.ndarray,
        center_freqs: np.ndarray,
        snrs_db: np.ndarray,
        rng: np.random.Generator,
    ) -> np.ndarray:
        """Sum frequency-shifted signals with frequency-dependent steering + noise."""
        n_elem = self.array.num_elements
        n_snap = self.num_snapshots
        noise_power = 1.0
        noise = np.sqrt(noise_power / 2.0) * (
            rng.standard_normal((n_elem, n_snap))
            + 1j * rng.standard_normal((n_elem, n_snap))
        )
        X = np.zeros((n_elem, n_snap), dtype=complex)
        for iq, az, el, f_k, snr_db in zip(
            source_iq, azimuths, elevations, center_freqs, snrs_db
        ):
            sv = self._wideband_steering_vector(az, el, f_k)
            sig_power = np.mean(np.abs(iq) ** 2)
            if sig_power > 0:
                scale = np.sqrt((10.0 ** (snr_db / 10.0)) * noise_power / sig_power)
                iq = iq * scale
            X += sv[:, np.newaxis] * iq[np.newaxis, :]
        return X + noise
