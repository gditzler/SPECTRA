"""Direction-finding dataset for ML-based DoA estimation."""

from __future__ import annotations

import warnings
from dataclasses import dataclass, field
from typing import Callable, List, Optional, Tuple, Union

import numpy as np
import torch

from spectra.arrays.array import AntennaArray
from spectra.arrays.calibration import CalibrationErrors
from spectra.datasets._base import BaseIQDataset
from spectra.datasets.iq_utils import truncate_pad
from spectra.impairments.compose import Compose
from spectra.scene.signal_desc import SignalDescription
from spectra.waveforms.base import Waveform


@dataclass
class DirectionFindingTarget:
    """Ground-truth labels for a direction-finding snapshot.

    Attributes:
        azimuths: Source azimuth angles in radians, shape ``(num_sources,)``.
        elevations: Source elevation angles in radians, shape ``(num_sources,)``.
        snrs: Per-source SNR in dB, shape ``(num_sources,)``.
        num_sources: Number of active sources.
        labels: Modulation label string per source.
        signal_descs: Full :class:`~spectra.scene.signal_desc.SignalDescription`
            per source, with DoA stored in ``modulation_params["doa"]``.
    """

    azimuths: np.ndarray
    elevations: np.ndarray
    snrs: np.ndarray
    num_sources: int
    labels: List[str]
    signal_descs: List[SignalDescription] = field(default_factory=list)


class DirectionFindingDataset(BaseIQDataset[Tuple[torch.Tensor, DirectionFindingTarget]]):
    """On-the-fly direction-finding IQ snapshot dataset.

    Generates multi-antenna IQ snapshots deterministically from
    ``(base_seed, idx)`` pairs using ``np.random.default_rng``. Safe for
    use with ``num_workers > 0``.

    .. note::
        PyTorch's ``default_collate`` cannot batch :class:`DirectionFindingTarget`
        directly. Use a custom ``collate_fn`` that returns targets as a list::

            def collate_fn(batch):
                return torch.stack([x for x, _ in batch]), [t for _, t in batch]

            loader = DataLoader(dataset, batch_size=8, collate_fn=collate_fn)

    The output tensor has shape ``[n_elements, 2, num_snapshots]`` where
    channel 0 is I and channel 1 is Q. Each element's IQ is formed by
    mixing the weighted steering-vector contributions of all sources.

    Args:
        array: :class:`~spectra.arrays.array.AntennaArray` defining the
            geometry and element patterns.
        signal_pool: List of :class:`~spectra.waveforms.base.Waveform`
            instances to sample from for each source.
        num_signals: Fixed number of sources (int) or ``(min, max)`` range
            (inclusive) to draw uniformly.
        num_snapshots: Number of IQ samples per antenna element.
        sample_rate: Receiver sample rate in Hz.
        snr_range: ``(min_db, max_db)`` per-source SNR drawn uniformly.
        azimuth_range: ``(min_rad, max_rad)`` azimuth sampling range.
            Defaults to ``(0, 2*pi)`` (full circle).
        elevation_range: ``(min_rad, max_rad)`` elevation range.
            Defaults to ``(-pi/2, pi/2)``.
        min_angular_separation: Minimum angular separation between sources in
            radians. If ``None``, no constraint is applied.
        calibration_errors: Optional :class:`~spectra.arrays.calibration.CalibrationErrors`
            to apply to every steering vector.
        impairments: Optional per-signal :class:`~spectra.impairments.compose.Compose`
            pipeline applied before spatial mixing.
        transform: Optional callable applied to the output tensor.
        num_samples: Total dataset size.
        seed: Base integer seed.
    """

    def __init__(
        self,
        array: AntennaArray,
        signal_pool: List[Waveform],
        num_signals: Union[int, Tuple[int, int]],
        num_snapshots: int,
        sample_rate: float,
        snr_range: Tuple[float, float],
        azimuth_range: Tuple[float, float] = (0.0, 2 * np.pi),
        elevation_range: Tuple[float, float] = (-np.pi / 2, np.pi / 2),
        min_angular_separation: Optional[float] = None,
        calibration_errors: Optional[CalibrationErrors] = None,
        impairments: Optional[Compose] = None,
        transform: Optional[Callable] = None,
        num_samples: int = 10000,
        seed: int = 0,
    ):
        if num_snapshots <= 0:
            raise ValueError(f"num_snapshots must be positive, got {num_snapshots}")
        super().__init__(num_samples=num_samples, seed=seed)
        self.array = array
        self.signal_pool = signal_pool
        self.num_signals = num_signals
        self.num_snapshots = num_snapshots
        self.sample_rate = sample_rate
        self.snr_range = snr_range
        self.azimuth_range = azimuth_range
        self.elevation_range = elevation_range
        self.min_angular_separation = min_angular_separation
        self.calibration_errors = calibration_errors
        self.impairments = impairments
        self.transform = transform

    def __getitem__(self, index: int) -> Tuple[torch.Tensor, DirectionFindingTarget]:
        rng = self._make_rng(index)

        # --- Determine number of sources ---
        if isinstance(self.num_signals, tuple):
            n_sources = int(rng.integers(self.num_signals[0], self.num_signals[1] + 1))
        else:
            n_sources = int(self.num_signals)

        # --- Sample angles, SNRs, waveforms for each source ---
        azimuths, elevations = self._sample_angles(rng, n_sources)
        snrs_db = rng.uniform(self.snr_range[0], self.snr_range[1], size=n_sources)

        # --- Generate per-source IQ signals ---
        signal_descs = []
        source_iq = []
        labels = []

        for k in range(n_sources):
            wf_idx = int(rng.integers(0, len(self.signal_pool)))
            waveform = self.signal_pool[wf_idx]
            sps = getattr(waveform, "samples_per_symbol", 8)
            num_symbols = self.num_snapshots // sps + 1
            sig_seed = int(rng.integers(0, 2**32))

            iq = waveform.generate(
                num_symbols=num_symbols,
                sample_rate=self.sample_rate,
                seed=sig_seed,
            )
            iq = truncate_pad(iq, self.num_snapshots)

            bw = waveform.bandwidth(self.sample_rate)
            desc = SignalDescription(
                t_start=0.0,
                t_stop=self.num_snapshots / self.sample_rate,
                f_low=-bw / 2,
                f_high=bw / 2,
                label=waveform.label,
                snr=float(snrs_db[k]),
                modulation_params={
                    "doa": {
                        "azimuth_rad": float(azimuths[k]),
                        "elevation_rad": float(elevations[k]),
                        "azimuth_spread_rad": None,
                        "elevation_spread_rad": None,
                    }
                },
            )

            if self.impairments is not None:
                iq, desc = self.impairments(iq, desc, sample_rate=self.sample_rate)

            source_iq.append(iq)
            signal_descs.append(desc)
            labels.append(waveform.label)

        # --- Spatial mixing ---
        X = self._spatial_mix(source_iq, azimuths, elevations, snrs_db, rng)
        # X shape: (n_elements, num_snapshots) complex

        # --- Convert to [n_elements, 2, num_snapshots] float32 ---
        tensor = np.stack([X.real, X.imag], axis=1).astype(np.float32)
        out = torch.from_numpy(tensor)

        if self.transform is not None:
            out = self.transform(out)

        target = DirectionFindingTarget(
            azimuths=azimuths,
            elevations=elevations,
            snrs=snrs_db,
            num_sources=n_sources,
            labels=labels,
            signal_descs=signal_descs,
        )
        return out, target

    def _sample_angles(
        self, rng: np.random.Generator, n_sources: int
    ) -> Tuple[np.ndarray, np.ndarray]:
        """Sample source angles with optional minimum angular separation constraint."""
        az_min, az_max = self.azimuth_range
        el_min, el_max = self.elevation_range

        if self.min_angular_separation is None or n_sources == 1:
            azimuths = rng.uniform(az_min, az_max, size=n_sources)
            elevations = rng.uniform(el_min, el_max, size=n_sources)
            return azimuths, elevations

        # Rejection sampling with max attempts
        max_attempts = 1000
        for _ in range(max_attempts):
            azimuths = rng.uniform(az_min, az_max, size=n_sources)
            elevations = rng.uniform(el_min, el_max, size=n_sources)
            if self._angles_are_separated(azimuths, elevations):
                return azimuths, elevations

        # Fallback: return last draw even if separation not met
        warnings.warn(
            f"Could not satisfy min_angular_separation={self.min_angular_separation:.4f} rad "
            f"for {n_sources} sources after {max_attempts} attempts. "
            "Returning angles that may violate the separation constraint.",
            UserWarning,
            stacklevel=2,
        )
        return azimuths, elevations

    def _angles_are_separated(
        self, azimuths: np.ndarray, elevations: np.ndarray
    ) -> bool:
        """Check that all pairs of angles have at least min_angular_separation."""
        assert self.min_angular_separation is not None
        n = len(azimuths)
        for i in range(n):
            for j in range(i + 1, n):
                sep = _angular_separation(
                    azimuths[i], elevations[i], azimuths[j], elevations[j]
                )
                if sep < self.min_angular_separation:
                    return False
        return True

    def _spatial_mix(
        self,
        source_iq: List[np.ndarray],
        azimuths: np.ndarray,
        elevations: np.ndarray,
        snrs_db: np.ndarray,
        rng: np.random.Generator,
    ) -> np.ndarray:
        """Compute the received multi-element signal X = sum_k a_k * s_k^T + N.

        Each source is amplitude-scaled against a fixed noise floor of 1.0 so
        that its per-element contribution achieves the target SNR (in dB). The
        returned SNR is therefore relative to that fixed noise floor.

        Returns:
            Complex array of shape ``(n_elements, num_snapshots)``.
        """
        n_elem = self.array.num_elements
        n_snap = self.num_snapshots
        noise_power = 1.0  # fixed reference noise power

        # Per-element independent complex Gaussian noise
        noise = np.sqrt(noise_power / 2.0) * (
            rng.standard_normal((n_elem, n_snap))
            + 1j * rng.standard_normal((n_elem, n_snap))
        )

        X = np.zeros((n_elem, n_snap), dtype=complex)
        for iq, az, el, snr_db in zip(source_iq, azimuths, elevations, snrs_db):
            sv = self.array.steering_vector(azimuth=az, elevation=el)  # (N,)
            if self.calibration_errors is not None:
                sv = self.calibration_errors.apply(sv)
            sig_power = np.mean(np.abs(iq) ** 2)
            if sig_power > 0:
                snr_linear = 10.0 ** (snr_db / 10.0)
                scale = np.sqrt(snr_linear * noise_power / sig_power)
                iq_scaled = iq * scale
            else:
                iq_scaled = iq
            X += sv[:, np.newaxis] * iq_scaled[np.newaxis, :]

        return X + noise


def _angular_separation(
    az1: float, el1: float, az2: float, el2: float
) -> float:
    """Great-circle angular separation between two directions in radians."""
    cos_sep = (
        np.sin(el1) * np.sin(el2)
        + np.cos(el1) * np.cos(el2) * np.cos(az1 - az2)
    )
    return float(np.arccos(np.clip(cos_sep, -1.0, 1.0)))
