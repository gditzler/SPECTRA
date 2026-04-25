"""DirectionFindingSNRSweepDataset: structured SNR × sample grid for DoA evaluation."""

from typing import List, Optional, Tuple, Union

import numpy as np
import torch
from torch.utils.data import Dataset

from spectra.arrays.array import AntennaArray
from spectra.datasets.direction_finding import DirectionFindingTarget, _angular_separation
from spectra.datasets.iq_utils import truncate_pad
from spectra.scene.signal_desc import SignalDescription
from spectra.waveforms.base import Waveform


class DirectionFindingSNRSweepDataset(Dataset[Tuple[torch.Tensor, DirectionFindingTarget, float]]):
    """Structured (SNR level × sample) grid for DoA algorithm evaluation.

    Items are indexed as ``snr_idx * samples_per_snr + sample_idx``.
    Seeding is ``(seed, snr_idx, sample_idx)`` — deterministic and
    DataLoader-worker safe.

    Returns ``(tensor, DirectionFindingTarget, snr_db)`` triples so that
    callers can bucket errors by SNR without any bookkeeping.

    Args:
        array: :class:`~spectra.arrays.array.AntennaArray` geometry.
        signal_pool: Waveforms to draw sources from.
        snr_levels: List of discrete SNR values in dB to sweep.
        samples_per_snr: Number of independent samples per SNR level.
        num_signals: Fixed source count or ``(min, max)`` range.
        num_snapshots: IQ samples per antenna element.
        sample_rate: Receiver sample rate in Hz.
        azimuth_range: ``(min_rad, max_rad)`` for source angles.
        elevation_range: ``(min_rad, max_rad)``. Default ``(0.0, 0.0)`` (2-D).
        min_angular_separation: Minimum pairwise source separation in radians.
        seed: Base seed.
    """

    def __init__(
        self,
        array: AntennaArray,
        signal_pool: List[Waveform],
        snr_levels: List[float],
        samples_per_snr: int,
        num_signals: Union[int, Tuple[int, int]],
        num_snapshots: int,
        sample_rate: float,
        azimuth_range: Tuple[float, float] = (0.0, np.pi),
        elevation_range: Tuple[float, float] = (0.0, 0.0),
        min_angular_separation: Optional[float] = None,
        seed: int = 0,
    ):
        self.array = array
        self.signal_pool = signal_pool
        self.snr_levels = list(snr_levels)
        self.samples_per_snr = samples_per_snr
        self.num_signals = num_signals
        self.num_snapshots = num_snapshots
        self.sample_rate = sample_rate
        self.azimuth_range = azimuth_range
        self.elevation_range = elevation_range
        self.min_angular_separation = min_angular_separation
        self.seed = seed

    def __len__(self) -> int:
        return len(self.snr_levels) * self.samples_per_snr

    def __getitem__(self, index: int) -> Tuple[torch.Tensor, DirectionFindingTarget, float]:
        snr_idx = index // self.samples_per_snr
        sample_idx = index % self.samples_per_snr
        snr_db = float(self.snr_levels[snr_idx])

        rng = np.random.default_rng(seed=(self.seed, snr_idx, sample_idx))

        # Number of sources
        if isinstance(self.num_signals, tuple):
            n_src = int(rng.integers(self.num_signals[0], self.num_signals[1] + 1))
        else:
            n_src = int(self.num_signals)

        # Sample angles
        az_min, az_max = self.azimuth_range
        el_min, el_max = self.elevation_range
        max_attempts = 500
        for _ in range(max_attempts):
            azimuths = rng.uniform(az_min, az_max, size=n_src)
            elevations = rng.uniform(el_min, el_max, size=n_src)
            if self.min_angular_separation is None or n_src == 1:
                break
            ok = True
            for i in range(n_src):
                for j in range(i + 1, n_src):
                    sep = _angular_separation(
                        azimuths[i], elevations[i], azimuths[j], elevations[j]
                    )
                    if sep < self.min_angular_separation:
                        ok = False
                        break
                if not ok:
                    break
            if ok:
                break

        snrs_db = np.full(n_src, snr_db)

        # Generate per-source IQ
        source_iq = []
        labels = []
        signal_descs = []
        for k in range(n_src):
            wf = self.signal_pool[int(rng.integers(0, len(self.signal_pool)))]
            sps = getattr(wf, "samples_per_symbol", 8)
            num_symbols = self.num_snapshots // sps + 1
            sig_seed = int(rng.integers(0, 2**32))
            iq = wf.generate(num_symbols=num_symbols, sample_rate=self.sample_rate, seed=sig_seed)
            iq = truncate_pad(iq, self.num_snapshots)
            bw = wf.bandwidth(self.sample_rate)
            desc = SignalDescription(
                t_start=0.0,
                t_stop=self.num_snapshots / self.sample_rate,
                f_low=-bw / 2,
                f_high=bw / 2,
                label=wf.label,
                snr=snr_db,
                modulation_params={
                    "doa": {
                        "azimuth_rad": float(azimuths[k]),
                        "elevation_rad": float(elevations[k]),
                    }
                },
            )
            source_iq.append(iq)
            labels.append(wf.label)
            signal_descs.append(desc)

        # Spatial mix
        n_elem = self.array.num_elements
        noise_power = 1.0
        noise = np.sqrt(noise_power / 2.0) * (
            rng.standard_normal((n_elem, self.num_snapshots))
            + 1j * rng.standard_normal((n_elem, self.num_snapshots))
        )
        X = np.zeros((n_elem, self.num_snapshots), dtype=complex)
        for iq, az, el, s_db in zip(source_iq, azimuths, elevations, snrs_db):
            sv = self.array.steering_vector(azimuth=az, elevation=el)
            sig_power = np.mean(np.abs(iq) ** 2)
            if sig_power > 0:
                scale = np.sqrt((10.0 ** (s_db / 10.0)) * noise_power / sig_power)
                iq = iq * scale
            X += sv[:, np.newaxis] * iq[np.newaxis, :]
        X += noise

        tensor = torch.from_numpy(
            np.stack([X.real, X.imag], axis=1).astype(np.float32)
        )
        target = DirectionFindingTarget(
            azimuths=azimuths,
            elevations=elevations,
            snrs=snrs_db,
            num_sources=n_src,
            labels=labels,
            signal_descs=signal_descs,
        )
        return tensor, target, snr_db
