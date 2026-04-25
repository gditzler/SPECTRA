"""SNRSweepDataset: structured (SNR × class × sample) grid for sweep evaluation.

Index layout for S levels, C classes, K samples/cell:
  snr_idx   = idx // (C * K)
  class_idx = (idx % (C * K)) // K
  cell_idx  = idx % K
  seed      = (base_seed, snr_idx, class_idx, cell_idx)
"""

from typing import Callable, List, Optional, Tuple

import numpy as np
import torch

from spectra.datasets._base import BaseIQDataset
from spectra.datasets.iq_utils import iq_to_tensor, truncate_pad
from spectra.impairments.compose import Compose
from spectra.waveforms.base import Waveform


class SNRSweepDataset(BaseIQDataset[Tuple[torch.Tensor, int, float]]):
    """Fixed (SNR × class × sample) grid. __getitem__ returns (Tensor, int, float)."""

    def __init__(
        self,
        waveform_pool: List[Waveform],
        snr_levels: List[float],
        samples_per_cell: int,
        num_iq_samples: int,
        sample_rate: float,
        impairments_fn: Callable[[float], Compose],
        seed: Optional[int] = None,
    ):
        self._C = len(waveform_pool)
        self._S = len(list(snr_levels))
        self._cell_size = self._C * samples_per_cell
        super().__init__(num_samples=self._S * self._cell_size, seed=seed)
        self.waveform_pool = waveform_pool
        self.snr_levels = list(snr_levels)
        self.samples_per_cell = samples_per_cell
        self.num_iq_samples = num_iq_samples
        self.sample_rate = sample_rate
        self.impairments_fn = impairments_fn

    def __getitem__(self, index: int) -> Tuple[torch.Tensor, int, float]:
        snr_idx = index // self._cell_size
        rem = index % self._cell_size
        class_idx = rem // self.samples_per_cell
        cell_idx = rem % self.samples_per_cell

        snr_db = self.snr_levels[snr_idx]
        waveform = self.waveform_pool[class_idx]

        rng = np.random.default_rng(seed=(self.seed, snr_idx, class_idx, cell_idx))
        sps = getattr(waveform, "samples_per_symbol", 8)
        num_symbols = self.num_iq_samples // sps + 1
        sig_seed = int(rng.integers(0, 2**32))
        noise_seed = int(rng.integers(0, 2**32))

        iq = waveform.generate(num_symbols=num_symbols, sample_rate=self.sample_rate, seed=sig_seed)
        iq = truncate_pad(iq, self.num_iq_samples)

        from spectra.scene.signal_desc import SignalDescription

        bw = waveform.bandwidth(self.sample_rate)
        desc = SignalDescription(
            t_start=0.0,
            t_stop=self.num_iq_samples / self.sample_rate,
            f_low=-bw / 2,
            f_high=bw / 2,
            label=waveform.label,
            snr=snr_db,
        )
        # Seed legacy global numpy state so impairments (e.g. AWGN) are reproducible
        np.random.seed(noise_seed)
        impairments = self.impairments_fn(snr_db)
        iq, _ = impairments(iq, desc, sample_rate=self.sample_rate)

        return iq_to_tensor(iq), class_idx, float(snr_db)
