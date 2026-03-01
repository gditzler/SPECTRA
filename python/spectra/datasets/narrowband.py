from typing import Callable, List, Optional, Tuple

import numpy as np
import torch
from torch.utils.data import Dataset

from spectra.impairments.compose import Compose
from spectra.waveforms.base import Waveform


class NarrowbandDataset(Dataset):
    def __init__(
        self,
        waveform_pool: List[Waveform],
        num_samples: int,
        num_iq_samples: int,
        sample_rate: float,
        impairments: Optional[Compose] = None,
        transform: Optional[Callable] = None,
        target_transform: Optional[Callable] = None,
        seed: Optional[int] = None,
    ):
        self.waveform_pool = waveform_pool
        self.num_samples = num_samples
        self.num_iq_samples = num_iq_samples
        self.sample_rate = sample_rate
        self.impairments = impairments
        self.transform = transform
        self.target_transform = target_transform
        self.seed = seed if seed is not None else 0

    def __len__(self) -> int:
        return self.num_samples

    def __getitem__(self, idx: int) -> Tuple[torch.Tensor, int]:
        rng = np.random.default_rng(seed=(self.seed, idx))

        # Pick a waveform class
        waveform_idx = int(rng.integers(0, len(self.waveform_pool)))
        waveform = self.waveform_pool[waveform_idx]

        # Generate enough symbols
        sps = getattr(waveform, "samples_per_symbol", 8)
        num_symbols = self.num_iq_samples // sps + 1
        sig_seed = int(rng.integers(0, 2**32))

        iq = waveform.generate(
            num_symbols=num_symbols,
            sample_rate=self.sample_rate,
            seed=sig_seed,
        )

        # Truncate to requested length
        iq = iq[: self.num_iq_samples]
        if len(iq) < self.num_iq_samples:
            padded = np.zeros(self.num_iq_samples, dtype=np.complex64)
            padded[: len(iq)] = iq
            iq = padded

        # Apply impairments
        if self.impairments is not None:
            from spectra.scene.signal_desc import SignalDescription
            desc = SignalDescription(
                t_start=0.0,
                t_stop=self.num_iq_samples / self.sample_rate,
                f_low=-waveform.bandwidth(self.sample_rate) / 2,
                f_high=waveform.bandwidth(self.sample_rate) / 2,
                label=waveform.label,
                snr=0.0,
            )
            iq, _ = self.impairments(iq, desc, sample_rate=self.sample_rate)

        # Convert to tensor: [2, num_iq_samples] (I and Q channels)
        if self.transform is not None:
            data = self.transform(iq)
        else:
            data = torch.tensor(
                np.stack([iq.real, iq.imag]), dtype=torch.float32
            )

        label = waveform_idx
        if self.target_transform is not None:
            label = self.target_transform(label)

        return data, label
