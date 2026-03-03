from typing import Callable, Dict, List, Optional, Tuple

import numpy as np
import torch
from torch.utils.data import Dataset

from spectra.impairments.compose import Compose
from spectra.waveforms.base import Waveform


class CyclostationaryDataset(Dataset):
    """Multi-representation dataset for cyclostationary signal processing.

    Unlike :class:`NarrowbandDataset` which applies a single transform,
    this dataset applies *multiple* named representations to the same IQ
    signal and returns a dictionary of tensors.  This supports
    multi-view learning and traditional AMC pipelines that combine
    cumulants, SCD, PSD, etc.

    Args:
        waveform_pool: List of :class:`Waveform` instances to draw from.
        num_samples: Total number of dataset samples.
        num_iq_samples: Length of each IQ capture.
        sample_rate: Sample rate in Hz.
        representations: Mapping of ``name -> callable``.  Each callable
            receives a 1-D ``np.ndarray`` of ``complex64`` and returns a
            ``torch.Tensor`` (e.g. ``{"scd": SCD(), "cum": Cumulants()}``).
        impairments: Optional :class:`Compose` pipeline applied to raw IQ
            before representation extraction.
        target_transform: Optional callable applied to the integer label.
        seed: Base seed for deterministic ``(seed, idx)`` seeding.

    Returns (per ``__getitem__``):
        ``(representations_dict, label)`` where *representations_dict* maps
        each name to its ``torch.Tensor`` output and *label* is an ``int``.
    """

    def __init__(
        self,
        waveform_pool: List[Waveform],
        num_samples: int,
        num_iq_samples: int,
        sample_rate: float,
        representations: Dict[str, Callable],
        impairments: Optional[Compose] = None,
        target_transform: Optional[Callable] = None,
        seed: Optional[int] = None,
    ):
        if not representations:
            raise ValueError("representations must be a non-empty dict")
        self.waveform_pool = waveform_pool
        self.num_samples = num_samples
        self.num_iq_samples = num_iq_samples
        self.sample_rate = sample_rate
        self.representations = representations
        self.impairments = impairments
        self.target_transform = target_transform
        self.seed = seed if seed is not None else 0

    def __len__(self) -> int:
        return self.num_samples

    def __getitem__(self, idx: int) -> Tuple[Dict[str, torch.Tensor], int]:
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

        # Truncate / pad to requested length
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

        # Apply each named representation
        data: Dict[str, torch.Tensor] = {}
        for name, transform in self.representations.items():
            data[name] = transform(iq)

        label = waveform_idx
        if self.target_transform is not None:
            label = self.target_transform(label)

        return data, label
