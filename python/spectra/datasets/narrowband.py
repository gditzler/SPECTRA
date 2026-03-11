from typing import Callable, Dict, List, Optional, Tuple

import numpy as np
import torch
from torch.utils.data import Dataset

from spectra.datasets.iq_utils import iq_to_tensor, truncate_pad
from spectra.impairments.compose import Compose
from spectra.waveforms.base import Waveform


class NarrowbandDataset(Dataset):
    """On-the-fly narrowband IQ dataset for AMC classification.

    Generates signals deterministically from ``(base_seed, idx)`` pairs using
    NumPy's ``default_rng(seed=(base_seed, idx))``. This makes the dataset safe
    for use with ``num_workers > 0`` — every worker produces the same sample
    for a given index regardless of process ordering.

    Args:
        waveform_pool: List of :class:`~spectra.waveforms.base.Waveform` instances.
            Class labels are the pool indices.
        num_samples: Total dataset size (number of IQ segments).
        num_iq_samples: Number of complex samples per segment.
        sample_rate: Receiver sample rate in Hz.
        impairments: Optional :class:`~spectra.impairments.compose.Compose`
            pipeline applied after generation.
        transform: Optional callable applied to the IQ tensor before returning.
            If ``None``, returns ``Tensor[2, num_iq_samples]`` (I/Q channels).
        target_transform: Optional callable applied to the integer class label.
        seed: Base integer seed. ``None`` gives non-deterministic behavior.

    Returns (from ``__getitem__``):
        Tuple of ``(iq_tensor, label_int)`` where ``iq_tensor`` has shape
        ``[2, num_iq_samples]`` (channel 0 = I, channel 1 = Q) and dtype
        ``float32``. If ``transform`` is applied (e.g., ``STFT``), shape changes
        accordingly.
    """

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
        class_weights: Optional[List[float]] = None,
        mimo_config: Optional[Dict] = None,
    ):
        self.waveform_pool = waveform_pool
        self.num_samples = num_samples
        self.num_iq_samples = num_iq_samples
        self.sample_rate = sample_rate
        self.impairments = impairments
        self.transform = transform
        self.target_transform = target_transform
        self.seed = seed if seed is not None else 0
        self.class_weights = class_weights
        self.mimo_config = mimo_config

    def __len__(self) -> int:
        return self.num_samples

    def __getitem__(self, idx: int) -> Tuple[torch.Tensor, int]:
        rng = np.random.default_rng(seed=(self.seed, idx))

        # Pick a waveform class
        if self.class_weights is not None:
            probs = np.array(self.class_weights, dtype=np.float64)
            probs /= probs.sum()
            waveform_idx = int(rng.choice(len(self.waveform_pool), p=probs))
        else:
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
        iq = truncate_pad(iq, self.num_iq_samples)

        # MIMO channel (optional)
        if self.mimo_config is not None:
            from spectra.impairments.mimo_channel import MIMOChannel

            n_tx = self.mimo_config.get("n_tx", 2)
            # Generate n_tx streams with different sub-seeds
            streams = np.stack(
                [
                    truncate_pad(
                        waveform.generate(
                            num_symbols=num_symbols,
                            sample_rate=self.sample_rate,
                            seed=sig_seed + tx,
                        ),
                        self.num_iq_samples,
                    )
                    for tx in range(n_tx)
                ]
            )  # (n_tx, N)
            from spectra.scene.signal_desc import SignalDescription as SD

            bw = waveform.bandwidth(self.sample_rate)
            mimo_desc = SD(
                t_start=0.0,
                t_stop=self.num_iq_samples / self.sample_rate,
                f_low=-bw / 2,
                f_high=bw / 2,
                label=waveform.label,
                snr=0.0,
            )
            mimo_ch = MIMOChannel(**{k: v for k, v in self.mimo_config.items()})
            rx_streams, _ = mimo_ch(streams, mimo_desc, sample_rate=self.sample_rate)
            # rx_streams: (n_rx, N)
            # Convert each RX antenna to [2, N] and stack -> [n_rx*2, N]
            data = torch.cat(
                [
                    torch.tensor(np.stack([rx.real, rx.imag]).astype(np.float32))
                    for rx in rx_streams
                ],
                dim=0,
            )

            label = waveform_idx
            if self.target_transform is not None:
                label = self.target_transform(label)
            return data, label

        # Apply impairments
        if self.impairments is not None:
            from spectra.scene.signal_desc import SignalDescription

            bw = waveform.bandwidth(self.sample_rate)
            desc = SignalDescription(
                t_start=0.0,
                t_stop=self.num_iq_samples / self.sample_rate,
                f_low=-bw / 2,
                f_high=bw / 2,
                label=waveform.label,
                snr=0.0,
            )
            iq, _ = self.impairments(iq, desc, sample_rate=self.sample_rate)

        # Convert to tensor: [2, num_iq_samples] (I and Q channels)
        if self.transform is not None:
            data = self.transform(iq)
        else:
            data = iq_to_tensor(iq)

        label = waveform_idx
        if self.target_transform is not None:
            label = self.target_transform(label)

        return data, label
