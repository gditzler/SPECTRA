"""Adapter wrapping a TorchSig v2.x iterable dataset into a map-style dataset
matching SPECTRA's (tensor[2,N], int) interface."""
from typing import Callable, List, Optional, Tuple

import numpy as np
import torch
from torch.utils.data import Dataset

from benchmarks.torchsig_compat.label_map import torchsig_to_canonical


class TorchSigAdapter(Dataset):
    """Materialize a TorchSig iterable dataset into a map-style dataset.

    TorchSig v2.x uses ``TorchSigIterableDataset`` which yields
    ``(complex64_array, str_label)`` pairs.  This adapter draws
    *num_samples* from the iterator and stores them as ``(tensor[2, N],
    int_label)`` pairs so they can be indexed, matching SPECTRA's
    ``NarrowbandDataset`` output format.

    Args:
        iterable_dataset: A TorchSig ``TorchSigIterableDataset`` instance.
        num_samples: Number of samples to materialize from the iterator.
        class_list: Canonical class names in index order.
        transform: Optional transform applied to the [2, N] tensor.
    """

    def __init__(
        self,
        iterable_dataset,
        num_samples: int,
        class_list: List[str],
        transform: Optional[Callable] = None,
    ):
        self._class_to_idx = {name: i for i, name in enumerate(class_list)}
        self._transform = transform
        self._data: List[Tuple[torch.Tensor, int]] = []

        for i, (iq, label_str) in enumerate(iterable_dataset):
            if i >= num_samples:
                break
            iq = np.asarray(iq)
            if np.iscomplexobj(iq):
                tensor = torch.tensor(
                    np.stack([iq.real, iq.imag], axis=0), dtype=torch.float32
                )
            else:
                tensor = torch.as_tensor(iq, dtype=torch.float32)
                if tensor.ndim == 2 and tensor.shape[0] != 2:
                    tensor = tensor.T

            try:
                canonical = torchsig_to_canonical(label_str)
                label = self._class_to_idx[canonical]
            except KeyError:
                label = self._class_to_idx.get(label_str, -1)

            self._data.append((tensor, label))

    def __len__(self) -> int:
        return len(self._data)

    def __getitem__(self, idx: int) -> Tuple[torch.Tensor, int]:
        tensor, label = self._data[idx]
        if self._transform is not None:
            tensor = self._transform(tensor)
        return tensor, label
