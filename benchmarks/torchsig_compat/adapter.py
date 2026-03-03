"""Adapter wrapping a TorchSig dataset to match SPECTRA's (tensor[2,N], int) interface."""
from typing import Callable, List, Optional, Tuple

import numpy as np
import torch
from torch.utils.data import Dataset

from benchmarks.torchsig_compat.label_map import torchsig_to_canonical


class TorchSigAdapter(Dataset):
    """Wrap a TorchSig dataset to return (tensor[2, N], int_label).

    TorchSig datasets return (complex_iq, metadata_dict). This adapter
    converts the complex IQ to a real-valued [2, N] tensor (matching
    SPECTRA's NarrowbandDataset format) and extracts the integer class
    label from the metadata.

    Args:
        dataset: A TorchSig dataset instance.
        class_list: Canonical class names in index order.
        class_key: Metadata key containing the TorchSig class string.
        transform: Optional transform applied to the [2, N] tensor.
    """

    def __init__(
        self,
        dataset,
        class_list: List[str],
        class_key: str = "class_name",
        transform: Optional[Callable] = None,
    ):
        self._dataset = dataset
        self._class_to_idx = {name: i for i, name in enumerate(class_list)}
        self._class_key = class_key
        self._transform = transform

    def __len__(self) -> int:
        return len(self._dataset)

    def __getitem__(self, idx: int) -> Tuple[torch.Tensor, int]:
        iq, meta = self._dataset[idx]

        # Convert complex IQ -> [2, N] real tensor
        iq = np.asarray(iq)
        if np.iscomplexobj(iq):
            data = torch.tensor(
                np.stack([iq.real, iq.imag], axis=0), dtype=torch.float32
            )
        else:
            data = torch.as_tensor(iq, dtype=torch.float32)
            if data.ndim == 2 and data.shape[0] != 2:
                data = data.T

        # Extract label
        class_str = meta[self._class_key]
        try:
            canonical = torchsig_to_canonical(class_str)
            label = self._class_to_idx[canonical]
        except KeyError:
            label = self._class_to_idx.get(class_str, -1)

        if self._transform is not None:
            data = self._transform(data)

        return data, label
