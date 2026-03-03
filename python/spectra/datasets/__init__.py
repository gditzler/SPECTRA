from typing import Dict, List, Tuple

import torch

from spectra.datasets.cyclo import CyclostationaryDataset
from spectra.datasets.metadata import DatasetMetadata, NarrowbandMetadata, WidebandMetadata
from spectra.datasets.narrowband import NarrowbandDataset
from spectra.datasets.wideband import WidebandDataset


def collate_fn(
    batch: List[Tuple[torch.Tensor, Dict]],
) -> Tuple[torch.Tensor, List[Dict]]:
    """Custom collate for variable-length detection targets."""
    data = torch.stack([item[0] for item in batch])
    targets = [item[1] for item in batch]
    return data, targets


__all__ = [
    "collate_fn",
    "CyclostationaryDataset",
    "DatasetMetadata",
    "NarrowbandDataset",
    "NarrowbandMetadata",
    "WidebandDataset",
    "WidebandMetadata",
]
