from typing import Dict, List, Tuple

import torch

from spectra.datasets.cyclo import CyclostationaryDataset
from spectra.datasets.direction_finding import DirectionFindingDataset, DirectionFindingTarget
from spectra.datasets.folder import SignalFolderDataset
from spectra.datasets.manifest import ManifestDataset
from spectra.datasets.metadata import DatasetMetadata, NarrowbandMetadata, WidebandMetadata
from spectra.datasets.mixing import CutMixDataset, MixUpDataset
from spectra.datasets.narrowband import NarrowbandDataset
from spectra.datasets.sampler import balanced_sampler
from spectra.datasets.snr_sweep import SNRSweepDataset
from spectra.datasets.wideband import WidebandDataset


def collate_fn(
    batch: List[Tuple[torch.Tensor, Dict]],
) -> Tuple[torch.Tensor, List[Dict]]:
    """Custom collate for variable-length detection targets."""
    data = torch.stack([item[0] for item in batch])
    targets = [item[1] for item in batch]
    return data, targets


__all__ = [
    "balanced_sampler",
    "collate_fn",
    "CutMixDataset",
    "CyclostationaryDataset",
    "DatasetMetadata",
    "DirectionFindingDataset",
    "DirectionFindingTarget",
    "ManifestDataset",
    "MixUpDataset",
    "NarrowbandDataset",
    "NarrowbandMetadata",
    "SignalFolderDataset",
    "SNRSweepDataset",
    "WidebandDataset",
    "WidebandMetadata",
]
