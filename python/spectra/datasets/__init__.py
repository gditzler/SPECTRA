from typing import Dict, List, Tuple

import torch

from spectra.datasets._base import BaseIQDataset
from spectra.datasets.configs import (
    BaseDatasetConfig,
    DirectionFindingConfig,
    NarrowbandConfig,
    RadarConfig,
    SNRSweepConfig,
    WidebandConfig,
)
from spectra.datasets.cyclo import CyclostationaryDataset
from spectra.datasets.df_snr_sweep import DirectionFindingSNRSweepDataset
from spectra.datasets.direction_finding import DirectionFindingDataset, DirectionFindingTarget
from spectra.datasets.folder import SignalFolderDataset
from spectra.datasets.manifest import ManifestDataset
from spectra.datasets.metadata import DatasetMetadata, NarrowbandMetadata, WidebandMetadata
from spectra.datasets.mixing import CutMixDataset, MixUpDataset
from spectra.datasets.narrowband import NarrowbandDataset
from spectra.datasets.profiling import DatasetProfile, DatasetProfiler
from spectra.datasets.radar import RadarDataset, RadarTarget
from spectra.datasets.radar_pipeline import RadarPipelineDataset, RadarPipelineTarget
from spectra.datasets.sampler import balanced_sampler
from spectra.datasets.snr_sweep import SNRSweepDataset
from spectra.datasets.spectrogram_dataset import SpectrogramDataset
from spectra.datasets.wideband import WidebandDataset
from spectra.datasets.wideband_df import WidebandDFTarget, WidebandDirectionFindingDataset


def collate_fn(
    batch: List[Tuple[torch.Tensor, Dict]],
) -> Tuple[torch.Tensor, List[Dict]]:
    """Custom collate for variable-length detection targets."""
    data = torch.stack([item[0] for item in batch])
    targets = [item[1] for item in batch]
    return data, targets


__all__ = [
    "balanced_sampler",
    "BaseDatasetConfig",
    "BaseIQDataset",
    "collate_fn",
    "CutMixDataset",
    "CyclostationaryDataset",
    "DatasetMetadata",
    "DatasetProfile",
    "DatasetProfiler",
    "DirectionFindingConfig",
    "DirectionFindingDataset",
    "DirectionFindingSNRSweepDataset",
    "DirectionFindingTarget",
    "ManifestDataset",
    "MixUpDataset",
    "NarrowbandConfig",
    "NarrowbandDataset",
    "NarrowbandMetadata",
    "RadarConfig",
    "RadarDataset",
    "RadarPipelineDataset",
    "RadarPipelineTarget",
    "RadarTarget",
    "SignalFolderDataset",
    "SNRSweepConfig",
    "SNRSweepDataset",
    "SpectrogramDataset",
    "WidebandConfig",
    "WidebandDataset",
    "WidebandDFTarget",
    "WidebandDirectionFindingDataset",
    "WidebandMetadata",
]
