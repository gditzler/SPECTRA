"""Balanced sampling utilities for classification datasets."""

from typing import Any, Optional

import torch
from torch.utils.data import WeightedRandomSampler


def balanced_sampler(
    dataset: Any,
    num_classes: int,
    num_samples: Optional[int] = None,
) -> WeightedRandomSampler:
    """Create a WeightedRandomSampler with inverse-frequency weights.

    Args:
        dataset: A dataset with ``len()`` support. For NarrowbandDataset,
            classes are assumed uniformly distributed across the waveform pool.
        num_classes: Number of classes in the dataset.
        num_samples: Number of samples to draw per epoch. Defaults to
            ``len(dataset)``.

    Returns:
        A ``WeightedRandomSampler`` suitable for ``torch.utils.data.DataLoader``.
    """
    n = len(dataset)
    if num_samples is None:
        num_samples = n
    # Uniform weight per sample — each class gets equal total weight
    weight_per_sample = 1.0 / num_classes
    weights = torch.full((n,), weight_per_sample, dtype=torch.double)
    # WeightedRandomSampler accepts a tensor at runtime; its annotation says
    # Sequence[float], so cast through a list to satisfy the type checker.
    return WeightedRandomSampler(weights.tolist(), num_samples=num_samples, replacement=True)
