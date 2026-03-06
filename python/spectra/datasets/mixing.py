"""Dataset wrappers for cross-sample MixUp and CutMix augmentations."""
from typing import Tuple

import numpy as np
import torch
from torch.utils.data import Dataset


class MixUpDataset(Dataset):
    """Wraps a classification dataset to apply MixUp between two samples.

    For each index, draws a second sample and blends the two tensors using
    a Beta-distributed mixing coefficient. Returns soft labels as a 3-tuple.

    Args:
        dataset: Base classification dataset returning ``(tensor, label)``.
        alpha: Beta distribution parameter for mixing coefficient.
    """

    def __init__(self, dataset: Dataset, alpha: float = 0.2):
        self._dataset = dataset
        self._alpha = alpha

    def __len__(self) -> int:
        return len(self._dataset)

    def __getitem__(self, idx: int) -> Tuple[torch.Tensor, Tuple[int, int, float]]:
        rng = np.random.default_rng(seed=(45, idx))
        x1, y1 = self._dataset[idx]
        idx2 = int(rng.integers(0, len(self._dataset)))
        x2, y2 = self._dataset[idx2]
        lam = float(rng.beta(self._alpha, self._alpha))
        mixed = lam * x1 + (1 - lam) * x2
        return mixed, (y1, y2, lam)


class CutMixDataset(Dataset):
    """Wraps a classification dataset to apply CutMix between two samples.

    For each index, draws a second sample and replaces a random segment
    of the first with the corresponding segment of the second.

    Args:
        dataset: Base classification dataset returning ``(tensor, label)``.
        alpha: Beta distribution parameter for cut ratio.
    """

    def __init__(self, dataset: Dataset, alpha: float = 1.0):
        self._dataset = dataset
        self._alpha = alpha

    def __len__(self) -> int:
        return len(self._dataset)

    def __getitem__(self, idx: int) -> Tuple[torch.Tensor, Tuple[int, int, float]]:
        rng = np.random.default_rng(seed=(45, idx))
        x1, y1 = self._dataset[idx]
        idx2 = int(rng.integers(0, len(self._dataset)))
        x2, y2 = self._dataset[idx2]
        lam = float(rng.beta(self._alpha, self._alpha))
        # CutMix: replace a segment
        n = x1.shape[-1]  # last dimension is sample count
        cut_len = int((1 - lam) * n)
        start = int(rng.integers(0, max(1, n - cut_len)))
        mixed = x1.clone()
        mixed[..., start : start + cut_len] = x2[..., start : start + cut_len]
        # Actual lambda based on cut proportion
        actual_lam = 1.0 - cut_len / n
        return mixed, (y1, y2, actual_lam)
