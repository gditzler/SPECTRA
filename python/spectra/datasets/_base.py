"""Base class for on-the-fly IQ datasets."""

from abc import abstractmethod
from typing import Optional

import numpy as np
from torch.utils.data import Dataset


class BaseIQDataset(Dataset):
    """Base class providing shared seed management and deterministic RNG creation."""

    def __init__(self, num_samples: int, seed: Optional[int] = None):
        self.num_samples = num_samples
        self.seed = seed if seed is not None else 0

    def __len__(self) -> int:
        return self.num_samples

    def _make_rng(self, idx: int) -> np.random.Generator:
        """Create a deterministic RNG from (base_seed, idx)."""
        return np.random.default_rng(seed=(self.seed, idx))

    @abstractmethod
    def __getitem__(self, idx: int):
        ...
