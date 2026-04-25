"""Base class for on-the-fly IQ datasets."""

from abc import abstractmethod
from typing import Optional, TypeVar

import numpy as np
from torch.utils.data import Dataset

T_item = TypeVar("T_item")


class BaseIQDataset(Dataset[T_item]):
    """Base class providing shared seed management and deterministic RNG creation."""

    def __init__(self, num_samples: int, seed: Optional[int] = None):
        self.num_samples = num_samples
        self.seed = seed if seed is not None else 0

    def __len__(self) -> int:
        return self.num_samples

    def _make_rng(self, index: int) -> np.random.Generator:
        """Create a deterministic RNG from (base_seed, index)."""
        return np.random.default_rng(seed=(self.seed, index))

    @abstractmethod
    def __getitem__(self, index: int) -> T_item:
        ...
