import hashlib
from typing import Any, Callable, Dict, Optional

from torch.utils.data import DataLoader, Dataset

from spectra.curriculum import CurriculumSchedule


def _epoch_seed(base_seed: int, epoch: int) -> int:
    """Deterministic epoch seed via hash.

    Uses SHA-256 to avoid Python's hash() randomization (PYTHONHASHSEED).
    Returns a positive integer in numpy's valid seed range.
    """
    h = hashlib.sha256(f"{base_seed}:{epoch}".encode()).hexdigest()
    return int(h[:8], 16)  # 32-bit unsigned int from first 8 hex chars


class StreamingDataLoader:
    """Epoch-aware DataLoader wrapper with optional curriculum scheduling.

    Each call to `.epoch(n)` builds a fresh dataset via the factory,
    injecting a unique deterministic seed and curriculum parameters.

    Parameters
    ----------
    dataset_factory : callable
        ``f(params: dict) -> Dataset``. Called once per epoch. The params dict
        always contains ``"seed"`` (int). If a curriculum is provided, it also
        contains the curriculum's interpolated parameters for that epoch.
    base_seed : int
        Root seed for deterministic generation.
    num_epochs : int
        Total number of epochs (used to compute curriculum progress).
    curriculum : CurriculumSchedule, optional
        Difficulty schedule. If None, only seed varies per epoch.
    **dataloader_kwargs
        Forwarded to ``torch.utils.data.DataLoader`` (batch_size, num_workers, etc.).
    """

    def __init__(
        self,
        dataset_factory: Callable[[Dict[str, Any]], Dataset],
        base_seed: int,
        num_epochs: int,
        curriculum: Optional[CurriculumSchedule] = None,
        **dataloader_kwargs: Any,
    ):
        self.dataset_factory = dataset_factory
        self.base_seed = base_seed
        self.num_epochs = num_epochs
        self.curriculum = curriculum
        self.dataloader_kwargs = dataloader_kwargs

    def epoch(self, epoch: int) -> DataLoader:
        """Build and return a DataLoader for the given epoch.

        Parameters
        ----------
        epoch : int
            Current epoch index (0-based).

        Returns
        -------
        DataLoader
            A standard PyTorch DataLoader wrapping the factory-built dataset.
        """
        seed = _epoch_seed(self.base_seed, epoch)

        params: Dict[str, Any] = {"seed": seed}

        if self.curriculum is not None:
            if self.num_epochs <= 1:
                progress = 0.0
            else:
                progress = epoch / (self.num_epochs - 1)
            curriculum_params = self.curriculum.at(progress)
            params.update(curriculum_params)

        dataset = self.dataset_factory(params)
        return DataLoader(dataset, **self.dataloader_kwargs)
