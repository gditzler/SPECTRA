import numpy as np
import torch


class ComplexTo2D:
    """Convert complex IQ array to [2, N] tensor (I and Q channels)."""

    def __call__(self, iq: np.ndarray) -> torch.Tensor:
        i = iq.real.astype(np.float32)
        q = iq.imag.astype(np.float32)
        return torch.from_numpy(np.stack([i, q], axis=0))
