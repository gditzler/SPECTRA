"""Shared utilities for dataset IQ processing."""
import numpy as np
import torch


def truncate_pad(iq: np.ndarray, num_iq_samples: int) -> np.ndarray:
    """Truncate or zero-pad IQ array to exactly num_iq_samples."""
    iq = iq[:num_iq_samples]
    if len(iq) < num_iq_samples:
        padded = np.zeros(num_iq_samples, dtype=np.complex64)
        padded[: len(iq)] = iq
        return padded
    return iq


def iq_to_tensor(iq: np.ndarray) -> torch.Tensor:
    """Convert complex64 IQ array to [2, N] float32 tensor."""
    buf = np.empty((2, len(iq)), dtype=np.float32)
    buf[0] = iq.real
    buf[1] = iq.imag
    return torch.from_numpy(buf)
