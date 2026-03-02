import numpy as np


class Normalize:
    """Zero-mean, unit-variance normalization per sample."""

    def __call__(self, iq: np.ndarray) -> np.ndarray:
        mean = np.mean(iq)
        std = np.std(iq)
        if std > 0:
            return ((iq - mean) / std).astype(iq.dtype)
        return (iq - mean).astype(iq.dtype)
