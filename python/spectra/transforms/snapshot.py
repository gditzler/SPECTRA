"""Snapshot matrix transform for DoA algorithm input."""

import numpy as np


class ToSnapshotMatrix:
    """Convert a real ``[n_elements, 2, num_snapshots]`` tensor to a complex
    ``[n_elements, num_snapshots]`` snapshot matrix.

    The input format (I and Q channels separated in dimension 1) matches the
    output of :class:`~spectra.datasets.DirectionFindingDataset`.
    The output format is suitable as input to classical DoA algorithms
    (MUSIC, ESPRIT, Capon).

    Example::

        transform = ToSnapshotMatrix()
        X = transform(data)  # data: [N, 2, T] → X: complex [N, T]
        R = (X @ X.conj().T) / X.shape[1]  # sample covariance
    """

    def __call__(self, x: np.ndarray) -> np.ndarray:
        """
        Args:
            x: Real array of shape ``[n_elements, 2, num_snapshots]``.
                Channel 0 is I, channel 1 is Q.

        Returns:
            Complex array of shape ``[n_elements, num_snapshots]``.
        """
        return x[:, 0, :] + 1j * x[:, 1, :]
