"""Reassigned Gabor Transform."""

import numpy as np
import torch

from spectra._rust import compute_reassigned_gabor as _compute_rgt


class ReassignedGabor:
    """Reassigned Gabor (spectrogram) Transform.

    Applies the reassignment method to the Gabor spectrogram (STFT with
    Gaussian window) to produce a sharper time-frequency representation.
    Each time-frequency cell's energy is relocated to the local centre of
    gravity, concentrating it away from spectral leakage.

    Computation is performed in Rust for performance.

    Args:
        nfft: FFT / window size in samples. Default 256.
        hop_length: Hop size between frames in samples. Default 64.
        sigma: Gaussian window standard deviation in samples.
            Smaller values → sharper time resolution; larger values →
            sharper frequency resolution.  Default ``nfft / 4``.

    Example::

        rgt = ReassignedGabor(nfft=256, hop_length=64, sigma=32.0)
        tensor = rgt(iq_samples)  # [1, 256, n_frames]
    """

    def __init__(
        self,
        nfft: int = 256,
        hop_length: int = 64,
        sigma: float | None = None,
    ):
        if nfft <= 0:
            raise ValueError(f"nfft must be positive, got {nfft}")
        if hop_length <= 0:
            raise ValueError(f"hop_length must be positive, got {hop_length}")
        _sigma = float(nfft) / 4.0 if sigma is None else float(sigma)
        if _sigma <= 0.0:
            raise ValueError(f"sigma must be positive, got {sigma}")
        self.nfft = nfft
        self.hop_length = hop_length
        self.sigma = _sigma

    def __call__(self, iq: np.ndarray) -> torch.Tensor:
        """Apply the Reassigned Gabor Transform to the input IQ signal.

        Args:
            iq: 1-D complex64 IQ array.

        Returns:
            ``torch.Tensor`` of shape ``[1, nfft, n_frames]`` (float32).
            Values are accumulated squared magnitudes (power), DC-centred
            along the frequency axis.
        """
        iq = np.ascontiguousarray(iq, dtype=np.complex64)
        result = np.asarray(_compute_rgt(iq, self.nfft, self.hop_length, self.sigma))
        return torch.from_numpy(result).unsqueeze(0).float()
