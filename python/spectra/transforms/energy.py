import numpy as np
import torch

from spectra._rust import compute_psd_welch as _compute_psd_welch


class PSD:
    """Power Spectral Density via Welch's averaged-periodogram method.

    Uses a Rust-backed Hann-windowed FFT with segment overlap.  The output is
    DC-centred.

    Args:
        nfft: FFT size per segment.
        overlap: Number of overlapping samples between adjacent segments.
        db_scale: If ``True``, convert to dB (``10 * log10``).

    Returns:
        ``torch.Tensor`` of shape ``[1, nfft]`` (``float32``).
    """

    def __init__(self, nfft: int = 256, overlap: int = 128, db_scale: bool = False):
        self.nfft = nfft
        self.overlap = overlap
        self.db_scale = db_scale

    def __call__(self, iq: np.ndarray) -> torch.Tensor:
        iq = np.ascontiguousarray(iq, dtype=np.complex64)
        psd = np.asarray(_compute_psd_welch(iq, self.nfft, self.overlap))
        if self.db_scale:
            psd = 10.0 * np.log10(psd + 1e-12)
        return torch.from_numpy(psd.astype(np.float32)).unsqueeze(0)


class EnergyDetector:
    """Energy-based signal detector using PSD thresholding.

    Computes the PSD in dB and returns a binary detection mask where bins
    exceeding the threshold are marked as detections.

    Args:
        nfft: FFT size per segment.
        overlap: Number of overlapping samples between adjacent segments.
        threshold_db: Detection threshold in dB.  Bins with PSD above
            this value (relative to the median noise floor) are flagged.
    """

    def __init__(self, nfft: int = 256, overlap: int = 128, threshold_db: float = 6.0):
        self.nfft = nfft
        self.overlap = overlap
        self.threshold_db = threshold_db

    def __call__(self, iq: np.ndarray) -> torch.Tensor:
        """Run energy detection on the input IQ signal.

        Args:
            iq: 1-D complex64 IQ array.

        Returns:
            ``torch.Tensor`` of shape ``[1, nfft]`` (``float32``), values in {0, 1}.
        """
        iq = np.ascontiguousarray(iq, dtype=np.complex64)
        psd = np.asarray(_compute_psd_welch(iq, self.nfft, self.overlap))
        psd_db = 10.0 * np.log10(psd + 1e-12)
        # Use median as noise floor estimate
        noise_floor = float(np.median(psd_db))
        mask = (psd_db > noise_floor + self.threshold_db).astype(np.float32)
        return torch.from_numpy(mask).unsqueeze(0)
