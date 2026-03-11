import numpy as np
import torch


class PSD:
    """Power Spectral Density via Welch's method.

    Returns a [1, nfft] tensor in dB scale with DC centered.
    """

    def __init__(self, nfft: int = 256, overlap: int = 0):
        self.nfft = nfft
        self.overlap = overlap

    def __call__(self, iq: np.ndarray) -> torch.Tensor:
        n = len(iq)
        nfft = self.nfft
        step = nfft - self.overlap if self.overlap > 0 else nfft

        # Handle short signals by zero-padding
        if n < nfft:
            padded = np.zeros(nfft, dtype=np.complex64)
            padded[:n] = iq
            iq = padded
            n = nfft

        # Segment and window
        window = np.hanning(nfft).astype(np.float32)
        window_power = np.sum(window**2)

        num_segments = max(1, (n - nfft) // step + 1)
        psd_accum = np.zeros(nfft, dtype=np.float64)

        for i in range(num_segments):
            start = i * step
            segment = iq[start : start + nfft]
            windowed = segment * window
            spectrum = np.fft.fft(windowed, n=nfft)
            psd_accum += np.abs(spectrum) ** 2

        psd_accum /= num_segments * window_power

        # fftshift for DC-centered output
        psd_shifted = np.fft.fftshift(psd_accum)

        # Convert to dB
        psd_db = 10.0 * np.log10(psd_shifted + 1e-20)

        return torch.from_numpy(psd_db.astype(np.float32)).unsqueeze(0)
