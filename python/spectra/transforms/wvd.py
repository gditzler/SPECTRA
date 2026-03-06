"""Wigner-Ville Distribution transform."""
import numpy as np
import torch

from spectra.transforms.csp_utils import format_csp_output


class WVD:
    """Wigner-Ville Distribution (time-frequency representation).

    W_x(t, f) = sum_tau x(t+tau) * conj(x(t-tau)) * exp(-j*2*pi*f*tau)

    Args:
        nfft: FFT size for frequency axis. Default 256.
        n_time: Number of time samples to compute (subsampled from input).
            Default None (use all input samples).
        output_format: "magnitude" (C=1), "log_magnitude" (C=1), or "real_imag" (C=2).

    Returns:
        torch.Tensor of shape [C, n_time, nfft].
    """

    def __init__(self, nfft: int = 256, n_time: int = None, output_format: str = "magnitude"):
        if output_format not in ("magnitude", "log_magnitude", "real_imag"):
            raise ValueError(
                f"Unknown output_format: {output_format!r}. "
                "Supported: 'magnitude', 'log_magnitude', 'real_imag'."
            )
        self.nfft = nfft
        self.n_time = n_time
        self.output_format = output_format

    def __call__(self, iq: np.ndarray) -> torch.Tensor:
        iq = np.ascontiguousarray(iq, dtype=np.complex128)
        N = len(iq)

        # Time indices to compute
        if self.n_time is not None and self.n_time < N:
            time_indices = np.linspace(0, N - 1, self.n_time, dtype=int)
        else:
            time_indices = np.arange(N)

        n_time = len(time_indices)
        wvd = np.zeros((n_time, self.nfft), dtype=np.complex128)

        for i, t in enumerate(time_indices):
            # Maximum lag for this time index
            max_tau = min(t, N - 1 - t, self.nfft // 2 - 1)
            if max_tau <= 0:
                continue
            taus = np.arange(-max_tau, max_tau + 1)
            # Lag product: x(t+tau) * conj(x(t-tau))
            lag_product = iq[t + taus] * np.conj(iq[t - taus])
            # Place in FFT buffer (centered)
            buf = np.zeros(self.nfft, dtype=np.complex128)
            buf_indices = taus % self.nfft
            buf[buf_indices] = lag_product
            wvd[i, :] = np.fft.fft(buf)

        # Apply fftshift along frequency axis for centered display
        wvd = np.fft.fftshift(wvd, axes=1)

        # Format output
        if self.output_format == "log_magnitude":
            mag = np.abs(wvd).astype(np.float32)
            mag = 10.0 * np.log10(mag + 1e-12)
            return torch.from_numpy(mag).unsqueeze(0).float()
        elif self.output_format == "magnitude":
            mag = np.abs(wvd).astype(np.float32)
            return torch.from_numpy(mag).unsqueeze(0).float()
        else:  # real_imag
            ri = np.stack([wvd.real.astype(np.float32), wvd.imag.astype(np.float32)], axis=0)
            return torch.from_numpy(ri).float()
