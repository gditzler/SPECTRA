"""Ambiguity Function transform."""
import numpy as np
import torch

from spectra.transforms.csp_utils import format_csp_output


class AmbiguityFunction:
    """Ambiguity Function (delay-Doppler representation).

    A_x(tau, nu) = sum_t x(t+tau/2) * conj(x(t-tau/2)) * exp(j*2*pi*nu*t)

    For discrete signals with integer lags:
    A_x(tau, nu) = sum_t x(t+tau) * conj(x(t)) * exp(j*2*pi*nu*t)

    Args:
        max_lag: Maximum delay in samples. Default 128.
        n_doppler: Number of Doppler bins. Default 256.
        output_format: "magnitude" (C=1), "mag_phase" (C=2), or "real_imag" (C=2).

    Returns:
        torch.Tensor of shape [C, n_doppler, 2*max_lag+1].
    """

    def __init__(self, max_lag: int = 128, n_doppler: int = 256, output_format: str = "magnitude"):
        if output_format not in ("magnitude", "mag_phase", "real_imag"):
            raise ValueError(
                f"Unknown output_format: {output_format!r}. "
                "Supported: 'magnitude', 'mag_phase', 'real_imag'."
            )
        self.max_lag = max_lag
        self.n_doppler = n_doppler
        self.output_format = output_format

    def __call__(self, iq: np.ndarray) -> torch.Tensor:
        iq = np.ascontiguousarray(iq, dtype=np.complex128)
        N = len(iq)
        n_lags = 2 * self.max_lag + 1

        ambiguity = np.zeros((self.n_doppler, n_lags), dtype=np.complex128)

        for lag_idx, tau in enumerate(range(-self.max_lag, self.max_lag + 1)):
            # Compute lag product: x(t+tau) * conj(x(t))
            if tau >= 0:
                t_start = 0
                t_end = N - tau
                lag_product = iq[t_start + tau : t_end + tau] * np.conj(iq[t_start:t_end])
            else:
                t_start = -tau
                t_end = N
                lag_product = iq[t_start + tau : t_end + tau] * np.conj(iq[t_start:t_end])

            # FFT for Doppler axis (truncates or zero-pads to n_doppler)
            ambiguity[:, lag_idx] = np.fft.fftshift(np.fft.fft(lag_product, n=self.n_doppler))

        return format_csp_output(ambiguity, self.output_format)
