"""Wigner-Ville Distribution transform (Rust-accelerated)."""

import numpy as np
import torch

from spectra._rust import compute_wvd as _compute_wvd
from spectra.transforms.csp_utils import format_csp_output


class WVD:
    """Wigner-Ville Distribution (time-frequency representation).

    W_x(t, f) = sum_tau x(t+tau) * conj(x(t-tau)) * exp(-j*2*pi*f*tau)

    Computation is performed in Rust for performance.

    Args:
        nfft: FFT size for the frequency axis (number of lag bins). Default 256.
        n_time: Number of output time samples (subsampled from input).
            Default ``None`` (use all input samples).
        output_format: ``"magnitude"`` (C=1), ``"mag_phase"`` (C=2),
            or ``"real_imag"`` (C=2). Default ``"magnitude"``.
        db_scale: Apply ``10 * log10`` to the magnitude (only for
            ``"magnitude"`` and ``"mag_phase"`` formats). Default ``False``.

    Returns:
        ``torch.Tensor`` of shape ``[C, n_time, nfft]``.
    """

    def __init__(
        self,
        nfft: int = 256,
        n_time: int | None = None,
        output_format: str = "magnitude",
        db_scale: bool = False,
    ):
        if output_format not in ("magnitude", "mag_phase", "real_imag"):
            raise ValueError(
                f"Unknown output_format: {output_format!r}. "
                "Supported: 'magnitude', 'mag_phase', 'real_imag'."
            )
        if nfft <= 0:
            raise ValueError(f"nfft must be positive, got {nfft}")
        self.nfft = nfft
        self.n_time = n_time
        self.output_format = output_format
        self.db_scale = db_scale

    def __call__(self, iq: np.ndarray) -> torch.Tensor:
        iq = np.ascontiguousarray(iq, dtype=np.complex64)
        n_time_arg = self.n_time if self.n_time is not None else 0
        wvd_complex = np.asarray(_compute_wvd(iq, self.nfft, n_time_arg))
        return format_csp_output(wvd_complex, self.output_format, self.db_scale)
