"""Choi-Williams Distribution transform."""

import numpy as np
import torch

from spectra._rust import compute_cwd as _compute_cwd
from spectra.transforms.csp_utils import format_csp_output


class CWD:
    """Choi-Williams Distribution (time-frequency representation).

    Computes a cross-term-suppressed time-frequency representation by applying
    an exponential kernel to the Wigner-Ville distribution.  The kernel
    ``exp(-theta^2 * tau^2 / sigma)`` is parameterised by ``sigma``:

    - Small ``sigma`` → strong cross-term suppression, reduced resolution.
    - Large ``sigma`` → approaches the WVD (less suppression, better resolution).

    Computation is performed in Rust for performance.

    Args:
        nfft: FFT size for the frequency axis (number of lag bins). Default 256.
        n_time: Number of output time samples (subsampled from input).
            Default ``None`` (use all input samples).
        sigma: Kernel parameter controlling cross-term suppression.
            Typical range: 0.1–10.0. Default 1.0.
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
        sigma: float = 1.0,
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
        if sigma <= 0:
            raise ValueError(f"sigma must be positive, got {sigma}")
        self.nfft = nfft
        self.n_time = n_time
        self.sigma = sigma
        self.output_format = output_format
        self.db_scale = db_scale

    def __call__(self, iq: np.ndarray) -> torch.Tensor:
        iq = np.ascontiguousarray(iq, dtype=np.complex64)
        n_time_arg = self.n_time if self.n_time is not None else 0
        cwd_complex = np.asarray(_compute_cwd(iq, self.nfft, n_time_arg, self.sigma))
        return format_csp_output(cwd_complex, self.output_format, self.db_scale)
