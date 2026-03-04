import numpy as np
import torch

from spectra._rust import compute_caf as _compute_caf
from spectra.transforms.csp_utils import format_csp_output


class CAF:
    """Cyclic Autocorrelation Function transform.

    Computes the CAF, which is the Fourier transform of the
    lag-product sequence with respect to cyclic frequency:

        R_x^alpha(tau) = (1/N) sum_n x[n+tau] conj(x[n]) exp(-j 2 pi alpha n / N)

    Args:
        n_alpha: Number of cyclic frequency bins (rows).
        max_lag: Maximum lag in samples (columns).
        output_format: ``"magnitude"`` (C=1), ``"mag_phase"`` (C=2),
            or ``"real_imag"`` (C=2).

    Returns:
        ``torch.Tensor`` of shape ``[C, n_alpha, max_lag]``.
    """

    def __init__(
        self,
        n_alpha: int = 256,
        max_lag: int = 128,
        output_format: str = "magnitude",
    ):
        if output_format not in ("magnitude", "mag_phase", "real_imag"):
            raise ValueError(
                f"Unknown output_format: {output_format!r}. "
                "Supported: 'magnitude', 'mag_phase', 'real_imag'."
            )
        self.n_alpha = n_alpha
        self.max_lag = max_lag
        self.output_format = output_format

    def __call__(self, iq: np.ndarray) -> torch.Tensor:
        iq = np.ascontiguousarray(iq, dtype=np.complex64)
        caf_complex = np.asarray(_compute_caf(iq, self.n_alpha, self.max_lag))
        return format_csp_output(caf_complex, self.output_format)
