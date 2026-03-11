import numpy as np
import torch

from spectra._rust import compute_scd_fam, compute_scd_s3ca, compute_scd_ssca
from spectra.transforms.csp_utils import format_csp_output


class SCD:
    """Spectral Correlation Density estimator.

    Computes the SCD using the Strip Spectral Correlation Algorithm (SSCA),
    the FFT Accumulation Method (FAM), or the Sparse Strip Spectral
    Correlation Analyzer (S3CA).

    Args:
        nfft: FFT size for the channeliser (spectral frequency resolution).
        n_alpha: Number of cyclic frequency bins.
        hop: Hop size for the channeliser.
        method: ``"ssca"`` (default), ``"fam"``, or ``"s3ca"`` (sparse).
        output_format: ``"magnitude"`` (C=1), ``"mag_phase"`` (C=2),
            or ``"real_imag"`` (C=2).
        db_scale: Apply ``10 * log10`` to the magnitude.
        kappa: Sparsity parameter for S3CA — number of significant
            cycle frequencies to recover per band. Only used when
            ``method="s3ca"``. Default: 8.
        seed: Random seed for S3CA deterministic randomization.
            Only used when ``method="s3ca"``. Default: 0.

    Returns:
        ``torch.Tensor`` of shape ``[C, nfft, n_alpha]``.
    """

    def __init__(
        self,
        nfft: int = 256,
        n_alpha: int = 256,
        hop: int = 64,
        method: str = "ssca",
        output_format: str = "magnitude",
        db_scale: bool = False,
        kappa: int = 8,
        seed: int = 0,
    ):
        if method not in ("ssca", "fam", "s3ca"):
            raise ValueError(f"Unknown method: {method!r}. Supported: 'ssca', 'fam', 's3ca'.")
        if output_format not in ("magnitude", "mag_phase", "real_imag"):
            raise ValueError(
                f"Unknown output_format: {output_format!r}. "
                "Supported: 'magnitude', 'mag_phase', 'real_imag'."
            )
        self.nfft = nfft
        self.n_alpha = n_alpha
        self.hop = hop
        self.method = method
        self.output_format = output_format
        self.db_scale = db_scale
        self.kappa = kappa
        self.seed = seed

    def __call__(self, iq: np.ndarray) -> torch.Tensor:
        iq = np.ascontiguousarray(iq, dtype=np.complex64)
        if self.method == "s3ca":
            scd_complex = np.asarray(
                compute_scd_s3ca(
                    iq,
                    self.nfft,
                    self.n_alpha,
                    self.hop,
                    self.kappa,
                    self.seed,
                )
            )
        elif self.method == "fam":
            scd_complex = np.asarray(compute_scd_fam(iq, self.nfft, self.n_alpha, self.hop))
        else:
            scd_complex = np.asarray(compute_scd_ssca(iq, self.nfft, self.n_alpha, self.hop))
        return format_csp_output(scd_complex, self.output_format, self.db_scale)
