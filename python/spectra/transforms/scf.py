import numpy as np
import torch

from spectra._rust import compute_psd_welch as _compute_psd_welch
from spectra._rust import compute_scd_ssca as _compute_scd_ssca
from spectra.transforms.csp_utils import format_csp_output


class SCF:
    """Spectral Coherence Function (normalised SCD).

    Computes the SCF by dividing the Spectral Correlation Density by the
    geometric mean of the PSD products:

        C_x^alpha(f) = S_x^alpha(f) / sqrt(S_x^0(f + alpha/2) * S_x^0(f - alpha/2))

    This normalisation bounds the output magnitude to [0, 1], making features
    more robust to signal power variations.

    Args:
        nfft: FFT size for the channeliser.
        n_alpha: Number of cyclic frequency bins.
        hop: Hop size for the channeliser.
        output_format: ``"magnitude"`` (C=1), ``"mag_phase"`` (C=2),
            or ``"real_imag"`` (C=2).
        eps: Small constant to avoid division by zero.
    """

    def __init__(
        self,
        nfft: int = 256,
        n_alpha: int = 256,
        hop: int = 64,
        output_format: str = "magnitude",
        eps: float = 1e-12,
    ):
        if output_format not in ("magnitude", "mag_phase", "real_imag"):
            raise ValueError(
                f"Unknown output_format: {output_format!r}. "
                "Supported: 'magnitude', 'mag_phase', 'real_imag'."
            )
        self.nfft = nfft
        self.n_alpha = n_alpha
        self.hop = hop
        self.output_format = output_format
        self.eps = eps

    def __call__(self, iq: np.ndarray) -> torch.Tensor:
        """Compute the SCF of the input IQ signal.

        Args:
            iq: 1-D complex64 IQ array.

        Returns:
            ``torch.Tensor`` of shape ``[C, nfft, n_alpha]``.
        """
        iq = np.ascontiguousarray(iq, dtype=np.complex64)

        # Compute SCD (complex, DC-centred)
        scd = np.asarray(_compute_scd_ssca(iq, self.nfft, self.n_alpha, self.hop))

        # Compute PSD (real, DC-centred) for normalisation
        psd = np.asarray(_compute_psd_welch(iq, self.nfft, self.nfft // 2))
        psd = np.maximum(psd, self.eps)

        # Normalise: SCF = SCD / sqrt(PSD(f+a/2) * PSD(f-a/2))
        psd_col = np.sqrt(psd).astype(np.float32)  # shape [nfft]
        denom = psd_col[:, None] * np.ones(self.n_alpha, dtype=np.float32)[None, :]
        scf_complex = scd / (denom + self.eps)

        return format_csp_output(scf_complex, self.output_format)
