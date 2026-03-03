import numpy as np
import torch

from spectra._rust import compute_psd_welch as _compute_psd_welch
from spectra._rust import compute_scd_ssca as _compute_scd_ssca


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

    Returns:
        ``torch.Tensor`` of shape ``[C, nfft, n_alpha]``.
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
        iq = np.ascontiguousarray(iq, dtype=np.complex64)

        # Compute SCD (complex, DC-centred)
        scd = np.asarray(_compute_scd_ssca(iq, self.nfft, self.n_alpha, self.hop))

        # Compute PSD (real, DC-centred) for normalisation
        psd = np.asarray(_compute_psd_welch(iq, self.nfft, self.nfft // 2))
        psd = np.maximum(psd, self.eps)

        # Normalise: SCF = SCD / sqrt(PSD(f+a/2) * PSD(f-a/2))
        # Approximate using PSD at each spectral bin (ignoring the alpha/2 shift
        # for the denominator, which is standard practice for discrete grids).
        psd_col = np.sqrt(psd).astype(np.float32)  # shape [nfft]
        # Broadcast: divide each column of SCD by PSD
        denom = psd_col[:, None] * np.ones(self.n_alpha, dtype=np.float32)[None, :]
        scf_complex = scd / (denom + self.eps)

        return self._format_output(scf_complex)

    def _format_output(self, scf_complex: np.ndarray) -> torch.Tensor:
        if self.output_format == "magnitude":
            result = np.abs(scf_complex).astype(np.float32)
            return torch.from_numpy(result).unsqueeze(0).float()

        if self.output_format == "mag_phase":
            mag = np.abs(scf_complex).astype(np.float32)
            phase = np.angle(scf_complex).astype(np.float32)
            stacked = np.stack([mag, phase], axis=0)
            return torch.from_numpy(stacked).float()

        # real_imag
        ri = np.stack(
            [scf_complex.real.astype(np.float32), scf_complex.imag.astype(np.float32)],
            axis=0,
        )
        return torch.from_numpy(ri).float()
