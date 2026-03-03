import numpy as np
import torch

from spectra._rust import compute_scd_fam, compute_scd_ssca


class SCD:
    """Spectral Correlation Density estimator.

    Computes the SCD using the Strip Spectral Correlation Algorithm (SSCA).
    Drop-in replacement for :class:`STFT` in any dataset's ``transform=``
    parameter.

    Args:
        nfft: FFT size for the channeliser (spectral frequency resolution).
        n_alpha: Number of cyclic frequency bins.
        hop: Hop size for the channeliser.
        method: ``"ssca"`` (default). ``"fam"`` will be available after
            Phase 1b.
        output_format: ``"magnitude"`` (C=1), ``"mag_phase"`` (C=2),
            or ``"real_imag"`` (C=2).
        db_scale: Apply ``10 * log10`` to the magnitude.

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
    ):
        if method not in ("ssca", "fam"):
            raise ValueError(f"Unknown method: {method!r}. Supported: 'ssca', 'fam'.")
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

    def __call__(self, iq: np.ndarray) -> torch.Tensor:
        iq = np.ascontiguousarray(iq, dtype=np.complex64)
        if self.method == "fam":
            scd_complex = np.asarray(
                compute_scd_fam(iq, self.nfft, self.n_alpha, self.hop)
            )
        else:
            scd_complex = np.asarray(
                compute_scd_ssca(iq, self.nfft, self.n_alpha, self.hop)
            )
        return self._format_output(scd_complex)

    def _format_output(self, scd_complex: np.ndarray) -> torch.Tensor:
        if self.output_format == "magnitude":
            result = np.abs(scd_complex).astype(np.float32)
            if self.db_scale:
                result = 10.0 * np.log10(result + 1e-12)
            return torch.from_numpy(result).unsqueeze(0).float()

        if self.output_format == "mag_phase":
            mag = np.abs(scd_complex).astype(np.float32)
            phase = np.angle(scd_complex).astype(np.float32)
            if self.db_scale:
                mag = 10.0 * np.log10(mag + 1e-12)
            stacked = np.stack([mag, phase], axis=0)
            return torch.from_numpy(stacked).float()

        # real_imag
        ri = np.stack(
            [scd_complex.real.astype(np.float32), scd_complex.imag.astype(np.float32)],
            axis=0,
        )
        return torch.from_numpy(ri).float()
