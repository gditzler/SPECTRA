"""Shared output formatting for CSP transforms (SCD, SCF, CAF)."""
import numpy as np
import torch


def format_csp_output(
    data: np.ndarray, output_format: str, db_scale: bool = False
) -> torch.Tensor:
    """Format complex CSP output into a torch tensor.

    Args:
        data: Complex numpy array (e.g. SCD, SCF, or CAF matrix).
        output_format: One of "magnitude", "mag_phase", "real_imag".
        db_scale: Apply 10*log10 to magnitude (only for "magnitude" and
            "mag_phase" formats).

    Returns:
        Tensor of shape [C, ...] where C=1 for magnitude, C=2 for others.
    """
    if output_format == "magnitude":
        result = np.abs(data).astype(np.float32)
        if db_scale:
            result = 10.0 * np.log10(result + 1e-12)
        return torch.from_numpy(result).unsqueeze(0).float()

    if output_format == "mag_phase":
        mag = np.abs(data).astype(np.float32)
        phase = np.angle(data).astype(np.float32)
        if db_scale:
            mag = 10.0 * np.log10(mag + 1e-12)
        stacked = np.stack([mag, phase], axis=0)
        return torch.from_numpy(stacked).float()

    # real_imag
    ri = np.stack(
        [data.real.astype(np.float32), data.imag.astype(np.float32)],
        axis=0,
    )
    return torch.from_numpy(ri).float()
