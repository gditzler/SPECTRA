import numpy as np
import torch

from spectra._rust import compute_cumulants as _compute_cumulants


class Cumulants:
    """Higher-order cumulant feature extractor for modulation classification.

    Computes 2nd- and 4th-order cumulants (C20, C21, C40, C41, C42) and,
    optionally, 6th-order cumulants (C60, C61, C62, C63).

    The input signal is zero-mean centred before computation.  The returned
    feature vector contains the **magnitudes** of the complex cumulants.

    Args:
        max_order: Maximum cumulant order.  ``4`` returns 5 features,
            ``6`` returns 9 features.
    """

    def __init__(self, max_order: int = 4):
        if max_order not in (4, 6):
            raise ValueError(f"max_order must be 4 or 6, got {max_order}")
        self.max_order = max_order

    def __call__(self, iq: np.ndarray) -> torch.Tensor:
        """Compute cumulant features of the input IQ signal.

        Args:
            iq: 1-D complex64 IQ array.

        Returns:
            ``torch.Tensor`` of shape ``[n_features]`` (``float32``).
        """
        iq = np.ascontiguousarray(iq, dtype=np.complex64)
        cumulant_values = np.asarray(_compute_cumulants(iq, self.max_order))
        features = np.abs(cumulant_values).astype(np.float32)
        return torch.from_numpy(features)
