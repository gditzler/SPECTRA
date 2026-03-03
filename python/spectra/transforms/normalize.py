import torch


class SpectrogramNormalize:
    """Normalize a spectrogram tensor for neural network input.

    Modes:
        "db": Convert to dB scale, then min-max normalize to [0, 1].
        "standardize": Zero-mean, unit-variance normalization.
    """

    def __init__(self, mode: str = "db"):
        if mode not in ("db", "standardize"):
            raise ValueError(f"mode must be 'db' or 'standardize', got '{mode}'")
        self.mode = mode

    def __call__(self, spec: torch.Tensor) -> torch.Tensor:
        if self.mode == "db":
            db = 10.0 * torch.log10(spec + 1e-20)
            min_val = db.min()
            max_val = db.max()
            denom = max_val - min_val
            if denom < 1e-10:
                return torch.zeros_like(db)
            return ((db - min_val) / denom).float()
        else:  # standardize
            mean = spec.mean()
            std = spec.std()
            if std < 1e-10:
                return torch.zeros_like(spec)
            return ((spec - mean) / std).float()
