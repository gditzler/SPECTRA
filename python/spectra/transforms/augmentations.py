import numpy as np


class CutOut:
    """Random rectangular masking (set region to zero)."""

    def __init__(self, max_length_fraction: float = 0.1):
        self._max_frac = max_length_fraction

    def __call__(self, iq: np.ndarray, rng: np.random.Generator = None) -> np.ndarray:
        if rng is None:
            rng = np.random.default_rng()
        out = iq.copy()
        length = int(rng.uniform(0, self._max_frac) * len(iq))
        if length > 0:
            start = rng.integers(0, max(1, len(iq) - length))
            out[start : start + length] = 0
        return out


class TimeReversal:
    """Flip signal in time."""

    def __call__(self, iq: np.ndarray, rng: np.random.Generator = None) -> np.ndarray:
        return iq[::-1].copy()


class ChannelSwap:
    """Swap I/Q channels (complex conjugate)."""

    def __call__(self, iq: np.ndarray, rng: np.random.Generator = None) -> np.ndarray:
        return iq.conj().copy()


class PatchShuffle:
    """Reorder signal patches."""

    def __init__(self, num_patches: int = 8):
        self._num_patches = num_patches

    def __call__(self, iq: np.ndarray, rng: np.random.Generator = None) -> np.ndarray:
        if rng is None:
            rng = np.random.default_rng()
        patches = np.array_split(iq, self._num_patches)
        rng.shuffle(patches)
        return np.concatenate(patches).astype(iq.dtype)


class RandomDropSamples:
    """Drop random samples with configurable fill strategy."""

    def __init__(self, drop_rate: float = 0.01, fill: str = "zero"):
        self._drop_rate = drop_rate
        self._fill = fill

    def __call__(self, iq: np.ndarray, rng: np.random.Generator = None) -> np.ndarray:
        if rng is None:
            rng = np.random.default_rng()
        out = iq.copy()
        mask = rng.random(len(iq)) < self._drop_rate
        if self._fill == "zero":
            out[mask] = 0
        elif self._fill == "ffill":
            for i in range(len(out)):
                if mask[i] and i > 0:
                    out[i] = out[i - 1]
        elif self._fill == "bfill":
            for i in range(len(out) - 2, -1, -1):
                if mask[i] and i < len(out) - 1:
                    out[i] = out[i + 1]
        elif self._fill == "mean":
            out[mask] = np.mean(iq)
        return out


class AddSlope:
    """Add linear amplitude trend."""

    def __init__(self, max_slope: float = 0.1):
        self._max_slope = max_slope

    def __call__(self, iq: np.ndarray, rng: np.random.Generator = None) -> np.ndarray:
        if rng is None:
            rng = np.random.default_rng()
        slope = rng.uniform(-self._max_slope, self._max_slope)
        ramp = np.linspace(0, slope, len(iq), dtype=np.float32)
        return (iq * (1.0 + ramp)).astype(iq.dtype)


class RandomMagRescale:
    """Random magnitude scaling."""

    def __init__(self, scale_range: tuple = (0.5, 2.0)):
        self._scale_range = scale_range

    def __call__(self, iq: np.ndarray, rng: np.random.Generator = None) -> np.ndarray:
        if rng is None:
            rng = np.random.default_rng()
        scale = rng.uniform(*self._scale_range)
        return (iq * scale).astype(iq.dtype)


class AGC:
    """Automatic Gain Control — normalize to target power."""

    def __init__(self, target_power: float = 1.0):
        self._target_power = target_power

    def __call__(self, iq: np.ndarray, rng: np.random.Generator = None) -> np.ndarray:
        power = np.mean(np.abs(iq) ** 2)
        if power > 0:
            scale = np.sqrt(self._target_power / power)
            return (iq * scale).astype(iq.dtype)
        return iq.copy()


class MixUp:
    """Blend IQ signal with a random permutation of itself.

    Args:
        alpha: Beta distribution parameter for mixing coefficient.
    """

    def __init__(self, alpha: float = 0.2):
        self._alpha = alpha

    def __call__(self, iq: np.ndarray, rng: np.random.Generator = None) -> np.ndarray:
        if rng is None:
            rng = np.random.default_rng()
        lam = rng.beta(self._alpha, self._alpha)
        perm = rng.permutation(len(iq))
        return (lam * iq + (1 - lam) * iq[perm]).astype(iq.dtype)


class CutMix:
    """Replace a random time segment with shuffled samples.

    Args:
        alpha: Beta distribution parameter for cut ratio.
    """

    def __init__(self, alpha: float = 1.0):
        self._alpha = alpha

    def __call__(self, iq: np.ndarray, rng: np.random.Generator = None) -> np.ndarray:
        if rng is None:
            rng = np.random.default_rng()
        lam = rng.beta(self._alpha, self._alpha)
        cut_len = int((1 - lam) * len(iq))
        start = rng.integers(0, max(1, len(iq) - cut_len))
        out = iq.copy()
        perm = rng.permutation(len(iq))
        out[start : start + cut_len] = iq[perm[start : start + cut_len]]
        return out
