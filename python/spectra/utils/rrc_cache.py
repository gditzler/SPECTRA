"""Cached RRC tap computation for waveform generators."""

from functools import lru_cache

import numpy as np

from spectra._rust import rrc_taps_py


@lru_cache(maxsize=32)
def cached_rrc_taps(rolloff: float, span: int, sps: int) -> np.ndarray:
    """Return RRC filter taps, cached by (rolloff, span, sps)."""
    return np.array(rrc_taps_py(rolloff, span, sps), dtype=np.float32)
