from typing import Optional, Tuple

import numpy as np

from spectra.utils.file_handlers.base_reader import FileReader, SignalMetadata
from spectra.utils.file_handlers.registry import register_reader


@register_reader
class RawIQReader(FileReader):
    """Read raw binary IQ files (``.cf32``, ``.cs16``, ``.raw``, ``.iq``, ``.bin``).

    Args:
        dtype: NumPy dtype for reading.  Use ``"complex64"`` for ``.cf32``
            files, ``"int16"`` for ``.cs16`` (interleaved I/Q pairs that
            are scaled to ``[-1, 1]``).
        sample_rate: Optional sample rate to include in metadata.
    """

    def __init__(self, dtype: str = "complex64", sample_rate: Optional[float] = None):
        self._dtype = np.dtype(dtype)
        self._sample_rate = sample_rate

    def read(self, path: str) -> Tuple[np.ndarray, SignalMetadata]:
        raw = np.fromfile(path, dtype=self._dtype)

        if np.issubdtype(self._dtype, np.complexfloating):
            iq = raw.astype(np.complex64)
        else:
            # Interleaved real samples: I0, Q0, I1, Q1, ...
            raw = raw.reshape(-1, 2)
            scale = 1.0
            if np.issubdtype(self._dtype, np.integer):
                scale = float(np.iinfo(self._dtype).max)
            iq = (
                raw[:, 0].astype(np.float32) / scale + 1j * raw[:, 1].astype(np.float32) / scale
            ).astype(np.complex64)

        meta = SignalMetadata(
            sample_rate=self._sample_rate,
            num_samples=len(iq),
            datatype=str(self._dtype),
        )
        return iq, meta

    @staticmethod
    def extensions() -> Tuple[str, ...]:
        return (".cf32", ".cs16", ".raw", ".iq", ".bin")
