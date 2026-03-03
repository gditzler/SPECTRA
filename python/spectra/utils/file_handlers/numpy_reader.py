from typing import Tuple

import numpy as np

from spectra.utils.file_handlers.base_reader import FileReader, SignalMetadata
from spectra.utils.file_handlers.registry import register_reader


@register_reader
class NumpyReader(FileReader):
    """Read NumPy ``.npy`` / ``.npz`` files containing IQ data.

    For ``.npz`` files, reads the array named *array_key* or falls back
    to the first array in the archive.
    """

    def __init__(self, array_key: str = "iq"):
        self._array_key = array_key

    def read(self, path: str) -> Tuple[np.ndarray, SignalMetadata]:
        if path.endswith(".npz"):
            data = np.load(path)
            if self._array_key in data:
                iq = data[self._array_key]
            else:
                iq = data[list(data.keys())[0]]
        else:
            iq = np.load(path)

        iq = iq.astype(np.complex64).ravel()
        meta = SignalMetadata(num_samples=len(iq))
        return iq, meta

    @staticmethod
    def extensions() -> Tuple[str, ...]:
        return (".npy", ".npz")
