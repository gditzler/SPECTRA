from typing import Tuple

import numpy as np

from spectra.utils.file_handlers.base_reader import FileReader, SignalMetadata
from spectra.utils.file_handlers.registry import register_reader

_FALLBACK_KEYS = ("data", "X", "samples", "iq_data")


@register_reader
class HDF5Reader(FileReader):
    """Read HDF5 (``.h5``, ``.hdf5``) files containing IQ data.

    Requires ``h5py``.  Install with::

        pip install 'spectra[io]'

    Args:
        iq_dataset: HDF5 dataset path containing IQ data.  If the key
            is not found, falls back to ``"data"``, ``"X"``, then the
            first dataset in the file.
    """

    def __init__(self, iq_dataset: str = "iq"):
        self._iq_dataset = iq_dataset

    def read(self, path: str) -> Tuple[np.ndarray, SignalMetadata]:
        import h5py

        with h5py.File(path, "r") as f:
            # Find the dataset
            key = self._resolve_key(f)
            raw = f[key][:]

            # Extract metadata from file-level attrs
            sample_rate = f.attrs.get("sample_rate")
            center_freq = f.attrs.get("center_frequency")

        # Convert to complex64
        if np.issubdtype(raw.dtype, np.complexfloating):
            iq = raw.astype(np.complex64).ravel()
        elif raw.ndim == 2 and raw.shape[1] == 2:
            # (N, 2) real/imag columns
            iq = (raw[:, 0].astype(np.float32) + 1j * raw[:, 1].astype(np.float32)).astype(
                np.complex64
            )
        else:
            # Assume interleaved real values
            raw = raw.ravel().astype(np.float32)
            iq = (raw[0::2] + 1j * raw[1::2]).astype(np.complex64)

        meta = SignalMetadata(
            sample_rate=float(sample_rate) if sample_rate is not None else None,
            center_frequency=float(center_freq) if center_freq is not None else None,
            num_samples=len(iq),
        )
        return iq, meta

    def _resolve_key(self, f) -> str:
        """Find the dataset key, trying primary then fallbacks."""
        if self._iq_dataset in f:
            return self._iq_dataset
        for key in _FALLBACK_KEYS:
            if key in f:
                return key
        # Last resort: first key in file
        keys = list(f.keys())
        if not keys:
            raise ValueError("HDF5 file contains no datasets")
        return keys[0]

    @staticmethod
    def extensions() -> Tuple[str, ...]:
        return (".h5", ".hdf5")
