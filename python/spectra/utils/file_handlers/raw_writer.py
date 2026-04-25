from typing import List, Optional, Tuple

import numpy as np

from spectra.utils.file_handlers.base_reader import SignalMetadata
from spectra.utils.file_handlers.base_writer import FileWriter
from spectra.utils.file_handlers.dataset_export import export_dataset_to_folder


class RawIQWriter(FileWriter):
    """Write IQ data as raw binary files.

    Supports ``.cf32`` (complex float32) and ``.cs16``
    (interleaved int16) formats, as well as generic
    ``.raw``, ``.iq``, and ``.bin`` extensions.

    Args:
        path: Full file path.
        dtype: Output dtype.  ``"complex64"`` writes native
            complex float32.  ``"int16"`` scales ``[-1, 1]``
            float to int16 range and interleaves I/Q.
    """

    def __init__(self, path: str, dtype: str = "complex64"):
        self._path = path
        self._dtype = np.dtype(dtype)

    def write(self, iq: np.ndarray, metadata: Optional[SignalMetadata] = None) -> None:
        """Write IQ samples to a raw binary file.

        Args:
            iq: 1-D ``complex64`` array.
            metadata: Ignored (raw format has no metadata).
        """
        iq = iq.astype(np.complex64)

        if np.issubdtype(self._dtype, np.complexfloating):
            iq.tofile(self._path)
        else:
            # Interleave I/Q and scale to integer range
            scale = float(np.iinfo(np.dtype(self._dtype).type).max)
            interleaved = np.empty(len(iq) * 2, dtype=np.float32)
            interleaved[0::2] = iq.real
            interleaved[1::2] = iq.imag
            (interleaved * scale).astype(self._dtype).tofile(self._path)

    @staticmethod
    def extensions() -> Tuple[str, ...]:
        return (".cf32", ".cs16", ".raw", ".iq", ".bin")

    @staticmethod
    def write_from_dataset(
        dataset,
        output_dir: str,
        class_list: Optional[List[str]] = None,
        max_samples: Optional[int] = None,
        dtype: str = "complex64",
    ) -> None:
        """Export a SPECTRA dataset to a folder of raw IQ files.

        Args:
            dataset: A SPECTRA dataset (NarrowbandDataset, etc.).
            output_dir: Root output directory.
            class_list: Ordered list of class names.
            max_samples: Cap the number of samples exported.
            dtype: Output dtype (default ``"complex64"`` → ``.cf32``).
        """
        ext = ".cs16" if np.dtype(dtype) == np.dtype("int16") else ".cf32"
        export_dataset_to_folder(
            dataset=dataset,
            output_dir=output_dir,
            writer_factory=lambda path: RawIQWriter(path, dtype=dtype),
            file_extension=ext,
            class_list=class_list,
            max_samples=max_samples,
        )
