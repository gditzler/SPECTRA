from typing import List, Optional, Tuple

import numpy as np

from spectra.utils.file_handlers.base_reader import SignalMetadata
from spectra.utils.file_handlers.base_writer import FileWriter
from spectra.utils.file_handlers.dataset_export import export_dataset_to_folder


class NumpyWriter(FileWriter):
    """Write IQ data as a NumPy ``.npy`` file.

    Args:
        path: Full file path including ``.npy`` extension.
    """

    def __init__(self, path: str):
        self._path = path

    def write(self, iq: np.ndarray, metadata: Optional[SignalMetadata] = None) -> None:
        """Write IQ samples to a ``.npy`` file.

        Args:
            iq: 1-D ``complex64`` array.
            metadata: Ignored (NumPy format has no metadata sidecar).
        """
        np.save(self._path, iq.astype(np.complex64))

    @staticmethod
    def extensions() -> Tuple[str, ...]:
        return (".npy",)

    @staticmethod
    def write_from_dataset(
        dataset,
        output_dir: str,
        class_list: Optional[List[str]] = None,
        max_samples: Optional[int] = None,
    ) -> None:
        """Export a SPECTRA dataset to a folder of ``.npy`` files.

        Creates an ImageFolder-compatible directory::

            output_dir/
                BPSK/
                    sample_000000.npy
                QPSK/
                    ...

        Args:
            dataset: A SPECTRA dataset (NarrowbandDataset, etc.).
            output_dir: Root output directory.
            class_list: Ordered list of class names.
            max_samples: Cap the number of samples exported.
        """
        export_dataset_to_folder(
            dataset=dataset,
            output_dir=output_dir,
            writer_factory=lambda path: NumpyWriter(path),
            file_extension=".npy",
            class_list=class_list,
            max_samples=max_samples,
        )
