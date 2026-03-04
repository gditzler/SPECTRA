from typing import List, Optional, Tuple

import numpy as np

from spectra.utils.file_handlers.base_reader import SignalMetadata
from spectra.utils.file_handlers.base_writer import FileWriter
from spectra.utils.file_handlers.dataset_export import export_dataset_to_folder


class HDF5Writer(FileWriter):
    """Write IQ data to an HDF5 (``.h5``) file.

    Requires ``h5py``.  Install with::

        pip install 'spectra[io]'

    Args:
        path: Full file path including ``.h5`` extension.
        iq_dataset: Name of the HDF5 dataset to store IQ data in.
    """

    def __init__(self, path: str, iq_dataset: str = "iq"):
        self._path = path
        self._iq_dataset = iq_dataset

    def write(self, iq: np.ndarray, metadata: Optional[SignalMetadata] = None) -> None:
        """Write IQ samples and optional metadata to an HDF5 file.

        Args:
            iq: 1-D ``complex64`` array.
            metadata: Optional metadata stored as file-level attributes.
        """
        import h5py

        with h5py.File(self._path, "w") as f:
            f.create_dataset(self._iq_dataset, data=iq.astype(np.complex64))
            if metadata is not None:
                if metadata.sample_rate is not None:
                    f.attrs["sample_rate"] = metadata.sample_rate
                if metadata.center_frequency is not None:
                    f.attrs["center_frequency"] = metadata.center_frequency

    @staticmethod
    def extensions() -> Tuple[str, ...]:
        return (".h5", ".hdf5")

    @staticmethod
    def write_from_dataset(
        dataset,
        output_dir: str,
        class_list: Optional[List[str]] = None,
        max_samples: Optional[int] = None,
    ) -> None:
        """Export a SPECTRA dataset to a folder of ``.h5`` files.

        Args:
            dataset: A SPECTRA dataset (NarrowbandDataset, etc.).
            output_dir: Root output directory.
            class_list: Ordered list of class names.
            max_samples: Cap the number of samples exported.
        """
        export_dataset_to_folder(
            dataset=dataset,
            output_dir=output_dir,
            writer_factory=lambda path: HDF5Writer(path),
            file_extension=".h5",
            class_list=class_list,
            max_samples=max_samples,
        )
