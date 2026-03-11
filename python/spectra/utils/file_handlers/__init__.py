from spectra.utils.file_handlers.base_reader import FileReader, SignalMetadata
from spectra.utils.file_handlers.base_writer import FileWriter
from spectra.utils.file_handlers.dataset_export import export_dataset_to_folder

# Always-available readers (numpy-only deps)
from spectra.utils.file_handlers.numpy_reader import NumpyReader

# Always-available writers (stdlib / numpy-only deps)
from spectra.utils.file_handlers.numpy_writer import NumpyWriter
from spectra.utils.file_handlers.raw_reader import RawIQReader
from spectra.utils.file_handlers.raw_writer import RawIQWriter
from spectra.utils.file_handlers.registry import (
    get_reader,
    register_reader,
    supported_extensions,
)
from spectra.utils.file_handlers.sigmf_reader import SigMFReader
from spectra.utils.file_handlers.sqlite_reader import SQLiteReader
from spectra.utils.file_handlers.sqlite_writer import SQLiteWriter
from spectra.utils.file_handlers.zarr_handler import ZarrHandler

# Optional readers
try:
    from spectra.utils.file_handlers.hdf5_reader import HDF5Reader  # noqa: F401
except ImportError:
    pass

# Optional writers
try:
    from spectra.utils.file_handlers.hdf5_writer import HDF5Writer  # noqa: F401
except ImportError:
    pass

__all__ = [
    "ZarrHandler",
    "FileReader",
    "FileWriter",
    "SignalMetadata",
    "export_dataset_to_folder",
    "get_reader",
    "register_reader",
    "supported_extensions",
    # Readers
    "NumpyReader",
    "RawIQReader",
    "SigMFReader",
    "SQLiteReader",
    # Writers
    "NumpyWriter",
    "RawIQWriter",
    "SQLiteWriter",
]
