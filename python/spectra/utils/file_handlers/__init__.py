from spectra.utils.file_handlers.zarr_handler import ZarrHandler
from spectra.utils.file_handlers.base_reader import FileReader, SignalMetadata
from spectra.utils.file_handlers.registry import (
    get_reader,
    register_reader,
    supported_extensions,
)

# Always-available readers (numpy-only deps)
from spectra.utils.file_handlers.numpy_reader import NumpyReader
from spectra.utils.file_handlers.raw_reader import RawIQReader
from spectra.utils.file_handlers.sigmf_reader import SigMFReader

# Optional readers
try:
    from spectra.utils.file_handlers.hdf5_reader import HDF5Reader
except ImportError:
    pass

__all__ = [
    "ZarrHandler",
    "FileReader",
    "SignalMetadata",
    "get_reader",
    "register_reader",
    "supported_extensions",
    "NumpyReader",
    "RawIQReader",
    "SigMFReader",
]
