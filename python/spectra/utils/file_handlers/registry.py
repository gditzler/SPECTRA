import os
from typing import Dict, Optional, Tuple, Type

from spectra.utils.file_handlers.base_reader import FileReader

_READER_REGISTRY: Dict[str, Type[FileReader]] = {}


def register_reader(reader_cls: Type[FileReader]) -> Type[FileReader]:
    """Register a FileReader for its declared extensions."""
    for ext in reader_cls.extensions():
        _READER_REGISTRY[ext.lower()] = reader_cls
    return reader_cls


def get_reader(
    path: str,
    reader_overrides: Optional[Dict[str, FileReader]] = None,
) -> FileReader:
    """Return a FileReader instance for the given file path."""
    _, ext = os.path.splitext(path)
    ext = ext.lower()

    if reader_overrides and ext in reader_overrides:
        return reader_overrides[ext]

    if ext not in _READER_REGISTRY:
        raise ValueError(
            f"No reader registered for extension '{ext}'. "
            f"Supported: {supported_extensions()}"
        )
    return _READER_REGISTRY[ext]()


def supported_extensions() -> Tuple[str, ...]:
    """Return all file extensions with registered readers."""
    return tuple(sorted(_READER_REGISTRY.keys()))
