from typing import Any, Optional

import numpy as np


class ZarrHandler:
    """Zarr format backend for dataset persistence.

    Compatible with both zarr v2 and v3 APIs.
    """

    def __init__(self, path: str):
        self._path = path
        # Loosely typed because the zarr v2/v3 group object types differ;
        # we duck-type at runtime via ``hasattr`` / ``getattr``.
        self._root: Any = None

    def open(self) -> None:
        import zarr

        # zarr v3 uses open_group, v2 uses open with mode
        if hasattr(zarr, "open_group"):
            try:
                self._root = zarr.open_group(self._path, mode="w")
            except TypeError:
                self._root = zarr.open(self._path, mode="w")
        else:
            self._root = zarr.open(self._path, mode="w")

    def create_array(self, name: str, shape: tuple, dtype, chunks: Optional[tuple] = None):
        if self._root is None:
            self.open()
        if chunks is None:
            chunks = (min(64, shape[0]),) + shape[1:] if len(shape) > 0 else shape
        root: Any = self._root
        # Try v3 API (create_array), fall back to v2 (create_dataset)
        if hasattr(root, "create_array"):
            return root.create_array(name, shape=shape, dtype=dtype, chunks=chunks)
        return root.create_dataset(name, shape=shape, dtype=dtype, chunks=chunks)

    def write_metadata(self, metadata: dict) -> None:
        if self._root is None:
            self.open()
        root: Any = self._root
        root.attrs.update(metadata)

    def close(self) -> None:
        self._root = None

    def read(self, name: Optional[str] = None):
        import zarr

        root: Any
        if hasattr(zarr, "open_group"):
            try:
                root = zarr.open_group(self._path, mode="r")
            except TypeError:
                root = zarr.open(self._path, mode="r")
        else:
            root = zarr.open(self._path, mode="r")
        if name is not None:
            return np.array(root[name])
        return root
