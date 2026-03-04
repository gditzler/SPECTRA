import sqlite3
from typing import List, Optional, Tuple

import numpy as np

from spectra.utils.file_handlers.base_reader import SignalMetadata
from spectra.utils.file_handlers.base_writer import FileWriter
from spectra.utils.file_handlers.dataset_export import _tensor_to_complex64

_SCHEMA = """
CREATE TABLE IF NOT EXISTS samples (
    id INTEGER PRIMARY KEY AUTOINCREMENT,
    label INTEGER,
    class_name TEXT,
    iq BLOB NOT NULL,
    num_samples INTEGER NOT NULL,
    sample_rate REAL
);

CREATE TABLE IF NOT EXISTS metadata (
    key TEXT PRIMARY KEY,
    value TEXT NOT NULL
);
"""


class SQLiteWriter(FileWriter):
    """Write IQ data and metadata to a SQLite database.

    Stores IQ samples as BLOBs (raw ``complex64`` bytes) alongside
    integer labels, class names, and per-sample metadata in a single
    ``.db`` file.  Uses Python's built-in ``sqlite3`` module (zero
    extra dependencies).

    Args:
        path: Full file path including ``.db`` extension.
        sample_rate: Default sample rate written to each row.
    """

    def __init__(self, path: str, sample_rate: Optional[float] = None):
        self._path = path
        self._sample_rate = sample_rate
        self._conn = sqlite3.connect(path)
        self._conn.executescript(_SCHEMA)

    def write(
        self,
        iq: np.ndarray,
        metadata: Optional[SignalMetadata] = None,
        *,
        label: Optional[int] = None,
        class_name: Optional[str] = None,
    ) -> None:
        """Insert a single IQ record into the database.

        Args:
            iq: 1-D ``complex64`` array.
            metadata: Optional signal metadata (``sample_rate`` is used
                if the instance default is ``None``).
            label: Integer class label.
            class_name: Human-readable class name.
        """
        iq = iq.astype(np.complex64)
        sr = self._sample_rate
        if metadata is not None and metadata.sample_rate is not None:
            sr = metadata.sample_rate

        self._conn.execute(
            "INSERT INTO samples (label, class_name, iq, num_samples, sample_rate) "
            "VALUES (?, ?, ?, ?, ?)",
            (label, class_name, iq.tobytes(), len(iq), sr),
        )

    def write_metadata(self, key: str, value: str) -> None:
        """Insert or replace a key-value pair in the metadata table."""
        self._conn.execute(
            "INSERT OR REPLACE INTO metadata (key, value) VALUES (?, ?)",
            (key, value),
        )

    def close(self) -> None:
        """Commit and close the database connection."""
        self._conn.commit()
        self._conn.close()

    @staticmethod
    def extensions() -> Tuple[str, ...]:
        return (".db",)

    @staticmethod
    def write_from_dataset(
        dataset,
        output_path: str,
        class_list: Optional[List[str]] = None,
        max_samples: Optional[int] = None,
        sample_rate: Optional[float] = None,
    ) -> None:
        """Export a SPECTRA dataset to a single SQLite database.

        Unlike the folder-based writers, all samples go into one
        ``.db`` file as rows in the ``samples`` table.

        Args:
            dataset: A SPECTRA dataset (NarrowbandDataset, etc.).
            output_path: Path for the ``.db`` file.
            class_list: Ordered list of class names.
            max_samples: Cap the number of rows exported.
            sample_rate: Sample rate stored per row.
        """
        writer = SQLiteWriter(output_path, sample_rate=sample_rate)

        n = len(dataset)
        if max_samples is not None:
            n = min(n, max_samples)

        for i in range(n):
            data, label = dataset[i]
            iq = _tensor_to_complex64(data)

            label_int = int(label) if not isinstance(label, int) else label
            cls_name = None
            if class_list is not None:
                cls_name = class_list[label_int]

            writer.write(iq, label=label_int, class_name=cls_name)

        writer.close()
