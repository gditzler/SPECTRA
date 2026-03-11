import sqlite3
from typing import Tuple

import numpy as np

from spectra.utils.file_handlers.base_reader import FileReader, SignalMetadata
from spectra.utils.file_handlers.registry import register_reader


@register_reader
class SQLiteReader(FileReader):
    """Read IQ data from a SQLite database (``.db``) created by :class:`SQLiteWriter`.

    Each call to :meth:`read` returns one row from the ``samples``
    table.  Use ``row_index`` to select which row (default: first).

    Args:
        row_index: Zero-based row offset into the ``samples`` table.
    """

    def __init__(self, row_index: int = 0):
        self._row_index = row_index

    def read(self, path: str) -> Tuple[np.ndarray, SignalMetadata]:
        conn = sqlite3.connect(path)
        row = conn.execute(
            "SELECT iq, num_samples, sample_rate, label, class_name "
            "FROM samples ORDER BY id LIMIT 1 OFFSET ?",
            (self._row_index,),
        ).fetchone()
        conn.close()

        if row is None:
            raise IndexError(f"Row index {self._row_index} out of range for {path}")

        iq = np.frombuffer(row[0], dtype=np.complex64)
        meta = SignalMetadata(
            num_samples=row[1],
            sample_rate=float(row[2]) if row[2] is not None else None,
            extra={"label": row[3], "class_name": row[4]},
        )
        return iq, meta

    @staticmethod
    def extensions() -> Tuple[str, ...]:
        return (".db",)
