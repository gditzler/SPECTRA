import json
import os
from typing import Tuple

import numpy as np

from spectra.utils.file_handlers.base_reader import FileReader, SignalMetadata
from spectra.utils.file_handlers.registry import register_reader

# Maps SigMF core:datatype strings to (numpy read dtype, is_interleaved)
_SIGMF_DTYPE_MAP = {
    "cf32_le": (np.complex64, False),
    "cf64_le": (np.complex128, False),
    "ci16_le": (np.int16, True),
    "ci8": (np.int8, True),
    "cu8": (np.uint8, True),
}


@register_reader
class SigMFReader(FileReader):
    """Read SigMF (``.sigmf-meta`` + ``.sigmf-data``) file pairs.

    Parses the JSON metadata directly — no ``sigmf`` package required.
    """

    def read(self, path: str) -> Tuple[np.ndarray, SignalMetadata]:
        """Read SigMF recording.

        Args:
            path: Path to the ``.sigmf-meta`` file.  The corresponding
                ``.sigmf-data`` file must exist alongside it.
        """
        # Derive data path
        if path.endswith(".sigmf-meta"):
            data_path = path[: -len(".sigmf-meta")] + ".sigmf-data"
        else:
            data_path = os.path.splitext(path)[0] + ".sigmf-data"

        if not os.path.exists(data_path):
            raise FileNotFoundError(
                f"SigMF data file not found: {data_path}"
            )

        # Parse metadata JSON
        with open(path) as f:
            meta_json = json.load(f)

        global_meta = meta_json.get("global", {})
        datatype_str = global_meta.get("core:datatype", "cf32_le")

        if datatype_str not in _SIGMF_DTYPE_MAP:
            raise ValueError(
                f"Unsupported SigMF datatype '{datatype_str}'. "
                f"Supported: {list(_SIGMF_DTYPE_MAP.keys())}"
            )

        read_dtype, is_interleaved = _SIGMF_DTYPE_MAP[datatype_str]

        # Read binary data
        raw = np.fromfile(data_path, dtype=read_dtype)

        if is_interleaved:
            raw = raw.reshape(-1, 2)
            if np.issubdtype(read_dtype, np.unsignedinteger):
                # Offset binary (e.g. cu8): center at 0 then scale
                info = np.iinfo(read_dtype)
                offset = (info.max + 1) / 2.0
                iq = (
                    (raw[:, 0].astype(np.float32) - offset) / offset
                    + 1j * (raw[:, 1].astype(np.float32) - offset) / offset
                ).astype(np.complex64)
            else:
                scale = float(np.iinfo(read_dtype).max)
                iq = (
                    raw[:, 0].astype(np.float32) / scale
                    + 1j * raw[:, 1].astype(np.float32) / scale
                ).astype(np.complex64)
        else:
            iq = raw.astype(np.complex64)

        # Extract metadata
        sample_rate = global_meta.get("core:sample_rate")
        captures = meta_json.get("captures", [])
        center_freq = None
        if captures:
            center_freq = captures[0].get("core:frequency")

        meta = SignalMetadata(
            sample_rate=sample_rate,
            center_frequency=center_freq,
            datatype=datatype_str,
            num_samples=len(iq),
            annotations=meta_json.get("annotations", []),
            extra={k: v for k, v in global_meta.items() if not k.startswith("core:")},
        )
        return iq, meta

    @staticmethod
    def extensions() -> Tuple[str, ...]:
        return (".sigmf-meta",)
