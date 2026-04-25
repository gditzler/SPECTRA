import json
import os
from typing import Any, Dict, List, Optional

import numpy as np


class SigMFWriter:
    """Write IQ data and metadata as SigMF file pairs.

    Creates a ``.sigmf-meta`` JSON metadata file and a
    ``.sigmf-data`` binary data file.

    Args:
        base_path: Path without extension.  Creates
            ``{base_path}.sigmf-meta`` and ``{base_path}.sigmf-data``.
        sample_rate: Sample rate in Hz.
        center_frequency: Center frequency in Hz.
        datatype: SigMF datatype string (default ``"cf32_le"``).
        extra_global: Additional key-value pairs for the global section.
    """

    def __init__(
        self,
        base_path: str,
        sample_rate: float,
        center_frequency: float = 0.0,
        datatype: str = "cf32_le",
        extra_global: Optional[Dict[str, Any]] = None,
    ):
        self._base_path = base_path
        self._sample_rate = sample_rate
        self._center_frequency = center_frequency
        self._datatype = datatype
        self._extra_global = extra_global or {}

    def write(
        self,
        iq: np.ndarray,
        annotations: Optional[List[Dict]] = None,
    ) -> None:
        """Write IQ samples and metadata to disk.

        Args:
            iq: 1-D ``complex64`` array.
            annotations: Optional list of SigMF annotation dicts.
        """
        # Write binary data
        data_path = self._base_path + ".sigmf-data"
        iq.astype(np.complex64).tofile(data_path)

        # Build metadata
        global_meta = {
            "core:datatype": self._datatype,
            "core:sample_rate": self._sample_rate,
            "core:version": "1.0.0",
        }
        global_meta.update(self._extra_global)

        meta = {
            "global": global_meta,
            "captures": [
                {
                    "core:sample_start": 0,
                    "core:frequency": self._center_frequency,
                }
            ],
            "annotations": annotations or [],
        }

        meta_path = self._base_path + ".sigmf-meta"
        with open(meta_path, "w") as f:
            json.dump(meta, f, indent=2)

    @staticmethod
    def write_from_dataset(
        dataset: Any,
        output_dir: str,
        sample_rate: float,
        class_list: Optional[List[str]] = None,
        max_samples: Optional[int] = None,
    ) -> None:
        """Export a SPECTRA dataset to SigMF folder structure.

        Creates an ImageFolder-compatible directory::

            output_dir/
                BPSK/
                    sample_000000.sigmf-meta
                    sample_000000.sigmf-data
                QPSK/
                    ...

        Args:
            dataset: A SPECTRA dataset (NarrowbandDataset, etc.).
            output_dir: Root output directory.
            sample_rate: Sample rate in Hz.
            class_list: Ordered list of class names.  If ``None``,
                uses integer labels as directory names.
            max_samples: Cap the number of samples exported.
        """
        os.makedirs(output_dir, exist_ok=True)
        n = len(dataset)
        if max_samples is not None:
            n = min(n, max_samples)

        for i in range(n):
            data, label = dataset[i]

            # Convert [2, N] tensor back to complex64
            if hasattr(data, "numpy"):
                arr = data.numpy()
            else:
                arr = np.asarray(data)

            if arr.ndim == 2 and arr.shape[0] == 2:
                iq = (arr[0] + 1j * arr[1]).astype(np.complex64)
            else:
                iq = arr.astype(np.complex64).ravel()

            # Determine class directory name
            if class_list is not None:
                label_int = int(label) if not isinstance(label, int) else label
                cls_name = class_list[label_int]
            else:
                cls_name = str(int(label) if not isinstance(label, int) else label)

            cls_dir = os.path.join(output_dir, cls_name)
            os.makedirs(cls_dir, exist_ok=True)

            base_path = os.path.join(cls_dir, f"sample_{i:06d}")
            writer = SigMFWriter(base_path=base_path, sample_rate=sample_rate)
            writer.write(iq)
