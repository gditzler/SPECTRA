import os
from typing import Callable, List, Optional

import numpy as np


def _tensor_to_complex64(data) -> np.ndarray:
    """Convert a ``[2, N]`` tensor or array to a 1-D ``complex64`` array.

    Handles:
    - PyTorch tensors with ``.numpy()``
    - NumPy arrays with shape ``(2, N)`` (I/Q rows)
    - Already-complex arrays (returned as-is after dtype cast)
    - 1-D real arrays (returned as complex view if possible)

    Args:
        data: Tensor or array to convert.

    Returns:
        1-D ``np.complex64`` array.
    """
    if hasattr(data, "numpy"):
        arr = data.numpy()
    else:
        arr = np.asarray(data)

    if np.iscomplexobj(arr):
        return arr.astype(np.complex64).ravel()

    if arr.ndim == 2 and arr.shape[0] == 2:
        return (arr[0] + 1j * arr[1]).astype(np.complex64)

    return arr.astype(np.complex64).ravel()


def export_dataset_to_folder(
    dataset,
    output_dir: str,
    writer_factory: Callable[[str], object],
    file_extension: str,
    class_list: Optional[List[str]] = None,
    max_samples: Optional[int] = None,
) -> None:
    """Export a SPECTRA dataset to an ImageFolder-compatible directory.

    Creates a folder structure::

        output_dir/
            ClassName/
                sample_000000.<ext>
                sample_000001.<ext>

    Args:
        dataset: A SPECTRA dataset with ``__len__`` and ``__getitem__``
            returning ``(data, label)`` tuples.
        output_dir: Root output directory.
        writer_factory: Callable that takes a file path (with extension)
            and returns a writer instance with a ``.write(iq)`` method.
        file_extension: Extension including dot (e.g. ``".npy"``).
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
        iq = _tensor_to_complex64(data)

        # Determine class directory name
        if class_list is not None:
            label_int = int(label) if not isinstance(label, int) else label
            cls_name = class_list[label_int]
        else:
            cls_name = str(int(label) if not isinstance(label, int) else label)

        cls_dir = os.path.join(output_dir, cls_name)
        os.makedirs(cls_dir, exist_ok=True)

        file_path = os.path.join(cls_dir, f"sample_{i:06d}{file_extension}")
        writer = writer_factory(file_path)
        writer.write(iq)
