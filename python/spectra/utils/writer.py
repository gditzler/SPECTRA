import numpy as np

from spectra.utils.file_handlers.zarr_handler import ZarrHandler


class DatasetWriter:
    """Write generated datasets to disk using a configurable backend."""

    def __init__(self, dataset, output_path: str, handler: str = "zarr"):
        self._dataset = dataset
        self._output_path = output_path
        if handler == "zarr":
            self._handler = ZarrHandler(output_path)
        else:
            raise ValueError(f"Unknown handler: {handler}")
        self._handler.open()

    def write(self, batch_size: int = 32, num_workers: int = 0, progress: bool = False):
        """Generate and write the dataset in batches."""
        import torch
        from torch.utils.data import DataLoader

        loader = DataLoader(
            self._dataset,
            batch_size=batch_size,
            num_workers=num_workers,
            shuffle=False,
        )

        data_list = []
        target_list = []

        iterator = loader
        if progress:
            try:
                from tqdm import tqdm

                iterator = tqdm(loader, desc="Writing dataset")
            except ImportError:
                pass

        for batch_data, batch_targets in iterator:
            data_np = batch_data.numpy()
            data_list.append(data_np)
            if isinstance(batch_targets, (list, tuple)):
                target_list.extend(batch_targets)
            else:
                target_list.append(
                    batch_targets.numpy()
                    if isinstance(batch_targets, torch.Tensor)
                    else batch_targets
                )

        # Write all data at once
        all_data = np.concatenate(data_list, axis=0)
        arr = self._handler.create_array("data", shape=all_data.shape, dtype=all_data.dtype)
        arr[:] = all_data

        # Write targets
        if target_list:
            if isinstance(target_list[0], (int, float, np.integer, np.floating)):
                all_targets = np.array(target_list)
                arr = self._handler.create_array(
                    "targets", shape=all_targets.shape, dtype=all_targets.dtype
                )
                arr[:] = all_targets
            elif isinstance(target_list[0], np.ndarray):
                all_targets = np.concatenate(target_list, axis=0)
                arr = self._handler.create_array(
                    "targets", shape=all_targets.shape, dtype=all_targets.dtype
                )
                arr[:] = all_targets

    def finalize(self, metadata: dict = None):
        """Flush and write metadata."""
        if metadata:
            self._handler.write_metadata(metadata)
        self._handler.close()
