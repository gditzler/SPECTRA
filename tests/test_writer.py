import os
import tempfile

import numpy as np
import pytest

try:
    import zarr

    HAS_ZARR = True
except ImportError:
    HAS_ZARR = False


def _open_zarr_group(path, mode):
    """Open zarr group compatible with both v2 and v3."""
    if hasattr(zarr, "open_group"):
        try:
            return zarr.open_group(path, mode=mode)
        except TypeError:
            return zarr.open(path, mode=mode)
    return zarr.open(path, mode=mode)


@pytest.mark.skipif(not HAS_ZARR, reason="zarr not installed")
class TestZarrHandler:
    def test_write_and_read(self):
        from spectra.utils.file_handlers import ZarrHandler

        with tempfile.TemporaryDirectory() as tmpdir:
            path = os.path.join(tmpdir, "test.zarr")
            handler = ZarrHandler(path)
            handler.open()
            data = np.random.randn(10, 4).astype(np.float32)
            arr = handler.create_array("data", shape=data.shape, dtype=data.dtype)
            arr[:] = data
            handler.write_metadata({"name": "test"})
            handler.close()

            # Read back
            read_data = handler.read("data")
            np.testing.assert_array_equal(read_data, data)

            root = handler.read()
            assert root.attrs["name"] == "test"


@pytest.mark.skipif(not HAS_ZARR, reason="zarr not installed")
class TestDatasetWriter:
    def test_write_narrowband(self):
        from spectra.waveforms import QPSK, BPSK
        from spectra.datasets import NarrowbandDataset
        from spectra.transforms import STFT
        from spectra.utils.writer import DatasetWriter

        ds = NarrowbandDataset(
            waveform_pool=[QPSK(), BPSK()],
            num_iq_samples=256,
            num_samples=8,
            sample_rate=1e6,
            seed=42,
            transform=STFT(nfft=64, hop_length=16),
        )
        with tempfile.TemporaryDirectory() as tmpdir:
            path = os.path.join(tmpdir, "test.zarr")
            writer = DatasetWriter(ds, path)
            writer.write(batch_size=4)
            writer.finalize(metadata={"dataset": "test"})

            # Verify
            root = _open_zarr_group(path, mode="r")
            assert "data" in root
            assert root["data"].shape[0] == 8
