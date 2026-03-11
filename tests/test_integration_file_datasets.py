import csv

import numpy as np
import pytest
import torch
from torch.utils.data import DataLoader


class TestEndToEndPipeline:
    def test_generate_export_reload(self, tmp_path):
        """Full pipeline: NarrowbandDataset -> SigMF -> SignalFolderDataset."""
        from spectra.datasets import NarrowbandDataset
        from spectra.datasets.folder import SignalFolderDataset
        from spectra.utils.file_handlers.sigmf_writer import SigMFWriter
        from spectra.waveforms import BPSK, QPSK

        ds = NarrowbandDataset(
            waveform_pool=[BPSK(), QPSK()],
            num_samples=8,
            num_iq_samples=256,
            sample_rate=1e6,
            seed=42,
        )
        out = str(tmp_path / "export")
        SigMFWriter.write_from_dataset(
            ds,
            output_dir=out,
            sample_rate=1e6,
            class_list=["BPSK", "QPSK"],
        )
        loaded = SignalFolderDataset(root=out, num_iq_samples=256)
        loader = DataLoader(loaded, batch_size=4)
        batch_data, batch_labels = next(iter(loader))
        assert batch_data.shape == (4, 2, 256)
        assert batch_data.dtype == torch.float32

    def test_folder_dataset_with_spectrogram(self, tmp_path):
        """SignalFolderDataset with Spectrogram transform."""
        from spectra.datasets.folder import SignalFolderDataset
        from spectra.transforms import Spectrogram

        cls_dir = tmp_path / "NOISE"
        cls_dir.mkdir()
        for i in range(2):
            iq = np.random.randn(512).astype(np.float32).view(np.complex64)
            np.save(str(cls_dir / f"rec_{i}.npy"), iq)

        ds = SignalFolderDataset(
            root=str(tmp_path),
            num_iq_samples=256,
            transform=Spectrogram(nfft=64, hop_length=16),
        )
        data, label = ds[0]
        assert data.ndim == 3  # [1, freq, time]
        assert label == 0

    def test_manifest_with_mixed_formats(self, tmp_path):
        """ManifestDataset loading .npy and .cf32 files."""
        from spectra.datasets.manifest import ManifestDataset

        # .npy file
        np.save(str(tmp_path / "a.npy"), np.zeros(128, dtype=np.complex64))
        # .cf32 file
        np.zeros(128, dtype=np.complex64).tofile(str(tmp_path / "b.cf32"))

        manifest = tmp_path / "manifest.csv"
        with open(str(manifest), "w", newline="") as f:
            w = csv.writer(f)
            w.writerow(["file", "label"])
            w.writerow(["a.npy", "ClassA"])
            w.writerow(["b.cf32", "ClassB"])

        ds = ManifestDataset(manifest_path=str(manifest), num_iq_samples=64)
        assert len(ds) == 2
        d0, l0 = ds[0]
        d1, l1 = ds[1]
        assert d0.shape == (2, 64)
        assert l0 != l1

    def test_generate_export_reload_numpy(self, tmp_path):
        """Full pipeline: NarrowbandDataset -> NumpyWriter -> SignalFolderDataset -> DataLoader."""
        from spectra.datasets import NarrowbandDataset
        from spectra.datasets.folder import SignalFolderDataset
        from spectra.utils.file_handlers.numpy_writer import NumpyWriter
        from spectra.waveforms import BPSK, QPSK

        ds = NarrowbandDataset(
            waveform_pool=[BPSK(), QPSK()],
            num_samples=8,
            num_iq_samples=128,
            sample_rate=1e6,
            seed=99,
        )
        out = str(tmp_path / "numpy_export")
        NumpyWriter.write_from_dataset(
            ds,
            output_dir=out,
            class_list=["BPSK", "QPSK"],
        )
        loaded = SignalFolderDataset(root=out, num_iq_samples=128)
        loader = DataLoader(loaded, batch_size=4)
        batch_data, batch_labels = next(iter(loader))
        assert batch_data.shape == (4, 2, 128)
        assert batch_data.dtype == torch.float32

    def test_generate_export_reload_raw(self, tmp_path):
        """Full pipeline: NarrowbandDataset -> RawIQWriter -> SignalFolderDataset -> DataLoader."""
        from spectra.datasets import NarrowbandDataset
        from spectra.datasets.folder import SignalFolderDataset
        from spectra.utils.file_handlers.raw_writer import RawIQWriter
        from spectra.waveforms import BPSK, QPSK

        ds = NarrowbandDataset(
            waveform_pool=[BPSK(), QPSK()],
            num_samples=8,
            num_iq_samples=128,
            sample_rate=1e6,
            seed=99,
        )
        out = str(tmp_path / "raw_export")
        RawIQWriter.write_from_dataset(
            ds,
            output_dir=out,
            class_list=["BPSK", "QPSK"],
        )
        loaded = SignalFolderDataset(root=out, num_iq_samples=128)
        loader = DataLoader(loaded, batch_size=4)
        batch_data, batch_labels = next(iter(loader))
        assert batch_data.shape == (4, 2, 128)
        assert batch_data.dtype == torch.float32

    def test_generate_export_reload_hdf5(self, tmp_path):
        """Full pipeline: NarrowbandDataset -> HDF5Writer -> SignalFolderDataset -> DataLoader."""
        pytest.importorskip("h5py")
        from spectra.datasets import NarrowbandDataset
        from spectra.datasets.folder import SignalFolderDataset
        from spectra.utils.file_handlers.hdf5_writer import HDF5Writer
        from spectra.waveforms import BPSK, QPSK

        ds = NarrowbandDataset(
            waveform_pool=[BPSK(), QPSK()],
            num_samples=8,
            num_iq_samples=128,
            sample_rate=1e6,
            seed=99,
        )
        out = str(tmp_path / "hdf5_export")
        HDF5Writer.write_from_dataset(
            ds,
            output_dir=out,
            class_list=["BPSK", "QPSK"],
        )
        loaded = SignalFolderDataset(root=out, num_iq_samples=128)
        loader = DataLoader(loaded, batch_size=4)
        batch_data, batch_labels = next(iter(loader))
        assert batch_data.shape == (4, 2, 128)
        assert batch_data.dtype == torch.float32

    def test_generate_export_to_sqlite(self, tmp_path):
        """Full pipeline: NarrowbandDataset -> SQLiteWriter -> verify SQLite contents."""
        import sqlite3

        from spectra.datasets import NarrowbandDataset
        from spectra.utils.file_handlers.sqlite_reader import SQLiteReader
        from spectra.utils.file_handlers.sqlite_writer import SQLiteWriter
        from spectra.waveforms import BPSK, QPSK

        ds = NarrowbandDataset(
            waveform_pool=[BPSK(), QPSK()],
            num_samples=8,
            num_iq_samples=128,
            sample_rate=1e6,
            seed=99,
        )
        db_path = str(tmp_path / "export.db")
        SQLiteWriter.write_from_dataset(
            ds,
            output_path=db_path,
            class_list=["BPSK", "QPSK"],
            sample_rate=1e6,
        )

        # Verify via raw SQL
        conn = sqlite3.connect(db_path)
        row_count = conn.execute("SELECT COUNT(*) FROM samples").fetchone()[0]
        assert row_count == 8
        classes = {r[0] for r in conn.execute("SELECT DISTINCT class_name FROM samples")}
        assert classes == {"BPSK", "QPSK"}
        conn.close()

        # Verify via SQLiteReader roundtrip
        reader = SQLiteReader(row_index=0)
        iq, meta = reader.read(db_path)
        assert iq.dtype == np.complex64
        assert len(iq) == 128
        assert meta.num_samples == 128
