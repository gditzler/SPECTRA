import os

import numpy as np
import pytest
import torch


class TestFileWriterABC:
    def test_cannot_instantiate(self):
        from spectra.utils.file_handlers.base_writer import FileWriter

        with pytest.raises(TypeError):
            FileWriter()


class TestTensorToComplex64:
    def test_2d_tensor(self):
        from spectra.utils.file_handlers.dataset_export import _tensor_to_complex64

        t = torch.tensor([[1.0, 2.0], [3.0, 4.0]])  # [2, 2]
        iq = _tensor_to_complex64(t)
        assert iq.dtype == np.complex64
        assert iq.shape == (2,)
        np.testing.assert_allclose(iq.real, [1.0, 2.0])
        np.testing.assert_allclose(iq.imag, [3.0, 4.0])

    def test_1d_complex_passthrough(self):
        from spectra.utils.file_handlers.dataset_export import _tensor_to_complex64

        arr = np.array([1 + 2j, 3 + 4j], dtype=np.complex64)
        result = _tensor_to_complex64(arr)
        np.testing.assert_array_equal(result, arr)

    def test_2d_ndarray(self):
        from spectra.utils.file_handlers.dataset_export import _tensor_to_complex64

        arr = np.array([[1.0, 2.0], [3.0, 4.0]])
        iq = _tensor_to_complex64(arr)
        assert iq.dtype == np.complex64
        assert iq.shape == (2,)


class TestNumpyWriter:
    def test_write_creates_file(self, tmp_path):
        from spectra.utils.file_handlers.numpy_writer import NumpyWriter

        iq = np.zeros(128, dtype=np.complex64)
        path = str(tmp_path / "signal.npy")
        writer = NumpyWriter(path)
        writer.write(iq)
        assert os.path.exists(path)

    def test_data_fidelity(self, tmp_path):
        from spectra.utils.file_handlers.numpy_writer import NumpyWriter

        iq = np.array([1 + 2j, 3 + 4j, 5 + 6j], dtype=np.complex64)
        path = str(tmp_path / "signal.npy")
        NumpyWriter(path).write(iq)
        loaded = np.load(path)
        np.testing.assert_array_equal(loaded, iq)

    def test_roundtrip_with_reader(self, tmp_path):
        from spectra.utils.file_handlers.numpy_reader import NumpyReader
        from spectra.utils.file_handlers.numpy_writer import NumpyWriter

        iq = np.array([1 + 2j, 3 + 4j], dtype=np.complex64)
        path = str(tmp_path / "rt.npy")
        NumpyWriter(path).write(iq)
        result, meta = NumpyReader().read(path)
        np.testing.assert_array_equal(result, iq)

    def test_extensions(self):
        from spectra.utils.file_handlers.numpy_writer import NumpyWriter

        assert NumpyWriter.extensions() == (".npy",)


class TestNumpyWriteFromDataset:
    def test_creates_folder_structure(self, tmp_path):
        from spectra.datasets import NarrowbandDataset
        from spectra.utils.file_handlers.numpy_writer import NumpyWriter
        from spectra.waveforms import BPSK, QPSK

        ds = NarrowbandDataset(
            waveform_pool=[BPSK(), QPSK()],
            num_samples=4,
            num_iq_samples=128,
            sample_rate=1e6,
            seed=42,
        )
        out = str(tmp_path / "export")
        NumpyWriter.write_from_dataset(ds, output_dir=out, class_list=["BPSK", "QPSK"])
        npy_files = []
        for root, dirs, files in os.walk(out):
            npy_files.extend(f for f in files if f.endswith(".npy"))
        assert len(npy_files) == 4

    def test_roundtrip_to_folder_dataset(self, tmp_path):
        from spectra.datasets import NarrowbandDataset
        from spectra.datasets.folder import SignalFolderDataset
        from spectra.utils.file_handlers.numpy_writer import NumpyWriter
        from spectra.waveforms import BPSK, QPSK

        ds = NarrowbandDataset(
            waveform_pool=[BPSK(), QPSK()],
            num_samples=4,
            num_iq_samples=128,
            sample_rate=1e6,
            seed=42,
        )
        out = str(tmp_path / "export")
        NumpyWriter.write_from_dataset(ds, output_dir=out, class_list=["BPSK", "QPSK"])
        loaded = SignalFolderDataset(root=out, num_iq_samples=128)
        assert len(loaded) == 4
        data, label = loaded[0]
        assert data.shape == (2, 128)

    def test_max_samples(self, tmp_path):
        from spectra.datasets import NarrowbandDataset
        from spectra.utils.file_handlers.numpy_writer import NumpyWriter
        from spectra.waveforms import BPSK

        ds = NarrowbandDataset(
            waveform_pool=[BPSK()],
            num_samples=10,
            num_iq_samples=64,
            sample_rate=1e6,
            seed=42,
        )
        out = str(tmp_path / "export")
        NumpyWriter.write_from_dataset(ds, output_dir=out, class_list=["BPSK"], max_samples=3)
        npy_files = []
        for root, dirs, files in os.walk(out):
            npy_files.extend(f for f in files if f.endswith(".npy"))
        assert len(npy_files) == 3


class TestRawIQWriter:
    def test_write_cf32_creates_file(self, tmp_path):
        from spectra.utils.file_handlers.raw_writer import RawIQWriter

        iq = np.zeros(128, dtype=np.complex64)
        path = str(tmp_path / "signal.cf32")
        RawIQWriter(path).write(iq)
        assert os.path.exists(path)

    def test_write_cf32_data_fidelity(self, tmp_path):
        from spectra.utils.file_handlers.raw_writer import RawIQWriter

        iq = np.array([1 + 2j, 3 + 4j, 5 + 6j], dtype=np.complex64)
        path = str(tmp_path / "signal.cf32")
        RawIQWriter(path).write(iq)
        loaded = np.fromfile(path, dtype=np.complex64)
        np.testing.assert_array_equal(loaded, iq)

    def test_write_cs16_interleaved(self, tmp_path):
        from spectra.utils.file_handlers.raw_writer import RawIQWriter

        iq = np.array([0.5 + 0.25j, -0.5 - 0.25j], dtype=np.complex64)
        path = str(tmp_path / "signal.cs16")
        RawIQWriter(path, dtype="int16").write(iq)
        raw = np.fromfile(path, dtype=np.int16)
        assert len(raw) == 4  # 2 complex samples * 2 (I + Q)

    def test_roundtrip_cf32_with_reader(self, tmp_path):
        from spectra.utils.file_handlers.raw_reader import RawIQReader
        from spectra.utils.file_handlers.raw_writer import RawIQWriter

        iq = np.array([1 + 2j, 3 + 4j], dtype=np.complex64)
        path = str(tmp_path / "rt.cf32")
        RawIQWriter(path).write(iq)
        result, _ = RawIQReader().read(path)
        np.testing.assert_array_equal(result, iq)

    def test_roundtrip_cs16_with_reader(self, tmp_path):
        from spectra.utils.file_handlers.raw_reader import RawIQReader
        from spectra.utils.file_handlers.raw_writer import RawIQWriter

        iq = np.array([0.5 + 0.25j, -0.5 - 0.25j], dtype=np.complex64)
        path = str(tmp_path / "rt.cs16")
        RawIQWriter(path, dtype="int16").write(iq)
        result, _ = RawIQReader(dtype="int16").read(path)
        np.testing.assert_allclose(result.real, iq.real, atol=1e-4)
        np.testing.assert_allclose(result.imag, iq.imag, atol=1e-4)

    def test_extensions(self):
        from spectra.utils.file_handlers.raw_writer import RawIQWriter

        assert RawIQWriter.extensions() == (".cf32", ".cs16", ".raw", ".iq", ".bin")


class TestRawIQWriteFromDataset:
    def test_creates_folder_structure(self, tmp_path):
        from spectra.datasets import NarrowbandDataset
        from spectra.utils.file_handlers.raw_writer import RawIQWriter
        from spectra.waveforms import BPSK, QPSK

        ds = NarrowbandDataset(
            waveform_pool=[BPSK(), QPSK()],
            num_samples=4,
            num_iq_samples=128,
            sample_rate=1e6,
            seed=42,
        )
        out = str(tmp_path / "export")
        RawIQWriter.write_from_dataset(ds, output_dir=out, class_list=["BPSK", "QPSK"])
        cf32_files = []
        for root, dirs, files in os.walk(out):
            cf32_files.extend(f for f in files if f.endswith(".cf32"))
        assert len(cf32_files) == 4

    def test_roundtrip_to_folder_dataset(self, tmp_path):
        from spectra.datasets import NarrowbandDataset
        from spectra.datasets.folder import SignalFolderDataset
        from spectra.utils.file_handlers.raw_writer import RawIQWriter
        from spectra.waveforms import BPSK, QPSK

        ds = NarrowbandDataset(
            waveform_pool=[BPSK(), QPSK()],
            num_samples=4,
            num_iq_samples=128,
            sample_rate=1e6,
            seed=42,
        )
        out = str(tmp_path / "export")
        RawIQWriter.write_from_dataset(ds, output_dir=out, class_list=["BPSK", "QPSK"])
        loaded = SignalFolderDataset(root=out, num_iq_samples=128)
        assert len(loaded) == 4
        data, label = loaded[0]
        assert data.shape == (2, 128)


try:
    import h5py

    HAS_H5PY = True
except ImportError:
    HAS_H5PY = False


@pytest.mark.skipif(not HAS_H5PY, reason="h5py not installed")
class TestHDF5Writer:
    def test_write_creates_file(self, tmp_path):
        from spectra.utils.file_handlers.hdf5_writer import HDF5Writer

        iq = np.zeros(128, dtype=np.complex64)
        path = str(tmp_path / "signal.h5")
        HDF5Writer(path).write(iq)
        assert os.path.exists(path)

    def test_data_fidelity(self, tmp_path):
        from spectra.utils.file_handlers.hdf5_writer import HDF5Writer

        iq = np.array([1 + 2j, 3 + 4j, 5 + 6j], dtype=np.complex64)
        path = str(tmp_path / "signal.h5")
        HDF5Writer(path).write(iq)
        with h5py.File(path, "r") as f:
            loaded = f["iq"][:]
        np.testing.assert_array_equal(loaded, iq)

    def test_metadata_as_attrs(self, tmp_path):
        from spectra.utils.file_handlers.base_reader import SignalMetadata
        from spectra.utils.file_handlers.hdf5_writer import HDF5Writer

        iq = np.zeros(64, dtype=np.complex64)
        path = str(tmp_path / "signal.h5")
        meta = SignalMetadata(sample_rate=2e6, center_frequency=915e6)
        HDF5Writer(path).write(iq, metadata=meta)
        with h5py.File(path, "r") as f:
            assert f.attrs["sample_rate"] == 2e6
            assert f.attrs["center_frequency"] == 915e6

    def test_custom_dataset_name(self, tmp_path):
        from spectra.utils.file_handlers.hdf5_writer import HDF5Writer

        iq = np.zeros(64, dtype=np.complex64)
        path = str(tmp_path / "signal.h5")
        HDF5Writer(path, iq_dataset="data").write(iq)
        with h5py.File(path, "r") as f:
            assert "data" in f
            assert "iq" not in f

    def test_roundtrip_with_reader(self, tmp_path):
        from spectra.utils.file_handlers.hdf5_reader import HDF5Reader
        from spectra.utils.file_handlers.hdf5_writer import HDF5Writer

        iq = np.array([1 + 2j, 3 + 4j], dtype=np.complex64)
        path = str(tmp_path / "rt.h5")
        HDF5Writer(path).write(iq)
        result, meta = HDF5Reader().read(path)
        np.testing.assert_array_equal(result, iq)

    def test_extensions(self):
        from spectra.utils.file_handlers.hdf5_writer import HDF5Writer

        assert HDF5Writer.extensions() == (".h5", ".hdf5")


@pytest.mark.skipif(not HAS_H5PY, reason="h5py not installed")
class TestHDF5WriteFromDataset:
    def test_creates_folder_structure(self, tmp_path):
        from spectra.datasets import NarrowbandDataset
        from spectra.utils.file_handlers.hdf5_writer import HDF5Writer
        from spectra.waveforms import BPSK, QPSK

        ds = NarrowbandDataset(
            waveform_pool=[BPSK(), QPSK()],
            num_samples=4,
            num_iq_samples=128,
            sample_rate=1e6,
            seed=42,
        )
        out = str(tmp_path / "export")
        HDF5Writer.write_from_dataset(ds, output_dir=out, class_list=["BPSK", "QPSK"])
        h5_files = []
        for root, dirs, files in os.walk(out):
            h5_files.extend(f for f in files if f.endswith(".h5"))
        assert len(h5_files) == 4

    def test_roundtrip_to_folder_dataset(self, tmp_path):
        from spectra.datasets import NarrowbandDataset
        from spectra.datasets.folder import SignalFolderDataset
        from spectra.utils.file_handlers.hdf5_writer import HDF5Writer
        from spectra.waveforms import BPSK, QPSK

        ds = NarrowbandDataset(
            waveform_pool=[BPSK(), QPSK()],
            num_samples=4,
            num_iq_samples=128,
            sample_rate=1e6,
            seed=42,
        )
        out = str(tmp_path / "export")
        HDF5Writer.write_from_dataset(ds, output_dir=out, class_list=["BPSK", "QPSK"])
        loaded = SignalFolderDataset(root=out, num_iq_samples=128)
        assert len(loaded) == 4
        data, label = loaded[0]
        assert data.shape == (2, 128)


class TestSQLiteWriter:
    def test_write_creates_db(self, tmp_path):
        import sqlite3

        from spectra.utils.file_handlers.sqlite_writer import SQLiteWriter

        iq = np.zeros(128, dtype=np.complex64)
        path = str(tmp_path / "data.db")
        writer = SQLiteWriter(path)
        writer.write(iq, label=0, class_name="BPSK")
        writer.close()
        assert os.path.exists(path)
        conn = sqlite3.connect(path)
        count = conn.execute("SELECT COUNT(*) FROM samples").fetchone()[0]
        conn.close()
        assert count == 1

    def test_data_fidelity(self, tmp_path):
        import sqlite3

        from spectra.utils.file_handlers.sqlite_writer import SQLiteWriter

        iq = np.array([1 + 2j, 3 + 4j], dtype=np.complex64)
        path = str(tmp_path / "data.db")
        writer = SQLiteWriter(path)
        writer.write(iq, label=0)
        writer.close()

        conn = sqlite3.connect(path)
        row = conn.execute("SELECT iq, num_samples FROM samples").fetchone()
        conn.close()
        loaded = np.frombuffer(row[0], dtype=np.complex64)
        np.testing.assert_array_equal(loaded, iq)
        assert row[1] == 2

    def test_schema_tables_exist(self, tmp_path):
        import sqlite3

        from spectra.utils.file_handlers.sqlite_writer import SQLiteWriter

        path = str(tmp_path / "data.db")
        writer = SQLiteWriter(path)
        writer.close()

        conn = sqlite3.connect(path)
        tables = {
            r[0]
            for r in conn.execute("SELECT name FROM sqlite_master WHERE type='table'").fetchall()
        }
        conn.close()
        assert "samples" in tables
        assert "metadata" in tables

    def test_write_with_metadata(self, tmp_path):
        import sqlite3

        from spectra.utils.file_handlers.sqlite_writer import SQLiteWriter

        path = str(tmp_path / "data.db")
        writer = SQLiteWriter(path, sample_rate=1e6)
        writer.write_metadata("description", "test dataset")
        writer.close()

        conn = sqlite3.connect(path)
        row = conn.execute("SELECT value FROM metadata WHERE key='description'").fetchone()
        conn.close()
        assert row[0] == "test dataset"

    def test_write_multiple_records(self, tmp_path):
        import sqlite3

        from spectra.utils.file_handlers.sqlite_writer import SQLiteWriter

        path = str(tmp_path / "data.db")
        writer = SQLiteWriter(path)
        for i in range(5):
            writer.write(np.zeros(64, dtype=np.complex64), label=i % 2)
        writer.close()

        conn = sqlite3.connect(path)
        count = conn.execute("SELECT COUNT(*) FROM samples").fetchone()[0]
        conn.close()
        assert count == 5

    def test_extensions(self):
        from spectra.utils.file_handlers.sqlite_writer import SQLiteWriter

        assert SQLiteWriter.extensions() == (".db",)


class TestSQLiteWriteFromDataset:
    def test_bulk_export(self, tmp_path):
        import sqlite3

        from spectra.datasets import NarrowbandDataset
        from spectra.utils.file_handlers.sqlite_writer import SQLiteWriter
        from spectra.waveforms import BPSK, QPSK

        ds = NarrowbandDataset(
            waveform_pool=[BPSK(), QPSK()],
            num_samples=6,
            num_iq_samples=128,
            sample_rate=1e6,
            seed=42,
        )
        path = str(tmp_path / "dataset.db")
        SQLiteWriter.write_from_dataset(
            ds, output_path=path, class_list=["BPSK", "QPSK"], sample_rate=1e6
        )
        conn = sqlite3.connect(path)
        count = conn.execute("SELECT COUNT(*) FROM samples").fetchone()[0]
        conn.close()
        assert count == 6

    def test_class_names_stored(self, tmp_path):
        import sqlite3

        from spectra.datasets import NarrowbandDataset
        from spectra.utils.file_handlers.sqlite_writer import SQLiteWriter
        from spectra.waveforms import BPSK, QPSK

        ds = NarrowbandDataset(
            waveform_pool=[BPSK(), QPSK()],
            num_samples=4,
            num_iq_samples=128,
            sample_rate=1e6,
            seed=42,
        )
        path = str(tmp_path / "dataset.db")
        SQLiteWriter.write_from_dataset(ds, output_path=path, class_list=["BPSK", "QPSK"])
        conn = sqlite3.connect(path)
        names = {r[0] for r in conn.execute("SELECT DISTINCT class_name FROM samples").fetchall()}
        conn.close()
        assert names.issubset({"BPSK", "QPSK"})

    def test_max_samples(self, tmp_path):
        import sqlite3

        from spectra.datasets import NarrowbandDataset
        from spectra.utils.file_handlers.sqlite_writer import SQLiteWriter
        from spectra.waveforms import BPSK

        ds = NarrowbandDataset(
            waveform_pool=[BPSK()],
            num_samples=10,
            num_iq_samples=64,
            sample_rate=1e6,
            seed=42,
        )
        path = str(tmp_path / "dataset.db")
        SQLiteWriter.write_from_dataset(ds, output_path=path, class_list=["BPSK"], max_samples=3)
        conn = sqlite3.connect(path)
        count = conn.execute("SELECT COUNT(*) FROM samples").fetchone()[0]
        conn.close()
        assert count == 3
