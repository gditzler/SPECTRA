import numpy as np
import pytest

from spectra.utils.file_handlers.base_reader import FileReader, SignalMetadata
from spectra.utils.file_handlers.registry import (
    get_reader,
    register_reader,
    supported_extensions,
)


class TestSignalMetadata:
    def test_defaults(self):
        meta = SignalMetadata()
        assert meta.sample_rate is None
        assert meta.center_frequency is None
        assert meta.num_samples is None
        assert meta.annotations == []
        assert meta.extra == {}

    def test_fields(self):
        meta = SignalMetadata(sample_rate=1e6, center_frequency=2.4e9, num_samples=1024)
        assert meta.sample_rate == 1e6
        assert meta.center_frequency == 2.4e9
        assert meta.num_samples == 1024


class TestReaderRegistry:
    def test_get_reader_unknown_extension_raises(self, tmp_path):
        path = str(tmp_path / "file.xyz")
        with pytest.raises(ValueError, match="No reader registered"):
            get_reader(path)

    def test_supported_extensions_returns_tuple(self):
        exts = supported_extensions()
        assert isinstance(exts, tuple)


class TestNumpyReader:
    def test_read_npy_complex64(self, tmp_path):
        from spectra.utils.file_handlers.numpy_reader import NumpyReader

        iq = np.random.randn(512).astype(np.float32).view(np.complex64)
        path = str(tmp_path / "data.npy")
        np.save(path, iq)
        reader = NumpyReader()
        result, meta = reader.read(path)
        np.testing.assert_array_equal(result, iq)
        assert result.dtype == np.complex64
        assert meta.num_samples == len(iq)

    def test_read_npz_with_iq_key(self, tmp_path):
        from spectra.utils.file_handlers.numpy_reader import NumpyReader

        iq = np.zeros(256, dtype=np.complex64)
        path = str(tmp_path / "data.npz")
        np.savez(path, iq=iq)
        reader = NumpyReader()
        result, meta = reader.read(path)
        assert result.dtype == np.complex64
        assert len(result) == 256

    def test_read_npz_first_array_fallback(self, tmp_path):
        from spectra.utils.file_handlers.numpy_reader import NumpyReader

        iq = np.zeros(128, dtype=np.complex64)
        path = str(tmp_path / "data.npz")
        np.savez(path, samples=iq)
        reader = NumpyReader(array_key="iq")
        result, _ = reader.read(path)
        assert len(result) == 128

    def test_extensions(self):
        from spectra.utils.file_handlers.numpy_reader import NumpyReader

        assert ".npy" in NumpyReader.extensions()
        assert ".npz" in NumpyReader.extensions()


class TestRawIQReader:
    def test_read_cf32(self, tmp_path):
        from spectra.utils.file_handlers.raw_reader import RawIQReader

        iq = np.array([1 + 2j, 3 + 4j, 5 + 6j], dtype=np.complex64)
        path = str(tmp_path / "data.cf32")
        iq.tofile(path)
        reader = RawIQReader(dtype="complex64")
        result, meta = reader.read(path)
        np.testing.assert_array_equal(result, iq)
        assert result.dtype == np.complex64

    def test_read_cs16_interleaved(self, tmp_path):
        from spectra.utils.file_handlers.raw_reader import RawIQReader

        interleaved = np.array([100, 200, 300, 400], dtype=np.int16)
        path = str(tmp_path / "data.cs16")
        interleaved.tofile(path)
        reader = RawIQReader(dtype="int16")
        result, meta = reader.read(path)
        assert result.dtype == np.complex64
        assert len(result) == 2
        np.testing.assert_allclose(result[0].real, 100 / 32767.0, atol=1e-4)

    def test_sample_rate_in_metadata(self, tmp_path):
        from spectra.utils.file_handlers.raw_reader import RawIQReader

        iq = np.zeros(4, dtype=np.complex64)
        path = str(tmp_path / "data.raw")
        iq.tofile(path)
        reader = RawIQReader(dtype="complex64", sample_rate=2e6)
        _, meta = reader.read(path)
        assert meta.sample_rate == 2e6

    def test_extensions(self):
        from spectra.utils.file_handlers.raw_reader import RawIQReader

        exts = RawIQReader.extensions()
        assert ".cf32" in exts
        assert ".cs16" in exts
        assert ".raw" in exts
        assert ".iq" in exts
        assert ".bin" in exts


class TestSigMFReader:
    def _write_sigmf_pair(self, tmp_path, iq, sample_rate=1e6, center_freq=2.4e9):
        """Helper: write a SigMF meta+data pair using raw JSON + tofile."""
        base = str(tmp_path / "recording")
        iq.astype(np.complex64).tofile(base + ".sigmf-data")
        meta = {
            "global": {
                "core:datatype": "cf32_le",
                "core:sample_rate": sample_rate,
                "core:version": "1.0.0",
            },
            "captures": [
                {"core:sample_start": 0, "core:frequency": center_freq}
            ],
            "annotations": [],
        }
        import json

        with open(base + ".sigmf-meta", "w") as f:
            json.dump(meta, f)
        return base + ".sigmf-meta"

    def test_read_cf32_le(self, tmp_path):
        from spectra.utils.file_handlers.sigmf_reader import SigMFReader

        iq = np.array([1 + 2j, 3 + 4j], dtype=np.complex64)
        path = self._write_sigmf_pair(tmp_path, iq)
        reader = SigMFReader()
        result, meta = reader.read(path)
        np.testing.assert_array_equal(result, iq)
        assert result.dtype == np.complex64

    def test_metadata_fields(self, tmp_path):
        from spectra.utils.file_handlers.sigmf_reader import SigMFReader

        iq = np.zeros(64, dtype=np.complex64)
        path = self._write_sigmf_pair(tmp_path, iq, sample_rate=2e6, center_freq=915e6)
        reader = SigMFReader()
        _, meta = reader.read(path)
        assert meta.sample_rate == 2e6
        assert meta.center_frequency == 915e6
        assert meta.num_samples == 64

    def test_missing_data_file_raises(self, tmp_path):
        import json

        from spectra.utils.file_handlers.sigmf_reader import SigMFReader

        meta_path = str(tmp_path / "orphan.sigmf-meta")
        with open(meta_path, "w") as f:
            json.dump(
                {
                    "global": {"core:datatype": "cf32_le"},
                    "captures": [],
                    "annotations": [],
                },
                f,
            )
        with pytest.raises(FileNotFoundError):
            SigMFReader().read(meta_path)

    def test_extensions(self):
        from spectra.utils.file_handlers.sigmf_reader import SigMFReader

        assert ".sigmf-meta" in SigMFReader.extensions()


try:
    import h5py

    HAS_H5PY = True
except ImportError:
    HAS_H5PY = False


@pytest.mark.skipif(not HAS_H5PY, reason="h5py not installed")
class TestHDF5Reader:
    def test_read_complex64_dataset(self, tmp_path):
        from spectra.utils.file_handlers.hdf5_reader import HDF5Reader

        iq = np.zeros(256, dtype=np.complex64)
        path = str(tmp_path / "data.h5")
        with h5py.File(path, "w") as f:
            f.create_dataset("iq", data=iq)
        reader = HDF5Reader(iq_dataset="iq")
        result, meta = reader.read(path)
        assert result.dtype == np.complex64
        assert len(result) == 256

    def test_read_real_imag_columns(self, tmp_path):
        from spectra.utils.file_handlers.hdf5_reader import HDF5Reader

        data = np.zeros((128, 2), dtype=np.float32)
        data[:, 0] = 1.0  # I channel
        data[:, 1] = 2.0  # Q channel
        path = str(tmp_path / "data.h5")
        with h5py.File(path, "w") as f:
            f.create_dataset("X", data=data)
        reader = HDF5Reader(iq_dataset="X")
        result, meta = reader.read(path)
        assert result.dtype == np.complex64
        assert len(result) == 128
        np.testing.assert_allclose(result.real, 1.0)
        np.testing.assert_allclose(result.imag, 2.0)

    def test_fallback_dataset_names(self, tmp_path):
        from spectra.utils.file_handlers.hdf5_reader import HDF5Reader

        iq = np.zeros(64, dtype=np.complex64)
        path = str(tmp_path / "data.h5")
        with h5py.File(path, "w") as f:
            f.create_dataset("X", data=iq)
        reader = HDF5Reader()  # default key is "iq", should fallback to "X"
        result, _ = reader.read(path)
        assert len(result) == 64

    def test_metadata_from_attrs(self, tmp_path):
        from spectra.utils.file_handlers.hdf5_reader import HDF5Reader

        iq = np.zeros(32, dtype=np.complex64)
        path = str(tmp_path / "data.h5")
        with h5py.File(path, "w") as f:
            f.create_dataset("iq", data=iq)
            f.attrs["sample_rate"] = 1e6
        reader = HDF5Reader()
        _, meta = reader.read(path)
        assert meta.sample_rate == 1e6


class TestRegistryWiring:
    def test_numpy_registered(self, tmp_path):
        from spectra.utils.file_handlers import get_reader

        path = str(tmp_path / "test.npy")
        np.save(path, np.zeros(4, dtype=np.complex64))
        reader = get_reader(path)
        assert type(reader).__name__ == "NumpyReader"

    def test_raw_registered(self, tmp_path):
        from spectra.utils.file_handlers import get_reader

        path = str(tmp_path / "test.cf32")
        np.zeros(4, dtype=np.complex64).tofile(path)
        reader = get_reader(path)
        assert type(reader).__name__ == "RawIQReader"

    def test_sigmf_registered(self):
        from spectra.utils.file_handlers import get_reader

        reader = get_reader("/fake/path.sigmf-meta")
        assert type(reader).__name__ == "SigMFReader"

    def test_supported_extensions_comprehensive(self):
        from spectra.utils.file_handlers import supported_extensions

        exts = supported_extensions()
        for ext in (".npy", ".npz", ".cf32", ".cs16", ".raw", ".iq", ".bin", ".sigmf-meta"):
            assert ext in exts


class TestSQLiteReader:
    def _make_db(self, tmp_path, n=3):
        """Create a test .db using SQLiteWriter."""
        from spectra.utils.file_handlers.sqlite_writer import SQLiteWriter

        path = str(tmp_path / "test.db")
        writer = SQLiteWriter(path, sample_rate=1e6)
        for i in range(n):
            iq = np.array([float(i) + 1j * float(i + 1)], dtype=np.complex64)
            writer.write(iq, label=i, class_name=f"class_{i}")
        writer.close()
        return path

    def test_read_single_record(self, tmp_path):
        from spectra.utils.file_handlers.sqlite_reader import SQLiteReader

        path = self._make_db(tmp_path)
        iq, meta = SQLiteReader().read(path)
        assert iq.dtype == np.complex64
        assert len(iq) == 1

    def test_read_metadata(self, tmp_path):
        from spectra.utils.file_handlers.sqlite_reader import SQLiteReader

        path = self._make_db(tmp_path)
        _, meta = SQLiteReader().read(path)
        assert meta.sample_rate == 1e6
        assert meta.num_samples == 1

    def test_read_by_index(self, tmp_path):
        from spectra.utils.file_handlers.sqlite_reader import SQLiteReader

        path = self._make_db(tmp_path, n=5)
        iq0, _ = SQLiteReader(row_index=0).read(path)
        iq2, _ = SQLiteReader(row_index=2).read(path)
        # Values differ because each row has a unique IQ value
        assert not np.array_equal(iq0, iq2)

    def test_roundtrip(self, tmp_path):
        from spectra.utils.file_handlers.sqlite_reader import SQLiteReader
        from spectra.utils.file_handlers.sqlite_writer import SQLiteWriter

        original = np.array([1 + 2j, 3 + 4j, 5 + 6j], dtype=np.complex64)
        path = str(tmp_path / "rt.db")
        writer = SQLiteWriter(path)
        writer.write(original, label=0)
        writer.close()

        loaded, meta = SQLiteReader().read(path)
        np.testing.assert_array_equal(loaded, original)

    def test_extensions(self):
        from spectra.utils.file_handlers.sqlite_reader import SQLiteReader

        assert SQLiteReader.extensions() == (".db",)
