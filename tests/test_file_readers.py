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
