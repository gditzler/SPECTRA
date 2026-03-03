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
