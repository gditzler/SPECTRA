import os
import tempfile


class TestDatasetMetadata:
    def test_to_dict(self):
        from spectra.datasets.metadata import DatasetMetadata

        meta = DatasetMetadata(name="test", num_samples=100)
        d = meta.to_dict()
        assert d["name"] == "test"
        assert d["num_samples"] == 100

    def test_yaml_roundtrip(self):
        from spectra.datasets.metadata import DatasetMetadata

        meta = DatasetMetadata(name="test_ds", num_samples=50, seed=123)
        with tempfile.NamedTemporaryFile(suffix=".yaml", delete=False) as f:
            path = f.name
        try:
            meta.to_yaml(path)
            loaded = DatasetMetadata.from_yaml(path)
            assert loaded.name == "test_ds"
            assert loaded.num_samples == 50
            assert loaded.seed == 123
        finally:
            os.unlink(path)


class TestNarrowbandMetadata:
    def test_from_dict(self):
        from spectra.datasets.metadata import NarrowbandMetadata

        meta = NarrowbandMetadata.from_dict(
            {
                "name": "nb",
                "num_samples": 10,
                "sample_rate": 1e6,
                "seed": 0,
                "waveform_labels": ["QPSK"],
                "num_iq_samples": 512,
                "snr_range": [0.0, 10.0],
            }
        )
        assert meta.snr_range == (0.0, 10.0)

    def test_yaml_roundtrip(self):
        from spectra.datasets.metadata import NarrowbandMetadata

        meta = NarrowbandMetadata(waveform_labels=["BPSK", "QPSK"])
        with tempfile.NamedTemporaryFile(suffix=".yaml", delete=False) as f:
            path = f.name
        try:
            meta.to_yaml(path)
            loaded = NarrowbandMetadata.from_yaml(path)
            assert loaded.waveform_labels == ["BPSK", "QPSK"]
        finally:
            os.unlink(path)


class TestWidebandMetadata:
    def test_from_dict(self):
        from spectra.datasets.metadata import WidebandMetadata

        meta = WidebandMetadata.from_dict(
            {
                "name": "wb",
                "num_samples": 10,
                "sample_rate": 1e6,
                "seed": 0,
                "capture_bandwidth": 2e6,
                "capture_duration": 1e-3,
                "num_signals_range": [1, 5],
            }
        )
        assert meta.num_signals_range == (1, 5)
