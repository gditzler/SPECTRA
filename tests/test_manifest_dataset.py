import csv
import json

import numpy as np
import pytest
import torch


class TestManifestDatasetCSV:
    @pytest.fixture
    def csv_manifest(self, tmp_path):
        """Create .npy files + CSV manifest."""
        for i in range(3):
            iq = np.zeros(256, dtype=np.complex64)
            np.save(str(tmp_path / f"rec_{i}.npy"), iq)
        manifest = tmp_path / "manifest.csv"
        with open(str(manifest), "w", newline="") as f:
            w = csv.writer(f)
            w.writerow(["file", "label"])
            w.writerow(["rec_0.npy", "BPSK"])
            w.writerow(["rec_1.npy", "QPSK"])
            w.writerow(["rec_2.npy", "BPSK"])
        return str(manifest)

    def test_len(self, csv_manifest):
        from spectra.datasets.manifest import ManifestDataset

        ds = ManifestDataset(manifest_path=csv_manifest, num_iq_samples=128)
        assert len(ds) == 3

    def test_classes(self, csv_manifest):
        from spectra.datasets.manifest import ManifestDataset

        ds = ManifestDataset(manifest_path=csv_manifest, num_iq_samples=128)
        assert ds.classes == ["BPSK", "QPSK"]

    def test_getitem_shape(self, csv_manifest):
        from spectra.datasets.manifest import ManifestDataset

        ds = ManifestDataset(manifest_path=csv_manifest, num_iq_samples=128)
        data, label = ds[0]
        assert data.shape == (2, 128)
        assert isinstance(label, int)


class TestManifestDatasetJSON:
    @pytest.fixture
    def json_manifest(self, tmp_path):
        for i in range(2):
            np.save(
                str(tmp_path / f"rec_{i}.npy"),
                np.zeros(64, dtype=np.complex64),
            )
        manifest = tmp_path / "manifest.json"
        entries = [
            {"file": "rec_0.npy", "label": "QAM16"},
            {"file": "rec_1.npy", "label": "QAM64"},
        ]
        with open(str(manifest), "w") as f:
            json.dump(entries, f)
        return str(manifest)

    def test_json_load(self, json_manifest):
        from spectra.datasets.manifest import ManifestDataset

        ds = ManifestDataset(manifest_path=json_manifest, num_iq_samples=32)
        assert len(ds) == 2
        data, label = ds[0]
        assert data.shape == (2, 32)

    def test_dataloader_compatible(self, json_manifest):
        from spectra.datasets.manifest import ManifestDataset
        from torch.utils.data import DataLoader

        ds = ManifestDataset(manifest_path=json_manifest, num_iq_samples=32)
        loader = DataLoader(ds, batch_size=2)
        batch_data, batch_labels = next(iter(loader))
        assert batch_data.shape == (2, 2, 32)
