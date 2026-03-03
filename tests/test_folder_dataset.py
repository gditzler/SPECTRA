import numpy as np
import pytest
import torch


class TestSignalFolderDataset:
    @pytest.fixture
    def folder_root(self, tmp_path):
        """Create root/BPSK/ and root/QPSK/ with .npy files."""
        for cls_name in ["BPSK", "QPSK"]:
            cls_dir = tmp_path / cls_name
            cls_dir.mkdir()
            for i in range(3):
                iq = np.random.randn(512).astype(np.float32).view(np.complex64)
                np.save(str(cls_dir / f"rec_{i:03d}.npy"), iq)
        return str(tmp_path)

    def test_len(self, folder_root):
        from spectra.datasets.folder import SignalFolderDataset

        ds = SignalFolderDataset(root=folder_root, num_iq_samples=128)
        assert len(ds) == 6  # 3 files x 2 classes

    def test_classes_sorted(self, folder_root):
        from spectra.datasets.folder import SignalFolderDataset

        ds = SignalFolderDataset(root=folder_root, num_iq_samples=128)
        assert ds.classes == ["BPSK", "QPSK"]

    def test_class_to_idx(self, folder_root):
        from spectra.datasets.folder import SignalFolderDataset

        ds = SignalFolderDataset(root=folder_root, num_iq_samples=128)
        assert ds.class_to_idx == {"BPSK": 0, "QPSK": 1}

    def test_getitem_returns_tensor_and_int(self, folder_root):
        from spectra.datasets.folder import SignalFolderDataset

        ds = SignalFolderDataset(root=folder_root, num_iq_samples=128)
        data, label = ds[0]
        assert isinstance(data, torch.Tensor)
        assert isinstance(label, int)

    def test_tensor_shape_default(self, folder_root):
        from spectra.datasets.folder import SignalFolderDataset

        ds = SignalFolderDataset(root=folder_root, num_iq_samples=128)
        data, _ = ds[0]
        assert data.shape == (2, 128)
        assert data.dtype == torch.float32

    def test_truncation(self, tmp_path):
        from spectra.datasets.folder import SignalFolderDataset

        cls_dir = tmp_path / "A"
        cls_dir.mkdir()
        iq = np.zeros(1024, dtype=np.complex64)
        np.save(str(cls_dir / "long.npy"), iq)
        ds = SignalFolderDataset(root=str(tmp_path), num_iq_samples=64)
        data, _ = ds[0]
        assert data.shape == (2, 64)

    def test_zero_padding(self, tmp_path):
        from spectra.datasets.folder import SignalFolderDataset

        cls_dir = tmp_path / "A"
        cls_dir.mkdir()
        iq = np.ones(32, dtype=np.complex64)
        np.save(str(cls_dir / "short.npy"), iq)
        ds = SignalFolderDataset(root=str(tmp_path), num_iq_samples=128)
        data, _ = ds[0]
        assert data.shape == (2, 128)
        # Last samples should be zero (padding)
        assert data[0, 32:].sum() == 0.0

    def test_with_transform(self, folder_root):
        from spectra.datasets.folder import SignalFolderDataset

        ds = SignalFolderDataset(
            root=folder_root,
            num_iq_samples=128,
            transform=lambda iq: torch.tensor([42.0]),
        )
        data, _ = ds[0]
        assert data.item() == 42.0

    def test_with_target_transform(self, folder_root):
        from spectra.datasets.folder import SignalFolderDataset

        ds = SignalFolderDataset(
            root=folder_root,
            num_iq_samples=128,
            target_transform=lambda x: x + 100,
        )
        _, label = ds[0]
        assert label >= 100

    def test_empty_root_raises(self, tmp_path):
        from spectra.datasets.folder import SignalFolderDataset

        with pytest.raises(FileNotFoundError):
            SignalFolderDataset(root=str(tmp_path), num_iq_samples=128)

    def test_dataloader_compatible(self, folder_root):
        from spectra.datasets.folder import SignalFolderDataset
        from torch.utils.data import DataLoader

        ds = SignalFolderDataset(root=folder_root, num_iq_samples=128)
        loader = DataLoader(ds, batch_size=2)
        batch_data, batch_labels = next(iter(loader))
        assert batch_data.shape == (2, 2, 128)
        assert batch_labels.shape == (2,)


class TestModuleExports:
    def test_import_from_datasets(self):
        from spectra.datasets import ManifestDataset, SignalFolderDataset

        assert SignalFolderDataset is not None
        assert ManifestDataset is not None

    def test_import_from_top_level(self):
        from spectra import ManifestDataset, SignalFolderDataset

        assert SignalFolderDataset is not None
        assert ManifestDataset is not None

    def test_import_readers_from_top_level(self):
        from spectra.utils.file_handlers import (
            FileReader,
            NumpyReader,
            RawIQReader,
            SigMFReader,
            SignalMetadata,
            get_reader,
        )

        assert FileReader is not None
        assert SignalMetadata is not None
        assert get_reader is not None
        assert NumpyReader is not None
        assert RawIQReader is not None
        assert SigMFReader is not None
