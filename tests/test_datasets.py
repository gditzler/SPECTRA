import numpy as np
import torch
import pytest


class TestNarrowbandDataset:
    def test_len(self):
        from spectra.datasets import NarrowbandDataset
        from spectra.waveforms import QPSK, BPSK
        ds = NarrowbandDataset(
            waveform_pool=[QPSK(samples_per_symbol=4), BPSK(samples_per_symbol=4)],
            num_samples=100,
            num_iq_samples=1024,
            sample_rate=1e6,
            seed=42,
        )
        assert len(ds) == 100

    def test_getitem_returns_tensor_and_label(self):
        from spectra.datasets import NarrowbandDataset
        from spectra.waveforms import QPSK, BPSK
        ds = NarrowbandDataset(
            waveform_pool=[QPSK(samples_per_symbol=4), BPSK(samples_per_symbol=4)],
            num_samples=100,
            num_iq_samples=1024,
            sample_rate=1e6,
            seed=42,
        )
        data, label = ds[0]
        assert isinstance(data, torch.Tensor)
        assert isinstance(label, int)
        assert data.shape == (2, 1024)  # I and Q channels
        assert data.dtype == torch.float32

    def test_deterministic(self):
        from spectra.datasets import NarrowbandDataset
        from spectra.waveforms import QPSK
        ds = NarrowbandDataset(
            waveform_pool=[QPSK(samples_per_symbol=4)],
            num_samples=10,
            num_iq_samples=512,
            sample_rate=1e6,
            seed=42,
        )
        d1, l1 = ds[0]
        d2, l2 = ds[0]
        torch.testing.assert_close(d1, d2)
        assert l1 == l2

    def test_different_indices_differ(self):
        from spectra.datasets import NarrowbandDataset
        from spectra.waveforms import QPSK, BPSK
        ds = NarrowbandDataset(
            waveform_pool=[QPSK(samples_per_symbol=4), BPSK(samples_per_symbol=4)],
            num_samples=100,
            num_iq_samples=512,
            sample_rate=1e6,
            seed=42,
        )
        d0, _ = ds[0]
        d1, _ = ds[1]
        assert not torch.equal(d0, d1)

    def test_with_dataloader(self):
        from spectra.datasets import NarrowbandDataset
        from spectra.waveforms import QPSK
        from torch.utils.data import DataLoader
        ds = NarrowbandDataset(
            waveform_pool=[QPSK(samples_per_symbol=4)],
            num_samples=32,
            num_iq_samples=256,
            sample_rate=1e6,
            seed=42,
        )
        loader = DataLoader(ds, batch_size=8)
        batch_data, batch_labels = next(iter(loader))
        assert batch_data.shape == (8, 2, 256)
        assert batch_labels.shape == (8,)


class TestWidebandDataset:
    @pytest.fixture
    def wideband_ds(self):
        from spectra.datasets import WidebandDataset
        from spectra.scene.composer import SceneConfig
        from spectra.waveforms import QPSK, BPSK
        from spectra.transforms.stft import STFT
        config = SceneConfig(
            capture_duration=1e-3,
            capture_bandwidth=1e6,
            sample_rate=2e6,
            num_signals=(2, 4),
            signal_pool=[QPSK(samples_per_symbol=4), BPSK(samples_per_symbol=4)],
            snr_range=(10, 20),
            allow_overlap=True,
        )
        return WidebandDataset(
            scene_config=config,
            num_samples=50,
            transform=STFT(nfft=128, hop_length=32),
            seed=42,
        )

    def test_len(self, wideband_ds):
        assert len(wideband_ds) == 50

    def test_getitem_returns_tensor_and_targets(self, wideband_ds):
        data, targets = wideband_ds[0]
        assert isinstance(data, torch.Tensor)
        assert data.ndim == 3  # [C, freq, time]
        assert isinstance(targets, dict)
        assert "boxes" in targets
        assert "labels" in targets
        assert "signal_descs" in targets

    def test_deterministic(self, wideband_ds):
        d1, t1 = wideband_ds[0]
        d2, t2 = wideband_ds[0]
        torch.testing.assert_close(d1, d2)
        torch.testing.assert_close(t1["boxes"], t2["boxes"])

    def test_boxes_match_signal_count(self, wideband_ds):
        _, targets = wideband_ds[0]
        num_signals = len(targets["signal_descs"])
        assert targets["boxes"].shape[0] == num_signals
        assert targets["labels"].shape[0] == num_signals


class TestCollate:
    def test_collate_fn(self):
        from spectra.datasets import collate_fn
        # Simulate two samples with different numbers of boxes
        batch = [
            (
                torch.randn(1, 64, 32),
                {"boxes": torch.randn(3, 4), "labels": torch.tensor([0, 1, 0]), "signal_descs": []},
            ),
            (
                torch.randn(1, 64, 32),
                {"boxes": torch.randn(2, 4), "labels": torch.tensor([1, 1]), "signal_descs": []},
            ),
        ]
        data, targets = collate_fn(batch)
        assert data.shape == (2, 1, 64, 32)
        assert isinstance(targets, list)
        assert len(targets) == 2
        assert targets[0]["boxes"].shape == (3, 4)
        assert targets[1]["boxes"].shape == (2, 4)
