"""Tests for the TorchSig adapter using a mock dataset."""
import numpy as np
import pytest
import torch
from benchmarks.torchsig_compat.adapter import TorchSigAdapter
from benchmarks.torchsig_compat.label_map import CANONICAL_CLASSES


class FakeTorchSigDataset:
    """Mock TorchSig dataset returning (complex_iq, metadata_dict)."""

    def __init__(self, n=100, num_iq=1024):
        self.n = n
        self.num_iq = num_iq

    def __len__(self):
        return self.n

    def __getitem__(self, idx):
        rng = np.random.default_rng(idx)
        iq = (rng.standard_normal(self.num_iq) +
              1j * rng.standard_normal(self.num_iq)).astype(np.complex64)
        meta = {
            "class_name": ["bpsk", "qpsk", "8psk", "16qam"][idx % 4],
            "snr": rng.uniform(0, 20),
        }
        return iq, meta


def test_adapter_len():
    ds = FakeTorchSigDataset(n=50)
    adapter = TorchSigAdapter(ds, class_list=CANONICAL_CLASSES[:4])
    assert len(adapter) == 50


def test_adapter_returns_tensor_and_int():
    ds = FakeTorchSigDataset(n=10, num_iq=512)
    adapter = TorchSigAdapter(ds, class_list=CANONICAL_CLASSES[:4])
    data, label = adapter[0]
    assert isinstance(data, torch.Tensor)
    assert data.shape == (2, 512)
    assert isinstance(label, int)
    assert 0 <= label < 4


def test_adapter_label_mapping():
    ds = FakeTorchSigDataset(n=4)
    adapter = TorchSigAdapter(ds, class_list=CANONICAL_CLASSES[:4])
    # idx=0 -> "bpsk" -> 0, idx=1 -> "qpsk" -> 1, etc.
    _, l0 = adapter[0]
    _, l1 = adapter[1]
    assert l0 == 0  # BPSK
    assert l1 == 1  # QPSK


def test_adapter_with_transform():
    ds = FakeTorchSigDataset(n=5, num_iq=512)
    transform = lambda x: x[:, :256]  # Crop
    adapter = TorchSigAdapter(ds, class_list=CANONICAL_CLASSES[:4],
                              transform=transform)
    data, _ = adapter[0]
    assert data.shape == (2, 256)
