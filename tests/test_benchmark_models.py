import pytest
import torch

pytest.importorskip("torchvision", reason="torchvision required for benchmark models")
from benchmarks.comparison.models import ResNetAMC


def test_resnet_forward_shape():
    model = ResNetAMC(num_classes=8, input_channels=1)
    x = torch.randn(4, 1, 128, 32)  # [B, C, freq, time]
    out = model(x)
    assert out.shape == (4, 8)


def test_resnet_default_channels():
    model = ResNetAMC(num_classes=8)
    x = torch.randn(2, 1, 64, 16)
    out = model(x)
    assert out.shape == (2, 8)


def test_resnet_two_channel_input():
    model = ResNetAMC(num_classes=8, input_channels=2)
    x = torch.randn(2, 2, 128, 32)
    out = model(x)
    assert out.shape == (2, 8)
