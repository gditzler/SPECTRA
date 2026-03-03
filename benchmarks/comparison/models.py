"""CNN model for the SPECTRA vs TorchSig benchmark."""
import torch
import torch.nn as nn
from torchvision.models import resnet18


class ResNetAMC(nn.Module):
    """ResNet-18 adapted for spectrogram-based modulation classification.

    Replaces the first conv layer to accept 1-channel (magnitude spectrogram)
    or 2-channel (I/Q spectrogram) input, and replaces the final FC layer
    for the target number of classes.

    Args:
        num_classes: Number of modulation classes.
        input_channels: 1 for magnitude STFT, 2 for I/Q.
    """

    def __init__(self, num_classes: int = 8, input_channels: int = 1):
        super().__init__()
        base = resnet18(weights=None)
        # Replace first conv: 7x7 -> 3x3 (spectrograms are small)
        base.conv1 = nn.Conv2d(
            input_channels, 64, kernel_size=3, stride=1, padding=1, bias=False
        )
        base.maxpool = nn.Identity()  # Skip aggressive downsampling
        base.fc = nn.Linear(base.fc.in_features, num_classes)
        self.net = base

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.net(x)
