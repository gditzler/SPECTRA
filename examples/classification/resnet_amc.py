"""
ResNet AMC Classifier on Spectrograms
======================================
Level: Advanced

Train a 2D ResNetAMC on spectrogram features (contrast with the 1D CNNAMC
in train_narrowband_cnn.py).

  - Generate a NarrowbandDataset with STFT spectrogram transform
  - Train ResNetAMC for a few epochs
  - Evaluate with confusion matrix

Run:
    python examples/classification/resnet_amc.py
"""

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

import matplotlib
import numpy as np

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import torch
import torch.nn as nn
from plot_helpers import savefig
from spectra.datasets import NarrowbandDataset
from spectra.impairments import AWGN
from spectra.metrics import accuracy, confusion_matrix
from spectra.models import ResNetAMC
from spectra.transforms import Spectrogram
from spectra.waveforms import BPSK, FSK, QAM16, QPSK
from torch.utils.data import DataLoader

sample_rate = 1e6
num_classes = 4
waveforms = [BPSK(), QPSK(), QAM16(), FSK()]

# ── 1. Build spectrogram dataset ────────────────────────────────────────────
spec_transform = Spectrogram(nfft=64, hop_length=16)

train_ds = NarrowbandDataset(
    waveform_pool=waveforms,
    num_samples=400,  # 4 classes × 100
    num_iq_samples=1024,
    sample_rate=sample_rate,
    impairments=AWGN(snr=15.0),
    transform=spec_transform,
    seed=42,
)

test_ds = NarrowbandDataset(
    waveform_pool=waveforms,
    num_samples=100,  # 4 classes × 25
    num_iq_samples=1024,
    sample_rate=sample_rate,
    impairments=AWGN(snr=15.0),
    transform=spec_transform,
    seed=999,
)

# Determine spectrogram shape from a sample
sample_spec, _ = train_ds[0]
if isinstance(sample_spec, np.ndarray):
    sample_spec = torch.from_numpy(sample_spec)
print(f"Spectrogram shape: {sample_spec.shape}")
in_channels = sample_spec.shape[0] if sample_spec.ndim == 3 else 1

# ── 2. DataLoaders ──────────────────────────────────────────────────────────
train_loader = DataLoader(train_ds, batch_size=32, shuffle=True, num_workers=0)
test_loader = DataLoader(test_ds, batch_size=32, shuffle=False, num_workers=0)

# ── 3. Model, loss, optimizer ────────────────────────────────────────────────
model = ResNetAMC(num_classes=num_classes, in_channels=in_channels)
criterion = nn.CrossEntropyLoss()
optimizer = torch.optim.Adam(model.parameters(), lr=1e-3)

print(f"\nResNetAMC: {sum(p.numel() for p in model.parameters()):,} parameters")

# ── 4. Train ─────────────────────────────────────────────────────────────────
num_epochs = 10
train_losses = []

for epoch in range(num_epochs):
    model.train()
    epoch_loss = 0.0
    for batch_x, batch_y in train_loader:
        if isinstance(batch_x, np.ndarray):
            batch_x = torch.from_numpy(batch_x)
        batch_x = batch_x.float()
        if batch_x.ndim == 2:
            batch_x = batch_x.unsqueeze(1)

        optimizer.zero_grad()
        logits = model(batch_x)
        loss = criterion(logits, batch_y)
        loss.backward()
        optimizer.step()
        epoch_loss += loss.item()

    avg_loss = epoch_loss / len(train_loader)
    train_losses.append(avg_loss)
    if (epoch + 1) % 2 == 0:
        print(f"  Epoch {epoch+1}/{num_epochs}: loss={avg_loss:.4f}")

# ── 5. Evaluate ──────────────────────────────────────────────────────────────
model.eval()
all_preds, all_labels = [], []
with torch.no_grad():
    for batch_x, batch_y in test_loader:
        if isinstance(batch_x, np.ndarray):
            batch_x = torch.from_numpy(batch_x)
        batch_x = batch_x.float()
        if batch_x.ndim == 2:
            batch_x = batch_x.unsqueeze(1)
        preds = model(batch_x).argmax(dim=1)
        all_preds.extend(preds.numpy())
        all_labels.extend(batch_y.numpy())

all_preds = np.array(all_preds)
all_labels = np.array(all_labels)
acc = accuracy(all_labels, all_preds)
cm = confusion_matrix(all_labels, all_preds, num_classes)
print(f"\nTest accuracy: {acc:.1%}")

# ── 6. Plot ──────────────────────────────────────────────────────────────────
fig, axes = plt.subplots(1, 2, figsize=(12, 4))

axes[0].plot(train_losses, "o-")
axes[0].set_xlabel("Epoch")
axes[0].set_ylabel("Loss")
axes[0].set_title("Training Loss")
axes[0].grid(True, alpha=0.3)

class_names = [w.label for w in waveforms]
im = axes[1].imshow(cm, cmap="Blues")
axes[1].set_xticks(range(num_classes))
axes[1].set_yticks(range(num_classes))
axes[1].set_xticklabels(class_names, rotation=45, fontsize=8)
axes[1].set_yticklabels(class_names, fontsize=8)
axes[1].set_xlabel("Predicted")
axes[1].set_ylabel("True")
axes[1].set_title(f"Confusion Matrix (Acc={acc:.1%})")
for i in range(num_classes):
    for j in range(num_classes):
        axes[1].text(j, i, f"{cm[i, j]}", ha="center", va="center", fontsize=9)
plt.colorbar(im, ax=axes[1])

fig.suptitle("ResNetAMC on Spectrograms", fontsize=13)
fig.tight_layout()
savefig("resnet_amc.png")
plt.close()

print("Done — ResNet AMC example saved.")
