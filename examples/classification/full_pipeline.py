"""
SPECTRA Example 06: Full Pipeline — Dataset Generation to Classification
=========================================================================
Level: Pro

Learn how to:
- Generate a reproducible AMC dataset with diverse waveforms
- Chain transforms and augmentations
- Train a simple CNN classifier on spectrograms
- Evaluate accuracy per class and per SNR
- Save dataset metadata for reproducibility
"""

import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import torch
import torch.nn as nn
from torch.utils.data import DataLoader, random_split
import spectra as sp
from plot_helpers import savefig

# ── 1. Dataset configuration ────────────────────────────────────────────────

sample_rate = 1e6
num_iq_samples = 1024
nfft = 64

waveform_pool = [
    sp.BPSK(),
    sp.QPSK(),
    sp.PSK8(),
    sp.QAM16(),
    sp.QAM64(),
    sp.FSK(),
    sp.GMSK(),
    sp.OFDM(),
]

class_names = [w.label for w in waveform_pool]
num_classes = len(class_names)

impairments = sp.Compose([
    sp.AWGN(snr_range=(0, 20)),
    sp.FrequencyOffset(max_offset=1000),
    sp.PhaseNoise(noise_power_db=-35),
])

# ── 2. Build train/val datasets ─────────────────────────────────────────────

dataset = sp.NarrowbandDataset(
    waveform_pool=waveform_pool,
    num_samples=1600,  # 200 per class
    num_iq_samples=num_iq_samples,
    sample_rate=sample_rate,
    impairments=impairments,
    transform=sp.STFT(nfft=nfft, hop_length=nfft // 4),
    seed=42,
)

train_size = int(0.8 * len(dataset))
val_size = len(dataset) - train_size
train_ds, val_ds = random_split(dataset, [train_size, val_size],
                                generator=torch.Generator().manual_seed(42))

print(f"Train: {len(train_ds)}, Val: {len(val_ds)}")
sample, label = train_ds[0]
print(f"Sample shape: {sample.shape}, label: {label} ({class_names[label]})")

train_loader = DataLoader(train_ds, batch_size=32, shuffle=True)
val_loader = DataLoader(val_ds, batch_size=32)

# ── 3. Simple CNN Classifier ────────────────────────────────────────────────

class SimpleCNN(nn.Module):
    def __init__(self, num_classes, input_channels=1):
        super().__init__()
        self.features = nn.Sequential(
            nn.Conv2d(input_channels, 32, 3, padding=1),
            nn.ReLU(),
            nn.AdaptiveAvgPool2d((8, 8)),
            nn.Conv2d(32, 64, 3, padding=1),
            nn.ReLU(),
            nn.AdaptiveAvgPool2d((4, 4)),
        )
        self.classifier = nn.Sequential(
            nn.Flatten(),
            nn.Linear(64 * 4 * 4, 128),
            nn.ReLU(),
            nn.Dropout(0.3),
            nn.Linear(128, num_classes),
        )

    def forward(self, x):
        return self.classifier(self.features(x))

model = SimpleCNN(num_classes)
optimizer = torch.optim.Adam(model.parameters(), lr=1e-3)
criterion = nn.CrossEntropyLoss()

# ── 4. Training loop ────────────────────────────────────────────────────────

num_epochs = 15
train_losses = []
val_accuracies = []

for epoch in range(num_epochs):
    model.train()
    epoch_loss = 0.0
    for batch_x, batch_y in train_loader:
        optimizer.zero_grad()
        out = model(batch_x)
        loss = criterion(out, batch_y)
        loss.backward()
        optimizer.step()
        epoch_loss += loss.item()
    train_losses.append(epoch_loss / len(train_loader))

    model.eval()
    correct = 0
    total = 0
    with torch.no_grad():
        for batch_x, batch_y in val_loader:
            preds = model(batch_x).argmax(dim=1)
            correct += (preds == batch_y).sum().item()
            total += len(batch_y)
    val_acc = correct / total
    val_accuracies.append(val_acc)

    if (epoch + 1) % 5 == 0:
        print(f"Epoch {epoch+1}/{num_epochs} — "
              f"Loss: {train_losses[-1]:.4f}, Val Acc: {val_acc:.2%}")

# ── 5. Plot training curves ─────────────────────────────────────────────────

fig, axes = plt.subplots(1, 2, figsize=(12, 4))
axes[0].plot(train_losses, linewidth=1.5)
axes[0].set_xlabel("Epoch")
axes[0].set_ylabel("Loss")
axes[0].set_title("Training Loss")
axes[0].grid(True, alpha=0.3)

axes[1].plot(val_accuracies, linewidth=1.5, color="tab:green")
axes[1].set_xlabel("Epoch")
axes[1].set_ylabel("Accuracy")
axes[1].set_title("Validation Accuracy")
axes[1].grid(True, alpha=0.3)
fig.tight_layout()
savefig("06_training_curves.png")

# ── 6. Per-class accuracy ───────────────────────────────────────────────────

model.eval()
class_correct = np.zeros(num_classes)
class_total = np.zeros(num_classes)

with torch.no_grad():
    for batch_x, batch_y in val_loader:
        preds = model(batch_x).argmax(dim=1)
        for pred, true in zip(preds, batch_y):
            class_total[true] += 1
            if pred == true:
                class_correct[true] += 1

class_acc = class_correct / np.maximum(class_total, 1)

fig, ax = plt.subplots(figsize=(10, 5))
bars = ax.bar(class_names, class_acc, color="steelblue")
ax.set_ylim(0, 1.05)
ax.set_xlabel("Modulation Class")
ax.set_ylabel("Accuracy")
ax.set_title("Per-Class Validation Accuracy")
ax.grid(True, alpha=0.3, axis="y")
for bar, acc in zip(bars, class_acc):
    ax.text(bar.get_x() + bar.get_width() / 2, bar.get_height() + 0.02,
            f"{acc:.0%}", ha="center", fontsize=9)
fig.tight_layout()
savefig("06_per_class_accuracy.png")

# ── 7. Confusion matrix ─────────────────────────────────────────────────────

confusion = np.zeros((num_classes, num_classes), dtype=int)
with torch.no_grad():
    for batch_x, batch_y in val_loader:
        preds = model(batch_x).argmax(dim=1)
        for pred, true in zip(preds, batch_y):
            confusion[true, pred] += 1

fig, ax = plt.subplots(figsize=(8, 7))
im = ax.imshow(confusion, cmap="Blues")
ax.set_xticks(range(num_classes))
ax.set_yticks(range(num_classes))
ax.set_xticklabels(class_names, rotation=45, ha="right")
ax.set_yticklabels(class_names)
ax.set_xlabel("Predicted")
ax.set_ylabel("True")
ax.set_title("Confusion Matrix")
for i in range(num_classes):
    for j in range(num_classes):
        ax.text(j, i, str(confusion[i, j]), ha="center", va="center",
                color="white" if confusion[i, j] > confusion.max() / 2 else "black")
fig.colorbar(im)
fig.tight_layout()
savefig("06_confusion_matrix.png")

# ── 8. Save dataset metadata ────────────────────────────────────────────────

metadata = sp.NarrowbandMetadata(
    name="amc_8class",
    num_samples=1600,
    sample_rate=sample_rate,
    seed=42,
    waveform_labels=class_names,
    num_iq_samples=num_iq_samples,
    snr_range=(0, 20),
)
metadata.to_yaml("outputs/06_dataset_metadata.yaml")
print(f"\nMetadata saved to outputs/06_dataset_metadata.yaml")

# Demonstrate reload
loaded = sp.NarrowbandMetadata.from_yaml("outputs/06_dataset_metadata.yaml")
print(f"Reloaded: {loaded.name}, {loaded.num_samples} samples, "
      f"{len(loaded.waveform_labels)} classes")

plt.close("all")
print("\nDone! Check examples/outputs/ for saved figures.")
