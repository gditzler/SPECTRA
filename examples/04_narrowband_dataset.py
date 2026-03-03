"""
SPECTRA Example 04: Narrowband Classification Dataset
=====================================================
Level: Advanced

Learn how to:
- Build a NarrowbandDataset for automatic modulation classification (AMC)
- Use PyTorch DataLoader for batched iteration
- Apply transforms and target transforms
- Visualize per-class spectrograms and IQ distributions
- Use FamilyName for family-level grouping
"""

import sys
sys.path.insert(0, ".")

import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import torch
from torch.utils.data import DataLoader
import spectra as sp
from plot_helpers import savefig

sample_rate = 1e6

# ── 1. Define waveform pool ─────────────────────────────────────────────────

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
print(f"Classes ({len(class_names)}): {class_names}")

# ── 2. Create dataset with impairments and transforms ────────────────────────
# NarrowbandDataset.__getitem__ returns (data, waveform_idx) where waveform_idx
# is an integer index into the waveform_pool. The optional target_transform
# receives this integer, so we leave it as-is for classification.

impairments = sp.Compose([
    sp.AWGN(snr_range=(5, 25)),
    sp.FrequencyOffset(max_offset=1000),
])

dataset = sp.NarrowbandDataset(
    waveform_pool=waveform_pool,
    num_samples=800,             # 100 per class
    num_iq_samples=1024,
    sample_rate=sample_rate,
    impairments=impairments,
    transform=sp.ComplexTo2D(),
    seed=42,
)

print(f"Dataset size: {len(dataset)}")
data, label = dataset[0]
print(f"Sample shape: {data.shape}, label: {label} ({class_names[label]})")

# ── 3. Iterate with DataLoader ───────────────────────────────────────────────

loader = DataLoader(dataset, batch_size=32, shuffle=True)
all_labels = []

for batch_data, batch_labels in loader:
    all_labels.extend(batch_labels.numpy())

all_labels = np.array(all_labels)

fig, ax = plt.subplots(figsize=(10, 5))
counts = [np.sum(all_labels == i) for i in range(len(class_names))]
ax.bar(class_names, counts, color="steelblue")
ax.set_xlabel("Modulation Class")
ax.set_ylabel("Count")
ax.set_title("Class Distribution in Dataset")
ax.grid(True, alpha=0.3, axis="y")
fig.tight_layout()
savefig("04_class_distribution.png")

# ── 4. Visualize one sample per class ────────────────────────────────────────

# Create a dataset without ComplexTo2D for raw IQ visualization
raw_dataset = sp.NarrowbandDataset(
    waveform_pool=waveform_pool,
    num_samples=800,
    num_iq_samples=1024,
    sample_rate=sample_rate,
    impairments=impairments,
    seed=42,
)

fig, axes = plt.subplots(2, 4, figsize=(16, 8))
seen = set()
idx = 0
while len(seen) < len(class_names) and idx < len(raw_dataset):
    data, label = raw_dataset[idx]
    if label not in seen:
        seen.add(label)
        ax = axes.flat[label]
        iq = data.numpy() if isinstance(data, torch.Tensor) else data
        if np.iscomplexobj(iq):
            spec = np.abs(np.fft.fftshift(
                np.array([np.fft.fft(iq[i:i+256], n=256) for i in range(0, len(iq)-256, 64)]).T
            ))
        else:
            spec = np.abs(np.fft.fftshift(
                np.array([np.fft.fft((iq[0, i:i+256] + 1j * iq[1, i:i+256]), n=256) for i in range(0, iq.shape[1]-256, 64)]).T
            ))
        spec_db = 10 * np.log10(spec + 1e-12)
        ax.imshow(spec_db, aspect="auto", origin="lower", cmap="viridis")
        ax.set_title(class_names[label])
    idx += 1
fig.suptitle("Spectrogram per Class (with impairments)", fontsize=14)
fig.tight_layout()
savefig("04_per_class_spectrograms.png")

# ── 5. Family-level grouping ────────────────────────────────────────────────
# FamilyName maps a waveform label string to its modulation family.

family_transform = sp.FamilyName()
families = set()
for wf in waveform_pool:
    families.add(family_transform(wf.label))
print(f"Modulation families: {sorted(families)}")

# ── 6. Multiple SNR visualization ───────────────────────────────────────────

snr_levels = [0, 5, 10, 20]
wf = sp.QAM16()
fig, axes = plt.subplots(1, 4, figsize=(16, 4))
for ax, snr in zip(axes, snr_levels):
    ds = sp.NarrowbandDataset(
        waveform_pool=[wf],
        num_samples=1,
        num_iq_samples=1024,
        sample_rate=sample_rate,
        impairments=sp.Compose([sp.AWGN(snr=snr)]),
        seed=42,
    )
    data, _ = ds[0]
    iq = data.numpy() if isinstance(data, torch.Tensor) else data
    if np.iscomplexobj(iq):
        pts = iq[:500]
    else:
        pts = (iq[0, :500] + 1j * iq[1, :500])
    ax.scatter(pts.real, pts.imag, s=3, alpha=0.5)
    ax.set_title(f"SNR = {snr} dB")
    ax.set_aspect("equal")
    ax.grid(True, alpha=0.3)
    ax.set_xlim(-3, 3)
    ax.set_ylim(-3, 3)
fig.suptitle("16QAM at Different SNR Levels", fontsize=14)
fig.tight_layout()
savefig("04_snr_comparison.png")

plt.close("all")
print("\nDone! Check examples/outputs/ for saved figures.")
