"""
Folder & Manifest Dataset Loading
==================================
Level: Intermediate

Demonstrate how to load pre-existing IQ recordings using:
  - SignalFolderDataset — ImageFolder-style (class-per-directory)
  - ManifestDataset — CSV/JSON manifest pointing to files

This example creates a small synthetic dataset on disk, then loads it
back via both dataset types.

Run:
    python examples/datasets/folder_and_manifest.py
"""

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

import json
import tempfile

import matplotlib
import numpy as np

matplotlib.use("Agg")
import matplotlib.pyplot as plt
from plot_helpers import savefig
from spectra.datasets import ManifestDataset, SignalFolderDataset
from spectra.waveforms import BPSK, QAM16, QPSK

sample_rate = 1e6
num_iq = 1024
seed_base = 42

# ── 1. Create synthetic dataset on disk ──────────────────────────────────────
tmpdir = tempfile.mkdtemp(prefix="spectra_folder_example_")
print(f"Temporary dataset directory: {tmpdir}")

waveforms = {"BPSK": BPSK(), "QPSK": QPSK(), "QAM16": QAM16()}
manifest_entries = []

for cls_name, wf in waveforms.items():
    cls_dir = Path(tmpdir) / "folder_dataset" / cls_name
    cls_dir.mkdir(parents=True, exist_ok=True)

    for i in range(10):
        iq = wf.generate(num_symbols=256, sample_rate=sample_rate, seed=seed_base + i)
        filepath = cls_dir / f"sample_{i:03d}.npy"
        np.save(filepath, iq[:num_iq])
        manifest_entries.append({
            "file": str(filepath),
            "label": cls_name,
        })

# Write manifest JSON
manifest_path = Path(tmpdir) / "manifest.json"
with open(manifest_path, "w") as f:
    json.dump(manifest_entries, f, indent=2)

# ── 2. Load via SignalFolderDataset ──────────────────────────────────────────
folder_root = Path(tmpdir) / "folder_dataset"
folder_ds = SignalFolderDataset(
    root=str(folder_root),
    num_iq_samples=num_iq,
)
print(f"\nSignalFolderDataset: {len(folder_ds)} samples")

iq_sample, label = folder_ds[0]
print(f"  Sample shape: {iq_sample.shape}, label: {label}")

# ── 3. Load via ManifestDataset ──────────────────────────────────────────────
manifest_ds = ManifestDataset(
    manifest_path=str(manifest_path),
    num_iq_samples=num_iq,
)
print(f"\nManifestDataset: {len(manifest_ds)} samples")

iq_sample_m, label_m = manifest_ds[0]
print(f"  Sample shape: {iq_sample_m.shape}, label: {label_m}")

# ── 4. Visualize samples from each ──────────────────────────────────────────
fig, axes = plt.subplots(2, 3, figsize=(12, 6))
for col, cls_name in enumerate(waveforms.keys()):
    # Folder dataset
    for idx in range(len(folder_ds)):
        iq, lbl = folder_ds[idx]
        if lbl == col:
            axes[0, col].plot(iq[:200].real, linewidth=0.5)
            axes[0, col].set_title(f"Folder: {cls_name}")
            axes[0, col].grid(True, alpha=0.3)
            break

    # Manifest dataset
    for idx in range(len(manifest_ds)):
        iq, lbl = manifest_ds[idx]
        if lbl == col:
            axes[1, col].plot(iq[:200].real, linewidth=0.5, color="tab:orange")
            axes[1, col].set_title(f"Manifest: {cls_name}")
            axes[1, col].grid(True, alpha=0.3)
            break

fig.suptitle("Folder vs Manifest Dataset Loading", fontsize=13)
fig.tight_layout()
savefig("folder_manifest_datasets.png")
plt.close()

print(f"\nDone — folder/manifest dataset examples saved. Temp dir: {tmpdir}")
