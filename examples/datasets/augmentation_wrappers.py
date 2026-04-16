"""
MixUp and CutMix Dataset Wrappers
==================================
Level: Intermediate

Demonstrate cross-sample augmentation wrappers:
  - MixUpDataset — blend two random samples with Beta-distributed weight
  - CutMixDataset — replace a random segment of one sample with another

Run:
    python examples/datasets/augmentation_wrappers.py
"""

import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from spectra.waveforms import BPSK, QPSK, QAM16, FSK
from spectra.impairments import AWGN
from spectra.datasets import NarrowbandDataset, MixUpDataset, CutMixDataset
from plot_helpers import savefig

sample_rate = 1e6

# ── 1. Build base dataset ───────────────────────────────────────────────────
base_ds = NarrowbandDataset(
    waveform_pool=[BPSK(), QPSK(), QAM16(), FSK()],
    num_samples=200,  # 4 classes × 50
    num_iq_samples=1024,
    sample_rate=sample_rate,
    impairments=AWGN(snr=15.0),
    seed=42,
)

# ── 2. Wrap with MixUp and CutMix ───────────────────────────────────────────
mixup_ds = MixUpDataset(base_ds, alpha=0.2)
cutmix_ds = CutMixDataset(base_ds, alpha=1.0)

print(f"Base dataset: {len(base_ds)} samples")
print(f"MixUp dataset: {len(mixup_ds)} samples")
print(f"CutMix dataset: {len(cutmix_ds)} samples")

# ── 3. Compare original vs augmented ────────────────────────────────────────
fig, axes = plt.subplots(3, 4, figsize=(14, 8))
for col in range(4):
    idx = col * 50  # first sample of each class
    iq_base, lbl_base = base_ds[idx]
    iq_mixup, lbl_mixup = mixup_ds[idx]
    iq_cutmix, lbl_cutmix = cutmix_ds[idx]

    n = 200
    axes[0, col].plot(iq_base[:n].real, linewidth=0.5)
    axes[0, col].set_title(f"Original (cls={lbl_base})", fontsize=9)
    axes[0, col].grid(True, alpha=0.3)

    axes[1, col].plot(iq_mixup[:n].real, linewidth=0.5, color="tab:green")
    axes[1, col].set_title("MixUp", fontsize=9)
    axes[1, col].grid(True, alpha=0.3)

    axes[2, col].plot(iq_cutmix[:n].real, linewidth=0.5, color="tab:red")
    axes[2, col].set_title("CutMix", fontsize=9)
    axes[2, col].grid(True, alpha=0.3)

fig.suptitle("MixUp vs CutMix Augmentation", fontsize=13)
fig.tight_layout()
savefig("augmentation_wrappers.png")
plt.close()

print("Done — augmentation wrappers example saved.")
