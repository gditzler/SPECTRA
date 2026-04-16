"""
SPECTRA Example 05: Wideband Scene Composition
===============================================
Level: Pro

Learn how to:
- Configure wideband RF scenes with SceneConfig
- Generate composite captures with multiple signals
- Visualize spectrograms with overlaid bounding boxes
- Convert physical units to pixel-space COCO labels
- Use WidebandDataset with DataLoader
"""

import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import matplotlib.patches as patches
import torch
import spectra as sp
from spectra.scene import STFTParams
from spectra.utils import dsp
from plot_helpers import savefig

# -- 1. Configure wideband scene --------------------------------------------------

signal_pool = [
    sp.QPSK(),
    sp.QAM16(),
    sp.FSK(),
    sp.OFDM(),
    sp.GMSK(),
    sp.BPSK(),
]

config = sp.SceneConfig(
    capture_duration=1e-3,       # 1 ms capture
    capture_bandwidth=10e6,      # 10 MHz capture BW
    sample_rate=10e6,            # 10 MHz sample rate
    num_signals=(2, 5),          # 2 to 5 signals per scene
    signal_pool=signal_pool,
    snr_range=(5, 25),
    allow_overlap=True,
)

# -- 2. Generate a scene -----------------------------------------------------------

composer = sp.Composer(config)
iq, signal_descs = composer.generate(seed=42)

print(f"IQ shape: {iq.shape} ({len(iq)} samples)")
print(f"Number of signals: {len(signal_descs)}")
for i, desc in enumerate(signal_descs):
    print(f"  Signal {i}: {desc.label}, "
          f"f=[{desc.f_low/1e6:.2f}, {desc.f_high/1e6:.2f}] MHz, "
          f"t=[{desc.t_start*1e3:.3f}, {desc.t_stop*1e3:.3f}] ms, "
          f"SNR={desc.snr:.1f} dB")

# -- 3. Compute spectrogram and overlay bounding boxes -----------------------------

nfft = 512
hop = 128

spec = dsp.compute_spectrogram(iq, nfft=nfft, hop=hop)
spec_db = 10 * np.log10(spec + 1e-12)

stft_params = STFTParams(
    nfft=nfft,
    hop_length=hop,
    sample_rate=config.sample_rate,
    num_samples=len(iq),
)

class_list = sorted(set(d.label for d in signal_descs))
targets = sp.to_coco(signal_descs, stft_params, class_list)

fig, ax = plt.subplots(figsize=(14, 6))
ax.imshow(spec_db, aspect="auto", origin="lower", cmap="viridis")

colors = plt.cm.Set1(np.linspace(0, 1, len(class_list)))
boxes = targets["boxes"].numpy()
labels = targets["labels"].numpy()

for box, label_idx in zip(boxes, labels):
    x1, y1, x2, y2 = box
    w, h = x2 - x1, y2 - y1
    color = colors[label_idx % len(colors)]
    rect = patches.Rectangle((x1, y1), w, h, linewidth=2,
                              edgecolor=color, facecolor="none")
    ax.add_patch(rect)
    ax.text(x1, y2 + 2, class_list[label_idx], color=color,
            fontsize=9, fontweight="bold",
            bbox=dict(boxstyle="round,pad=0.2", facecolor="black", alpha=0.5))

ax.set_xlabel("Time Bin")
ax.set_ylabel("Frequency Bin")
ax.set_title("Wideband Scene with COCO Bounding Boxes")
fig.tight_layout()
savefig("05_wideband_scene.png")

# -- 4. Generate multiple scenes ---------------------------------------------------

fig, axes = plt.subplots(2, 3, figsize=(18, 10))
for idx, ax in enumerate(axes.flat):
    iq_i, descs_i = composer.generate(seed=idx * 7)
    spec_i = dsp.compute_spectrogram(iq_i, nfft=nfft, hop=hop)
    spec_db_i = 10 * np.log10(spec_i + 1e-12)
    ax.imshow(spec_db_i, aspect="auto", origin="lower", cmap="viridis")

    stft_p = STFTParams(nfft=nfft, hop_length=hop,
                        sample_rate=config.sample_rate, num_samples=len(iq_i))
    cls = sorted(set(d.label for d in descs_i))
    tgt = sp.to_coco(descs_i, stft_p, cls)
    for box, li in zip(tgt["boxes"].numpy(), tgt["labels"].numpy()):
        x1, y1, x2, y2 = box
        rect = patches.Rectangle((x1, y1), x2 - x1, y2 - y1,
                                  linewidth=1.5, edgecolor="red", facecolor="none")
        ax.add_patch(rect)
    signal_labels = [d.label for d in descs_i]
    ax.set_title(f"Scene {idx}: {', '.join(signal_labels)}", fontsize=9)

fig.suptitle("Multiple Wideband Scenes", fontsize=14)
fig.tight_layout()
savefig("05_multiple_scenes.png")

# -- 5. WidebandDataset with DataLoader --------------------------------------------

wideband_ds = sp.WidebandDataset(
    scene_config=config,
    num_samples=16,
    transform=sp.STFT(nfft=512, hop_length=128),
    seed=42,
)

loader = torch.utils.data.DataLoader(
    wideband_ds,
    batch_size=4,
    collate_fn=sp.collate_fn,
)

for batch_data, batch_targets in loader:
    print(f"Batch shape: {batch_data.shape}")
    print(f"Num targets in first sample: {len(batch_targets[0]['boxes'])}")
    break

# Visualize batch
fig, axes = plt.subplots(1, 4, figsize=(18, 4))
batch_data_np = batch_data.numpy()
for i, ax in enumerate(axes):
    ax.imshow(batch_data_np[i, 0], aspect="auto", origin="lower", cmap="viridis")
    n_sigs = len(batch_targets[i]["boxes"])
    ax.set_title(f"Sample {i} ({n_sigs} signals)")
fig.suptitle("WidebandDataset -- DataLoader Batch", fontsize=14)
fig.tight_layout()
savefig("05_wideband_batch.png")

plt.close("all")
print("\nDone! Check examples/outputs/ for saved figures.")
