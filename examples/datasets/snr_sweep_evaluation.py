"""
SNR Sweep Evaluation Dataset
=============================
Level: Advanced

Demonstrate the SNRSweepDataset for structured AMC evaluation:
  - Build a (SNR x class x sample) grid
  - Iterate and visualize samples at each SNR
  - Use evaluate_snr_sweep for per-SNR accuracy

Run:
    python examples/datasets/snr_sweep_evaluation.py
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
from spectra.datasets import SNRSweepDataset
from spectra.transforms import Spectrogram
from plot_helpers import savefig

sample_rate = 1e6

# ── 1. Build the SNR sweep dataset ──────────────────────────────────────────
waveform_pool = [BPSK(), QPSK(), QAM16(), FSK()]
snr_levels = [-5.0, 0.0, 5.0, 10.0, 15.0, 20.0]

dataset = SNRSweepDataset(
    waveform_pool=waveform_pool,
    snr_levels=snr_levels,
    samples_per_cell=5,
    num_iq_samples=1024,
    sample_rate=sample_rate,
    impairments_fn=lambda snr: AWGN(snr=snr),
    seed=42,
)

print(f"SNRSweepDataset: {len(dataset)} total samples")
print(f"  SNR levels: {snr_levels}")
print(f"  Classes: {len(waveform_pool)}")
print(f"  Samples per cell: 5")
print(f"  Grid: {len(snr_levels)} x {len(waveform_pool)} x 5 = {len(dataset)}")

# ── 2. Visualize samples across SNR levels ───────────────────────────────────
class_names = [w.label for w in waveform_pool]
fig, axes = plt.subplots(len(waveform_pool), len(snr_levels),
                         figsize=(3 * len(snr_levels), 3 * len(waveform_pool)))

spec_xform = Spectrogram(nfft=64, hop_length=16)

for cls_idx in range(len(waveform_pool)):
    for snr_idx, snr in enumerate(snr_levels):
        idx = (snr_idx * len(waveform_pool) + cls_idx) * 5
        iq, label, snr_val = dataset[idx]

        iq_np = iq.numpy() if hasattr(iq, "numpy") else iq
        # SNRSweepDataset returns [2, N] (I, Q channels); convert to complex 1D
        if iq_np.ndim == 2 and iq_np.shape[0] == 2:
            iq_complex = iq_np[0] + 1j * iq_np[1]
        else:
            iq_complex = iq_np
        spec = spec_xform(iq_complex.astype(np.complex64))
        spec = spec.numpy() if hasattr(spec, "numpy") else spec
        if spec.ndim == 3:
            spec = spec[0]  # take the single channel [freq, time]

        axes[cls_idx, snr_idx].imshow(
            10 * np.log10(np.abs(spec) + 1e-12),
            aspect="auto", origin="lower", cmap="viridis",
        )
        if cls_idx == 0:
            axes[cls_idx, snr_idx].set_title(f"SNR={snr} dB", fontsize=9)
        if snr_idx == 0:
            axes[cls_idx, snr_idx].set_ylabel(class_names[cls_idx], fontsize=9)
        axes[cls_idx, snr_idx].set_xticks([])
        axes[cls_idx, snr_idx].set_yticks([])

fig.suptitle("SNR Sweep: Spectrograms by Class and SNR", fontsize=13)
fig.tight_layout()
savefig("snr_sweep_grid.png")
plt.close()

print("Done — SNR sweep evaluation example saved.")
