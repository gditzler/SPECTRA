# examples/15_radar_processing.py
"""Example 15 — Radar Range Profile Processing with Matched Filter and CFAR
============================================================================
Level: Intermediate

This example shows how to:
  1. Build a RadarDataset with LFM and coded-pulse waveforms
  2. Visualise a matched-filter range profile
  3. Apply CA-CFAR and OS-CFAR detectors
  4. Compute detection probability vs SNR over 200 samples

Run:
    python examples/15_radar_processing.py
"""

import sys
from collections import Counter
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

import matplotlib.pyplot as plt
import numpy as np
from plot_helpers import OUTPUT_DIR, savefig
from spectra.algorithms import ca_cfar, matched_filter, os_cfar
from spectra.datasets.radar import RadarDataset
from spectra.waveforms import LFM, BarkerCodedPulse, PolyphaseCodedPulse

# ── Configuration ──────────────────────────────────────────────────────────────

SAMPLE_RATE    = 1e6
NUM_RANGE_BINS = 512
SNR_RANGE      = (5.0, 25.0)
N_SAMPLES      = 200
SEED           = 42

GUARD    = 4
TRAINING = 16
PFA      = 1e-4

# ── 1. Build Dataset ───────────────────────────────────────────────────────────

waveform_pool = [LFM(), BarkerCodedPulse(), PolyphaseCodedPulse(code_type="p4")]

ds = RadarDataset(
    waveform_pool=waveform_pool,
    num_range_bins=NUM_RANGE_BINS,
    sample_rate=SAMPLE_RATE,
    snr_range=SNR_RANGE,
    num_targets_range=(1, 3),
    num_samples=N_SAMPLES,
    seed=SEED,
)
print(f"Dataset: {len(ds)} samples, waveforms: {[w.label for w in waveform_pool]}\n")

# ── 2. Inspect One Range Profile ──────────────────────────────────────────────

data, target = ds[0]
print(f"Sample 0: {target.num_targets} target(s), waveform={target.waveform_label}")
print(f"  Target range bins: {target.range_bins}")
print(f"  Target SNRs (dB):  {target.snrs}\n")

fig, ax = plt.subplots(figsize=(10, 3))
ax.plot(data.numpy(), linewidth=0.8, color="steelblue")
for rb in target.range_bins:
    ax.axvline(rb, color="crimson", linestyle="--", linewidth=1.2, label=f"Target @ bin {rb}")
ax.set_xlabel("Range bin")
ax.set_ylabel("Normalised MF power")
ax.set_title(f"Matched-Filter Range Profile — {target.waveform_label}")
ax.legend(fontsize=8)
ax.grid(True, alpha=0.3)
plt.tight_layout()
savefig("15_range_profile.png")

# ── 3. CFAR Detection on One Sample ───────────────────────────────────────────

# Re-generate raw IQ so we have un-normalised power for CFAR
rng = np.random.default_rng((SEED, 0))
wf = LFM()
pulse = wf.generate(num_symbols=4, sample_rate=SAMPLE_RATE, seed=0)[:NUM_RANGE_BINS // 4]
_noise_re = rng.standard_normal(NUM_RANGE_BINS)
_noise_im = rng.standard_normal(NUM_RANGE_BINS)
noise = np.sqrt(0.5) * (_noise_re + 1j * _noise_im)
for rb in target.range_bins:
    snr_lin = 10 ** (target.snrs[0] / 10)
    amp = np.sqrt(snr_lin / (np.mean(np.abs(pulse)**2) + 1e-30))
    if rb + len(pulse) <= NUM_RANGE_BINS:
        noise[rb:rb+len(pulse)] += amp * pulse
mf_raw = np.abs(matched_filter(noise, pulse)[:NUM_RANGE_BINS]) ** 2

det_ca = ca_cfar(mf_raw, guard_cells=GUARD, training_cells=TRAINING, pfa=PFA)
det_os = os_cfar(mf_raw, guard_cells=GUARD, training_cells=TRAINING, pfa=PFA)

fig, axes = plt.subplots(3, 1, figsize=(10, 8), sharex=True)
axes[0].plot(10 * np.log10(mf_raw + 1e-30), linewidth=0.8)
for rb in target.range_bins:
    axes[0].axvline(rb, color="crimson", linestyle="--", linewidth=1.2)
axes[0].set_ylabel("MF Power (dB)")
axes[0].set_title("Matched Filter Output")

det_ca_bins = np.where(det_ca)[0]
if len(det_ca_bins) > 0:
    axes[1].stem(
        det_ca_bins, np.ones(len(det_ca_bins)),
        linefmt="C1-", markerfmt="C1o", basefmt="k",
    )
axes[1].set_ylabel("CA-CFAR")
axes[1].set_ylim(-0.1, 1.5)

det_os_bins = np.where(det_os)[0]
if len(det_os_bins) > 0:
    axes[2].stem(
        det_os_bins, np.ones(len(det_os_bins)),
        linefmt="C2-", markerfmt="C2o", basefmt="k",
    )
axes[2].set_ylabel("OS-CFAR")
axes[2].set_ylim(-0.1, 1.5)
axes[2].set_xlabel("Range bin")
plt.tight_layout()
savefig("15_cfar_detections.png")
print(f"CA-CFAR detections: {det_ca_bins}")
print(f"OS-CFAR detections: {det_os_bins}")
print(f"True target bins:   {target.range_bins}\n")

# ── 4. Dataset Overview: Waveform Mix ─────────────────────────────────────────

labels_seen = [ds[i][1].waveform_label for i in range(min(100, len(ds)))]
counts = Counter(labels_seen)
fig, ax = plt.subplots(figsize=(6, 3))
ax.bar(counts.keys(), counts.values(), color="steelblue")
ax.set_ylabel("Count")
ax.set_title("Waveform distribution in first 100 samples")
ax.grid(True, axis="y", alpha=0.3)
plt.tight_layout()
savefig("15_waveform_distribution.png")

print(f"\nAll figures saved to {OUTPUT_DIR}")
