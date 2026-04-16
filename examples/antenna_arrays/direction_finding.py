"""Example 13 — Direction-Finding Dataset with MUSIC and ESPRIT
================================================================
Level: Intermediate

This example shows how to:
  1. Configure a Uniform Linear Array (ULA) with isotropic elements
  2. Build a DirectionFindingDataset with multiple sources
  3. Convert raw snapshots to a complex snapshot matrix
  4. Apply MUSIC to estimate the azimuth spectrum
  5. Apply ESPRIT to directly estimate source azimuths
  6. Compare estimated angles to ground truth

Run:
    python examples/13_direction_finding.py
"""

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

import matplotlib.pyplot as plt
import numpy as np
import torch
from torch.utils.data import DataLoader

from spectra.algorithms import esprit, find_peaks_doa, music
from spectra.arrays import ula
from spectra.datasets import DirectionFindingDataset
from spectra.waveforms import BPSK, QAM16, QPSK

from plot_helpers import OUTPUT_DIR, savefig

# ── Configuration ─────────────────────────────────────────────────────────────

FREQ_HZ = 2.4e9        # carrier frequency
N_ELEMENTS = 8         # ULA elements
SPACING = 0.5          # half-wavelength spacing
N_SNAPSHOTS = 256      # IQ samples per antenna per item
SAMPLE_RATE = 1e6      # Hz
N_SOURCES = 2          # fixed number of concurrent sources
SNR_RANGE = (15.0, 25.0)  # dB
N_SAMPLES = 200        # dataset size
SEED = 42

# Azimuth scan for MUSIC (in radians, 0° = along array axis, 90° = broadside)
SCAN_DEG = np.linspace(5, 175, 512)
SCAN_RAD = np.deg2rad(SCAN_DEG)

# ── 1. Array and Dataset Setup ─────────────────────────────────────────────────

arr = ula(num_elements=N_ELEMENTS, spacing=SPACING, frequency=FREQ_HZ)
print(f"ULA: {arr.num_elements} elements, {SPACING}λ spacing")
print(f"Element positions (wavelengths):\n  x = {arr.positions[:, 0]}\n")

signal_pool = [
    BPSK(samples_per_symbol=4),
    QPSK(samples_per_symbol=4),
    QAM16(samples_per_symbol=4),
]

ds = DirectionFindingDataset(
    array=arr,
    signal_pool=signal_pool,
    num_signals=N_SOURCES,
    num_snapshots=N_SNAPSHOTS,
    sample_rate=SAMPLE_RATE,
    snr_range=SNR_RANGE,
    azimuth_range=(np.deg2rad(10), np.deg2rad(170)),  # avoid near-endfire
    elevation_range=(0.0, 0.0),                        # 2-D scenario: el=0
    min_angular_separation=np.deg2rad(15),
    num_samples=N_SAMPLES,
    seed=SEED,
)
print(f"Dataset: {len(ds)} samples, {N_SOURCES} sources each\n")

# ── 2. Visualise Array Geometry ───────────────────────────────────────────────

fig, ax = plt.subplots(figsize=(8, 2))
ax.scatter(arr.positions[:, 0], arr.positions[:, 1], s=120, zorder=3)
for i, (x, y) in enumerate(arr.positions):
    ax.annotate(
        f"{i}", (x, y), textcoords="offset points", xytext=(0, 8), ha="center", fontsize=8
    )
ax.set_xlabel("x (wavelengths)")
ax.set_title(f"ULA Geometry — {N_ELEMENTS} elements, d = {SPACING}λ")
ax.set_yticks([])
ax.grid(True, alpha=0.3)
plt.tight_layout()
savefig("13_array_geometry.png")

# ── 3. Inspect One Sample ──────────────────────────────────────────────────────

data, target = ds[0]  # data: Tensor [N_elem, 2, N_snap]
print("Sample 0 ground truth:")
for k in range(target.num_sources):
    print(
        f"  Source {k}: az = {np.rad2deg(target.azimuths[k]):.1f}°, "
        f"SNR = {target.snrs[k]:.1f} dB, label = {target.labels[k]}"
    )
print()

fig, axes = plt.subplots(N_ELEMENTS, 1, figsize=(10, 12), sharex=True)
t = np.arange(N_SNAPSHOTS) / SAMPLE_RATE * 1e6  # µs
for i, ax in enumerate(axes):
    iq = data[i, 0, :].numpy()  # I channel
    ax.plot(t, iq, linewidth=0.6)
    ax.set_ylabel(f"Ant {i}", fontsize=7)
    ax.set_yticks([])
axes[-1].set_xlabel("Time (µs)")
fig.suptitle("Received I-channel per antenna element (sample 0)")
plt.tight_layout()
savefig("13_snapshot_iq.png")

# ── 4. MUSIC Pseudospectrum ────────────────────────────────────────────────────

# Convert [N_elem, 2, T] → complex [N_elem, T]
X = data[:, 0, :].numpy() + 1j * data[:, 1, :].numpy()  # (N, T)

spectrum = music(X, num_sources=N_SOURCES, array=arr, scan_angles=SCAN_RAD, elevation=0.0)
peaks_music = find_peaks_doa(spectrum, SCAN_RAD, num_peaks=N_SOURCES)

fig, ax = plt.subplots(figsize=(8, 4))
ax.plot(SCAN_DEG, 10 * np.log10(spectrum / spectrum.max()), color="steelblue", linewidth=1.2)
for k, az_true in enumerate(target.azimuths):
    ax.axvline(
        np.rad2deg(az_true),
        color="crimson",
        linestyle="--",
        linewidth=1.5,
        label=f"True az {np.rad2deg(az_true):.1f}°",
    )
for az_est in peaks_music:
    ax.axvline(
        np.rad2deg(az_est),
        color="orange",
        linestyle=":",
        linewidth=1.5,
    )
ax.set_xlabel("Azimuth (degrees from x-axis)")
ax.set_ylabel("Pseudospectrum (dB, normalised)")
ax.set_title(
    f"MUSIC — {N_ELEMENTS}-element ULA, {N_SOURCES} sources, SNR ∈ {SNR_RANGE} dB"
)
ax.legend(fontsize=8)
ax.grid(True, alpha=0.3)
plt.tight_layout()
savefig("13_music_spectrum.png")
print(f"MUSIC peaks:   {np.rad2deg(peaks_music)}")
print(f"True azimuths: {np.rad2deg(np.sort(target.azimuths))}")

# ── 5. ESPRIT Estimates ────────────────────────────────────────────────────────

az_esprit = esprit(X, num_sources=N_SOURCES, spacing=SPACING)
print(f"\nESPRIT estimates: {np.rad2deg(az_esprit)}")
print(f"True azimuths:    {np.rad2deg(np.sort(target.azimuths))}\n")

# ── 6. Batch RMSE Comparison ───────────────────────────────────────────────────


def collate_fn(batch):
    return torch.stack([x for x, _ in batch]), [t for _, t in batch]


loader = DataLoader(ds, batch_size=1, collate_fn=collate_fn)

music_errors, esprit_errors = [], []

for i, (batch_data, batch_targets) in enumerate(loader):
    if i >= 100:
        break
    tgt = batch_targets[0]
    Xb = batch_data[0, :, 0, :].numpy() + 1j * batch_data[0, :, 1, :].numpy()
    true_az = np.sort(tgt.azimuths)

    # MUSIC
    sp_val = music(Xb, num_sources=N_SOURCES, array=arr, scan_angles=SCAN_RAD)
    est_music = find_peaks_doa(sp_val, SCAN_RAD, num_peaks=N_SOURCES)
    for j, az_t in enumerate(true_az):
        music_errors.append(abs(est_music[j % len(est_music)] - az_t))

    # ESPRIT
    est_esprit = esprit(Xb, num_sources=N_SOURCES, spacing=SPACING)
    for j, az_t in enumerate(true_az):
        esprit_errors.append(abs(est_esprit[j % len(est_esprit)] - az_t))

rmse_music = np.rad2deg(np.sqrt(np.mean(np.array(music_errors) ** 2)))
rmse_esprit = np.rad2deg(np.sqrt(np.mean(np.array(esprit_errors) ** 2)))
print(f"RMSE over 100 samples:")
print(f"  MUSIC:  {rmse_music:.2f}°")
print(f"  ESPRIT: {rmse_esprit:.2f}°")

fig, ax = plt.subplots(figsize=(6, 4))
ax.bar(["MUSIC", "ESPRIT"], [rmse_music, rmse_esprit], color=["steelblue", "seagreen"])
ax.set_ylabel("RMSE (degrees)")
ax.set_title(f"DoA Estimation RMSE — 100 samples, SNR ∈ {SNR_RANGE} dB")
ax.grid(True, axis="y", alpha=0.3)
plt.tight_layout()
savefig("13_rmse_comparison.png")

print(f"\nAll figures saved to {OUTPUT_DIR}")
