# examples/14_beamforming.py
"""Example 14 — Beamforming with a Uniform Linear Array
=======================================================
Level: Intermediate

This example shows how to:
  1. Build a DirectionFindingDataset with a desired source and an interferer
  2. Apply delay-and-sum, MVDR, and LCMV beamformers to the snapshot matrix
  3. Visualise the output SNR of each beamformer
  4. Compare beam patterns (DAS vs MVDR vs LCMV with null)

Run:
    python examples/14_beamforming.py
"""

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

import matplotlib.pyplot as plt
import numpy as np
from plot_helpers import OUTPUT_DIR, savefig
from spectra.algorithms import compute_beam_pattern, delay_and_sum, lcmv, mvdr
from spectra.algorithms.beamforming import _mvdr_weights
from spectra.arrays import ula
from spectra.datasets import DirectionFindingDataset
from spectra.waveforms import QPSK

# ── Configuration ─────────────────────────────────────────────────────────────

FREQ_HZ     = 2.4e9
N_ELEMENTS  = 8
SPACING     = 0.5
N_SNAPSHOTS = 512
SAMPLE_RATE = 1e6
SNR_SOURCE  = 10.0               # dB
SNR_INTER   = 20.0               # dB (strong interferer)
SEED        = 0

SCAN_DEG = np.linspace(1, 179, 512)
SCAN_RAD = np.deg2rad(SCAN_DEG)

# ── 1. Build Dataset and Get One Sample ───────────────────────────────────────

arr = ula(num_elements=N_ELEMENTS, spacing=SPACING, frequency=FREQ_HZ)

ds = DirectionFindingDataset(
    array=arr,
    signal_pool=[QPSK(samples_per_symbol=4)],
    num_signals=2,
    num_snapshots=N_SNAPSHOTS,
    sample_rate=SAMPLE_RATE,
    snr_range=(SNR_SOURCE, SNR_INTER),
    azimuth_range=(np.deg2rad(10), np.deg2rad(170)),
    elevation_range=(0.0, 0.0),
    min_angular_separation=np.deg2rad(30),
    num_samples=50,
    seed=SEED,
)

data, target = ds[0]
X = data[:, 0, :].numpy() + 1j * data[:, 1, :].numpy()  # (N, T)
print(f"Sources: az = {np.rad2deg(target.azimuths)} degrees")
print(f"SNRs:    {target.snrs} dB\n")

az_desired    = float(target.azimuths[0])
az_interferer = float(target.azimuths[1])

# ── 2. Apply Beamformers ───────────────────────────────────────────────────────

y_das  = delay_and_sum(X, arr, target_az=az_desired)
y_mvdr = mvdr(X, arr, target_az=az_desired)
y_lcmv = lcmv(
    X, arr,
    constraints=[(az_desired, 0.0), (az_interferer, 0.0)],
    responses=[1.0 + 0j, 0.0 + 0j],
)

print("Output power comparison:")
for name, y in [("DAS", y_das), ("MVDR", y_mvdr), ("LCMV", y_lcmv)]:
    print(f"  {name}: {10*np.log10(np.mean(np.abs(y)**2)):.1f} dB")

# ── 3. Beam Pattern Comparison ────────────────────────────────────────────────

w_das  = arr.steering_vector(azimuth=az_desired, elevation=0.0).conj() / N_ELEMENTS
w_mvdr = _mvdr_weights(X, arr, target_az=az_desired)
w_lcmv = lcmv(
    X, arr,
    constraints=[(az_desired, 0.0), (az_interferer, 0.0)],
    responses=[1.0 + 0j, 0.0 + 0j],
    return_weights=True,
)

fig, ax = plt.subplots(figsize=(9, 4))
_beamformers = [
    ("DAS", w_das, "steelblue"),
    ("MVDR", w_mvdr, "seagreen"),
    ("LCMV", w_lcmv, "darkorange"),
]
for name, w, color in _beamformers:
    pattern = compute_beam_pattern(w, arr, SCAN_RAD)
    ax.plot(SCAN_DEG, 10 * np.log10(pattern + 1e-12), label=name, color=color, linewidth=1.3)

ax.axvline(np.rad2deg(az_desired),    color="crimson", linestyle="--", linewidth=1.2,
           label=f"Desired {np.rad2deg(az_desired):.0f}°")
ax.axvline(np.rad2deg(az_interferer), color="black",   linestyle=":",  linewidth=1.2,
           label=f"Interferer {np.rad2deg(az_interferer):.0f}°")
ax.set_xlabel("Azimuth (degrees)")
ax.set_ylabel("Normalised beam pattern (dB)")
ax.set_title(f"Beam Pattern Comparison — {N_ELEMENTS}-element ULA")
ax.set_ylim(-50, 5)
ax.legend(fontsize=8)
ax.grid(True, alpha=0.3)
plt.tight_layout()
savefig("14_beam_patterns.png")

# ── 4. Output Spectra ─────────────────────────────────────────────────────────

freqs = np.fft.fftshift(np.fft.fftfreq(N_SNAPSHOTS, d=1.0 / SAMPLE_RATE))
fig, axes = plt.subplots(3, 1, figsize=(10, 9), sharex=True)
for ax, name, y in zip(axes, ["DAS", "MVDR", "LCMV"], [y_das, y_mvdr, y_lcmv]):
    psd = np.abs(np.fft.fftshift(np.fft.fft(y))) ** 2
    ax.plot(freqs / 1e3, 10 * np.log10(psd + 1e-30), linewidth=0.8)
    ax.set_ylabel(f"{name}\nPower (dB)")
    ax.grid(True, alpha=0.3)
axes[-1].set_xlabel("Frequency (kHz)")
fig.suptitle("Output Spectrum per Beamformer")
plt.tight_layout()
savefig("14_output_spectra.png")

print(f"\nFigures saved to {OUTPUT_DIR}")
