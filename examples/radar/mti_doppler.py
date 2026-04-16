"""
MTI and Doppler Filter Banks
=============================
Level: Intermediate

Demonstrate SPECTRA's Moving Target Indication and Doppler processing:
  - single_pulse_canceller — first-order clutter suppression
  - double_pulse_canceller — second-order clutter suppression
  - doppler_filter_bank — DFT-based Doppler filtering

Run:
    python examples/radar/mti_doppler.py
"""

import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from spectra.algorithms import single_pulse_canceller, double_pulse_canceller, doppler_filter_bank
from plot_helpers import savefig

rng = np.random.default_rng(42)

# ── 1. Simulate a pulse matrix ──────────────────────────────────────────────
num_pulses = 32
num_range_bins = 256
prf = 1000.0  # Hz

# Ground clutter: strong, zero Doppler
clutter = 10.0 * rng.standard_normal((num_pulses, num_range_bins)).astype(np.complex128)

# Moving target: moderate amplitude, non-zero Doppler
target_range_bin = 100
target_doppler_hz = 200.0
target_amplitude = 3.0
for p in range(num_pulses):
    phase = 2 * np.pi * target_doppler_hz * p / prf
    clutter[p, target_range_bin] += target_amplitude * np.exp(1j * phase)

# Add noise
pulse_matrix = clutter + 0.5 * (rng.standard_normal(clutter.shape) + 1j * rng.standard_normal(clutter.shape))

# ── 2. Apply MTI cancellers ─────────────────────────────────────────────────
spc = single_pulse_canceller(pulse_matrix)
dpc = double_pulse_canceller(pulse_matrix)

# ── 3. Apply Doppler filter bank ────────────────────────────────────────────
dfb = doppler_filter_bank(pulse_matrix)

# ── 4. Plot results ─────────────────────────────────────────────────────────
fig, axes = plt.subplots(2, 2, figsize=(12, 10))

# Raw pulse matrix (one pulse)
axes[0, 0].plot(20 * np.log10(np.abs(pulse_matrix[0]) + 1e-12), linewidth=0.5)
axes[0, 0].axvline(target_range_bin, color="red", linestyle="--", linewidth=1, label="Target")
axes[0, 0].set_title("Raw Pulse (single CPI)")
axes[0, 0].set_xlabel("Range Bin")
axes[0, 0].set_ylabel("dB")
axes[0, 0].legend()
axes[0, 0].grid(True, alpha=0.3)

# Single pulse canceller
axes[0, 1].plot(20 * np.log10(np.abs(spc[0]) + 1e-12), linewidth=0.5)
axes[0, 1].axvline(target_range_bin, color="red", linestyle="--", linewidth=1, label="Target")
axes[0, 1].set_title("After Single Pulse Canceller")
axes[0, 1].set_xlabel("Range Bin")
axes[0, 1].set_ylabel("dB")
axes[0, 1].legend()
axes[0, 1].grid(True, alpha=0.3)

# Double pulse canceller
axes[1, 0].plot(20 * np.log10(np.abs(dpc[0]) + 1e-12), linewidth=0.5)
axes[1, 0].axvline(target_range_bin, color="red", linestyle="--", linewidth=1, label="Target")
axes[1, 0].set_title("After Double Pulse Canceller")
axes[1, 0].set_xlabel("Range Bin")
axes[1, 0].set_ylabel("dB")
axes[1, 0].legend()
axes[1, 0].grid(True, alpha=0.3)

# Range-Doppler map
axes[1, 1].imshow(
    20 * np.log10(np.abs(dfb) + 1e-12),
    aspect="auto", origin="lower", cmap="hot",
    extent=[0, num_range_bins, -prf / 2, prf / 2],
)
axes[1, 1].axhline(target_doppler_hz, color="cyan", linestyle="--", linewidth=1)
axes[1, 1].axvline(target_range_bin, color="cyan", linestyle="--", linewidth=1)
axes[1, 1].set_title("Range-Doppler Map")
axes[1, 1].set_xlabel("Range Bin")
axes[1, 1].set_ylabel("Doppler (Hz)")

fig.suptitle("MTI Clutter Suppression & Doppler Processing", fontsize=14)
fig.tight_layout()
savefig("mti_doppler.png")
plt.close()

print("Done — MTI/Doppler example saved.")
