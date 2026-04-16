"""
Array Geometries and Calibration
================================
Level: Intermediate

Demonstrate SPECTRA array construction and calibration:
  - ULA — Uniform Linear Array
  - UCA — Uniform Circular Array
  - Rectangular — 2D planar array
  - CalibrationErrors — per-element gain/phase offsets
  - Steering vector computation

Run:
    python examples/antenna_arrays/array_geometries.py
"""

import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from spectra.antennas import IsotropicElement
from spectra.arrays import ula, uca, rectangular, AntennaArray, CalibrationErrors
from plot_helpers import savefig

freq = 1e9

# ── 1. Create three array geometries ────────────────────────────────────────
element = IsotropicElement(frequency=freq)
arr_ula = ula(num_elements=8, spacing=0.5, element=element, frequency=freq)
arr_uca = uca(num_elements=8, radius=1.0, element=element, frequency=freq)
arr_rect = rectangular(
    rows=3, cols=4, spacing_x=0.5, spacing_y=0.5, element=element, frequency=freq,
)

# ── 2. Plot array geometries ────────────────────────────────────────────────
fig, axes = plt.subplots(1, 3, figsize=(15, 4))
for ax, arr, name in [
    (axes[0], arr_ula, f"ULA ({arr_ula.num_elements} elements)"),
    (axes[1], arr_uca, f"UCA ({arr_uca.num_elements} elements)"),
    (axes[2], arr_rect, f"Rectangular ({arr_rect.num_elements} elements)"),
]:
    pos = arr.positions
    ax.scatter(pos[:, 0], pos[:, 1], s=100, zorder=3)
    for i, (x, y) in enumerate(pos[:, :2]):
        ax.annotate(f"{i}", (x, y), textcoords="offset points", xytext=(0, 8),
                    ha="center", fontsize=8)
    ax.set_title(name)
    ax.set_xlabel("x (m)")
    ax.set_ylabel("y (m)")
    ax.set_aspect("equal")
    ax.grid(True, alpha=0.3)

fig.suptitle("Array Geometries", fontsize=13)
fig.tight_layout()
savefig("array_geometries.png")
plt.close()

# ── 3. Steering vectors ─────────────────────────────────────────────────────
azimuths = np.linspace(-90, 90, 181)
fig, axes = plt.subplots(1, 3, figsize=(15, 4))
for ax, arr, name in [
    (axes[0], arr_ula, "ULA"),
    (axes[1], arr_uca, "UCA"),
    (axes[2], arr_rect, "Rectangular"),
]:
    beampattern = np.zeros(len(azimuths))
    # Steer to 0 deg (broadside)
    w = arr.steering_vector(azimuth=np.deg2rad(0), elevation=0.0)
    w = w / np.linalg.norm(w)
    for i, az in enumerate(azimuths):
        a = arr.steering_vector(azimuth=np.deg2rad(az), elevation=0.0)
        beampattern[i] = np.abs(w.conj() @ a) ** 2

    bp_db = 10 * np.log10(beampattern / beampattern.max() + 1e-12)
    ax.plot(azimuths, bp_db, linewidth=1.2)
    ax.set_title(f"{name} — Broadside Beam")
    ax.set_xlabel("Azimuth (deg)")
    ax.set_ylabel("dB")
    ax.set_ylim([-30, 1])
    ax.grid(True, alpha=0.3)

fig.suptitle("Conventional Beamforming (Broadside Steering)", fontsize=13)
fig.tight_layout()
savefig("array_steering_vectors.png")
plt.close()

# ── 4. Calibration errors ───────────────────────────────────────────────────
rng = np.random.default_rng(42)
cal = CalibrationErrors.random(num_elements=8, gain_std_db=1.0, phase_std_rad=0.1, rng=rng)

print("Calibration errors:")
print(f"  Gain offsets (dB): {cal.gain_offsets_db}")
print(f"  Phase offsets (rad): {cal.phase_offsets_rad}")

# Compare ideal vs calibrated steering
sv_ideal = arr_ula.steering_vector(azimuth=np.deg2rad(30), elevation=0.0)
sv_cal = cal.apply(sv_ideal)

fig, axes = plt.subplots(1, 2, figsize=(10, 4))
axes[0].stem(range(8), np.abs(sv_ideal), linefmt="b-", markerfmt="bo", basefmt="gray")
axes[0].stem(range(8), np.abs(sv_cal), linefmt="r--", markerfmt="rs", basefmt="gray")
axes[0].set_title("Magnitude: Ideal (blue) vs Calibrated (red)")
axes[0].set_xlabel("Element")
axes[0].grid(True, alpha=0.3)

axes[1].stem(range(8), np.angle(sv_ideal), linefmt="b-", markerfmt="bo", basefmt="gray")
axes[1].stem(range(8), np.angle(sv_cal), linefmt="r--", markerfmt="rs", basefmt="gray")
axes[1].set_title("Phase: Ideal (blue) vs Calibrated (red)")
axes[1].set_xlabel("Element")
axes[1].set_ylabel("Phase (rad)")
axes[1].grid(True, alpha=0.3)

fig.suptitle("Effect of Calibration Errors on Steering Vector", fontsize=12)
fig.tight_layout()
savefig("array_calibration.png")
plt.close()

print("Done — array geometry examples saved.")
