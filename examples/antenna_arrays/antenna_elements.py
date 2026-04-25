"""
Antenna Element Radiation Patterns
===================================
Level: Intermediate

Demonstrate all SPECTRA antenna element types and their radiation patterns:
  - IsotropicElement — unit gain everywhere
  - ShortDipoleElement — sin(theta) pattern
  - HalfWaveDipoleElement — cos(pi/2*cos(theta))/sin(theta)
  - CosinePowerElement — cosine^n patch pattern
  - YagiElement — directional multi-element

Run:
    python examples/antenna_arrays/antenna_elements.py
"""

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

import matplotlib
import numpy as np

matplotlib.use("Agg")
import matplotlib.pyplot as plt
from plot_helpers import savefig
from spectra.antennas import (
    CosinePowerElement,
    HalfWaveDipoleElement,
    IsotropicElement,
    ShortDipoleElement,
    YagiElement,
)

freq = 1e9  # 1 GHz

# ── 1. Create antenna elements ──────────────────────────────────────────────
elements = [
    ("Isotropic", IsotropicElement(frequency=freq)),
    ("Short Dipole (z)", ShortDipoleElement(axis="z", frequency=freq)),
    ("Half-Wave Dipole", HalfWaveDipoleElement(axis="z", frequency=freq)),
    ("Cosine Power (n=1.5)", CosinePowerElement(exponent=1.5, frequency=freq)),
    ("Cosine Power (n=4)", CosinePowerElement(exponent=4.0, frequency=freq)),
    ("Yagi (3 elem)", YagiElement(n_elements=3, frequency=freq)),
    ("Yagi (5 elem)", YagiElement(n_elements=5, frequency=freq)),
]

# ── 2. Plot azimuth patterns (elevation=0) ──────────────────────────────────
azimuths = np.linspace(-180, 180, 361)
elevation = 0.0

fig, axes = plt.subplots(2, 4, figsize=(16, 8), subplot_kw={"projection": "polar"})
axes_flat = axes.flat

for idx, (name, element) in enumerate(elements):
    ax = axes_flat[idx]
    gains = np.array([
        element.pattern(np.deg2rad(az), np.deg2rad(elevation))
        for az in azimuths
    ])
    gains_db = 10 * np.log10(np.maximum(gains, 1e-12))
    gains_db = np.maximum(gains_db, -30)  # clip for display
    ax.plot(np.deg2rad(azimuths), gains_db + 30, linewidth=1.5)  # shift so 0 dB is at radius 30
    ax.set_title(name, fontsize=9, pad=12)
    ax.set_rticks([0, 10, 20, 30])
    ax.set_yticklabels(["-30", "-20", "-10", "0 dB"], fontsize=6)

# Hide unused subplot
axes_flat[-1].set_visible(False)

fig.suptitle("Antenna Element Azimuth Patterns (elevation=0)", fontsize=14)
fig.tight_layout()
savefig("antenna_elements_azimuth.png")
plt.close()

# ── 3. Elevation cut (azimuth=0) ────────────────────────────────────────────
elevations = np.linspace(-90, 90, 181)
azimuth = 0.0

fig, ax = plt.subplots(figsize=(10, 5))
for name, element in elements:
    gains = np.array([
        element.pattern(np.deg2rad(azimuth), np.deg2rad(el))
        for el in elevations
    ])
    gains_db = 10 * np.log10(np.maximum(gains, 1e-12))
    ax.plot(elevations, gains_db, linewidth=1.2, label=name)

ax.set_xlabel("Elevation (degrees)")
ax.set_ylabel("Gain (dB)")
ax.set_title("Elevation Pattern Cut (azimuth=0)")
ax.legend(fontsize=8)
ax.grid(True, alpha=0.3)
ax.set_ylim([-30, 15])
fig.tight_layout()
savefig("antenna_elements_elevation.png")
plt.close()

print("Done — antenna element examples saved.")
