"""
Terrestrial Propagation Models — Comprehensive Demo
====================================================
Level: Intermediate

Visualizes SPECTRA's full terrestrial propagation model library:
  - FreeSpacePathLoss / ITU_R_P525 (with P.676 absorption)
  - LogDistancePL
  - OkumuraHataPL / COST231HataPL
  - 3GPP 38.901: UMa, UMi, RMa, InH (LOS probability + path loss)
  - ITU-R P.1411

Produces four plots:
  1. PL vs distance overlay (all models @ 2.1 GHz urban)
  2. PL vs frequency with and without P.676 absorption
  3. 38.901 UMa LOS probability curve
  4. Shadow-fading histograms

Run:
    python examples/environment/terrestrial_propagation_models.py
"""

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
from plot_helpers import savefig
from spectra.environment.propagation import (
    ITU_R_P525,
    ITU_R_P1411,
    COST231HataPL,
    FreeSpacePathLoss,
    GPP38901InH,
    GPP38901RMa,
    GPP38901UMa,
    GPP38901UMi,
    LogDistancePL,
    OkumuraHataPL,
)

# ── 1. PL vs distance overlay (2.1 GHz, urban) ─────────────────────────────
freq = 2.1e9
distances = np.logspace(1.5, 3.7, 200)  # 30 m to 5 km

models = [
    ("Free Space", FreeSpacePathLoss()),
    ("Log-Distance n=3.5", LogDistancePL(n=3.5)),
    ("Okumura-Hata Urban", OkumuraHataPL(
        h_bs_m=50.0, h_ms_m=1.5,
        environment="urban_small_medium",
        strict_range=False,  # 2.1 GHz is outside Hata's envelope
    )),
    ("COST-231 Hata Urban", COST231HataPL(environment="urban")),
    ("38.901 UMa (LOS)", GPP38901UMa(h_bs_m=25.0, h_ut_m=1.5, los_mode="force_los")),
    ("38.901 UMa (NLOS)", GPP38901UMa(h_bs_m=25.0, h_ut_m=1.5, los_mode="force_nlos")),
    ("38.901 UMi (LOS)", GPP38901UMi(h_bs_m=10.0, h_ut_m=1.5, los_mode="force_los")),
    ("P.1411 Urban HR (LOS)", ITU_R_P1411(environment="urban_high_rise", los_mode="force_los")),
]

plt.figure(figsize=(10, 6))
for name, m in models:
    pl = []
    for d in distances:
        try:
            # Use seed=0 for deterministic shadow fading
            r = m(d, freq, seed=0)
            # Subtract shadow fading to plot the mean
            pl.append(r.path_loss_db - r.shadow_fading_db)
        except ValueError:
            pl.append(np.nan)
    plt.plot(distances / 1e3, pl, linewidth=1.3, label=name)

plt.xlabel("Distance (km)")
plt.ylabel("Mean Path Loss (dB)")
plt.title(f"Terrestrial Propagation Models @ {freq / 1e9:.1f} GHz")
plt.legend(fontsize=8, loc="lower right")
plt.grid(True, alpha=0.3)
plt.xscale("log")
plt.tight_layout()
savefig("propagation_models_overlay.png")
plt.close()

# ── 2. PL vs frequency: P.525 clean vs P.525 + P.676 ───────────────────────
freqs = np.logspace(9, 11, 300)  # 1 GHz to 100 GHz
d = 1000.0  # 1 km horizontal link

p525_clean = ITU_R_P525(include_gaseous=False)
p525_absorb = ITU_R_P525(include_gaseous=True)

pl_clean = []
pl_absorb = []
for f in freqs:
    try:
        pl_clean.append(p525_clean(d, f).path_loss_db)
        pl_absorb.append(p525_absorb(d, f).path_loss_db)
    except ValueError:
        # P.676 helper raises above 100 GHz; set to NaN for plotting
        pl_clean.append(np.nan)
        pl_absorb.append(np.nan)

plt.figure(figsize=(10, 5))
plt.plot(freqs / 1e9, pl_clean, label="P.525 only", linewidth=1.5)
plt.plot(freqs / 1e9, pl_absorb, label="P.525 + P.676", linewidth=1.5)
plt.xlabel("Frequency (GHz)")
plt.ylabel("Path Loss (dB)")
plt.title("ITU-R P.525 with and without P.676 Gaseous Absorption (1 km)")
plt.legend()
plt.grid(True, alpha=0.3)
plt.xscale("log")
plt.tight_layout()
savefig("propagation_p525_p676.png")
plt.close()

# ── 3. 38.901 UMa LOS probability ──────────────────────────────────────────
uma = GPP38901UMa(h_bs_m=25.0, h_ut_m=1.5)
d_range = np.linspace(10.0, 5000.0, 500)
p_los = [uma._los_probability(d) for d in d_range]

plt.figure(figsize=(10, 5))
plt.plot(d_range, p_los, linewidth=1.5, color="tab:purple")
plt.xlabel("2D Distance (m)")
plt.ylabel("LOS Probability")
plt.title("3GPP 38.901 UMa LOS Probability vs Distance (h_UT = 1.5 m)")
plt.grid(True, alpha=0.3)
plt.ylim(0, 1.05)
plt.tight_layout()
savefig("propagation_38901_los_probability.png")
plt.close()

# ── 4. Shadow-fading histograms ────────────────────────────────────────────
N = 10_000
scenarios = [
    ("38.901 UMa LOS (σ=4)", GPP38901UMa(h_bs_m=25.0, h_ut_m=1.5, los_mode="force_los")),
    ("38.901 UMa NLOS (σ=6)", GPP38901UMa(h_bs_m=25.0, h_ut_m=1.5, los_mode="force_nlos")),
    (
        "38.901 InH LOS (σ=3)",
        GPP38901InH(h_bs_m=3.0, h_ut_m=1.0, los_mode="force_los", strict_range=False),
    ),
    ("38.901 RMa LOS (σ=4)", GPP38901RMa(h_bs_m=35.0, h_ut_m=1.5, los_mode="force_los")),
]

fig, axes = plt.subplots(2, 2, figsize=(12, 7))
for ax, (name, m) in zip(axes.flat, scenarios):
    samples = [m(500.0, 3.5e9, seed=i).shadow_fading_db for i in range(N)]
    ax.hist(samples, bins=60, alpha=0.7, color="tab:orange", edgecolor="black")
    ax.axvline(0.0, color="red", linestyle="--", linewidth=1.0)
    ax.set_title(name)
    ax.set_xlabel("Shadow Fading (dB)")
    ax.set_ylabel("Count")
    ax.grid(True, alpha=0.3)
fig.suptitle("Shadow Fading Realizations (10 000 samples)")
fig.tight_layout()
savefig("propagation_shadow_fading_histograms.png")
plt.close()

print("Done — four propagation plots saved.")
