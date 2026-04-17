"""
Environment & Propagation Modeling
===================================
Level: Intermediate

Demonstrate SPECTRA's environment simulation module:
  - Position, Emitter, ReceiverConfig
  - Propagation models: FreeSpacePathLoss, LogDistancePL, OkumuraHataPL, GPP38901UMa
  - Environment.compute() — link budget computation
  - link_params_to_impairments — auto-generate impairment chains
  - propagation_presets — quick model selection

Run:
    python examples/environment/propagation_and_links.py
"""

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

import matplotlib
import numpy as np

matplotlib.use("Agg")
import matplotlib.pyplot as plt
from plot_helpers import savefig
from spectra.environment import (
    Emitter,
    Environment,
    FreeSpacePathLoss,
    LogDistancePL,
    Position,
    ReceiverConfig,
    link_params_to_impairments,
    propagation_presets,
)
from spectra.environment.propagation import (
    GPP38901UMa,
    OkumuraHataPL,
)
from spectra.waveforms import BPSK, QPSK

sample_rate = 1e6

# ── 1. Propagation model comparison ─────────────────────────────────────────
freq = 900e6  # 900 MHz
distances = np.logspace(1, 4, 100)  # 10 m to 10 km

fspl = FreeSpacePathLoss()
logd = LogDistancePL(n=3.5)
hata = OkumuraHataPL(h_bs_m=50.0, h_ms_m=1.5, environment="urban_small_medium")
uma = GPP38901UMa(h_bs_m=25.0, h_ut_m=1.5, los_mode="force_nlos")

plt.figure(figsize=(10, 5))
for model, name in [
    (fspl, "Free Space"),
    (logd, "Log-Distance (n=3.5)"),
    (hata, "Okumura-Hata (urban, 50m)"),
    (uma, "38.901 UMa (NLOS)"),
]:
    losses = []
    for d in distances:
        try:
            r = model(d, freq, seed=0)
            losses.append(r.path_loss_db - r.shadow_fading_db)
        except ValueError:
            losses.append(np.nan)
    plt.plot(distances / 1e3, losses, linewidth=1.5, label=name)

plt.xlabel("Distance (km)")
plt.ylabel("Path Loss (dB)")
plt.title(f"Propagation Model Comparison @ {freq / 1e6:.0f} MHz")
plt.legend()
plt.grid(True, alpha=0.3)
plt.xscale("log")
plt.tight_layout()
savefig("propagation_models.png")
plt.close()

# ── 2. Multi-emitter environment ────────────────────────────────────────────
rx = ReceiverConfig(
    position=Position(x=0, y=0),
    noise_figure_db=6.0,
    bandwidth_hz=sample_rate,
)

emitters = [
    Emitter(
        waveform=QPSK(samples_per_symbol=8, rolloff=0.35),
        position=Position(x=500, y=300),
        power_dbm=30.0,
        freq_hz=freq,
        velocity_mps=(10.0, 5.0),
    ),
    Emitter(
        waveform=BPSK(samples_per_symbol=8, rolloff=0.35),
        position=Position(x=-200, y=800),
        power_dbm=25.0,
        freq_hz=freq,
    ),
    Emitter(
        waveform=QPSK(samples_per_symbol=8, rolloff=0.35),
        position=Position(x=1000, y=-100),
        power_dbm=35.0,
        freq_hz=freq,
        velocity_mps=(-15.0, 0.0),
    ),
]

env = Environment(
    propagation=FreeSpacePathLoss(),
    emitters=emitters,
    receiver=rx,
)

link_params_list = env.compute(seed=42)

print("Link Budget Results:")
print(
    f"{'Emitter':>8} {'Distance':>10} {'PathLoss':>10} {'SNR':>8} {'Doppler':>10} {'Rx Power':>10}"
)
for i, lp in enumerate(link_params_list):
    print(
        f"  {i:>5d} {lp.distance_m:>9.1f}m {lp.path_loss_db:>9.1f}dB "
        f"{lp.snr_db:>7.1f}dB {lp.doppler_hz:>9.1f}Hz {lp.received_power_dbm:>9.1f}dBm"
    )

# ── 3. Plot environment geometry ─────────────────────────────────────────────
fig, ax = plt.subplots(figsize=(8, 8))
ax.scatter(
    rx.position.x, rx.position.y, s=200, marker="^", color="blue", zorder=5, label="Receiver"
)
for i, em in enumerate(emitters):
    ax.scatter(em.position.x, em.position.y, s=100, marker="o", color="red", zorder=5)
    ax.annotate(
        f"TX{i}\n{em.power_dbm:.0f}dBm",
        (em.position.x, em.position.y),
        textcoords="offset points",
        xytext=(10, 10),
        fontsize=8,
    )
    ax.plot([rx.position.x, em.position.x], [rx.position.y, em.position.y], "k--", alpha=0.3)

ax.set_xlabel("x (m)")
ax.set_ylabel("y (m)")
ax.set_title("Multi-Emitter Environment")
ax.legend()
ax.grid(True, alpha=0.3)
ax.set_aspect("equal")
fig.tight_layout()
savefig("environment_geometry.png")
plt.close()

# ── 4. Convert link params to impairments ────────────────────────────────────
print("\nGenerating impairment chains from link params:")
for i, lp in enumerate(link_params_list):
    imp_chain = link_params_to_impairments(lp)
    print(f"  Emitter {i}: {imp_chain}")

# ── 5. Propagation presets ───────────────────────────────────────────────────
print(f"\nAvailable presets: {list(propagation_presets.keys())}")

print("\nDone — environment/propagation examples saved.")
