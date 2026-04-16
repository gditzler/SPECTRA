"""
Environment & Propagation Modeling
===================================
Level: Intermediate

Demonstrate SPECTRA's environment simulation module:
  - Position, Emitter, ReceiverConfig
  - Propagation models: FreeSpacePathLoss, LogDistancePL, COST231HataPL
  - Environment.compute() — link budget computation
  - link_params_to_impairments — auto-generate impairment chains
  - propagation_presets — quick model selection

Run:
    python examples/environment/propagation_and_links.py
"""

import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from spectra.waveforms import QPSK, BPSK
from spectra.environment import (
    Position, Emitter, ReceiverConfig, Environment,
    FreeSpacePathLoss, LogDistancePL, COST231HataPL,
    link_params_to_impairments, propagation_presets,
)
from plot_helpers import savefig

sample_rate = 1e6

# ── 1. Propagation model comparison ─────────────────────────────────────────
freq = 900e6  # 900 MHz
distances = np.logspace(1, 4, 100)  # 10 m to 10 km

fspl = FreeSpacePathLoss(frequency_hz=freq)
logd = LogDistancePL(frequency_hz=freq, path_loss_exp=3.5)
cost231 = COST231HataPL(frequency_hz=freq, is_urban=True)

plt.figure(figsize=(10, 5))
for model, name in [(fspl, "Free Space"), (logd, "Log-Distance (n=3.5)"), (cost231, "COST231-Hata (urban)")]:
    losses = []
    for d in distances:
        result = model.compute(d)
        losses.append(result.path_loss_db)
    plt.plot(distances / 1e3, losses, linewidth=1.5, label=name)

plt.xlabel("Distance (km)")
plt.ylabel("Path Loss (dB)")
plt.title(f"Propagation Model Comparison @ {freq/1e6:.0f} MHz")
plt.legend()
plt.grid(True, alpha=0.3)
plt.xscale("log")
plt.tight_layout()
savefig("propagation_models.png")
plt.close()

# ── 2. Multi-emitter environment ────────────────────────────────────────────
rx = ReceiverConfig(
    position=Position(x_m=0, y_m=0),
    noise_figure_db=6.0,
    bandwidth_hz=sample_rate,
)

emitters = [
    Emitter(
        waveform=QPSK(samples_per_symbol=8, rolloff=0.35),
        position=Position(x_m=500, y_m=300),
        power_dbm=30.0,
        freq_hz=freq,
        velocity_mps=(10.0, 5.0),
    ),
    Emitter(
        waveform=BPSK(samples_per_symbol=8, rolloff=0.35),
        position=Position(x_m=-200, y_m=800),
        power_dbm=25.0,
        freq_hz=freq,
    ),
    Emitter(
        waveform=QPSK(samples_per_symbol=8, rolloff=0.35),
        position=Position(x_m=1000, y_m=-100),
        power_dbm=35.0,
        freq_hz=freq,
        velocity_mps=(-15.0, 0.0),
    ),
]

env = Environment(
    propagation=FreeSpacePathLoss(frequency_hz=freq),
    emitters=emitters,
    receiver=rx,
)

link_params_list = env.compute(seed=42)

print("Link Budget Results:")
print(f"{'Emitter':>8} {'Distance':>10} {'PathLoss':>10} {'SNR':>8} {'Doppler':>10} {'Rx Power':>10}")
for i, lp in enumerate(link_params_list):
    print(f"  {i:>5d} {lp.distance_m:>9.1f}m {lp.path_loss_db:>9.1f}dB "
          f"{lp.snr_db:>7.1f}dB {lp.doppler_hz:>9.1f}Hz {lp.received_power_dbm:>9.1f}dBm")

# ── 3. Plot environment geometry ─────────────────────────────────────────────
fig, ax = plt.subplots(figsize=(8, 8))
ax.scatter(rx.position.x_m, rx.position.y_m, s=200, marker="^", color="blue", zorder=5, label="Receiver")
for i, em in enumerate(emitters):
    ax.scatter(em.position.x_m, em.position.y_m, s=100, marker="o", color="red", zorder=5)
    ax.annotate(f"TX{i}\n{em.power_dbm:.0f}dBm", (em.position.x_m, em.position.y_m),
                textcoords="offset points", xytext=(10, 10), fontsize=8)
    ax.plot([rx.position.x_m, em.position.x_m], [rx.position.y_m, em.position.y_m],
            "k--", alpha=0.3)

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
presets = propagation_presets()
print(f"\nAvailable presets: {list(presets.keys())}")

print("\nDone — environment/propagation examples saved.")
