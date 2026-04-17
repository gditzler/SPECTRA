"""
Urban 5G Scene — Propagation → Auto-Impairment Chain
=====================================================
Level: Advanced

Demonstrates the end-to-end flow:
  1. Build an Environment with GPP38901UMa propagation @ 3.5 GHz.
  2. Environment.compute() populates LinkParams with delay spread,
     K-factor, and angular spread from the 38.901 large-scale params.
  3. link_params_to_impairments() auto-emits a TDLChannel scaled to
     the delay spread, with the Rician K-factor baked in (TDL-D base).
  4. Apply the chain to QPSK and plot before/after spectrograms.

Run:
    python examples/environment/urban_5g_scene.py
"""

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
from plot_helpers import savefig
from spectra.environment import (
    Emitter,
    Environment,
    Position,
    ReceiverConfig,
    link_params_to_impairments,
)
from spectra.environment.propagation import GPP38901UMa
from spectra.impairments import TDLChannel
from spectra.scene import SignalDescription
from spectra.waveforms import QPSK

sample_rate = 10e6

# ── 1. Set up the environment ────────────────────────────────────────────
receiver = ReceiverConfig(
    position=Position(0.0, 0.0),
    noise_figure_db=7.0,
    bandwidth_hz=sample_rate,
)

emitter = Emitter(
    waveform=QPSK(samples_per_symbol=8, rolloff=0.25),
    position=Position(250.0, 100.0),  # ~270 m away
    power_dbm=30.0,
    freq_hz=3.5e9,
    velocity_mps=(5.0, 0.0),
)

env = Environment(
    propagation=GPP38901UMa(h_bs_m=25.0, h_ut_m=1.5, los_mode="stochastic"),
    emitters=[emitter],
    receiver=receiver,
)

# ── 2. Compute link params and inspect the populated fields ──────────────
lp = env.compute(seed=42)[0]
print("Link parameters from 38.901 UMa:")
print(f"  distance       = {lp.distance_m:.1f} m")
print(f"  path_loss      = {lp.path_loss_db:.1f} dB")
print(f"  shadow_fading  = {lp.shadow_fading_db:+.2f} dB")
print(f"  snr            = {lp.snr_db:.1f} dB")
print(f"  doppler        = {lp.doppler_hz:.1f} Hz")
print(f"  rms_delay_spread = {lp.rms_delay_spread_s * 1e9:.1f} ns")
print(f"  k_factor       = {lp.k_factor_db}")
print(f"  angular_spread = {lp.angular_spread_deg:.1f} deg")

# ── 3. Auto-generate the impairment chain ─────────────────────────────────
chain = link_params_to_impairments(lp)
print(f"\nAuto-generated impairment chain ({len(chain)} stages):")
for i, t in enumerate(chain):
    extra = ""
    if isinstance(t, TDLChannel):
        extra = f" (base={t._profile_name}, k={t._profile.get('k_factor_db')})"
    print(f"  {i}: {type(t).__name__}{extra}")

# ── 4. Apply the chain to a QPSK signal ──────────────────────────────────
iq_clean = emitter.waveform.generate(num_symbols=4096, sample_rate=sample_rate, seed=7)
desc = SignalDescription(
    t_start=0.0,
    t_stop=len(iq_clean) / sample_rate,
    f_low=-sample_rate / 2,
    f_high=sample_rate / 2,
    label="QPSK",
    snr=lp.snr_db,
)

iq = iq_clean.copy()
for t in chain:
    iq, desc = t(iq, desc, sample_rate=sample_rate)

# ── 5. Plot spectrograms before/after ────────────────────────────────────
fig, axes = plt.subplots(1, 2, figsize=(12, 5), sharey=True)
for ax, data, title in [
    (axes[0], iq_clean, "Clean QPSK (before propagation)"),
    (axes[1], iq, "After 38.901 UMa + auto-impairments"),
]:
    ax.specgram(data, NFFT=256, Fs=sample_rate, noverlap=128, cmap="viridis")
    ax.set_title(title)
    ax.set_xlabel("Time (s)")
axes[0].set_ylabel("Frequency (Hz)")
fig.tight_layout()
savefig("urban_5g_scene_spectrograms.png")
plt.close()

print("\nDone — urban_5g_scene_spectrograms.png saved.")
