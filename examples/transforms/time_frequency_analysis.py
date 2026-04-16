"""
Advanced Time-Frequency Analysis
================================
Level: Intermediate

Demonstrate SPECTRA transforms not covered by the cyclostationary examples:
  - ReassignedGabor — reassigned spectrogram for sharper TF localization
  - InstantaneousFrequency — analytic-signal IF estimation
  - AmbiguityFunction — delay-Doppler analysis

Run:
    python examples/transforms/time_frequency_analysis.py
"""

import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from spectra.waveforms.lfm import LFM
from spectra.waveforms import QPSK
from spectra.transforms import ReassignedGabor, InstantaneousFrequency, AmbiguityFunction, Spectrogram
from spectra.scene import SignalDescription
from plot_helpers import savefig

sample_rate = 1e6
seed = 42

# ── 1. ReassignedGabor vs standard Spectrogram ──────────────────────────────
lfm = LFM(bandwidth_fraction=0.2, samples_per_pulse=256)
iq_lfm = lfm.generate(num_symbols=4, sample_rate=sample_rate, seed=seed)
desc = SignalDescription(
    t_start=0.0, t_stop=len(iq_lfm) / sample_rate,
    f_low=-sample_rate / 4, f_high=sample_rate / 4,
    label="LFM", snr=30.0,
)

spec_xform = Spectrogram(nfft=128, hop_length=16)
rg_xform = ReassignedGabor(nfft=128, hop_length=16)

spec_out = spec_xform(iq_lfm)
rg_out = rg_xform(iq_lfm)
spec_out = spec_out.numpy() if hasattr(spec_out, "numpy") else spec_out
rg_out = rg_out.numpy() if hasattr(rg_out, "numpy") else rg_out

fig, axes = plt.subplots(1, 2, figsize=(12, 5))
for ax, data, title in [
    (axes[0], spec_out, "Standard Spectrogram"),
    (axes[1], rg_out, "Reassigned Gabor"),
]:
    if data.ndim == 3:
        data = data[0]  # take first (and only) channel [freq, time]
    ax.imshow(
        10 * np.log10(np.abs(data) + 1e-12),
        aspect="auto", origin="lower", cmap="viridis",
    )
    ax.set_title(title)
    ax.set_xlabel("Time Frame")
    ax.set_ylabel("Frequency Bin")

fig.suptitle("LFM Chirp — Spectrogram vs Reassigned Gabor", fontsize=13)
fig.tight_layout()
savefig("tf_reassigned_gabor.png")
plt.close()

# ── 2. Instantaneous Frequency ──────────────────────────────────────────────
if_xform = InstantaneousFrequency()
iq_qpsk = QPSK(samples_per_symbol=8, rolloff=0.35).generate(
    num_symbols=128, sample_rate=sample_rate, seed=seed)

if_out = if_xform(iq_lfm)
if_qpsk = if_xform(iq_qpsk)
if_out = if_out.numpy() if hasattr(if_out, "numpy") else if_out
if_qpsk = if_qpsk.numpy() if hasattr(if_qpsk, "numpy") else if_qpsk

fig, axes = plt.subplots(2, 1, figsize=(10, 6))
axes[0].plot(if_out[:500], linewidth=0.8)
axes[0].set_title("LFM — Instantaneous Frequency")
axes[0].set_ylabel("Freq (normalized)")
axes[0].grid(True, alpha=0.3)

axes[1].plot(if_qpsk[:500], linewidth=0.8, color="tab:orange")
axes[1].set_title("QPSK — Instantaneous Frequency")
axes[1].set_ylabel("Freq (normalized)")
axes[1].set_xlabel("Sample")
axes[1].grid(True, alpha=0.3)
fig.tight_layout()
savefig("tf_instantaneous_frequency.png")
plt.close()

# ── 3. Ambiguity Function ───────────────────────────────────────────────────
af_xform = AmbiguityFunction()
af_lfm = af_xform(iq_lfm)
af_lfm = af_lfm.numpy() if hasattr(af_lfm, "numpy") else af_lfm

fig, ax = plt.subplots(figsize=(8, 6))
if af_lfm.ndim == 3:
    af_plot = af_lfm[0]  # take first channel [delay, doppler]
else:
    af_plot = np.abs(af_lfm)
ax.imshow(
    10 * np.log10(af_plot + 1e-12),
    aspect="auto", origin="lower", cmap="hot",
)
ax.set_title("LFM — Ambiguity Function (Delay-Doppler)")
ax.set_xlabel("Doppler Bin")
ax.set_ylabel("Delay Bin")
fig.tight_layout()
savefig("tf_ambiguity_function.png")
plt.close()

print("Done — time-frequency analysis examples saved.")
