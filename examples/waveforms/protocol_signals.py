"""
Protocol & Aviation/Maritime Waveforms
======================================
Level: Intermediate

Demonstrate SPECTRA's protocol waveform generators:
  - ADS-B (Automatic Dependent Surveillance-Broadcast)
  - Mode S (Secondary Surveillance Radar)
  - AIS (Automatic Identification System)
  - ACARS (Aircraft Communications Addressing and Reporting System)
  - DME (Distance Measuring Equipment)
  - ILS Localizer (Instrument Landing System)

Run:
    python examples/waveforms/protocol_signals.py
"""

import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from spectra.waveforms import ADSB, ModeS, AIS, ACARS, DME, ILS_Localizer
from plot_helpers import savefig, plot_psd

sample_rate = 2e6
seed = 42

# ── 1. Generate each protocol waveform ──────────────────────────────────────
protocols = [
    ("ADS-B", ADSB()),
    ("Mode S", ModeS()),
    ("AIS", AIS()),
    ("ACARS", ACARS()),
    ("DME", DME()),
    ("ILS Localizer", ILS_Localizer()),
]

fig, axes = plt.subplots(len(protocols), 2, figsize=(14, 3 * len(protocols)))
for row, (name, waveform) in enumerate(protocols):
    iq = waveform.generate(num_symbols=256, sample_rate=sample_rate, seed=seed)
    print(f"{name}: label={waveform.label}, samples={len(iq)}")

    # Time domain
    n = min(500, len(iq))
    axes[row, 0].plot(iq[:n].real, linewidth=0.6)
    axes[row, 0].plot(iq[:n].imag, linewidth=0.6, alpha=0.7)
    axes[row, 0].set_title(f"{name} — Time Domain")
    axes[row, 0].set_ylabel("Amplitude")
    axes[row, 0].grid(True, alpha=0.3)

    # PSD
    nfft = 1024
    freqs = np.fft.fftshift(np.fft.fftfreq(nfft, d=1.0 / sample_rate))
    spectrum = np.fft.fftshift(np.fft.fft(iq[:nfft], n=nfft))
    psd = 10 * np.log10(np.abs(spectrum) ** 2 + 1e-12)
    axes[row, 1].plot(freqs / 1e3, psd, linewidth=0.8, color="tab:green")
    axes[row, 1].set_title(f"{name} — PSD")
    axes[row, 1].set_ylabel("dB")
    axes[row, 1].grid(True, alpha=0.3)

axes[-1, 0].set_xlabel("Sample")
axes[-1, 1].set_xlabel("Frequency (kHz)")
fig.suptitle("Aviation & Maritime Protocol Waveforms", fontsize=14, y=1.01)
fig.tight_layout()
savefig("protocol_signals.png")
plt.close()

# ── 2. Bandwidth comparison ─────────────────────────────────────────────────
print("\nBandwidth summary:")
for name, waveform in protocols:
    bw = waveform.bandwidth(sample_rate)
    print(f"  {name:15s}: {bw / 1e3:.1f} kHz")

print("\nDone — protocol signal examples saved.")
