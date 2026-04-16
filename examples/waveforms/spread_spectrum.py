"""
Spread Spectrum Waveforms
=========================
Level: Intermediate

Demonstrate SPECTRA's spread-spectrum waveform generators:
  - DSSS-BPSK and DSSS-QPSK (direct-sequence)
  - FHSS (frequency-hopping)
  - THSS (time-hopping)
  - CDMA Forward and Reverse links
  - ChirpSS (chirp spread spectrum)

Run:
    python examples/waveforms/spread_spectrum.py
"""

import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from spectra.waveforms import (
    DSSS_BPSK, DSSS_QPSK, FHSS, THSS, CDMA_Forward, CDMA_Reverse, ChirpSS,
)
from plot_helpers import savefig, plot_psd

sample_rate = 1e6
num_symbols = 256
seed = 42

# ── 1. DSSS-BPSK vs DSSS-QPSK ──────────────────────────────────────────────
dsss_bpsk = DSSS_BPSK(chips_per_symbol=31)
dsss_qpsk = DSSS_QPSK(chips_per_symbol=31)

iq_bpsk = dsss_bpsk.generate(num_symbols=num_symbols, sample_rate=sample_rate, seed=seed)
iq_qpsk = dsss_qpsk.generate(num_symbols=num_symbols, sample_rate=sample_rate, seed=seed)

fig, axes = plt.subplots(2, 1, figsize=(10, 6))
axes[0].plot(iq_bpsk[:500].real, linewidth=0.5)
axes[0].set_title(f"DSSS-BPSK — {dsss_bpsk.label}")
axes[0].set_ylabel("In-Phase")
axes[0].grid(True, alpha=0.3)
axes[1].plot(iq_qpsk[:500].real, linewidth=0.5, color="tab:orange")
axes[1].set_title(f"DSSS-QPSK — {dsss_qpsk.label}")
axes[1].set_ylabel("In-Phase")
axes[1].set_xlabel("Sample")
axes[1].grid(True, alpha=0.3)
fig.tight_layout()
savefig("spread_spectrum_dsss.png")
plt.close()

# ── 2. FHSS — Frequency Hopping ─────────────────────────────────────────────
fhss = FHSS(num_hops=16, hop_bandwidth=50e3)
iq_fhss = fhss.generate(num_symbols=num_symbols, sample_rate=sample_rate, seed=seed)

fig, axes = plt.subplots(2, 1, figsize=(10, 6))
nfft = 256
hop_len = len(iq_fhss) // nfft
spec = np.array([
    np.fft.fftshift(np.abs(np.fft.fft(iq_fhss[i * nfft:(i + 1) * nfft])) ** 2)
    for i in range(hop_len)
])
axes[0].imshow(
    10 * np.log10(spec.T + 1e-12), aspect="auto", origin="lower", cmap="viridis",
)
axes[0].set_title(f"FHSS Spectrogram — {fhss.label}")
axes[0].set_ylabel("Frequency Bin")
axes[1].plot(iq_fhss[:1000].real, linewidth=0.5)
axes[1].set_title("FHSS Time Domain")
axes[1].set_xlabel("Sample")
axes[1].grid(True, alpha=0.3)
fig.tight_layout()
savefig("spread_spectrum_fhss.png")
plt.close()

# ── 3. THSS — Time Hopping ──────────────────────────────────────────────────
thss = THSS(num_slots=8, slot_duration_symbols=4)
iq_thss = thss.generate(num_symbols=num_symbols, sample_rate=sample_rate, seed=seed)

plt.figure(figsize=(10, 3))
plt.plot(np.abs(iq_thss[:2000]), linewidth=0.5)
plt.title(f"THSS Envelope — {thss.label}")
plt.xlabel("Sample")
plt.ylabel("|IQ|")
plt.grid(True, alpha=0.3)
plt.tight_layout()
savefig("spread_spectrum_thss.png")
plt.close()

# ── 4. CDMA Forward and Reverse ─────────────────────────────────────────────
cdma_fwd = CDMA_Forward(num_users=4, code_length=64)
cdma_rev = CDMA_Reverse(num_users=4, code_length=64)

iq_fwd = cdma_fwd.generate(num_symbols=num_symbols, sample_rate=sample_rate, seed=seed)
iq_rev = cdma_rev.generate(num_symbols=num_symbols, sample_rate=sample_rate, seed=seed)

fig, axes = plt.subplots(1, 2, figsize=(12, 4))
for ax, iq, label in [(axes[0], iq_fwd, "CDMA Forward"), (axes[1], iq_rev, "CDMA Reverse")]:
    ax.scatter(iq[:500].real, iq[:500].imag, s=1, alpha=0.4)
    ax.set_title(label)
    ax.set_xlabel("I")
    ax.set_ylabel("Q")
    ax.set_aspect("equal")
    ax.grid(True, alpha=0.3)
fig.tight_layout()
savefig("spread_spectrum_cdma.png")
plt.close()

# ── 5. ChirpSS ──────────────────────────────────────────────────────────────
chirpss = ChirpSS(spreading_factor=128)
iq_css = chirpss.generate(num_symbols=64, sample_rate=sample_rate, seed=seed)

plot_psd(iq_css, sample_rate, title=f"ChirpSS PSD — {chirpss.label}")
savefig("spread_spectrum_chirpss.png")
plt.close()

# ── 6. PSD Comparison ───────────────────────────────────────────────────────
fig, axes = plt.subplots(3, 2, figsize=(12, 10))
waveforms = [
    ("DSSS-BPSK", iq_bpsk), ("DSSS-QPSK", iq_qpsk),
    ("FHSS", iq_fhss), ("THSS", iq_thss),
    ("CDMA Fwd", iq_fwd), ("ChirpSS", iq_css),
]
nfft = 1024
for ax, (name, iq) in zip(axes.flat, waveforms):
    freqs = np.fft.fftshift(np.fft.fftfreq(nfft, d=1.0 / sample_rate))
    spectrum = np.fft.fftshift(np.fft.fft(iq[:nfft], n=nfft))
    psd = 10 * np.log10(np.abs(spectrum) ** 2 + 1e-12)
    ax.plot(freqs / 1e3, psd, linewidth=0.8)
    ax.set_title(name)
    ax.set_xlabel("Freq (kHz)")
    ax.set_ylabel("dB")
    ax.grid(True, alpha=0.3)
fig.suptitle("Spread Spectrum PSD Comparison", fontsize=14)
fig.tight_layout()
savefig("spread_spectrum_psd_comparison.png")
plt.close()

print("Done — spread spectrum examples saved.")
