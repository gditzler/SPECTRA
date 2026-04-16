"""
SPECTRA Example 01: Basic Waveform Generation
==============================================
Level: Novice

Learn how to:
- Generate digital modulation waveforms (BPSK, QPSK, 16QAM, FSK)
- Visualize IQ time-domain signals
- Plot constellation diagrams
- View power spectral density (PSD)
"""

import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import spectra as sp
from plot_helpers import savefig, plot_iq_time, plot_constellation, plot_psd

# ── 1. Generate a QPSK signal ───────────────────────────────────────────────

waveform = sp.QPSK(samples_per_symbol=8, rolloff=0.35)
sample_rate = 1e6  # 1 MHz
iq = waveform.generate(num_symbols=512, sample_rate=sample_rate, seed=42)

print(f"Waveform: {waveform.label}")
print(f"IQ shape: {iq.shape}, dtype: {iq.dtype}")
print(f"Bandwidth: {waveform.bandwidth(sample_rate) / 1e3:.1f} kHz")

# ── 2. Visualize QPSK ───────────────────────────────────────────────────────

plot_iq_time(iq, title="QPSK — Time Domain")
savefig("01_qpsk_time.png")

plot_constellation(iq, title="QPSK — Constellation")
savefig("01_qpsk_constellation.png")

plot_psd(iq, sample_rate, title="QPSK — Power Spectral Density")
savefig("01_qpsk_psd.png")

# ── 3. Compare multiple modulation schemes ───────────────────────────────────

waveforms = [
    sp.BPSK(),
    sp.QPSK(),
    sp.QAM16(),
    sp.QAM64(),
    sp.PSK8(),
    sp.OOK(),
]

fig, axes = plt.subplots(2, 3, figsize=(14, 9))
for ax, wf in zip(axes.flat, waveforms):
    iq_i = wf.generate(num_symbols=512, sample_rate=sample_rate, seed=42)
    pts = iq_i[: min(2000, len(iq_i))]
    ax.scatter(pts.real, pts.imag, s=2, alpha=0.4)
    ax.set_title(wf.label)
    ax.set_aspect("equal")
    ax.grid(True, alpha=0.3)
fig.suptitle("Constellation Diagrams — Digital Modulation Comparison", fontsize=14)
fig.tight_layout()
savefig("01_constellation_grid.png")

# ── 4. Compare PSD of different waveforms ────────────────────────────────────

waveforms_psd = [
    ("BPSK", sp.BPSK()),
    ("QPSK", sp.QPSK()),
    ("16QAM", sp.QAM16()),
    ("OFDM-64", sp.OFDM()),
    ("FSK", sp.FSK()),
    ("GMSK", sp.GMSK()),
]

fig, axes = plt.subplots(2, 3, figsize=(15, 8))
nfft = 1024
freqs = np.fft.fftshift(np.fft.fftfreq(nfft, d=1.0 / sample_rate))
for ax, (name, wf) in zip(axes.flat, waveforms_psd):
    iq_i = wf.generate(num_symbols=512, sample_rate=sample_rate, seed=42)
    spectrum = np.fft.fftshift(np.fft.fft(iq_i[:nfft], n=nfft))
    psd = 10 * np.log10(np.abs(spectrum) ** 2 + 1e-12)
    ax.plot(freqs / 1e3, psd, linewidth=0.8)
    ax.set_title(name)
    ax.set_xlabel("Freq (kHz)")
    ax.set_ylabel("Power (dB)")
    ax.grid(True, alpha=0.3)
fig.suptitle("Power Spectral Density — Modulation Comparison", fontsize=14)
fig.tight_layout()
savefig("01_psd_grid.png")

# ── 5. Analog modulations: AM and FM ────────────────────────────────────────

analog_waveforms = [
    ("AM-DSB", sp.AMDSB()),
    ("AM-SSB (USB)", sp.AMUSB()),
    ("FM", sp.FM()),
    ("Tone", sp.Tone()),
]

fig, axes = plt.subplots(2, 2, figsize=(12, 8))
for ax, (name, wf) in zip(axes.flat, analog_waveforms):
    iq_i = wf.generate(num_symbols=1024, sample_rate=sample_rate, seed=42)
    spectrum = np.fft.fftshift(np.fft.fft(iq_i[:nfft], n=nfft))
    psd = 10 * np.log10(np.abs(spectrum) ** 2 + 1e-12)
    ax.plot(freqs / 1e3, psd, linewidth=0.8)
    ax.set_title(name)
    ax.set_xlabel("Freq (kHz)")
    ax.set_ylabel("Power (dB)")
    ax.grid(True, alpha=0.3)
fig.suptitle("Analog Modulation Spectra", fontsize=14)
fig.tight_layout()
savefig("01_analog_psd.png")

plt.close("all")
print("\nDone! Check examples/outputs/ for saved figures.")
