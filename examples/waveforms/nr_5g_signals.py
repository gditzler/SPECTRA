"""
5G NR Signal Generation
=======================
Level: Advanced

Demonstrate SPECTRA's 5G New Radio waveform generators:
  - NR_OFDM — generic NR OFDM symbol
  - NR_PDSCH — Physical Downlink Shared Channel
  - NR_PUSCH — Physical Uplink Shared Channel
  - NR_PRACH — Physical Random Access Channel
  - NR_SSB — Synchronization Signal Block (PSS + SSS + DMRS)

Run:
    python examples/waveforms/nr_5g_signals.py
"""

import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from spectra.waveforms import NR_OFDM, NR_PDSCH, NR_PUSCH, NR_PRACH, NR_SSB
from plot_helpers import savefig

sample_rate = 30.72e6  # 30.72 MHz (common NR rate)
seed = 42

# ── 1. Generate each NR waveform ────────────────────────────────────────────
nr_waveforms = [
    ("NR OFDM", NR_OFDM()),
    ("NR PDSCH", NR_PDSCH()),
    ("NR PUSCH", NR_PUSCH()),
    ("NR PRACH", NR_PRACH()),
    ("NR SSB", NR_SSB()),
]

fig, axes = plt.subplots(len(nr_waveforms), 2, figsize=(14, 3 * len(nr_waveforms)))
for row, (name, waveform) in enumerate(nr_waveforms):
    iq = waveform.generate(num_symbols=128, sample_rate=sample_rate, seed=seed)
    print(f"{name}: label={waveform.label}, samples={len(iq)}, BW={waveform.bandwidth(sample_rate)/1e6:.2f} MHz")

    # Time domain (first 1000 samples)
    n = min(1000, len(iq))
    axes[row, 0].plot(iq[:n].real, linewidth=0.4)
    axes[row, 0].set_title(f"{name} — Time Domain")
    axes[row, 0].set_ylabel("I")
    axes[row, 0].grid(True, alpha=0.3)

    # Spectrogram
    nfft = 256
    hop = 64
    num_frames = (len(iq) - nfft) // hop
    if num_frames > 0:
        spec = np.array([
            np.abs(np.fft.fftshift(np.fft.fft(
                iq[i * hop:i * hop + nfft] * np.hanning(nfft)
            ))) ** 2
            for i in range(num_frames)
        ])
        axes[row, 1].imshow(
            10 * np.log10(spec.T + 1e-12),
            aspect="auto", origin="lower", cmap="viridis",
        )
    axes[row, 1].set_title(f"{name} — Spectrogram")
    axes[row, 1].set_ylabel("Freq Bin")

axes[-1, 0].set_xlabel("Sample")
axes[-1, 1].set_xlabel("Time Frame")
fig.suptitle("5G New Radio Waveforms", fontsize=14, y=1.01)
fig.tight_layout()
savefig("nr_5g_signals.png")
plt.close()

# ── 2. PSD overlay ──────────────────────────────────────────────────────────
plt.figure(figsize=(10, 5))
nfft = 2048
freqs = np.fft.fftshift(np.fft.fftfreq(nfft, d=1.0 / sample_rate)) / 1e6
for name, waveform in nr_waveforms:
    iq = waveform.generate(num_symbols=128, sample_rate=sample_rate, seed=seed)
    spectrum = np.fft.fftshift(np.fft.fft(iq[:nfft], n=nfft))
    psd = 10 * np.log10(np.abs(spectrum) ** 2 + 1e-12)
    plt.plot(freqs, psd, linewidth=0.8, label=name, alpha=0.8)

plt.xlabel("Frequency (MHz)")
plt.ylabel("Power (dB)")
plt.title("5G NR — PSD Comparison")
plt.legend(fontsize=8)
plt.grid(True, alpha=0.3)
plt.tight_layout()
savefig("nr_5g_psd_comparison.png")
plt.close()

print("Done — 5G NR examples saved.")
