"""
SPECTRA Example 03: Transforms and Spectrograms
================================================
Level: Intermediate

Learn how to:
- Compute spectrograms with STFT and Spectrogram transforms
- Convert complex IQ to 2-channel format
- Normalize signals
- Apply data augmentations (CutOut, TimeReversal, PatchShuffle, etc.)
- Use DSP utilities for filtering and resampling
"""

import sys
sys.path.insert(0, ".")

import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import spectra as sp
from spectra.utils import dsp
from plot_helpers import savefig

sample_rate = 1e6

# ── 1. STFT Spectrogram ─────────────────────────────────────────────────────

waveforms = [
    ("QPSK", sp.QPSK()),
    ("OFDM-64", sp.OFDM()),
    ("FSK", sp.FSK()),
    ("LFM Chirp", sp.LFM()),
    ("GMSK", sp.GMSK()),
    ("AM-DSB", sp.AMDSB()),
]

fig, axes = plt.subplots(2, 3, figsize=(15, 8))
for ax, (name, wf) in zip(axes.flat, waveforms):
    iq = wf.generate(num_symbols=1024, sample_rate=sample_rate, seed=42)
    spec = dsp.compute_spectrogram(iq, nfft=256, hop=64)
    spec_db = 10 * np.log10(spec + 1e-12)
    ax.imshow(spec_db, aspect="auto", origin="lower", cmap="viridis")
    ax.set_title(name)
    ax.set_xlabel("Time Bin")
    ax.set_ylabel("Frequency Bin")
fig.suptitle("Spectrograms of Different Modulations", fontsize=14)
fig.tight_layout()
savefig("03_spectrogram_grid.png")

# ── 2. ComplexTo2D Transform ─────────────────────────────────────────────────

iq = sp.QPSK().generate(num_symbols=256, sample_rate=sample_rate, seed=42)
c2d = sp.ComplexTo2D()
two_channel = c2d(iq)

print(f"Input shape: {iq.shape}, dtype: {iq.dtype}")
print(f"Output shape: {two_channel.shape}")  # [2, N]

fig, axes = plt.subplots(2, 1, figsize=(10, 4), sharex=True)
axes[0].plot(two_channel[0, :200], linewidth=0.8)
axes[0].set_ylabel("I Channel")
axes[0].set_title("ComplexTo2D — Two-Channel Representation")
axes[0].grid(True, alpha=0.3)
axes[1].plot(two_channel[1, :200], linewidth=0.8, color="tab:orange")
axes[1].set_ylabel("Q Channel")
axes[1].set_xlabel("Sample Index")
axes[1].grid(True, alpha=0.3)
fig.tight_layout()
savefig("03_complex_to_2d.png")

# ── 3. Normalize Transform ──────────────────────────────────────────────────

norm = sp.Normalize()
iq_raw = sp.QAM64().generate(num_symbols=512, sample_rate=sample_rate, seed=42)
iq_normed = norm(iq_raw)

fig, axes = plt.subplots(1, 2, figsize=(12, 4))
for ax, (data, title) in zip(axes, [(iq_raw, "Before Normalize"), (iq_normed, "After Normalize")]):
    ax.plot(data[:200].real, label="I", linewidth=0.8)
    ax.plot(data[:200].imag, label="Q", linewidth=0.8)
    ax.set_title(title)
    ax.legend()
    ax.grid(True, alpha=0.3)
print(f"Before — mean: {iq_raw.mean():.4f}, std: {np.std(iq_raw):.4f}")
print(f"After  — mean: {iq_normed.mean():.4f}, std: {np.std(iq_normed):.4f}")
fig.tight_layout()
savefig("03_normalize.png")

# ── 4. Data Augmentations ───────────────────────────────────────────────────

iq = sp.QPSK().generate(num_symbols=512, sample_rate=sample_rate, seed=42)

augmentations = [
    ("Original", None),
    ("CutOut", sp.CutOut(max_length_fraction=0.15)),
    ("TimeReversal", sp.TimeReversal()),
    ("PatchShuffle", sp.PatchShuffle(num_patches=8)),
    ("RandomDropSamples", sp.RandomDropSamples(drop_rate=0.05, fill="zero")),
    ("AddSlope", sp.AddSlope(max_slope=0.3)),
]

fig, axes = plt.subplots(2, 3, figsize=(15, 8))
for ax, (name, aug) in zip(axes.flat, augmentations):
    if aug is None:
        iq_aug = iq
    else:
        iq_aug = aug(iq.copy())
    spec = dsp.compute_spectrogram(iq_aug, nfft=256, hop=64)
    spec_db = 10 * np.log10(spec + 1e-12)
    ax.imshow(spec_db, aspect="auto", origin="lower", cmap="viridis")
    ax.set_title(name)
fig.suptitle("Data Augmentations — Spectrogram View", fontsize=14)
fig.tight_layout()
savefig("03_augmentations.png")

# ── 5. DSP Utilities: Filter Design ─────────────────────────────────────────

fig, axes = plt.subplots(1, 3, figsize=(15, 4))

# Low-pass filter
lp_taps = dsp.low_pass(num_taps=101, cutoff=0.25)
axes[0].plot(lp_taps, linewidth=0.8)
axes[0].set_title("Low-Pass Filter (cutoff=0.25)")
axes[0].set_xlabel("Tap Index")
axes[0].grid(True, alpha=0.3)

# SRRC filter
srrc = dsp.srrc_taps(num_symbols=10, rolloff=0.35, sps=8)
axes[1].plot(srrc, linewidth=0.8)
axes[1].set_title("SRRC Filter (rolloff=0.35, sps=8)")
axes[1].set_xlabel("Tap Index")
axes[1].grid(True, alpha=0.3)

# Gaussian filter
gauss = dsp.gaussian_taps(bt=0.3, span=4, sps=8)
axes[2].plot(gauss, linewidth=0.8)
axes[2].set_title("Gaussian Filter (BT=0.3)")
axes[2].set_xlabel("Tap Index")
axes[2].grid(True, alpha=0.3)

fig.suptitle("DSP Utilities — Filter Taps", fontsize=14)
fig.tight_layout()
savefig("03_filter_taps.png")

# ── 6. Frequency Shifting ───────────────────────────────────────────────────

iq = sp.QPSK().generate(num_symbols=512, sample_rate=sample_rate, seed=42)
iq_shifted = dsp.frequency_shift(iq, offset=200e3, sample_rate=sample_rate)

nfft = 1024
freqs = np.fft.fftshift(np.fft.fftfreq(nfft, d=1.0 / sample_rate))

fig, axes = plt.subplots(1, 2, figsize=(12, 4))
for ax, (data, title) in zip(axes, [(iq, "Original"), (iq_shifted, "Shifted +200 kHz")]):
    spectrum = np.fft.fftshift(np.fft.fft(data[:nfft], n=nfft))
    psd = 10 * np.log10(np.abs(spectrum) ** 2 + 1e-12)
    ax.plot(freqs / 1e3, psd, linewidth=0.8)
    ax.set_title(title)
    ax.set_xlabel("Freq (kHz)")
    ax.set_ylabel("Power (dB)")
    ax.grid(True, alpha=0.3)
fig.suptitle("DSP Utilities — Frequency Shift", fontsize=14)
fig.tight_layout()
savefig("03_freq_shift.png")

plt.close("all")
print("\nDone! Check examples/outputs/ for saved figures.")
