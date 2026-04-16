"""
Alignment & Normalization Transforms
=====================================
Level: Intermediate

Demonstrate SPECTRA's alignment transforms for domain-adaptation preprocessing:
  - DCRemove — remove DC offset
  - PowerNormalize — scale to target RMS power
  - AGCNormalize — automatic gain control
  - ClipNormalize — clip outliers and scale
  - SpectralWhitening — flatten frequency response
  - NoiseFloorMatch — match noise floor levels

Run:
    python examples/transforms/alignment_transforms.py
"""

import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from spectra.waveforms import QPSK
from spectra.impairments import DCOffset, AWGN, Compose
from spectra.scene import SignalDescription
from spectra.transforms.alignment import (
    DCRemove, PowerNormalize, AGCNormalize, ClipNormalize,
    SpectralWhitening, NoiseFloorMatch,
)
from plot_helpers import savefig

sample_rate = 1e6
waveform = QPSK(samples_per_symbol=8, rolloff=0.35)
iq_clean = waveform.generate(num_symbols=512, sample_rate=sample_rate, seed=42)
desc = SignalDescription(sample_rate=sample_rate, num_iq_samples=len(iq_clean))

# Apply DC offset + noise to create a "dirty" signal
dirty_pipeline = Compose([DCOffset(offset=0.3 + 0.2j), AWGN(snr=15.0)])
iq_dirty, desc_dirty = dirty_pipeline(iq_clean.copy(), desc)


def plot_before_after(iq_before, iq_after, title_before, title_after, filename):
    """Plot time domain and PSD before/after a transform."""
    fig, axes = plt.subplots(2, 2, figsize=(12, 6))
    n = 300
    nfft = 1024

    # Time domain
    axes[0, 0].plot(iq_before[:n].real, linewidth=0.5, label="I")
    axes[0, 0].plot(iq_before[:n].imag, linewidth=0.5, label="Q")
    axes[0, 0].set_title(title_before)
    axes[0, 0].legend(fontsize=7)
    axes[0, 0].grid(True, alpha=0.3)

    axes[0, 1].plot(iq_after[:n].real, linewidth=0.5, label="I")
    axes[0, 1].plot(iq_after[:n].imag, linewidth=0.5, label="Q")
    axes[0, 1].set_title(title_after)
    axes[0, 1].legend(fontsize=7)
    axes[0, 1].grid(True, alpha=0.3)

    # PSD
    for col, iq in enumerate([iq_before, iq_after]):
        freqs = np.fft.fftshift(np.fft.fftfreq(nfft, d=1.0 / sample_rate))
        spectrum = np.fft.fftshift(np.fft.fft(iq[:nfft], n=nfft))
        psd = 10 * np.log10(np.abs(spectrum) ** 2 + 1e-12)
        axes[1, col].plot(freqs / 1e3, psd, linewidth=0.8)
        axes[1, col].set_xlabel("Freq (kHz)")
        axes[1, col].set_ylabel("dB")
        axes[1, col].grid(True, alpha=0.3)

    fig.tight_layout()
    savefig(filename)
    plt.close()


# ── 1. DCRemove ──────────────────────────────────────────────────────────────
dc_remove = DCRemove()
iq_dc_removed, _ = dc_remove(iq_dirty.copy(), desc_dirty)
plot_before_after(iq_dirty, iq_dc_removed, "Before DCRemove", "After DCRemove",
                  "alignment_dc_remove.png")

# ── 2. PowerNormalize ────────────────────────────────────────────────────────
power_norm = PowerNormalize(target_power_dbfs=-20.0)
iq_pnorm, _ = power_norm(iq_dirty.copy(), desc_dirty)
plot_before_after(iq_dirty, iq_pnorm, "Before PowerNormalize", "After PowerNormalize (-20 dBFS)",
                  "alignment_power_normalize.png")

# ── 3. AGCNormalize ──────────────────────────────────────────────────────────
agc = AGCNormalize(method="rms", target_level=1.0)
iq_agc, _ = agc(iq_dirty.copy(), desc_dirty)
plot_before_after(iq_dirty, iq_agc, "Before AGC", "After AGC (RMS=1.0)",
                  "alignment_agc.png")

# ── 4. ClipNormalize ─────────────────────────────────────────────────────────
clip = ClipNormalize(clip_sigma=2.0)
iq_clipped, _ = clip(iq_dirty.copy(), desc_dirty)
plot_before_after(iq_dirty, iq_clipped, "Before ClipNormalize", "After ClipNormalize (2σ)",
                  "alignment_clip.png")

# ── 5. SpectralWhitening ─────────────────────────────────────────────────────
whiten = SpectralWhitening(smoothing_window=64)
iq_whitened, _ = whiten(iq_dirty.copy(), desc_dirty)
plot_before_after(iq_dirty, iq_whitened, "Before Whitening", "After Spectral Whitening",
                  "alignment_whitening.png")

# ── 6. NoiseFloorMatch ───────────────────────────────────────────────────────
nf_match = NoiseFloorMatch(target_noise_floor_db=-40.0)
iq_nf, _ = nf_match(iq_dirty.copy(), desc_dirty)
plot_before_after(iq_dirty, iq_nf, "Before NoiseFloorMatch", "After NoiseFloorMatch (-40 dB)",
                  "alignment_noise_floor.png")

# ── 7. Summary comparison ────────────────────────────────────────────────────
transforms = [
    ("Original (dirty)", iq_dirty),
    ("DCRemove", iq_dc_removed),
    ("PowerNormalize", iq_pnorm),
    ("AGCNormalize", iq_agc),
    ("ClipNormalize", iq_clipped),
    ("SpectralWhitening", iq_whitened),
    ("NoiseFloorMatch", iq_nf),
]

fig, axes = plt.subplots(1, len(transforms), figsize=(3 * len(transforms), 3))
for ax, (name, iq) in zip(axes, transforms):
    ax.scatter(iq[:500].real, iq[:500].imag, s=1, alpha=0.3)
    ax.set_title(name, fontsize=8)
    ax.set_aspect("equal")
    ax.grid(True, alpha=0.3)

fig.suptitle("Alignment Transforms — Constellation Comparison", fontsize=12)
fig.tight_layout()
savefig("alignment_comparison.png")
plt.close()

print("Done — alignment transform examples saved.")
