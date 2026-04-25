"""
SPECTRA Example 19: Wigner-Ville Distribution — Time-Frequency Analysis
========================================================================
Level: Intermediate

Learn how to:
- Compute the WVD for single-component and multi-component signals
- Observe cross-terms in multi-component WVD output
- Use different output formats and dB scaling
- Compare WVD and CWD for cross-term suppression
"""

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

import matplotlib
import numpy as np

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import spectra as sp
from plot_helpers import savefig

# ── 1. Single Tone — WVD Energy Concentration ────────────────────────────────

sample_rate = 1e6
n_samples = 512
t = np.arange(n_samples) / sample_rate

f0 = 0.15e6
iq_tone = np.exp(1j * 2 * np.pi * f0 * t).astype(np.complex64)

wvd = sp.WVD(nfft=256, output_format="magnitude", db_scale=True)
out = wvd(iq_tone).squeeze(0).numpy()

fig, ax = plt.subplots(figsize=(8, 5))
ax.imshow(out.T, aspect="auto", origin="lower", cmap="inferno", interpolation="nearest")
ax.set_title(f"WVD — Single Tone at {f0/1e3:.0f} kHz")
ax.set_xlabel("Time Sample")
ax.set_ylabel("Frequency Bin")
fig.tight_layout()
savefig("19_wvd_single_tone.png")
print(f"Single tone WVD: shape {out.shape}")


# ── 2. Two-Tone Signal — Cross-Terms ─────────────────────────────────────────

f1, f2 = 0.1e6, 0.3e6
iq_two_tone = (np.exp(1j * 2 * np.pi * f1 * t) + np.exp(1j * 2 * np.pi * f2 * t)).astype(
    np.complex64
)

wvd_out = sp.WVD(nfft=256, output_format="magnitude")(iq_two_tone).squeeze(0).numpy()
cwd_out = sp.CWD(nfft=256, sigma=1.0, output_format="magnitude")(iq_two_tone).squeeze(0).numpy()

fig, axes = plt.subplots(1, 2, figsize=(14, 5))

axes[0].imshow(
    10 * np.log10(wvd_out.T + 1e-12),
    aspect="auto", origin="lower", cmap="inferno", interpolation="nearest",
)
axes[0].set_title("WVD — Two Tones (cross-terms visible)")
axes[0].set_xlabel("Time Sample")
axes[0].set_ylabel("Frequency Bin")

axes[1].imshow(
    10 * np.log10(cwd_out.T + 1e-12),
    aspect="auto", origin="lower", cmap="inferno", interpolation="nearest",
)
axes[1].set_title("CWD (σ=1.0) — Cross-terms suppressed")
axes[1].set_xlabel("Time Sample")
axes[1].set_ylabel("Frequency Bin")

fig.suptitle("WVD vs CWD — Two-Tone Cross-Term Comparison", fontsize=14)
fig.tight_layout()
savefig("19_wvd_vs_cwd_two_tone.png")


# ── 3. Linear Chirp ──────────────────────────────────────────────────────────

chirp_wf = sp.LFM(bandwidth_fraction=0.6, samples_per_pulse=1024)
chirp_iq = chirp_wf.generate(num_symbols=1, sample_rate=sample_rate, seed=42)[:1024]

wvd_chirp = sp.WVD(nfft=256, output_format="magnitude", db_scale=True)
out_chirp = wvd_chirp(chirp_iq).squeeze(0).numpy()

fig, ax = plt.subplots(figsize=(8, 5))
ax.imshow(out_chirp.T, aspect="auto", origin="lower", cmap="inferno", interpolation="nearest")
ax.set_title("WVD — Linear Chirp")
ax.set_xlabel("Time Sample")
ax.set_ylabel("Frequency Bin")
fig.tight_layout()
savefig("19_wvd_chirp.png")


# ── 4. Output Formats ────────────────────────────────────────────────────────

tone_iq = np.exp(1j * 2 * np.pi * 0.15 * np.arange(256)).astype(np.complex64)

print("\nOutput formats:")
for fmt in ["magnitude", "mag_phase", "real_imag"]:
    wvd = sp.WVD(nfft=64, output_format=fmt)
    out = wvd(tone_iq)
    print(f"  output_format={fmt!r:14s} -> shape {tuple(out.shape)}, dtype {out.dtype}")

# dB scaling
wvd_db = sp.WVD(nfft=64, output_format="magnitude", db_scale=True)
out_db = wvd_db(tone_iq)
print(
    f"  magnitude+db_scale     -> shape {tuple(out_db.shape)}, "
    f"range [{out_db.min():.1f}, {out_db.max():.1f}]"
)


plt.close("all")
print("\nDone! Check examples/outputs/ for saved figures.")
