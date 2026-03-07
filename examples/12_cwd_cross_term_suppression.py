"""
SPECTRA Example 12: Choi-Williams Distribution — Cross-Term Suppression
=======================================================================
Level: Intermediate

Learn how to:
- Compute the Choi-Williams Distribution (CWD) for multi-component signals
- Compare CWD cross-term suppression against the Wigner-Ville Distribution
- Explore the effect of the sigma parameter on resolution vs. cross-terms
"""

import sys
sys.path.insert(0, ".")

import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

import spectra as sp
from plot_helpers import savefig


# ── 1. Generate a Two-Tone Signal ───────────────────────────────────────────
# Two complex tones at different frequencies produce strong cross-terms in
# the WVD.  The CWD kernel suppresses these artefacts.

sample_rate = 1e6
n_samples = 512
t = np.arange(n_samples) / sample_rate

f1, f2 = 0.1e6, 0.3e6
iq = (np.exp(1j * 2 * np.pi * f1 * t) + np.exp(1j * 2 * np.pi * f2 * t)).astype(
    np.complex64
)

print(f"Two-tone signal: {f1/1e3:.0f} kHz + {f2/1e3:.0f} kHz, {n_samples} samples")


# ── 2. WVD vs CWD Side-by-Side ──────────────────────────────────────────────

nfft = 256
wvd_transform = sp.WVD(nfft=nfft, output_format="magnitude")
cwd_transform = sp.CWD(nfft=nfft, sigma=1.0, output_format="magnitude")

wvd_out = wvd_transform(iq).squeeze(0).numpy()
cwd_out = cwd_transform(iq).squeeze(0).numpy()

fig, axes = plt.subplots(1, 2, figsize=(14, 5))

im0 = axes[0].imshow(
    10 * np.log10(wvd_out.T + 1e-12),
    aspect="auto",
    origin="lower",
    cmap="inferno",
    interpolation="nearest",
)
axes[0].set_title("Wigner-Ville Distribution")
axes[0].set_xlabel("Time Sample")
axes[0].set_ylabel("Frequency Bin")
fig.colorbar(im0, ax=axes[0], label="dB")

im1 = axes[1].imshow(
    10 * np.log10(cwd_out.T + 1e-12),
    aspect="auto",
    origin="lower",
    cmap="inferno",
    interpolation="nearest",
)
axes[1].set_title("Choi-Williams Distribution (σ = 1.0)")
axes[1].set_xlabel("Time Sample")
axes[1].set_ylabel("Frequency Bin")
fig.colorbar(im1, ax=axes[1], label="dB")

fig.suptitle("WVD vs CWD — Two-Tone Cross-Term Suppression", fontsize=14)
fig.tight_layout()
savefig("12_wvd_vs_cwd.png")


# ── 3. Sigma Sweep ──────────────────────────────────────────────────────────
# Show how different sigma values affect cross-term suppression and
# time-frequency resolution.

sigmas = [0.1, 0.5, 1.0, 5.0]

fig, axes = plt.subplots(1, len(sigmas), figsize=(16, 4))
for ax, sigma in zip(axes, sigmas):
    cwd = sp.CWD(nfft=nfft, sigma=sigma, output_format="magnitude")
    out = cwd(iq).squeeze(0).numpy()
    im = ax.imshow(
        10 * np.log10(out.T + 1e-12),
        aspect="auto",
        origin="lower",
        cmap="inferno",
        interpolation="nearest",
    )
    ax.set_title(f"σ = {sigma}")
    ax.set_xlabel("Time Sample")
    if ax is axes[0]:
        ax.set_ylabel("Frequency Bin")
fig.suptitle("CWD — Effect of Sigma on Cross-Term Suppression", fontsize=14)
fig.tight_layout()
savefig("12_cwd_sigma_sweep.png")


# ── 4. Chirp Signal ─────────────────────────────────────────────────────────
# A linear chirp (LFM) signal shows how the CWD tracks instantaneous
# frequency while suppressing artefacts.

chirp_wf = sp.LFM(bandwidth_fraction=0.6, samples_per_pulse=1024)
chirp_iq = chirp_wf.generate(num_symbols=1, sample_rate=sample_rate, seed=42)[:1024]

cwd_chirp = sp.CWD(nfft=256, sigma=1.0, output_format="magnitude")
wvd_chirp = sp.WVD(nfft=256, output_format="magnitude")

cwd_out = cwd_chirp(chirp_iq).squeeze(0).numpy()
wvd_out = wvd_chirp(chirp_iq).squeeze(0).numpy()

fig, axes = plt.subplots(1, 2, figsize=(14, 5))

im0 = axes[0].imshow(
    10 * np.log10(wvd_out.T + 1e-12),
    aspect="auto",
    origin="lower",
    cmap="inferno",
    interpolation="nearest",
)
axes[0].set_title("WVD — Linear Chirp")
axes[0].set_xlabel("Time Sample")
axes[0].set_ylabel("Frequency Bin")
fig.colorbar(im0, ax=axes[0], label="dB")

im1 = axes[1].imshow(
    10 * np.log10(cwd_out.T + 1e-12),
    aspect="auto",
    origin="lower",
    cmap="inferno",
    interpolation="nearest",
)
axes[1].set_title("CWD — Linear Chirp (σ = 1.0)")
axes[1].set_xlabel("Time Sample")
axes[1].set_ylabel("Frequency Bin")
fig.colorbar(im1, ax=axes[1], label="dB")

fig.suptitle("Time-Frequency Tracking — Linear Chirp", fontsize=14)
fig.tight_layout()
savefig("12_cwd_chirp.png")


# ── 5. Output Formats ───────────────────────────────────────────────────────
# Demonstrate the three output format modes.

tone_iq = np.exp(1j * 2 * np.pi * 0.15 * np.arange(256)).astype(np.complex64)

for fmt in ["magnitude", "mag_phase", "real_imag"]:
    cwd = sp.CWD(nfft=64, sigma=1.0, output_format=fmt)
    out = cwd(tone_iq)
    print(f"  output_format={fmt!r:14s} → shape {tuple(out.shape)}, dtype {out.dtype}")


plt.close("all")
print("\nDone! Check examples/outputs/ for saved figures.")
