"""
SPECTRA Example 07: CSP Feature Visualization
==============================================
Level: Intermediate

Learn how to:
- Compute Spectral Correlation Density (SCD) for different modulations
- Compare Spectral Coherence Function (SCF) between signal types
- Visualize Cyclic Autocorrelation Function (CAF) heatmaps
- Extract and compare higher-order cumulant features
- Use the Rust-backed PSD and energy detector transforms
"""

import sys
sys.path.insert(0, ".")

import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

import spectra as sp
from plot_helpers import savefig


# ── 1. Generate Test Signals ─────────────────────────────────────────────────

sample_rate = 1e6
num_symbols = 512
num_iq = 4096

waveforms = {
    "BPSK": sp.BPSK(),
    "QPSK": sp.QPSK(),
    "16QAM": sp.QAM16(),
    "Noise": sp.Noise(),
}

signals = {}
for name, wf in waveforms.items():
    iq = wf.generate(num_symbols=num_symbols, sample_rate=sample_rate, seed=42)
    signals[name] = iq[:num_iq]

print(f"Generated {len(signals)} signals, each {num_iq} samples at {sample_rate/1e6:.0f} MHz")


# ── 2. SCD Magnitude Grid ────────────────────────────────────────────────────

scd_transform = sp.SCD(nfft=128, n_alpha=128, hop=32, output_format="magnitude")

fig, axes = plt.subplots(2, 2, figsize=(12, 10))
for ax, (name, iq) in zip(axes.flat, signals.items()):
    scd_tensor = scd_transform(iq)  # [1, nfft, n_alpha]
    scd_mag = scd_tensor.squeeze(0).numpy()
    im = ax.imshow(
        10 * np.log10(scd_mag + 1e-12),
        aspect="auto",
        origin="lower",
        cmap="inferno",
        interpolation="nearest",
    )
    ax.set_title(f"SCD — {name}")
    ax.set_xlabel("Cyclic Frequency Bin")
    ax.set_ylabel("Spectral Frequency Bin")
    fig.colorbar(im, ax=ax, label="dB")
fig.suptitle("Spectral Correlation Density Comparison", fontsize=14)
fig.tight_layout()
savefig("07_scd_comparison.png")


# ── 3. SCF Comparison (BPSK vs QPSK) ─────────────────────────────────────────

scf_transform = sp.SCF(nfft=128, n_alpha=128, hop=32, output_format="magnitude")

fig, axes = plt.subplots(1, 2, figsize=(12, 5))
for ax, name in zip(axes, ["BPSK", "QPSK"]):
    scf_tensor = scf_transform(signals[name])  # [1, nfft, n_alpha]
    scf_mag = scf_tensor.squeeze(0).numpy()
    im = ax.imshow(
        scf_mag,
        aspect="auto",
        origin="lower",
        cmap="viridis",
        interpolation="nearest",
    )
    ax.set_title(f"SCF — {name}")
    ax.set_xlabel("Cyclic Frequency Bin")
    ax.set_ylabel("Spectral Frequency Bin")
    fig.colorbar(im, ax=ax, label="Coherence")
fig.suptitle("Spectral Coherence Function — BPSK vs QPSK", fontsize=14)
fig.tight_layout()
savefig("07_scf_comparison.png")


# ── 4. CAF Heatmap for BPSK ──────────────────────────────────────────────────

caf_transform = sp.CAF(n_alpha=128, max_lag=64, output_format="magnitude")

caf_tensor = caf_transform(signals["BPSK"])  # [1, n_alpha, max_lag]
caf_mag = caf_tensor.squeeze(0).numpy()

fig, ax = plt.subplots(figsize=(10, 6))
im = ax.imshow(
    10 * np.log10(caf_mag + 1e-12),
    aspect="auto",
    origin="lower",
    cmap="magma",
    interpolation="nearest",
)
ax.set_title("Cyclic Autocorrelation Function — BPSK")
ax.set_xlabel("Lag (samples)")
ax.set_ylabel("Cyclic Frequency Bin")
fig.colorbar(im, ax=ax, label="dB")
fig.tight_layout()
savefig("07_caf_bpsk.png")


# ── 5. Cumulant Bar Chart ────────────────────────────────────────────────────

cum_transform = sp.Cumulants(max_order=4)
cumulant_labels = ["|C20|", "|C21|", "|C40|", "|C41|", "|C42|"]

cumulant_data = {}
for name, iq in signals.items():
    feats = cum_transform(iq).numpy()
    cumulant_data[name] = feats

x = np.arange(len(cumulant_labels))
width = 0.2
fig, ax = plt.subplots(figsize=(10, 5))
for i, (name, feats) in enumerate(cumulant_data.items()):
    ax.bar(x + i * width, feats, width, label=name)
ax.set_xticks(x + width * 1.5)
ax.set_xticklabels(cumulant_labels)
ax.set_ylabel("Magnitude")
ax.set_title("Cumulant Feature Comparison Across Modulations")
ax.legend()
ax.grid(True, alpha=0.3, axis="y")
fig.tight_layout()
savefig("07_cumulant_comparison.png")


# ── 6. Rust-Backed PSD Comparison ────────────────────────────────────────────

psd_transform = sp.PSD(nfft=256, overlap=128, db_scale=True)

fig, axes = plt.subplots(2, 2, figsize=(12, 8))
for ax, (name, iq) in zip(axes.flat, signals.items()):
    psd_tensor = psd_transform(iq)  # [1, nfft]
    psd_db = psd_tensor.squeeze(0).numpy()
    freqs = np.linspace(-sample_rate / 2, sample_rate / 2, len(psd_db)) / 1e3
    ax.plot(freqs, psd_db, linewidth=0.8)
    ax.set_title(f"PSD — {name}")
    ax.set_xlabel("Frequency (kHz)")
    ax.set_ylabel("Power (dB)")
    ax.grid(True, alpha=0.3)
fig.suptitle("Rust-Backed Welch PSD Comparison", fontsize=14)
fig.tight_layout()
savefig("07_psd_comparison.png")


# ── 7. Energy Detection ──────────────────────────────────────────────────────

# Create a tone signal embedded in noise
tone = sp.Tone(frequency=0.25e6)
tone_iq = tone.generate(num_symbols=num_iq, sample_rate=sample_rate, seed=42)[:num_iq]
noise_iq = sp.Noise().generate(num_symbols=num_iq, sample_rate=sample_rate, seed=99)[:num_iq]
# Mix: tone at 10 dB SNR above noise
tone_power = np.mean(np.abs(tone_iq) ** 2)
noise_power = np.mean(np.abs(noise_iq) ** 2)
snr_linear = 10.0
scale = np.sqrt(tone_power / (noise_power * snr_linear))
mixed = tone_iq + noise_iq * scale

psd_mixed = psd_transform(mixed).squeeze(0).numpy()
detector = sp.EnergyDetector(nfft=256, overlap=128, threshold_db=6.0)
detection_mask = detector(mixed).squeeze(0).numpy()

freqs = np.linspace(-sample_rate / 2, sample_rate / 2, len(psd_mixed)) / 1e3

fig, axes = plt.subplots(2, 1, figsize=(10, 6), sharex=True)
axes[0].plot(freqs, psd_mixed, linewidth=0.8)
axes[0].set_ylabel("Power (dB)")
axes[0].set_title("PSD — Tone + Noise (10 dB SNR)")
axes[0].grid(True, alpha=0.3)

axes[1].fill_between(freqs, detection_mask, alpha=0.5, color="tab:red")
axes[1].set_ylabel("Detection")
axes[1].set_xlabel("Frequency (kHz)")
axes[1].set_title("Energy Detector Output (threshold = 6 dB)")
axes[1].set_ylim(-0.1, 1.1)
axes[1].grid(True, alpha=0.3)
fig.tight_layout()
savefig("07_energy_detection.png")


plt.close("all")
print("\nDone! Check examples/outputs/ for saved figures.")
