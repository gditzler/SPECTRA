"""
SPECTRA Example 02: Impairments and Channel Effects
====================================================
Level: Intermediate

Learn how to:
- Apply AWGN at various SNR levels
- Simulate frequency offset and phase noise
- Model multipath fading (Rayleigh/Rician)
- Chain multiple impairments with Compose
- Visualize before/after effects on constellations and spectra
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
from spectra.scene import SignalDescription

# ── Helper: create a SignalDescription for impairments ───────────────────────

def make_desc(label="QPSK", bw=125e3, sample_rate=1e6):
    return SignalDescription(
        t_start=0.0,
        t_stop=1e-3,
        f_low=-bw / 2,
        f_high=bw / 2,
        label=label,
        snr=20.0,
    )

sample_rate = 1e6

# ── 1. AWGN at different SNR levels ─────────────────────────────────────────

waveform = sp.QPSK()
iq_clean = waveform.generate(num_symbols=1024, sample_rate=sample_rate, seed=42)

fig, axes = plt.subplots(1, 4, figsize=(18, 4))
snr_values = [None, 20, 10, 0]
titles = ["Clean", "SNR = 20 dB", "SNR = 10 dB", "SNR = 0 dB"]

for ax, snr, title in zip(axes, snr_values, titles):
    if snr is None:
        iq_plot = iq_clean
    else:
        impairment = sp.AWGN(snr=snr)
        desc = make_desc()
        iq_plot, _ = impairment(iq_clean.copy(), desc, sample_rate=sample_rate)
    pts = iq_plot[:2000]
    ax.scatter(pts.real, pts.imag, s=2, alpha=0.4)
    ax.set_title(title)
    ax.set_aspect("equal")
    ax.grid(True, alpha=0.3)
    ax.set_xlim(-3, 3)
    ax.set_ylim(-3, 3)

fig.suptitle("QPSK Constellation Under AWGN", fontsize=14)
fig.tight_layout()
savefig("02_awgn_snr_comparison.png")

# ── 2. Frequency offset ─────────────────────────────────────────────────────

iq_clean_16qam = sp.QAM16().generate(num_symbols=1024, sample_rate=sample_rate, seed=42)

fig, axes = plt.subplots(1, 3, figsize=(14, 4))
offsets = [0, 500, 5000]

for ax, offset in zip(axes, offsets):
    if offset == 0:
        iq_plot = iq_clean_16qam
    else:
        imp = sp.FrequencyOffset(offset=offset)
        desc = make_desc(label="16QAM")
        iq_plot, _ = imp(iq_clean_16qam.copy(), desc, sample_rate=sample_rate)
    pts = iq_plot[:2000]
    ax.scatter(pts.real, pts.imag, s=2, alpha=0.4)
    ax.set_title(f"Freq Offset = {offset} Hz")
    ax.set_aspect("equal")
    ax.grid(True, alpha=0.3)

fig.suptitle("16QAM Under Frequency Offset", fontsize=14)
fig.tight_layout()
savefig("02_freq_offset.png")

# ── 3. Phase noise ──────────────────────────────────────────────────────────

fig, axes = plt.subplots(1, 3, figsize=(14, 4))
noise_powers = [-50, -30, -15]

for ax, pn in zip(axes, noise_powers):
    imp = sp.PhaseNoise(noise_power_db=pn)
    desc = make_desc(label="16QAM")
    iq_plot, _ = imp(iq_clean_16qam.copy(), desc)
    pts = iq_plot[:2000]
    ax.scatter(pts.real, pts.imag, s=2, alpha=0.4)
    ax.set_title(f"Phase Noise = {pn} dB")
    ax.set_aspect("equal")
    ax.grid(True, alpha=0.3)

fig.suptitle("16QAM Under Phase Noise", fontsize=14)
fig.tight_layout()
savefig("02_phase_noise.png")

# ── 4. IQ Imbalance ─────────────────────────────────────────────────────────

fig, axes = plt.subplots(1, 3, figsize=(14, 4))
imbalances = [(0, 0), (3, 5), (5, 15)]

for ax, (amp, phase) in zip(axes, imbalances):
    if amp == 0 and phase == 0:
        iq_plot = iq_clean_16qam
        ax.set_title("Clean")
    else:
        imp = sp.IQImbalance(amplitude_imbalance_db=amp, phase_imbalance_deg=phase)
        desc = make_desc(label="16QAM")
        iq_plot, _ = imp(iq_clean_16qam.copy(), desc)
        ax.set_title(f"Amp={amp}dB, Phase={phase}deg")
    pts = iq_plot[:2000]
    ax.scatter(pts.real, pts.imag, s=2, alpha=0.4)
    ax.set_aspect("equal")
    ax.grid(True, alpha=0.3)

fig.suptitle("16QAM Under IQ Imbalance", fontsize=14)
fig.tight_layout()
savefig("02_iq_imbalance.png")

# ── 5. Composing multiple impairments ────────────────────────────────────────

channel = sp.Compose([
    sp.AWGN(snr=15),
    sp.FrequencyOffset(max_offset=500),
    sp.PhaseNoise(noise_power_db=-35),
    sp.IQImbalance(amplitude_imbalance_db=1.0, phase_imbalance_deg=3.0),
])

fig, axes = plt.subplots(1, 2, figsize=(10, 5))
for ax, (iq_data, title) in zip(axes, [(iq_clean_16qam, "Clean"), (None, "After Channel")]):
    if iq_data is None:
        desc = make_desc(label="16QAM")
        iq_data, _ = channel(iq_clean_16qam.copy(), desc, sample_rate=sample_rate)
    pts = iq_data[:2000]
    ax.scatter(pts.real, pts.imag, s=3, alpha=0.4)
    ax.set_title(title)
    ax.set_aspect("equal")
    ax.grid(True, alpha=0.3)

fig.suptitle("16QAM — Realistic Channel Model (Compose)", fontsize=14)
fig.tight_layout()
savefig("02_composed_channel.png")

# ── 6. Fading channels ──────────────────────────────────────────────────────

nfft = 1024
freqs = np.fft.fftshift(np.fft.fftfreq(nfft, d=1.0 / sample_rate))
iq_qpsk = sp.QPSK().generate(num_symbols=2048, sample_rate=sample_rate, seed=42)

fig, axes = plt.subplots(1, 3, figsize=(15, 4))
fading_models = [
    ("No Fading", None),
    ("Rayleigh", sp.RayleighFading(num_taps=8, doppler_spread=100.0)),
    ("Rician (K=5)", sp.RicianFading(k_factor=5.0, num_taps=8)),
]

for ax, (name, fading) in zip(axes, fading_models):
    if fading is None:
        iq_f = iq_qpsk
    else:
        desc = make_desc()
        iq_f, _ = fading(iq_qpsk.copy(), desc, sample_rate=sample_rate)
    spectrum = np.fft.fftshift(np.fft.fft(iq_f[:nfft], n=nfft))
    psd = 10 * np.log10(np.abs(spectrum) ** 2 + 1e-12)
    ax.plot(freqs / 1e3, psd, linewidth=0.8)
    ax.set_title(name)
    ax.set_xlabel("Freq (kHz)")
    ax.set_ylabel("Power (dB)")
    ax.grid(True, alpha=0.3)

fig.suptitle("QPSK Under Fading Channels", fontsize=14)
fig.tight_layout()
savefig("02_fading_channels.png")

plt.close("all")
print("\nDone! Check examples/outputs/ for saved figures.")
