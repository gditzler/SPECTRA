"""
Advanced Impairments Showcase
=============================
Level: Intermediate

Demonstrate impairments NOT covered in the getting_started examples:
  - ColoredNoise (pink/red)
  - DopplerShift (velocity-based)
  - FrequencyDrift (linear drift)
  - IQImbalance (gain and phase mismatch)
  - DCOffset
  - Quantization (ADC bit depth)
  - SampleRateOffset (PPM error)
  - FractionalDelay and SamplingJitter
  - TDLChannel (3GPP standardized)
  - PassbandRipple and SpectralInversion
  - RappPA and SalehPA (power amplifier distortion)
  - AdjacentChannelInterference
  - IntermodulationProducts

Run:
    python examples/impairments/advanced_impairments.py
"""

import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from spectra.waveforms import QPSK
from spectra.impairments import (
    ColoredNoise, DopplerShift, FrequencyDrift, IQImbalance, DCOffset,
    Quantization, SampleRateOffset, FractionalDelay, SamplingJitter,
    TDLChannel, PassbandRipple, SpectralInversion,
    RappPA, SalehPA, AdjacentChannelInterference, IntermodulationProducts,
)
from spectra.scene import SignalDescription
from plot_helpers import savefig

sample_rate = 1e6
waveform = QPSK(samples_per_symbol=8, rolloff=0.35)
iq_clean = waveform.generate(num_symbols=512, sample_rate=sample_rate, seed=42)
desc = SignalDescription(sample_rate=sample_rate, num_iq_samples=len(iq_clean))


def apply_and_plot(impairment, name, ax_time, ax_const):
    """Apply impairment, plot time domain and constellation."""
    iq_out, _ = impairment(iq_clean.copy(), desc)
    n = min(200, len(iq_out))
    ax_time.plot(iq_out[:n].real, linewidth=0.5)
    ax_time.set_title(name, fontsize=9)
    ax_time.grid(True, alpha=0.3)
    pts = iq_out[:1000]
    ax_const.scatter(pts.real, pts.imag, s=1, alpha=0.3)
    ax_const.set_aspect("equal")
    ax_const.grid(True, alpha=0.3)


# ── 1. Grid of impairments ──────────────────────────────────────────────────
impairments = [
    ("Colored Noise (pink)", ColoredNoise(snr=15.0, color="pink")),
    ("Doppler Shift (100 Hz)", DopplerShift(fd_hz=100.0)),
    ("Frequency Drift (500 Hz)", FrequencyDrift(max_drift=500.0)),
    ("IQ Imbalance (2dB, 10°)", IQImbalance(amplitude_imbalance_db=2.0, phase_imbalance_deg=10.0)),
    ("DC Offset", DCOffset(offset=0.2 + 0.1j)),
    ("Quantization (4-bit)", Quantization(num_bits=4)),
    ("Sample Rate Offset (50 ppm)", SampleRateOffset(ppm=50.0)),
    ("Fractional Delay (0.3)", FractionalDelay(delay=0.3)),
    ("Sampling Jitter (0.05)", SamplingJitter(std_samples=0.05)),
    ("TDL-A Channel", TDLChannel(profile="TDL-A", doppler_hz=5.0)),
    ("Passband Ripple (2dB)", PassbandRipple(max_ripple_db=2.0, num_ripples=5)),
    ("Spectral Inversion", SpectralInversion()),
    ("Rapp PA (p=2)", RappPA(smoothness=2.0, saturation=1.0)),
    ("Saleh PA", SalehPA(alpha_a=2.0, beta_a=1.0, alpha_p=1.0, beta_p=1.0)),
    ("Adj. Channel Intf.", AdjacentChannelInterference(power_db=-10.0)),
    ("IMD3 (IIP3=20dB)", IntermodulationProducts(iip3_db=20.0)),
]

rows = 4
cols = 4
fig, axes = plt.subplots(rows * 2, cols, figsize=(16, rows * 5))
for idx, (name, imp) in enumerate(impairments):
    r = (idx // cols) * 2
    c = idx % cols
    apply_and_plot(imp, name, axes[r, c], axes[r + 1, c])

fig.suptitle("Advanced Impairments: Time Domain (top) & Constellation (bottom)", fontsize=13)
fig.tight_layout()
savefig("advanced_impairments_grid.png")
plt.close()

# ── 2. PA AM/AM curves ──────────────────────────────────────────────────────
input_amp = np.linspace(0, 2.0, 200)
input_signal = input_amp.astype(np.complex128)
dummy_desc = SignalDescription(sample_rate=sample_rate, num_iq_samples=len(input_signal))

rapp = RappPA(smoothness=3.0, saturation=1.0)
saleh = SalehPA(alpha_a=2.0, beta_a=1.0, alpha_p=1.0, beta_p=1.0)

rapp_out, _ = rapp(input_signal.copy(), dummy_desc)
saleh_out, _ = saleh(input_signal.copy(), dummy_desc)

fig, ax = plt.subplots(figsize=(8, 5))
ax.plot(input_amp, np.abs(rapp_out), label="Rapp PA (p=3)", linewidth=1.5)
ax.plot(input_amp, np.abs(saleh_out), label="Saleh PA", linewidth=1.5)
ax.plot(input_amp, input_amp, "--", color="gray", label="Linear", linewidth=1)
ax.set_xlabel("Input Amplitude")
ax.set_ylabel("Output Amplitude")
ax.set_title("Power Amplifier AM/AM Characteristics")
ax.legend()
ax.grid(True, alpha=0.3)
fig.tight_layout()
savefig("advanced_impairments_pa_curves.png")
plt.close()

print("Done — advanced impairments examples saved.")
