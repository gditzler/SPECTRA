# examples/16_wideband_direction_finding.py
"""Example 16 — Wideband Direction-Finding Dataset
===================================================
Level: Intermediate-Advanced

This example shows how to:
  1. Build a WidebandDirectionFindingDataset with 3 co-channel sources
  2. Visualise the multi-antenna wideband spectrogram
  3. Apply MUSIC sub-band processing at each source's center frequency
  4. Compare per-frequency estimated azimuths to ground truth

Run:
    python examples/16_wideband_direction_finding.py
"""

import os
import sys
sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))
sys.path.insert(0, os.path.dirname(__file__))

import numpy as np
import matplotlib.pyplot as plt
import torch

from spectra.arrays import ula
from spectra.datasets import WidebandDirectionFindingDataset
from spectra.algorithms import music, find_peaks_doa
from spectra.waveforms import BPSK, QPSK, QAM16

from plot_helpers import savefig, OUTPUT_DIR

# ── Configuration ──────────────────────────────────────────────────────────────

FREQ_HZ       = 2.4e9
N_ELEMENTS    = 8
SPACING       = 0.5
N_SNAPSHOTS   = 512
SAMPLE_RATE   = 20e6       # 20 MHz wideband
CAPTURE_BW    = 16e6       # usable ±8 MHz
N_SOURCES     = 3
SNR_RANGE     = (15.0, 25.0)
N_SAMPLES     = 100
SEED          = 42
MIN_FREQ_SEP  = 3e6        # 3 MHz minimum spacing
MIN_ANG_SEP   = np.deg2rad(15)

SCAN_DEG = np.linspace(5, 175, 512)
SCAN_RAD = np.deg2rad(SCAN_DEG)

# ── 1. Build Dataset ───────────────────────────────────────────────────────────

arr = ula(num_elements=N_ELEMENTS, spacing=SPACING, frequency=FREQ_HZ)
signal_pool = [BPSK(samples_per_symbol=2), QPSK(samples_per_symbol=2), QAM16(samples_per_symbol=2)]

ds = WidebandDirectionFindingDataset(
    array=arr,
    signal_pool=signal_pool,
    num_signals=N_SOURCES,
    num_snapshots=N_SNAPSHOTS,
    sample_rate=SAMPLE_RATE,
    capture_bandwidth=CAPTURE_BW,
    snr_range=SNR_RANGE,
    azimuth_range=(np.deg2rad(10), np.deg2rad(170)),
    elevation_range=(0.0, 0.0),
    min_freq_separation=MIN_FREQ_SEP,
    min_angular_separation=MIN_ANG_SEP,
    num_samples=N_SAMPLES,
    seed=SEED,
)
print(f"Dataset: {len(ds)} wideband captures, {N_SOURCES} sources each")

# ── 2. Inspect One Sample ──────────────────────────────────────────────────────

data, target = ds[0]
print("\nSample 0 ground truth:")
for k in range(target.num_signals):
    print(f"  Source {k}: az={np.rad2deg(target.azimuths[k]):.1f}°, "
          f"f_c={target.center_freqs[k]/1e6:+.1f} MHz, "
          f"SNR={target.snrs[k]:.1f} dB, label={target.labels[k]}")

# ── 3. Multi-Antenna Spectrogram (element 0) ──────────────────────────────────

iq0 = data[0, 0, :].numpy() + 1j * data[0, 1, :].numpy()
NFFT = 128
hop = 32
freqs_stft = np.fft.fftshift(np.fft.fftfreq(NFFT, d=1.0 / SAMPLE_RATE))
n_frames = (N_SNAPSHOTS - NFFT) // hop + 1
spec = np.zeros((NFFT, n_frames), dtype=complex)
for fi in range(n_frames):
    seg = iq0[fi * hop: fi * hop + NFFT]
    spec[:, fi] = np.fft.fftshift(np.fft.fft(seg * np.hanning(NFFT)))
spec_db = 10 * np.log10(np.abs(spec) ** 2 + 1e-30)

fig, ax = plt.subplots(figsize=(10, 4))
ax.imshow(
    spec_db,
    aspect="auto", origin="lower",
    extent=[0, n_frames, freqs_stft[0] / 1e6, freqs_stft[-1] / 1e6],
    cmap="viridis",
)
for fc in target.center_freqs:
    ax.axhline(fc / 1e6, color="crimson", linestyle="--", linewidth=1.2)
ax.set_xlabel("Frame")
ax.set_ylabel("Frequency (MHz)")
ax.set_title("Wideband Spectrogram — element 0 (sample 0)")
plt.tight_layout()
savefig("16_wideband_spectrogram.png")

# ── 4. Sub-Band MUSIC at Each Source Frequency ────────────────────────────────

X_full = data[:, 0, :].numpy() + 1j * data[:, 1, :].numpy()  # (N, T)

fig, axes = plt.subplots(N_SOURCES, 1, figsize=(9, 3 * N_SOURCES), sharex=True)
for k, ax in enumerate(axes):
    fc = target.center_freqs[k]
    true_az = target.azimuths[k]

    # Frequency-shift down so source k is at DC, then low-pass via zeroed FFT bins
    t = np.arange(N_SNAPSHOTS) / SAMPLE_RATE
    X_shift = X_full * np.exp(-1j * 2 * np.pi * fc * t)[np.newaxis, :]
    BW_sub = 2e6
    fft_len = N_SNAPSHOTS
    sub_bins = int(BW_sub / SAMPLE_RATE * fft_len)
    X_fft = np.fft.fft(X_shift, axis=1)
    X_fft[:, sub_bins // 2: fft_len - sub_bins // 2] = 0
    X_sub = np.fft.ifft(X_fft, axis=1)

    spectrum = music(X_sub, num_sources=1, array=arr, scan_angles=SCAN_RAD)
    peaks = find_peaks_doa(spectrum, SCAN_RAD, num_peaks=1)

    ax.plot(SCAN_DEG, 10 * np.log10(spectrum / spectrum.max()), color="steelblue", linewidth=1.0)
    ax.axvline(np.rad2deg(true_az), color="crimson", linestyle="--", linewidth=1.5,
               label=f"True {np.rad2deg(true_az):.1f}°")
    ax.axvline(np.rad2deg(peaks[0]), color="orange", linestyle=":", linewidth=1.5,
               label=f"Est. {np.rad2deg(peaks[0]):.1f}°")
    ax.set_ylabel(f"Source {k}\n{fc/1e6:+.1f} MHz")
    ax.legend(fontsize=8)
    ax.grid(True, alpha=0.3)
axes[-1].set_xlabel("Azimuth (degrees)")
fig.suptitle("Sub-Band MUSIC per Source")
plt.tight_layout()
savefig("16_subband_music.png")

# ── 5. Wideband Capture Statistics ────────────────────────────────────────────

all_freqs_mhz = []
for i in range(len(ds)):
    _, t = ds[i]
    all_freqs_mhz.extend([f / 1e6 for f in t.center_freqs])

fig, ax = plt.subplots(figsize=(7, 3))
ax.hist(all_freqs_mhz, bins=40, color="steelblue", edgecolor="white")
ax.set_xlabel("Center frequency (MHz relative to DC)")
ax.set_ylabel("Count")
ax.set_title(f"Source frequency distribution ({N_SAMPLES} samples × {N_SOURCES} sources)")
ax.grid(True, axis="y", alpha=0.3)
plt.tight_layout()
savefig("16_freq_distribution.png")

print(f"\nAll figures saved to {OUTPUT_DIR}")
