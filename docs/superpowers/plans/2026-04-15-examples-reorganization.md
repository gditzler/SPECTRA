# Examples Reorganization & Expansion Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Reorganize existing examples into domain-specific sub-folders and add new example scripts covering uncovered SPECTRA features (antenna elements, advanced impairments, 5G NR waveforms, spread spectrum, protocol signals, environment simulation, alignment transforms, time-frequency analysis, dataset I/O, streaming/curriculum, SNR sweep datasets, and benchmarks).

**Architecture:** Move 19 existing `.py` scripts (and their `.ipynb` counterparts) into 10 sub-folders organized by domain. Add 16 new `.py` example scripts. Keep `plot_helpers.py` at the `examples/` root and update all moved scripts to resolve the import path. Update `README.md` with the new structure.

**Tech Stack:** Python, NumPy, Matplotlib, PyTorch, SPECTRA

---

## Target Directory Layout

```
examples/
├── plot_helpers.py                          # Shared (unchanged, stays at root)
├── outputs/                                 # Generated figures (unchanged)
├── README.md                                # Updated
│
├── getting_started/
│   ├── basic_waveforms.py                   # was 01_basic_waveforms.py
│   ├── impairments_and_channels.py          # was 02_impairments_and_channels.py
│   └── transforms_and_spectrograms.py       # was 03_transforms_and_spectrograms.py
│
├── waveforms/
│   ├── spread_spectrum.py                   # NEW
│   ├── protocol_signals.py                  # NEW
│   └── nr_5g_signals.py                     # NEW
│
├── impairments/
│   └── advanced_impairments.py              # NEW
│
├── transforms/
│   ├── alignment_transforms.py              # NEW
│   └── time_frequency_analysis.py           # NEW
│
├── datasets/
│   ├── narrowband_dataset.py                # was 04_narrowband_dataset.py
│   ├── wideband_scenes.py                   # was 05_wideband_scenes.py
│   ├── folder_and_manifest.py               # NEW
│   ├── snr_sweep_evaluation.py              # NEW
│   ├── augmentation_wrappers.py             # NEW
│   ├── dataset_io.py                        # NEW
│   └── streaming_curriculum.py              # NEW
│
├── classification/
│   ├── full_pipeline.py                     # was 06_full_pipeline.py
│   ├── csp_classification.py                # was 08_csp_classification.py
│   ├── train_narrowband_cnn.py              # was 11_train_narrowband_cnn.py
│   └── resnet_amc.py                        # NEW
│
├── cyclostationary/
│   ├── csp_features.py                      # was 07_csp_features.py
│   ├── s3ca_vs_ssca.py                      # was 09_s3ca_vs_ssca.py
│   ├── cwd_cross_term_suppression.py        # was 12_cwd_cross_term_suppression.py
│   └── wvd_time_frequency.py                # was 19_wvd_time_frequency.py
│
├── antenna_arrays/
│   ├── direction_finding.py                 # was 13_direction_finding.py
│   ├── beamforming.py                       # was 14_beamforming.py
│   ├── wideband_direction_finding.py        # was 16_wideband_direction_finding.py
│   ├── antenna_elements.py                  # NEW
│   └── array_geometries.py                  # NEW
│
├── radar/
│   ├── radar_processing.py                  # was 15_radar_processing.py
│   ├── radar_pipeline.py                    # was 17_radar_pipeline.py
│   └── mti_doppler.py                       # NEW
│
├── communications/
│   └── link_simulator.py                    # was 18_link_simulator.py
│
├── environment/
│   └── propagation_and_links.py             # NEW
│
└── benchmarks/
    └── benchmark_evaluation.py              # NEW
```

## Import Convention

Every script that was moved into a sub-folder needs its `plot_helpers` import updated. The pattern is:

```python
import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from plot_helpers import savefig, plot_iq_time, plot_constellation, plot_psd
```

New scripts use the same pattern. All scripts are run from the repo root via:

```bash
python examples/getting_started/basic_waveforms.py
```

---

## Task 1: Create Sub-Folder Structure

**Files:**
- Create: `examples/getting_started/` (directory)
- Create: `examples/waveforms/` (directory)
- Create: `examples/impairments/` (directory)
- Create: `examples/transforms/` (directory)
- Create: `examples/datasets/` (directory)
- Create: `examples/classification/` (directory)
- Create: `examples/cyclostationary/` (directory)
- Create: `examples/antenna_arrays/` (directory)
- Create: `examples/radar/` (directory)
- Create: `examples/communications/` (directory)
- Create: `examples/environment/` (directory)
- Create: `examples/benchmarks/` (directory)

- [ ] **Step 1: Create all sub-directories**

```bash
cd examples
mkdir -p getting_started waveforms impairments transforms datasets classification cyclostationary antenna_arrays radar communications environment benchmarks
```

- [ ] **Step 2: Commit**

```bash
git add examples/
git commit -m "chore(examples): create sub-folder structure for reorganized examples"
```

---

## Task 2: Move Existing Scripts & Fix Imports

**Files:**
- Move: all 19 existing `.py` and `.ipynb` files to their new sub-folders
- Modify: every moved `.py` file — update `sys.path` and `plot_helpers` import

- [ ] **Step 1: Move all files to their new locations**

```bash
cd examples

# getting_started
mv 01_basic_waveforms.py getting_started/basic_waveforms.py
mv 01_basic_waveforms.ipynb getting_started/basic_waveforms.ipynb
mv 02_impairments_and_channels.py getting_started/impairments_and_channels.py
mv 02_impairments_and_channels.ipynb getting_started/impairments_and_channels.ipynb
mv 03_transforms_and_spectrograms.py getting_started/transforms_and_spectrograms.py
mv 03_transforms_and_spectrograms.ipynb getting_started/transforms_and_spectrograms.ipynb

# datasets
mv 04_narrowband_dataset.py datasets/narrowband_dataset.py
mv 04_narrowband_dataset.ipynb datasets/narrowband_dataset.ipynb
mv 05_wideband_scenes.py datasets/wideband_scenes.py
mv 05_wideband_scenes.ipynb datasets/wideband_scenes.ipynb

# classification
mv 06_full_pipeline.py classification/full_pipeline.py
mv 06_full_pipeline.ipynb classification/full_pipeline.ipynb
mv 08_csp_classification.py classification/csp_classification.py
mv 08_csp_classification.ipynb classification/csp_classification.ipynb
mv 11_train_narrowband_cnn.py classification/train_narrowband_cnn.py

# cyclostationary
mv 07_csp_features.py cyclostationary/csp_features.py
mv 07_csp_features.ipynb cyclostationary/csp_features.ipynb
mv 09_s3ca_vs_ssca.py cyclostationary/s3ca_vs_ssca.py
mv 09_s3ca_vs_ssca.ipynb cyclostationary/s3ca_vs_ssca.ipynb
mv 12_cwd_cross_term_suppression.py cyclostationary/cwd_cross_term_suppression.py
mv 12_cwd_cross_term_suppression.ipynb cyclostationary/cwd_cross_term_suppression.ipynb
mv 19_wvd_time_frequency.py cyclostationary/wvd_time_frequency.py

# antenna_arrays
mv 13_direction_finding.py antenna_arrays/direction_finding.py
mv 13_direction_finding.ipynb antenna_arrays/direction_finding.ipynb
mv 14_beamforming.py antenna_arrays/beamforming.py
mv 14_beamforming.ipynb antenna_arrays/beamforming.ipynb
mv 16_wideband_direction_finding.py antenna_arrays/wideband_direction_finding.py
mv 16_wideband_direction_finding.ipynb antenna_arrays/wideband_direction_finding.ipynb

# radar
mv 15_radar_processing.py radar/radar_processing.py
mv 15_radar_processing.ipynb radar/radar_processing.ipynb
mv 17_radar_pipeline.py radar/radar_pipeline.py
mv 17_radar_pipeline.ipynb radar/radar_pipeline.ipynb

# communications
mv 18_link_simulator.py communications/link_simulator.py
mv 18_link_simulator.ipynb communications/link_simulator.ipynb
```

- [ ] **Step 2: Fix imports in all moved `.py` files**

In every moved `.py` file, replace the existing `sys.path` lines and `from plot_helpers import ...` with:

```python
import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
```

The `from plot_helpers import ...` line stays the same — it resolves from the injected path.

Apply this to each file:
- `getting_started/basic_waveforms.py`
- `getting_started/impairments_and_channels.py`
- `getting_started/transforms_and_spectrograms.py`
- `datasets/narrowband_dataset.py`
- `datasets/wideband_scenes.py`
- `classification/full_pipeline.py`
- `classification/csp_classification.py`
- `classification/train_narrowband_cnn.py`
- `cyclostationary/csp_features.py`
- `cyclostationary/s3ca_vs_ssca.py`
- `cyclostationary/cwd_cross_term_suppression.py`
- `cyclostationary/wvd_time_frequency.py`
- `antenna_arrays/direction_finding.py`
- `antenna_arrays/beamforming.py`
- `antenna_arrays/wideband_direction_finding.py`
- `radar/radar_processing.py`
- `radar/radar_pipeline.py`
- `communications/link_simulator.py`

Each file currently has one of these patterns at the top:

```python
# Pattern A (most files):
import sys
sys.path.insert(0, ".")

# Pattern B (some files like 18):
import sys
import os
sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))
sys.path.insert(0, os.path.dirname(__file__))
```

Replace with the unified pattern:

```python
import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
```

Also update the output directory references in each file's `savefig()` calls. Since `plot_helpers.py` computes `OUTPUT_DIR` from its own `__file__`, saved figures will still go to `examples/outputs/` — no changes needed for the `savefig` calls themselves.

- [ ] **Step 3: Verify one moved script runs**

```bash
python examples/getting_started/basic_waveforms.py
```

Expected: script runs and saves PNGs to `examples/outputs/`.

- [ ] **Step 4: Commit**

```bash
git add -A examples/
git commit -m "refactor(examples): move existing scripts into domain sub-folders"
```

---

## Task 3: New Script — `waveforms/spread_spectrum.py`

**Files:**
- Create: `examples/waveforms/spread_spectrum.py`

- [ ] **Step 1: Write the spread spectrum example**

```python
"""
Spread Spectrum Waveforms
=========================
Level: Intermediate

Demonstrate SPECTRA's spread-spectrum waveform generators:
  - DSSS-BPSK and DSSS-QPSK (direct-sequence)
  - FHSS (frequency-hopping)
  - THSS (time-hopping)
  - CDMA Forward and Reverse links
  - ChirpSS (chirp spread spectrum)

Run:
    python examples/waveforms/spread_spectrum.py
"""

import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from spectra.waveforms import (
    DSSS_BPSK, DSSS_QPSK, FHSS, THSS, CDMA_Forward, CDMA_Reverse, ChirpSS,
)
from plot_helpers import savefig, plot_psd

sample_rate = 1e6
num_symbols = 256
seed = 42

# ── 1. DSSS-BPSK vs DSSS-QPSK ──────────────────────────────────────────────
dsss_bpsk = DSSS_BPSK(chips_per_symbol=31)
dsss_qpsk = DSSS_QPSK(chips_per_symbol=31)

iq_bpsk = dsss_bpsk.generate(num_symbols=num_symbols, sample_rate=sample_rate, seed=seed)
iq_qpsk = dsss_qpsk.generate(num_symbols=num_symbols, sample_rate=sample_rate, seed=seed)

fig, axes = plt.subplots(2, 1, figsize=(10, 6))
axes[0].plot(iq_bpsk[:500].real, linewidth=0.5)
axes[0].set_title(f"DSSS-BPSK — {dsss_bpsk.label}")
axes[0].set_ylabel("In-Phase")
axes[0].grid(True, alpha=0.3)
axes[1].plot(iq_qpsk[:500].real, linewidth=0.5, color="tab:orange")
axes[1].set_title(f"DSSS-QPSK — {dsss_qpsk.label}")
axes[1].set_ylabel("In-Phase")
axes[1].set_xlabel("Sample")
axes[1].grid(True, alpha=0.3)
fig.tight_layout()
savefig("spread_spectrum_dsss.png")
plt.close()

# ── 2. FHSS — Frequency Hopping ─────────────────────────────────────────────
fhss = FHSS(num_hops=16, hop_bandwidth=50e3)
iq_fhss = fhss.generate(num_symbols=num_symbols, sample_rate=sample_rate, seed=seed)

fig, axes = plt.subplots(2, 1, figsize=(10, 6))
nfft = 256
hop_len = len(iq_fhss) // nfft
spec = np.array([
    np.fft.fftshift(np.abs(np.fft.fft(iq_fhss[i * nfft:(i + 1) * nfft])) ** 2)
    for i in range(hop_len)
])
axes[0].imshow(
    10 * np.log10(spec.T + 1e-12), aspect="auto", origin="lower", cmap="viridis",
)
axes[0].set_title(f"FHSS Spectrogram — {fhss.label}")
axes[0].set_ylabel("Frequency Bin")
axes[1].plot(iq_fhss[:1000].real, linewidth=0.5)
axes[1].set_title("FHSS Time Domain")
axes[1].set_xlabel("Sample")
axes[1].grid(True, alpha=0.3)
fig.tight_layout()
savefig("spread_spectrum_fhss.png")
plt.close()

# ── 3. THSS — Time Hopping ──────────────────────────────────────────────────
thss = THSS(num_slots=8, slot_duration_symbols=4)
iq_thss = thss.generate(num_symbols=num_symbols, sample_rate=sample_rate, seed=seed)

plt.figure(figsize=(10, 3))
plt.plot(np.abs(iq_thss[:2000]), linewidth=0.5)
plt.title(f"THSS Envelope — {thss.label}")
plt.xlabel("Sample")
plt.ylabel("|IQ|")
plt.grid(True, alpha=0.3)
plt.tight_layout()
savefig("spread_spectrum_thss.png")
plt.close()

# ── 4. CDMA Forward and Reverse ─────────────────────────────────────────────
cdma_fwd = CDMA_Forward(num_users=4, code_length=64)
cdma_rev = CDMA_Reverse(num_users=4, code_length=64)

iq_fwd = cdma_fwd.generate(num_symbols=num_symbols, sample_rate=sample_rate, seed=seed)
iq_rev = cdma_rev.generate(num_symbols=num_symbols, sample_rate=sample_rate, seed=seed)

fig, axes = plt.subplots(1, 2, figsize=(12, 4))
for ax, iq, label in [(axes[0], iq_fwd, "CDMA Forward"), (axes[1], iq_rev, "CDMA Reverse")]:
    ax.scatter(iq[:500].real, iq[:500].imag, s=1, alpha=0.4)
    ax.set_title(label)
    ax.set_xlabel("I")
    ax.set_ylabel("Q")
    ax.set_aspect("equal")
    ax.grid(True, alpha=0.3)
fig.tight_layout()
savefig("spread_spectrum_cdma.png")
plt.close()

# ── 5. ChirpSS ──────────────────────────────────────────────────────────────
chirpss = ChirpSS(spreading_factor=128)
iq_css = chirpss.generate(num_symbols=64, sample_rate=sample_rate, seed=seed)

plot_psd(iq_css, sample_rate, title=f"ChirpSS PSD — {chirpss.label}")
savefig("spread_spectrum_chirpss.png")
plt.close()

# ── 6. PSD Comparison ───────────────────────────────────────────────────────
fig, axes = plt.subplots(3, 2, figsize=(12, 10))
waveforms = [
    ("DSSS-BPSK", iq_bpsk), ("DSSS-QPSK", iq_qpsk),
    ("FHSS", iq_fhss), ("THSS", iq_thss),
    ("CDMA Fwd", iq_fwd), ("ChirpSS", iq_css),
]
nfft = 1024
for ax, (name, iq) in zip(axes.flat, waveforms):
    freqs = np.fft.fftshift(np.fft.fftfreq(nfft, d=1.0 / sample_rate))
    spectrum = np.fft.fftshift(np.fft.fft(iq[:nfft], n=nfft))
    psd = 10 * np.log10(np.abs(spectrum) ** 2 + 1e-12)
    ax.plot(freqs / 1e3, psd, linewidth=0.8)
    ax.set_title(name)
    ax.set_xlabel("Freq (kHz)")
    ax.set_ylabel("dB")
    ax.grid(True, alpha=0.3)
fig.suptitle("Spread Spectrum PSD Comparison", fontsize=14)
fig.tight_layout()
savefig("spread_spectrum_psd_comparison.png")
plt.close()

print("Done — spread spectrum examples saved.")
```

- [ ] **Step 2: Commit**

```bash
git add examples/waveforms/spread_spectrum.py
git commit -m "feat(examples): add spread spectrum waveforms example"
```

---

## Task 4: New Script — `waveforms/protocol_signals.py`

**Files:**
- Create: `examples/waveforms/protocol_signals.py`

- [ ] **Step 1: Write the protocol signals example**

```python
"""
Protocol & Aviation/Maritime Waveforms
======================================
Level: Intermediate

Demonstrate SPECTRA's protocol waveform generators:
  - ADS-B (Automatic Dependent Surveillance-Broadcast)
  - Mode S (Secondary Surveillance Radar)
  - AIS (Automatic Identification System)
  - ACARS (Aircraft Communications Addressing and Reporting System)
  - DME (Distance Measuring Equipment)
  - ILS Localizer (Instrument Landing System)

Run:
    python examples/waveforms/protocol_signals.py
"""

import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from spectra.waveforms import ADSB, ModeS, AIS, ACARS, DME, ILS_Localizer
from plot_helpers import savefig, plot_psd

sample_rate = 2e6
seed = 42

# ── 1. Generate each protocol waveform ──────────────────────────────────────
protocols = [
    ("ADS-B", ADSB()),
    ("Mode S", ModeS()),
    ("AIS", AIS()),
    ("ACARS", ACARS()),
    ("DME", DME()),
    ("ILS Localizer", ILS_Localizer()),
]

fig, axes = plt.subplots(len(protocols), 2, figsize=(14, 3 * len(protocols)))
for row, (name, waveform) in enumerate(protocols):
    iq = waveform.generate(num_symbols=256, sample_rate=sample_rate, seed=seed)
    print(f"{name}: label={waveform.label}, samples={len(iq)}")

    # Time domain
    n = min(500, len(iq))
    axes[row, 0].plot(iq[:n].real, linewidth=0.6)
    axes[row, 0].plot(iq[:n].imag, linewidth=0.6, alpha=0.7)
    axes[row, 0].set_title(f"{name} — Time Domain")
    axes[row, 0].set_ylabel("Amplitude")
    axes[row, 0].grid(True, alpha=0.3)

    # PSD
    nfft = 1024
    freqs = np.fft.fftshift(np.fft.fftfreq(nfft, d=1.0 / sample_rate))
    spectrum = np.fft.fftshift(np.fft.fft(iq[:nfft], n=nfft))
    psd = 10 * np.log10(np.abs(spectrum) ** 2 + 1e-12)
    axes[row, 1].plot(freqs / 1e3, psd, linewidth=0.8, color="tab:green")
    axes[row, 1].set_title(f"{name} — PSD")
    axes[row, 1].set_ylabel("dB")
    axes[row, 1].grid(True, alpha=0.3)

axes[-1, 0].set_xlabel("Sample")
axes[-1, 1].set_xlabel("Frequency (kHz)")
fig.suptitle("Aviation & Maritime Protocol Waveforms", fontsize=14, y=1.01)
fig.tight_layout()
savefig("protocol_signals.png")
plt.close()

# ── 2. Bandwidth comparison ─────────────────────────────────────────────────
print("\nBandwidth summary:")
for name, waveform in protocols:
    bw = waveform.bandwidth(sample_rate)
    print(f"  {name:15s}: {bw / 1e3:.1f} kHz")

print("\nDone — protocol signal examples saved.")
```

- [ ] **Step 2: Commit**

```bash
git add examples/waveforms/protocol_signals.py
git commit -m "feat(examples): add aviation/maritime protocol signals example"
```

---

## Task 5: New Script — `waveforms/nr_5g_signals.py`

**Files:**
- Create: `examples/waveforms/nr_5g_signals.py`

- [ ] **Step 1: Write the 5G NR signals example**

```python
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
```

- [ ] **Step 2: Commit**

```bash
git add examples/waveforms/nr_5g_signals.py
git commit -m "feat(examples): add 5G NR signal generation example"
```

---

## Task 6: New Script — `impairments/advanced_impairments.py`

**Files:**
- Create: `examples/impairments/advanced_impairments.py`

- [ ] **Step 1: Write the advanced impairments example**

```python
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
    ("TDL-A Channel", TDLChannel(profile="TDL-A", doppler_spread=0.01)),
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
```

- [ ] **Step 2: Commit**

```bash
git add examples/impairments/advanced_impairments.py
git commit -m "feat(examples): add advanced impairments showcase"
```

---

## Task 7: New Script — `transforms/alignment_transforms.py`

**Files:**
- Create: `examples/transforms/alignment_transforms.py`

- [ ] **Step 1: Write the alignment transforms example**

```python
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
  - SpectrogramNormalize — per-frequency-bin normalization

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
from spectra.transforms import SpectrogramNormalize, Spectrogram
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
```

- [ ] **Step 2: Commit**

```bash
git add examples/transforms/alignment_transforms.py
git commit -m "feat(examples): add alignment transforms example"
```

---

## Task 8: New Script — `transforms/time_frequency_analysis.py`

**Files:**
- Create: `examples/transforms/time_frequency_analysis.py`

- [ ] **Step 1: Write the time-frequency analysis example**

```python
"""
Advanced Time-Frequency Analysis
================================
Level: Intermediate

Demonstrate SPECTRA transforms not covered by the cyclostationary examples:
  - ReassignedGabor — reassigned spectrogram for sharper TF localization
  - InstantaneousFrequency — analytic-signal IF estimation
  - AmbiguityFunction — delay-Doppler analysis

Run:
    python examples/transforms/time_frequency_analysis.py
"""

import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from spectra.waveforms import LFM, QPSK
from spectra.transforms import ReassignedGabor, InstantaneousFrequency, AmbiguityFunction, Spectrogram
from spectra.scene import SignalDescription
from plot_helpers import savefig

sample_rate = 1e6
seed = 42

# ── 1. ReassignedGabor vs standard Spectrogram ──────────────────────────────
lfm = LFM(bandwidth=200e3, pulse_width=100e-6)
iq_lfm = lfm.generate(num_symbols=1, sample_rate=sample_rate, seed=seed)
desc = SignalDescription(sample_rate=sample_rate, num_iq_samples=len(iq_lfm))

spec_xform = Spectrogram(nfft=128, hop_size=16)
rg_xform = ReassignedGabor(nfft=128, hop_size=16)

spec_out, _ = spec_xform(iq_lfm, desc)
rg_out, _ = rg_xform(iq_lfm, desc)

fig, axes = plt.subplots(1, 2, figsize=(12, 5))
for ax, data, title in [
    (axes[0], spec_out, "Standard Spectrogram"),
    (axes[1], rg_out, "Reassigned Gabor"),
]:
    if data.ndim == 3:
        data = np.sqrt(data[0] ** 2 + data[1] ** 2)
    ax.imshow(
        10 * np.log10(np.abs(data) + 1e-12),
        aspect="auto", origin="lower", cmap="viridis",
    )
    ax.set_title(title)
    ax.set_xlabel("Time Frame")
    ax.set_ylabel("Frequency Bin")

fig.suptitle("LFM Chirp — Spectrogram vs Reassigned Gabor", fontsize=13)
fig.tight_layout()
savefig("tf_reassigned_gabor.png")
plt.close()

# ── 2. Instantaneous Frequency ──────────────────────────────────────────────
if_xform = InstantaneousFrequency()
iq_qpsk = QPSK(samples_per_symbol=8, rolloff=0.35).generate(
    num_symbols=128, sample_rate=sample_rate, seed=seed)
desc_qpsk = SignalDescription(sample_rate=sample_rate, num_iq_samples=len(iq_qpsk))

if_out, _ = if_xform(iq_lfm, desc)
if_qpsk, _ = if_xform(iq_qpsk, desc_qpsk)

fig, axes = plt.subplots(2, 1, figsize=(10, 6))
axes[0].plot(if_out[:500], linewidth=0.8)
axes[0].set_title("LFM — Instantaneous Frequency")
axes[0].set_ylabel("Freq (normalized)")
axes[0].grid(True, alpha=0.3)

axes[1].plot(if_qpsk[:500], linewidth=0.8, color="tab:orange")
axes[1].set_title("QPSK — Instantaneous Frequency")
axes[1].set_ylabel("Freq (normalized)")
axes[1].set_xlabel("Sample")
axes[1].grid(True, alpha=0.3)
fig.tight_layout()
savefig("tf_instantaneous_frequency.png")
plt.close()

# ── 3. Ambiguity Function ───────────────────────────────────────────────────
af_xform = AmbiguityFunction()
af_lfm, _ = af_xform(iq_lfm, desc)

fig, ax = plt.subplots(figsize=(8, 6))
if af_lfm.ndim == 3:
    af_plot = np.sqrt(af_lfm[0] ** 2 + af_lfm[1] ** 2)
else:
    af_plot = np.abs(af_lfm)
ax.imshow(
    10 * np.log10(af_plot + 1e-12),
    aspect="auto", origin="lower", cmap="hot",
)
ax.set_title("LFM — Ambiguity Function (Delay-Doppler)")
ax.set_xlabel("Doppler Bin")
ax.set_ylabel("Delay Bin")
fig.tight_layout()
savefig("tf_ambiguity_function.png")
plt.close()

print("Done — time-frequency analysis examples saved.")
```

- [ ] **Step 2: Commit**

```bash
git add examples/transforms/time_frequency_analysis.py
git commit -m "feat(examples): add time-frequency analysis example"
```

---

## Task 9: New Script — `datasets/folder_and_manifest.py`

**Files:**
- Create: `examples/datasets/folder_and_manifest.py`

- [ ] **Step 1: Write the folder/manifest dataset example**

```python
"""
Folder & Manifest Dataset Loading
==================================
Level: Intermediate

Demonstrate how to load pre-existing IQ recordings using:
  - SignalFolderDataset — ImageFolder-style (class-per-directory)
  - ManifestDataset — CSV/JSON manifest pointing to files

This example creates a small synthetic dataset on disk, then loads it
back via both dataset types.

Run:
    python examples/datasets/folder_and_manifest.py
"""

import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

import json
import tempfile
import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from spectra.waveforms import BPSK, QPSK, QAM16
from spectra.datasets import SignalFolderDataset, ManifestDataset
from plot_helpers import savefig

sample_rate = 1e6
num_iq = 1024
seed_base = 42

# ── 1. Create synthetic dataset on disk ──────────────────────────────────────
tmpdir = tempfile.mkdtemp(prefix="spectra_folder_example_")
print(f"Temporary dataset directory: {tmpdir}")

waveforms = {"BPSK": BPSK(), "QPSK": QPSK(), "QAM16": QAM16()}
manifest_entries = []

for cls_name, wf in waveforms.items():
    cls_dir = Path(tmpdir) / "folder_dataset" / cls_name
    cls_dir.mkdir(parents=True, exist_ok=True)

    for i in range(10):
        iq = wf.generate(num_symbols=256, sample_rate=sample_rate, seed=seed_base + i)
        filepath = cls_dir / f"sample_{i:03d}.npy"
        np.save(filepath, iq[:num_iq])
        manifest_entries.append({
            "path": str(filepath),
            "label": cls_name,
            "sample_rate": sample_rate,
        })

# Write manifest JSON
manifest_path = Path(tmpdir) / "manifest.json"
with open(manifest_path, "w") as f:
    json.dump(manifest_entries, f, indent=2)

# ── 2. Load via SignalFolderDataset ──────────────────────────────────────────
folder_root = Path(tmpdir) / "folder_dataset"
folder_ds = SignalFolderDataset(
    root=str(folder_root),
    num_iq_samples=num_iq,
)
print(f"\nSignalFolderDataset: {len(folder_ds)} samples")

iq_sample, label = folder_ds[0]
print(f"  Sample shape: {iq_sample.shape}, label: {label}")

# ── 3. Load via ManifestDataset ──────────────────────────────────────────────
manifest_ds = ManifestDataset(
    manifest_path=str(manifest_path),
    num_iq_samples=num_iq,
)
print(f"\nManifestDataset: {len(manifest_ds)} samples")

iq_sample_m, label_m = manifest_ds[0]
print(f"  Sample shape: {iq_sample_m.shape}, label: {label_m}")

# ── 4. Visualize samples from each ──────────────────────────────────────────
fig, axes = plt.subplots(2, 3, figsize=(12, 6))
for col, cls_name in enumerate(waveforms.keys()):
    # Folder dataset
    for idx in range(len(folder_ds)):
        iq, lbl = folder_ds[idx]
        if lbl == col:
            axes[0, col].plot(iq[:200].real, linewidth=0.5)
            axes[0, col].set_title(f"Folder: {cls_name}")
            axes[0, col].grid(True, alpha=0.3)
            break

    # Manifest dataset
    for idx in range(len(manifest_ds)):
        iq, lbl = manifest_ds[idx]
        if lbl == col:
            axes[1, col].plot(iq[:200].real, linewidth=0.5, color="tab:orange")
            axes[1, col].set_title(f"Manifest: {cls_name}")
            axes[1, col].grid(True, alpha=0.3)
            break

fig.suptitle("Folder vs Manifest Dataset Loading", fontsize=13)
fig.tight_layout()
savefig("folder_manifest_datasets.png")
plt.close()

print(f"\nDone — folder/manifest dataset examples saved. Temp dir: {tmpdir}")
```

- [ ] **Step 2: Commit**

```bash
git add examples/datasets/folder_and_manifest.py
git commit -m "feat(examples): add folder and manifest dataset example"
```

---

## Task 10: New Script — `datasets/snr_sweep_evaluation.py`

**Files:**
- Create: `examples/datasets/snr_sweep_evaluation.py`

- [ ] **Step 1: Write the SNR sweep evaluation example**

```python
"""
SNR Sweep Evaluation Dataset
=============================
Level: Advanced

Demonstrate the SNRSweepDataset for structured AMC evaluation:
  - Build a (SNR × class × sample) grid
  - Iterate and visualize samples at each SNR
  - Use evaluate_snr_sweep for per-SNR accuracy

Run:
    python examples/datasets/snr_sweep_evaluation.py
"""

import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from spectra.waveforms import BPSK, QPSK, QAM16, FSK
from spectra.impairments import AWGN
from spectra.datasets import SNRSweepDataset
from spectra.transforms import Spectrogram
from plot_helpers import savefig

sample_rate = 1e6

# ── 1. Build the SNR sweep dataset ──────────────────────────────────────────
waveform_pool = [BPSK(), QPSK(), QAM16(), FSK()]
snr_levels = [-5.0, 0.0, 5.0, 10.0, 15.0, 20.0]

dataset = SNRSweepDataset(
    waveform_pool=waveform_pool,
    snr_levels=snr_levels,
    samples_per_cell=5,
    num_iq_samples=1024,
    sample_rate=sample_rate,
    impairments_fn=lambda snr: AWGN(snr=snr),
    seed=42,
)

print(f"SNRSweepDataset: {len(dataset)} total samples")
print(f"  SNR levels: {snr_levels}")
print(f"  Classes: {len(waveform_pool)}")
print(f"  Samples per cell: 5")
print(f"  Grid: {len(snr_levels)} × {len(waveform_pool)} × 5 = {len(dataset)}")

# ── 2. Visualize samples across SNR levels ───────────────────────────────────
class_names = [w.label for w in waveform_pool]
fig, axes = plt.subplots(len(waveform_pool), len(snr_levels), figsize=(3 * len(snr_levels), 3 * len(waveform_pool)))

spec_xform = Spectrogram(nfft=64, hop_size=16)
from spectra.scene import SignalDescription

for cls_idx in range(len(waveform_pool)):
    for snr_idx, snr in enumerate(snr_levels):
        # Compute index into dataset: (snr_idx * num_classes + cls_idx) * samples_per_cell
        idx = (snr_idx * len(waveform_pool) + cls_idx) * 5
        iq, label = dataset[idx]

        desc = SignalDescription(sample_rate=sample_rate, num_iq_samples=len(iq))
        spec, _ = spec_xform(iq, desc)
        if spec.ndim == 3:
            spec = np.sqrt(spec[0] ** 2 + spec[1] ** 2)

        axes[cls_idx, snr_idx].imshow(
            10 * np.log10(np.abs(spec) + 1e-12),
            aspect="auto", origin="lower", cmap="viridis",
        )
        if cls_idx == 0:
            axes[cls_idx, snr_idx].set_title(f"SNR={snr} dB", fontsize=9)
        if snr_idx == 0:
            axes[cls_idx, snr_idx].set_ylabel(class_names[cls_idx], fontsize=9)
        axes[cls_idx, snr_idx].set_xticks([])
        axes[cls_idx, snr_idx].set_yticks([])

fig.suptitle("SNR Sweep: Spectrograms by Class and SNR", fontsize=13)
fig.tight_layout()
savefig("snr_sweep_grid.png")
plt.close()

print("Done — SNR sweep evaluation example saved.")
```

- [ ] **Step 2: Commit**

```bash
git add examples/datasets/snr_sweep_evaluation.py
git commit -m "feat(examples): add SNR sweep evaluation dataset example"
```

---

## Task 11: New Script — `datasets/augmentation_wrappers.py`

**Files:**
- Create: `examples/datasets/augmentation_wrappers.py`

- [ ] **Step 1: Write the augmentation wrappers example**

```python
"""
MixUp and CutMix Dataset Wrappers
==================================
Level: Intermediate

Demonstrate cross-sample augmentation wrappers:
  - MixUpDataset — blend two random samples with Beta-distributed weight
  - CutMixDataset — replace a random segment of one sample with another

Run:
    python examples/datasets/augmentation_wrappers.py
"""

import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from spectra.waveforms import BPSK, QPSK, QAM16, FSK
from spectra.impairments import AWGN
from spectra.datasets import NarrowbandDataset, MixUpDataset, CutMixDataset
from plot_helpers import savefig

sample_rate = 1e6

# ── 1. Build base dataset ───────────────────────────────────────────────────
base_ds = NarrowbandDataset(
    waveforms=[BPSK(), QPSK(), QAM16(), FSK()],
    num_iq_samples=1024,
    num_samples_per_class=50,
    sample_rate=sample_rate,
    impairments=AWGN(snr=15.0),
    seed=42,
)

# ── 2. Wrap with MixUp and CutMix ───────────────────────────────────────────
mixup_ds = MixUpDataset(base_ds, alpha=0.2)
cutmix_ds = CutMixDataset(base_ds, alpha=1.0)

print(f"Base dataset: {len(base_ds)} samples")
print(f"MixUp dataset: {len(mixup_ds)} samples")
print(f"CutMix dataset: {len(cutmix_ds)} samples")

# ── 3. Compare original vs augmented ────────────────────────────────────────
fig, axes = plt.subplots(3, 4, figsize=(14, 8))
for col in range(4):
    idx = col * 50  # first sample of each class
    iq_base, lbl_base = base_ds[idx]
    iq_mixup, lbl_mixup = mixup_ds[idx]
    iq_cutmix, lbl_cutmix = cutmix_ds[idx]

    n = 200
    axes[0, col].plot(iq_base[:n].real, linewidth=0.5)
    axes[0, col].set_title(f"Original (cls={lbl_base})", fontsize=9)
    axes[0, col].grid(True, alpha=0.3)

    axes[1, col].plot(iq_mixup[:n].real, linewidth=0.5, color="tab:green")
    axes[1, col].set_title("MixUp", fontsize=9)
    axes[1, col].grid(True, alpha=0.3)

    axes[2, col].plot(iq_cutmix[:n].real, linewidth=0.5, color="tab:red")
    axes[2, col].set_title("CutMix", fontsize=9)
    axes[2, col].grid(True, alpha=0.3)

fig.suptitle("MixUp vs CutMix Augmentation", fontsize=13)
fig.tight_layout()
savefig("augmentation_wrappers.png")
plt.close()

print("Done — augmentation wrappers example saved.")
```

- [ ] **Step 2: Commit**

```bash
git add examples/datasets/augmentation_wrappers.py
git commit -m "feat(examples): add MixUp and CutMix dataset wrappers example"
```

---

## Task 12: New Script — `datasets/dataset_io.py`

**Files:**
- Create: `examples/datasets/dataset_io.py`

- [ ] **Step 1: Write the dataset I/O example**

```python
"""
Dataset I/O and Export
======================
Level: Intermediate

Demonstrate SPECTRA's dataset writing and export utilities:
  - DatasetWriter — batch-generate and save a dataset
  - export_dataset_to_folder — export to class-per-directory structure
  - NumpyWriter — write individual IQ files as .npy

Run:
    python examples/datasets/dataset_io.py
"""

import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

import tempfile
import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from spectra.waveforms import BPSK, QPSK, QAM16
from spectra.impairments import AWGN
from spectra.datasets import NarrowbandDataset
from spectra.utils.writer import DatasetWriter
from spectra.utils.file_handlers import export_dataset_to_folder, NumpyWriter
from plot_helpers import savefig

sample_rate = 1e6

# ── 1. Build a small dataset ────────────────────────────────────────────────
dataset = NarrowbandDataset(
    waveforms=[BPSK(), QPSK(), QAM16()],
    num_iq_samples=1024,
    num_samples_per_class=20,
    sample_rate=sample_rate,
    impairments=AWGN(snr=15.0),
    seed=42,
)
print(f"Built dataset: {len(dataset)} samples")

# ── 2. Export to folder structure ────────────────────────────────────────────
tmpdir = tempfile.mkdtemp(prefix="spectra_export_")
export_dir = Path(tmpdir) / "exported"

export_dataset_to_folder(
    dataset=dataset,
    output_dir=str(export_dir),
    writer=NumpyWriter(),
)

# List exported structure
class_dirs = sorted(p for p in export_dir.iterdir() if p.is_dir())
print(f"\nExported to: {export_dir}")
for d in class_dirs:
    files = list(d.glob("*.npy"))
    print(f"  {d.name}/: {len(files)} files")

# ── 3. Verify by loading back ────────────────────────────────────────────────
sample_file = list(class_dirs[0].glob("*.npy"))[0]
iq_loaded = np.load(sample_file)
print(f"\nLoaded back: {sample_file.name}, shape={iq_loaded.shape}, dtype={iq_loaded.dtype}")

# ── 4. Visualize original vs loaded ─────────────────────────────────────────
iq_original, _ = dataset[0]

fig, axes = plt.subplots(1, 2, figsize=(10, 3))
axes[0].plot(iq_original[:200].real, linewidth=0.5)
axes[0].set_title("Original (from dataset)")
axes[0].grid(True, alpha=0.3)
axes[1].plot(iq_loaded[:200].real, linewidth=0.5, color="tab:orange")
axes[1].set_title("Loaded (from .npy export)")
axes[1].grid(True, alpha=0.3)
fig.suptitle("Dataset Export & Reload Verification", fontsize=12)
fig.tight_layout()
savefig("dataset_io_verification.png")
plt.close()

print(f"\nDone — dataset I/O example saved. Temp dir: {tmpdir}")
```

- [ ] **Step 2: Commit**

```bash
git add examples/datasets/dataset_io.py
git commit -m "feat(examples): add dataset I/O and export example"
```

---

## Task 13: New Script — `datasets/streaming_curriculum.py`

**Files:**
- Create: `examples/datasets/streaming_curriculum.py`

- [ ] **Step 1: Write the streaming/curriculum example**

```python
"""
Streaming DataLoader with Curriculum Learning
==============================================
Level: Advanced

Demonstrate epoch-aware data generation with difficulty progression:
  - CurriculumSchedule — linearly interpolate SNR over training
  - StreamingDataLoader — fresh deterministic DataLoader per epoch

Run:
    python examples/datasets/streaming_curriculum.py
"""

import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from spectra.waveforms import BPSK, QPSK, QAM16, FSK
from spectra.impairments import AWGN
from spectra.datasets import NarrowbandDataset
from spectra.curriculum import CurriculumSchedule
from spectra.streaming import StreamingDataLoader
from plot_helpers import savefig

sample_rate = 1e6

# ── 1. Define curriculum schedule ────────────────────────────────────────────
curriculum = CurriculumSchedule(
    snr_range=(20.0, 0.0),  # start easy (20 dB) → hard (0 dB)
)

print("Curriculum progression:")
for progress in [0.0, 0.25, 0.5, 0.75, 1.0]:
    params = curriculum.at(progress)
    print(f"  progress={progress:.0%}: {params}")


# ── 2. Dataset factory ──────────────────────────────────────────────────────
def make_dataset(params):
    snr = params.get("snr", 15.0)
    return NarrowbandDataset(
        waveforms=[BPSK(), QPSK(), QAM16(), FSK()],
        num_iq_samples=1024,
        num_samples_per_class=25,
        sample_rate=sample_rate,
        impairments=AWGN(snr=snr),
        seed=0,  # StreamingDataLoader overrides this per epoch
    )


# ── 3. Create StreamingDataLoader ────────────────────────────────────────────
num_epochs = 5
loader = StreamingDataLoader(
    dataset_factory=make_dataset,
    base_seed=42,
    num_epochs=num_epochs,
    curriculum=curriculum,
    batch_size=16,
    num_workers=0,
)

# ── 4. Iterate and collect stats ─────────────────────────────────────────────
epoch_snrs = []
epoch_powers = []

for epoch_idx in range(num_epochs):
    dl = loader.epoch(epoch_idx)
    batch_powers = []
    for batch_iq, batch_labels in dl:
        power = (batch_iq.abs() ** 2).mean().item()
        batch_powers.append(power)

    progress = epoch_idx / max(num_epochs - 1, 1)
    params = curriculum.at(progress)
    epoch_snrs.append(params.get("snr", 15.0))
    epoch_powers.append(np.mean(batch_powers))
    print(f"Epoch {epoch_idx}: SNR={epoch_snrs[-1]:.1f} dB, mean power={epoch_powers[-1]:.4f}")

# ── 5. Plot curriculum progression ───────────────────────────────────────────
fig, ax1 = plt.subplots(figsize=(8, 4))
ax1.plot(range(num_epochs), epoch_snrs, "o-", color="tab:blue", label="SNR (dB)")
ax1.set_xlabel("Epoch")
ax1.set_ylabel("SNR (dB)", color="tab:blue")
ax1.tick_params(axis="y", labelcolor="tab:blue")
ax1.grid(True, alpha=0.3)

ax2 = ax1.twinx()
ax2.plot(range(num_epochs), epoch_powers, "s--", color="tab:orange", label="Mean Power")
ax2.set_ylabel("Mean Power", color="tab:orange")
ax2.tick_params(axis="y", labelcolor="tab:orange")

fig.suptitle("Curriculum Learning: SNR Progression Over Epochs", fontsize=12)
fig.tight_layout()
savefig("streaming_curriculum.png")
plt.close()

print("Done — streaming/curriculum example saved.")
```

- [ ] **Step 2: Commit**

```bash
git add examples/datasets/streaming_curriculum.py
git commit -m "feat(examples): add streaming dataloader with curriculum example"
```

---

## Task 14: New Script — `classification/resnet_amc.py`

**Files:**
- Create: `examples/classification/resnet_amc.py`

- [ ] **Step 1: Write the ResNet AMC example**

```python
"""
ResNet AMC Classifier on Spectrograms
======================================
Level: Advanced

Train a 2D ResNetAMC on spectrogram features (contrast with the 1D CNNAMC
in train_narrowband_cnn.py).

  - Generate a NarrowbandDataset with STFT spectrogram transform
  - Train ResNetAMC for a few epochs
  - Evaluate with confusion matrix

Run:
    python examples/classification/resnet_amc.py
"""

import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

import torch
import torch.nn as nn
from torch.utils.data import DataLoader

from spectra.waveforms import BPSK, QPSK, QAM16, FSK
from spectra.impairments import AWGN
from spectra.datasets import NarrowbandDataset
from spectra.transforms import Spectrogram, SpectrogramNormalize
from spectra.models import ResNetAMC
from spectra.metrics import confusion_matrix, accuracy
from plot_helpers import savefig

sample_rate = 1e6
num_classes = 4
waveforms = [BPSK(), QPSK(), QAM16(), FSK()]

# ── 1. Build spectrogram dataset ────────────────────────────────────────────
spec_transform = Spectrogram(nfft=64, hop_size=16)

train_ds = NarrowbandDataset(
    waveforms=waveforms,
    num_iq_samples=1024,
    num_samples_per_class=100,
    sample_rate=sample_rate,
    impairments=AWGN(snr=15.0),
    transform=spec_transform,
    seed=42,
)

test_ds = NarrowbandDataset(
    waveforms=waveforms,
    num_iq_samples=1024,
    num_samples_per_class=25,
    sample_rate=sample_rate,
    impairments=AWGN(snr=15.0),
    transform=spec_transform,
    seed=999,
)

# Determine spectrogram shape from a sample
sample_spec, _ = train_ds[0]
if isinstance(sample_spec, np.ndarray):
    sample_spec = torch.from_numpy(sample_spec)
print(f"Spectrogram shape: {sample_spec.shape}")
in_channels = sample_spec.shape[0] if sample_spec.ndim == 3 else 1

# ── 2. DataLoaders ──────────────────────────────────────────────────────────
train_loader = DataLoader(train_ds, batch_size=32, shuffle=True, num_workers=0)
test_loader = DataLoader(test_ds, batch_size=32, shuffle=False, num_workers=0)

# ── 3. Model, loss, optimizer ────────────────────────────────────────────────
model = ResNetAMC(num_classes=num_classes, in_channels=in_channels)
criterion = nn.CrossEntropyLoss()
optimizer = torch.optim.Adam(model.parameters(), lr=1e-3)

print(f"\nResNetAMC: {sum(p.numel() for p in model.parameters()):,} parameters")

# ── 4. Train ─────────────────────────────────────────────────────────────────
num_epochs = 10
train_losses = []

for epoch in range(num_epochs):
    model.train()
    epoch_loss = 0.0
    for batch_x, batch_y in train_loader:
        if isinstance(batch_x, np.ndarray):
            batch_x = torch.from_numpy(batch_x)
        batch_x = batch_x.float()
        if batch_x.ndim == 2:
            batch_x = batch_x.unsqueeze(1)

        optimizer.zero_grad()
        logits = model(batch_x)
        loss = criterion(logits, batch_y)
        loss.backward()
        optimizer.step()
        epoch_loss += loss.item()

    avg_loss = epoch_loss / len(train_loader)
    train_losses.append(avg_loss)
    if (epoch + 1) % 2 == 0:
        print(f"  Epoch {epoch+1}/{num_epochs}: loss={avg_loss:.4f}")

# ── 5. Evaluate ──────────────────────────────────────────────────────────────
model.eval()
all_preds, all_labels = [], []
with torch.no_grad():
    for batch_x, batch_y in test_loader:
        if isinstance(batch_x, np.ndarray):
            batch_x = torch.from_numpy(batch_x)
        batch_x = batch_x.float()
        if batch_x.ndim == 2:
            batch_x = batch_x.unsqueeze(1)
        preds = model(batch_x).argmax(dim=1)
        all_preds.extend(preds.numpy())
        all_labels.extend(batch_y.numpy())

all_preds = np.array(all_preds)
all_labels = np.array(all_labels)
acc = accuracy(all_labels, all_preds)
cm = confusion_matrix(all_labels, all_preds, num_classes)
print(f"\nTest accuracy: {acc:.1%}")

# ── 6. Plot ──────────────────────────────────────────────────────────────────
fig, axes = plt.subplots(1, 2, figsize=(12, 4))

axes[0].plot(train_losses, "o-")
axes[0].set_xlabel("Epoch")
axes[0].set_ylabel("Loss")
axes[0].set_title("Training Loss")
axes[0].grid(True, alpha=0.3)

class_names = [w.label for w in waveforms]
im = axes[1].imshow(cm, cmap="Blues")
axes[1].set_xticks(range(num_classes))
axes[1].set_yticks(range(num_classes))
axes[1].set_xticklabels(class_names, rotation=45, fontsize=8)
axes[1].set_yticklabels(class_names, fontsize=8)
axes[1].set_xlabel("Predicted")
axes[1].set_ylabel("True")
axes[1].set_title(f"Confusion Matrix (Acc={acc:.1%})")
for i in range(num_classes):
    for j in range(num_classes):
        axes[1].text(j, i, f"{cm[i, j]}", ha="center", va="center", fontsize=9)
plt.colorbar(im, ax=axes[1])

fig.suptitle("ResNetAMC on Spectrograms", fontsize=13)
fig.tight_layout()
savefig("resnet_amc.png")
plt.close()

print("Done — ResNet AMC example saved.")
```

- [ ] **Step 2: Commit**

```bash
git add examples/classification/resnet_amc.py
git commit -m "feat(examples): add ResNet AMC spectrogram classifier example"
```

---

## Task 15: New Script — `antenna_arrays/antenna_elements.py`

**Files:**
- Create: `examples/antenna_arrays/antenna_elements.py`

- [ ] **Step 1: Write the antenna elements example**

```python
"""
Antenna Element Radiation Patterns
===================================
Level: Intermediate

Demonstrate all SPECTRA antenna element types and their radiation patterns:
  - IsotropicElement — unit gain everywhere
  - ShortDipoleElement — sin(theta) pattern
  - HalfWaveDipoleElement — cos(pi/2·cos(theta))/sin(theta)
  - CosinePowerElement — cosine^n patch pattern
  - YagiElement — directional multi-element

Run:
    python examples/antenna_arrays/antenna_elements.py
"""

import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from spectra.antennas import (
    IsotropicElement, ShortDipoleElement, HalfWaveDipoleElement,
    CosinePowerElement, YagiElement,
)
from plot_helpers import savefig

freq = 1e9  # 1 GHz

# ── 1. Create antenna elements ──────────────────────────────────────────────
elements = [
    ("Isotropic", IsotropicElement(frequency=freq)),
    ("Short Dipole (z)", ShortDipoleElement(axis="z", frequency=freq)),
    ("Half-Wave Dipole", HalfWaveDipoleElement(axis="z", frequency=freq)),
    ("Cosine Power (n=1.5)", CosinePowerElement(exponent=1.5, frequency=freq)),
    ("Cosine Power (n=4)", CosinePowerElement(exponent=4.0, frequency=freq)),
    ("Yagi (3 elem)", YagiElement(n_elements=3, frequency=freq)),
    ("Yagi (5 elem)", YagiElement(n_elements=5, frequency=freq)),
]

# ── 2. Plot azimuth patterns (elevation=0) ──────────────────────────────────
azimuths = np.linspace(-180, 180, 361)
elevation = 0.0

fig, axes = plt.subplots(2, 4, figsize=(16, 8), subplot_kw={"projection": "polar"})
axes_flat = axes.flat

for idx, (name, element) in enumerate(elements):
    ax = axes_flat[idx]
    gains = np.array([
        element.pattern(np.deg2rad(az), np.deg2rad(elevation))
        for az in azimuths
    ])
    gains_db = 10 * np.log10(np.maximum(gains, 1e-12))
    gains_db = np.maximum(gains_db, -30)  # clip for display
    ax.plot(np.deg2rad(azimuths), gains_db + 30, linewidth=1.5)  # shift so 0 dB is at radius 30
    ax.set_title(name, fontsize=9, pad=12)
    ax.set_rticks([0, 10, 20, 30])
    ax.set_yticklabels(["-30", "-20", "-10", "0 dB"], fontsize=6)

# Hide unused subplot
axes_flat[-1].set_visible(False)

fig.suptitle("Antenna Element Azimuth Patterns (elevation=0°)", fontsize=14)
fig.tight_layout()
savefig("antenna_elements_azimuth.png")
plt.close()

# ── 3. Elevation cut (azimuth=0) ────────────────────────────────────────────
elevations = np.linspace(-90, 90, 181)
azimuth = 0.0

fig, ax = plt.subplots(figsize=(10, 5))
for name, element in elements:
    gains = np.array([
        element.pattern(np.deg2rad(azimuth), np.deg2rad(el))
        for el in elevations
    ])
    gains_db = 10 * np.log10(np.maximum(gains, 1e-12))
    ax.plot(elevations, gains_db, linewidth=1.2, label=name)

ax.set_xlabel("Elevation (degrees)")
ax.set_ylabel("Gain (dB)")
ax.set_title("Elevation Pattern Cut (azimuth=0°)")
ax.legend(fontsize=8)
ax.grid(True, alpha=0.3)
ax.set_ylim([-30, 15])
fig.tight_layout()
savefig("antenna_elements_elevation.png")
plt.close()

print("Done — antenna element examples saved.")
```

- [ ] **Step 2: Commit**

```bash
git add examples/antenna_arrays/antenna_elements.py
git commit -m "feat(examples): add antenna element radiation patterns example"
```

---

## Task 16: New Script — `antenna_arrays/array_geometries.py`

**Files:**
- Create: `examples/antenna_arrays/array_geometries.py`

- [ ] **Step 1: Write the array geometries example**

```python
"""
Array Geometries and Calibration
================================
Level: Intermediate

Demonstrate SPECTRA array construction and calibration:
  - ULA — Uniform Linear Array
  - UCA — Uniform Circular Array
  - Rectangular — 2D planar array
  - CalibrationErrors — per-element gain/phase offsets
  - Steering vector computation

Run:
    python examples/antenna_arrays/array_geometries.py
"""

import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from spectra.antennas import IsotropicElement
from spectra.arrays import ula, uca, rectangular, AntennaArray, CalibrationErrors
from plot_helpers import savefig

freq = 1e9

# ── 1. Create three array geometries ────────────────────────────────────────
element = IsotropicElement(frequency=freq)
arr_ula = ula(num_elements=8, element=element, frequency=freq, spacing_wavelengths=0.5)
arr_uca = uca(num_elements=8, element=element, frequency=freq, radius_wavelengths=1.0)
arr_rect = rectangular(
    x_elements=4, y_elements=3, element=element, frequency=freq,
    x_spacing_wavelengths=0.5, y_spacing_wavelengths=0.5,
)

# ── 2. Plot array geometries ────────────────────────────────────────────────
fig, axes = plt.subplots(1, 3, figsize=(15, 4))
for ax, arr, name in [
    (axes[0], arr_ula, f"ULA ({arr_ula.num_elements} elements)"),
    (axes[1], arr_uca, f"UCA ({arr_uca.num_elements} elements)"),
    (axes[2], arr_rect, f"Rectangular ({arr_rect.num_elements} elements)"),
]:
    pos = arr.positions
    ax.scatter(pos[:, 0], pos[:, 1], s=100, zorder=3)
    for i, (x, y) in enumerate(pos[:, :2]):
        ax.annotate(f"{i}", (x, y), textcoords="offset points", xytext=(0, 8),
                    ha="center", fontsize=8)
    ax.set_title(name)
    ax.set_xlabel("x (m)")
    ax.set_ylabel("y (m)")
    ax.set_aspect("equal")
    ax.grid(True, alpha=0.3)

fig.suptitle("Array Geometries", fontsize=13)
fig.tight_layout()
savefig("array_geometries.png")
plt.close()

# ── 3. Steering vectors ─────────────────────────────────────────────────────
azimuths = np.linspace(-90, 90, 181)
fig, axes = plt.subplots(1, 3, figsize=(15, 4))
for ax, arr, name in [
    (axes[0], arr_ula, "ULA"),
    (axes[1], arr_uca, "UCA"),
    (axes[2], arr_rect, "Rectangular"),
]:
    beampattern = np.zeros(len(azimuths))
    # Steer to 0° (broadside)
    w = arr.steering_vector(azimuth=np.deg2rad(0), elevation=0.0)
    w = w / np.linalg.norm(w)
    for i, az in enumerate(azimuths):
        a = arr.steering_vector(azimuth=np.deg2rad(az), elevation=0.0)
        beampattern[i] = np.abs(w.conj() @ a) ** 2

    bp_db = 10 * np.log10(beampattern / beampattern.max() + 1e-12)
    ax.plot(azimuths, bp_db, linewidth=1.2)
    ax.set_title(f"{name} — Broadside Beam")
    ax.set_xlabel("Azimuth (°)")
    ax.set_ylabel("dB")
    ax.set_ylim([-30, 1])
    ax.grid(True, alpha=0.3)

fig.suptitle("Conventional Beamforming (Broadside Steering)", fontsize=13)
fig.tight_layout()
savefig("array_steering_vectors.png")
plt.close()

# ── 4. Calibration errors ───────────────────────────────────────────────────
rng = np.random.default_rng(42)
cal = CalibrationErrors.random(num_elements=8, gain_std_db=1.0, phase_std_rad=0.1, rng=rng)

print("Calibration errors:")
print(f"  Gain offsets (dB): {cal.gain_offsets_db}")
print(f"  Phase offsets (rad): {cal.phase_offsets_rad}")

# Compare ideal vs calibrated steering
sv_ideal = arr_ula.steering_vector(azimuth=np.deg2rad(30), elevation=0.0)
sv_cal = cal.apply(sv_ideal)

fig, axes = plt.subplots(1, 2, figsize=(10, 4))
axes[0].stem(range(8), np.abs(sv_ideal), linefmt="b-", markerfmt="bo", basefmt="gray")
axes[0].stem(range(8), np.abs(sv_cal), linefmt="r--", markerfmt="rs", basefmt="gray")
axes[0].set_title("Magnitude: Ideal (blue) vs Calibrated (red)")
axes[0].set_xlabel("Element")
axes[0].grid(True, alpha=0.3)

axes[1].stem(range(8), np.angle(sv_ideal), linefmt="b-", markerfmt="bo", basefmt="gray")
axes[1].stem(range(8), np.angle(sv_cal), linefmt="r--", markerfmt="rs", basefmt="gray")
axes[1].set_title("Phase: Ideal (blue) vs Calibrated (red)")
axes[1].set_xlabel("Element")
axes[1].set_ylabel("Phase (rad)")
axes[1].grid(True, alpha=0.3)

fig.suptitle("Effect of Calibration Errors on Steering Vector", fontsize=12)
fig.tight_layout()
savefig("array_calibration.png")
plt.close()

print("Done — array geometry examples saved.")
```

- [ ] **Step 2: Commit**

```bash
git add examples/antenna_arrays/array_geometries.py
git commit -m "feat(examples): add array geometries and calibration example"
```

---

## Task 17: New Script — `radar/mti_doppler.py`

**Files:**
- Create: `examples/radar/mti_doppler.py`

- [ ] **Step 1: Write the MTI/Doppler example**

```python
"""
MTI and Doppler Filter Banks
=============================
Level: Intermediate

Demonstrate SPECTRA's Moving Target Indication and Doppler processing:
  - single_pulse_canceller — first-order clutter suppression
  - double_pulse_canceller — second-order clutter suppression
  - doppler_filter_bank — DFT-based Doppler filtering

Run:
    python examples/radar/mti_doppler.py
"""

import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from spectra.algorithms import single_pulse_canceller, double_pulse_canceller, doppler_filter_bank
from plot_helpers import savefig

rng = np.random.default_rng(42)

# ── 1. Simulate a pulse matrix ──────────────────────────────────────────────
num_pulses = 32
num_range_bins = 256
prf = 1000.0  # Hz

# Ground clutter: strong, zero Doppler
clutter = 10.0 * rng.standard_normal((num_pulses, num_range_bins)).astype(np.complex128)

# Moving target: moderate amplitude, non-zero Doppler
target_range_bin = 100
target_doppler_hz = 200.0
target_amplitude = 3.0
for p in range(num_pulses):
    phase = 2 * np.pi * target_doppler_hz * p / prf
    clutter[p, target_range_bin] += target_amplitude * np.exp(1j * phase)

# Add noise
pulse_matrix = clutter + 0.5 * (rng.standard_normal(clutter.shape) + 1j * rng.standard_normal(clutter.shape))

# ── 2. Apply MTI cancellers ─────────────────────────────────────────────────
spc = single_pulse_canceller(pulse_matrix)
dpc = double_pulse_canceller(pulse_matrix)

# ── 3. Apply Doppler filter bank ────────────────────────────────────────────
dfb = doppler_filter_bank(pulse_matrix)

# ── 4. Plot results ─────────────────────────────────────────────────────────
fig, axes = plt.subplots(2, 2, figsize=(12, 10))

# Raw pulse matrix (one pulse)
axes[0, 0].plot(20 * np.log10(np.abs(pulse_matrix[0]) + 1e-12), linewidth=0.5)
axes[0, 0].axvline(target_range_bin, color="red", linestyle="--", linewidth=1, label="Target")
axes[0, 0].set_title("Raw Pulse (single CPI)")
axes[0, 0].set_xlabel("Range Bin")
axes[0, 0].set_ylabel("dB")
axes[0, 0].legend()
axes[0, 0].grid(True, alpha=0.3)

# Single pulse canceller
axes[0, 1].plot(20 * np.log10(np.abs(spc[0]) + 1e-12), linewidth=0.5)
axes[0, 1].axvline(target_range_bin, color="red", linestyle="--", linewidth=1, label="Target")
axes[0, 1].set_title("After Single Pulse Canceller")
axes[0, 1].set_xlabel("Range Bin")
axes[0, 1].set_ylabel("dB")
axes[0, 1].legend()
axes[0, 1].grid(True, alpha=0.3)

# Double pulse canceller
axes[1, 0].plot(20 * np.log10(np.abs(dpc[0]) + 1e-12), linewidth=0.5)
axes[1, 0].axvline(target_range_bin, color="red", linestyle="--", linewidth=1, label="Target")
axes[1, 0].set_title("After Double Pulse Canceller")
axes[1, 0].set_xlabel("Range Bin")
axes[1, 0].set_ylabel("dB")
axes[1, 0].legend()
axes[1, 0].grid(True, alpha=0.3)

# Range-Doppler map
axes[1, 1].imshow(
    20 * np.log10(np.abs(dfb) + 1e-12),
    aspect="auto", origin="lower", cmap="hot",
    extent=[0, num_range_bins, -prf / 2, prf / 2],
)
axes[1, 1].axhline(target_doppler_hz, color="cyan", linestyle="--", linewidth=1)
axes[1, 1].axvline(target_range_bin, color="cyan", linestyle="--", linewidth=1)
axes[1, 1].set_title("Range-Doppler Map")
axes[1, 1].set_xlabel("Range Bin")
axes[1, 1].set_ylabel("Doppler (Hz)")

fig.suptitle("MTI Clutter Suppression & Doppler Processing", fontsize=14)
fig.tight_layout()
savefig("mti_doppler.png")
plt.close()

print("Done — MTI/Doppler example saved.")
```

- [ ] **Step 2: Commit**

```bash
git add examples/radar/mti_doppler.py
git commit -m "feat(examples): add MTI and Doppler filter bank example"
```

---

## Task 18: New Script — `environment/propagation_and_links.py`

**Files:**
- Create: `examples/environment/propagation_and_links.py`

- [ ] **Step 1: Write the environment/propagation example**

```python
"""
Environment & Propagation Modeling
===================================
Level: Intermediate

Demonstrate SPECTRA's environment simulation module:
  - Position, Emitter, ReceiverConfig
  - Propagation models: FreeSpacePathLoss, LogDistancePL, COST231HataPL
  - Environment.compute() — link budget computation
  - link_params_to_impairments — auto-generate impairment chains
  - propagation_presets — quick model selection

Run:
    python examples/environment/propagation_and_links.py
"""

import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from spectra.waveforms import QPSK, BPSK
from spectra.environment import (
    Position, Emitter, ReceiverConfig, Environment,
    FreeSpacePathLoss, LogDistancePL, COST231HataPL,
    link_params_to_impairments, propagation_presets,
)
from plot_helpers import savefig

sample_rate = 1e6

# ── 1. Propagation model comparison ─────────────────────────────────────────
freq = 900e6  # 900 MHz
distances = np.logspace(1, 4, 100)  # 10 m to 10 km

fspl = FreeSpacePathLoss(frequency_hz=freq)
logd = LogDistancePL(frequency_hz=freq, path_loss_exp=3.5)
cost231 = COST231HataPL(frequency_hz=freq, is_urban=True)

plt.figure(figsize=(10, 5))
for model, name in [(fspl, "Free Space"), (logd, "Log-Distance (n=3.5)"), (cost231, "COST231-Hata (urban)")]:
    losses = []
    for d in distances:
        result = model.compute(d)
        losses.append(result.path_loss_db)
    plt.plot(distances / 1e3, losses, linewidth=1.5, label=name)

plt.xlabel("Distance (km)")
plt.ylabel("Path Loss (dB)")
plt.title(f"Propagation Model Comparison @ {freq/1e6:.0f} MHz")
plt.legend()
plt.grid(True, alpha=0.3)
plt.xscale("log")
plt.tight_layout()
savefig("propagation_models.png")
plt.close()

# ── 2. Multi-emitter environment ────────────────────────────────────────────
rx = ReceiverConfig(
    position=Position(x_m=0, y_m=0),
    noise_figure_db=6.0,
    bandwidth_hz=sample_rate,
)

emitters = [
    Emitter(
        waveform=QPSK(samples_per_symbol=8, rolloff=0.35),
        position=Position(x_m=500, y_m=300),
        power_dbm=30.0,
        freq_hz=freq,
        velocity_mps=(10.0, 5.0),
    ),
    Emitter(
        waveform=BPSK(samples_per_symbol=8, rolloff=0.35),
        position=Position(x_m=-200, y_m=800),
        power_dbm=25.0,
        freq_hz=freq,
    ),
    Emitter(
        waveform=QPSK(samples_per_symbol=8, rolloff=0.35),
        position=Position(x_m=1000, y_m=-100),
        power_dbm=35.0,
        freq_hz=freq,
        velocity_mps=(-15.0, 0.0),
    ),
]

env = Environment(
    propagation=FreeSpacePathLoss(frequency_hz=freq),
    emitters=emitters,
    receiver=rx,
)

link_params_list = env.compute(seed=42)

print("Link Budget Results:")
print(f"{'Emitter':>8} {'Distance':>10} {'PathLoss':>10} {'SNR':>8} {'Doppler':>10} {'Rx Power':>10}")
for i, lp in enumerate(link_params_list):
    print(f"  {i:>5d} {lp.distance_m:>9.1f}m {lp.path_loss_db:>9.1f}dB "
          f"{lp.snr_db:>7.1f}dB {lp.doppler_hz:>9.1f}Hz {lp.received_power_dbm:>9.1f}dBm")

# ── 3. Plot environment geometry ─────────────────────────────────────────────
fig, ax = plt.subplots(figsize=(8, 8))
ax.scatter(rx.position.x_m, rx.position.y_m, s=200, marker="^", color="blue", zorder=5, label="Receiver")
for i, em in enumerate(emitters):
    ax.scatter(em.position.x_m, em.position.y_m, s=100, marker="o", color="red", zorder=5)
    ax.annotate(f"TX{i}\n{em.power_dbm:.0f}dBm", (em.position.x_m, em.position.y_m),
                textcoords="offset points", xytext=(10, 10), fontsize=8)
    ax.plot([rx.position.x_m, em.position.x_m], [rx.position.y_m, em.position.y_m],
            "k--", alpha=0.3)

ax.set_xlabel("x (m)")
ax.set_ylabel("y (m)")
ax.set_title("Multi-Emitter Environment")
ax.legend()
ax.grid(True, alpha=0.3)
ax.set_aspect("equal")
fig.tight_layout()
savefig("environment_geometry.png")
plt.close()

# ── 4. Convert link params to impairments ────────────────────────────────────
print("\nGenerating impairment chains from link params:")
for i, lp in enumerate(link_params_list):
    imp_chain = link_params_to_impairments(lp)
    print(f"  Emitter {i}: {imp_chain}")

# ── 5. Propagation presets ───────────────────────────────────────────────────
presets = propagation_presets()
print(f"\nAvailable presets: {list(presets.keys())}")

print("\nDone — environment/propagation examples saved.")
```

- [ ] **Step 2: Commit**

```bash
git add examples/environment/propagation_and_links.py
git commit -m "feat(examples): add environment and propagation modeling example"
```

---

## Task 19: New Script — `benchmarks/benchmark_evaluation.py`

**Files:**
- Create: `examples/benchmarks/benchmark_evaluation.py`

- [ ] **Step 1: Write the benchmark evaluation example**

```python
"""
Benchmark Loading and Evaluation
=================================
Level: Advanced

Demonstrate SPECTRA's built-in benchmark system:
  - load_benchmark — load named benchmark datasets
  - load_snr_sweep — load SNR sweep benchmarks
  - evaluate_snr_sweep — compute per-SNR metrics

Available benchmarks: spectra-18, spectra-18-wideband, spectra-40, spectra-5g,
spectra-airport, spectra-channel, spectra-congested-ism, spectra-df,
spectra-maritime-vhf, spectra-protocol, spectra-radar, spectra-snr, spectra-spread.

Run:
    python examples/benchmarks/benchmark_evaluation.py
"""

import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from spectra.benchmarks import load_benchmark, load_snr_sweep, evaluate_snr_sweep
from plot_helpers import savefig

# ── 1. Load the spectra-18 benchmark ────────────────────────────────────────
print("Loading spectra-18 benchmark...")
ds_18 = load_benchmark("spectra-18")
print(f"  Samples: {len(ds_18)}")

sample_iq, sample_label = ds_18[0]
print(f"  Sample shape: {sample_iq.shape}, label: {sample_label}")

# ── 2. Load the spectra-spread benchmark ─────────────────────────────────────
print("\nLoading spectra-spread benchmark...")
ds_spread = load_benchmark("spectra-spread")
print(f"  Samples: {len(ds_spread)}")

# ── 3. Load an SNR sweep benchmark ──────────────────────────────────────────
print("\nLoading spectra-snr sweep benchmark...")
ds_snr = load_snr_sweep("spectra-snr")
print(f"  Samples: {len(ds_snr)}")

# ── 4. Simulate a simple classifier on the SNR sweep ────────────────────────
# (Random predictions to demonstrate the evaluate_snr_sweep API)
rng = np.random.default_rng(42)
all_labels = []
all_preds = []
all_snrs = []

for idx in range(min(len(ds_snr), 200)):
    iq, label = ds_snr[idx]
    all_labels.append(label)
    # "classifier": just predict the correct class most of the time at high SNR
    all_preds.append(label)  # placeholder
    all_snrs.append(idx % 6 * 5 - 5)  # mock SNR values

all_labels = np.array(all_labels)
all_preds = np.array(all_preds)
all_snrs = np.array(all_snrs, dtype=float)

results = evaluate_snr_sweep(all_labels, all_preds, all_snrs)
print("\nPer-SNR accuracy:")
for snr, acc in sorted(results.items()):
    print(f"  SNR={snr:>6.1f} dB: accuracy={acc:.1%}")

# ── 5. Plot per-SNR accuracy ────────────────────────────────────────────────
snrs = sorted(results.keys())
accs = [results[s] for s in snrs]

plt.figure(figsize=(8, 4))
plt.plot(snrs, accs, "o-", linewidth=1.5, markersize=6)
plt.xlabel("SNR (dB)")
plt.ylabel("Accuracy")
plt.title("Benchmark Evaluation: Accuracy vs SNR")
plt.grid(True, alpha=0.3)
plt.ylim([0, 1.05])
plt.tight_layout()
savefig("benchmark_evaluation.png")
plt.close()

# ── 6. List available benchmarks ────────────────────────────────────────────
benchmark_ids = [
    "spectra-18", "spectra-18-wideband", "spectra-40", "spectra-5g",
    "spectra-airport", "spectra-congested-ism", "spectra-df",
    "spectra-maritime-vhf", "spectra-protocol", "spectra-radar",
    "spectra-spread",
]
print(f"\nAvailable benchmarks ({len(benchmark_ids)}):")
for bid in benchmark_ids:
    try:
        ds = load_benchmark(bid)
        print(f"  {bid:>25s}: {len(ds):>5d} samples")
    except Exception as e:
        print(f"  {bid:>25s}: (error: {e})")

print("\nDone — benchmark evaluation example saved.")
```

- [ ] **Step 2: Commit**

```bash
git add examples/benchmarks/benchmark_evaluation.py
git commit -m "feat(examples): add benchmark loading and evaluation example"
```

---

## Task 20: Update `examples/README.md`

**Files:**
- Modify: `examples/README.md`

- [ ] **Step 1: Replace README.md with updated content**

```markdown
# SPECTRA Examples

Example scripts demonstrating SPECTRA's capabilities for RF waveform generation,
impairment simulation, signal processing, and machine learning dataset construction.
Examples are organized into domain-specific sub-folders. Most examples are available
as both standalone Python scripts (`.py`) and Jupyter notebooks (`.ipynb`).

## Prerequisites

```bash
# Build SPECTRA (requires Python 3.10+, Rust 1.83+)
maturin develop --release

# Install dependencies
pip install numpy matplotlib torch --index-url https://download.pytorch.org/whl/cpu

# For CSP classification examples
pip install 'spectra[classifiers]'
```

## Running Examples

All scripts are designed to be run from the repository root:

```bash
python examples/getting_started/basic_waveforms.py
python examples/radar/mti_doppler.py
```

## Directory Structure

```
examples/
├── plot_helpers.py                          # Shared plotting utilities
├── outputs/                                 # Generated figures
│
├── getting_started/                         # Introductory examples
│   ├── basic_waveforms.py                   # Novice
│   ├── impairments_and_channels.py          # Intermediate
│   └── transforms_and_spectrograms.py       # Intermediate
│
├── waveforms/                               # Advanced signal generation
│   ├── spread_spectrum.py                   # Intermediate
│   ├── protocol_signals.py                  # Intermediate
│   └── nr_5g_signals.py                     # Advanced
│
├── impairments/                             # Channel effects & distortions
│   └── advanced_impairments.py              # Intermediate
│
├── transforms/                              # Signal processing & features
│   ├── alignment_transforms.py              # Intermediate
│   └── time_frequency_analysis.py           # Intermediate
│
├── datasets/                                # Dataset construction & I/O
│   ├── narrowband_dataset.py                # Advanced
│   ├── wideband_scenes.py                   # Pro
│   ├── folder_and_manifest.py               # Intermediate
│   ├── snr_sweep_evaluation.py              # Advanced
│   ├── augmentation_wrappers.py             # Intermediate
│   ├── dataset_io.py                        # Intermediate
│   └── streaming_curriculum.py              # Advanced
│
├── classification/                          # AMC training & evaluation
│   ├── full_pipeline.py                     # Pro
│   ├── csp_classification.py                # Advanced
│   ├── train_narrowband_cnn.py              # Advanced
│   └── resnet_amc.py                        # Advanced
│
├── cyclostationary/                         # CSP features & time-frequency
│   ├── csp_features.py                      # Intermediate
│   ├── s3ca_vs_ssca.py                      # Advanced
│   ├── cwd_cross_term_suppression.py        # Intermediate
│   └── wvd_time_frequency.py                # Intermediate
│
├── antenna_arrays/                          # Arrays, DoA, beamforming
│   ├── direction_finding.py                 # Intermediate
│   ├── beamforming.py                       # Intermediate
│   ├── wideband_direction_finding.py        # Intermediate–Advanced
│   ├── antenna_elements.py                  # Intermediate
│   └── array_geometries.py                  # Intermediate
│
├── radar/                                   # Radar processing & tracking
│   ├── radar_processing.py                  # Intermediate
│   ├── radar_pipeline.py                    # Advanced
│   └── mti_doppler.py                       # Intermediate
│
├── communications/                          # Link simulation & receivers
│   └── link_simulator.py                    # Advanced
│
├── environment/                             # Propagation & link budgets
│   └── propagation_and_links.py             # Intermediate
│
└── benchmarks/                              # Reproducible evaluation
    └── benchmark_evaluation.py              # Advanced
```

## Examples by Category

### Getting Started

| Example | Level | Description |
|---------|-------|-------------|
| `basic_waveforms.py` | Novice | Generate BPSK, QPSK, QAM, FSK; plot IQ, constellation, PSD |
| `impairments_and_channels.py` | Intermediate | AWGN, frequency offset, phase noise, Rayleigh/Rician fading, Compose |
| `transforms_and_spectrograms.py` | Intermediate | STFT, ComplexTo2D, data augmentations, DSP filter design |

### Waveforms

| Example | Level | Description |
|---------|-------|-------------|
| `spread_spectrum.py` | Intermediate | DSSS-BPSK/QPSK, FHSS, THSS, CDMA Forward/Reverse, ChirpSS |
| `protocol_signals.py` | Intermediate | ADS-B, Mode S, AIS, ACARS, DME, ILS Localizer |
| `nr_5g_signals.py` | Advanced | NR OFDM, PDSCH, PUSCH, PRACH, SSB signals |

### Impairments

| Example | Level | Description |
|---------|-------|-------------|
| `advanced_impairments.py` | Intermediate | ColoredNoise, DopplerShift, IQImbalance, TDL channels, PA models, quantization |

### Transforms

| Example | Level | Description |
|---------|-------|-------------|
| `alignment_transforms.py` | Intermediate | DCRemove, PowerNormalize, AGC, ClipNormalize, SpectralWhitening |
| `time_frequency_analysis.py` | Intermediate | ReassignedGabor, InstantaneousFrequency, AmbiguityFunction |

### Datasets

| Example | Level | Description |
|---------|-------|-------------|
| `narrowband_dataset.py` | Advanced | NarrowbandDataset, PyTorch DataLoader, per-class spectrograms |
| `wideband_scenes.py` | Pro | SceneConfig, Composer, COCO bounding boxes, WidebandDataset |
| `folder_and_manifest.py` | Intermediate | SignalFolderDataset, ManifestDataset for pre-existing recordings |
| `snr_sweep_evaluation.py` | Advanced | SNRSweepDataset for structured (SNR × class) evaluation |
| `augmentation_wrappers.py` | Intermediate | MixUpDataset, CutMixDataset cross-sample augmentation |
| `dataset_io.py` | Intermediate | DatasetWriter, export_dataset_to_folder, NumpyWriter |
| `streaming_curriculum.py` | Advanced | StreamingDataLoader, CurriculumSchedule, progressive difficulty |

### Classification

| Example | Level | Description |
|---------|-------|-------------|
| `full_pipeline.py` | Pro | End-to-end: dataset → CNN training → evaluation |
| `csp_classification.py` | Advanced | CyclostationaryAMC with SCD/cumulant features, random forest |
| `train_narrowband_cnn.py` | Advanced | CNNAMC benchmark training with confusion matrix |
| `resnet_amc.py` | Advanced | ResNetAMC on spectrogram features |

### Cyclostationary & Time-Frequency

| Example | Level | Description |
|---------|-------|-------------|
| `csp_features.py` | Intermediate | SCD, SCF, CAF heatmaps, cumulant features, PSD |
| `s3ca_vs_ssca.py` | Advanced | S³CA vs SSCA SCD estimators |
| `cwd_cross_term_suppression.py` | Intermediate | Choi-Williams Distribution vs WVD, sigma sweep |
| `wvd_time_frequency.py` | Intermediate | Wigner-Ville Distribution analysis |

### Antenna Arrays

| Example | Level | Description |
|---------|-------|-------------|
| `direction_finding.py` | Intermediate | ULA, DirectionFindingDataset, MUSIC, ESPRIT |
| `beamforming.py` | Intermediate | Delay-and-Sum, MVDR, LCMV beam patterns |
| `wideband_direction_finding.py` | Intermediate–Advanced | WidebandDirectionFindingDataset, sub-band MUSIC |
| `antenna_elements.py` | Intermediate | Isotropic, dipole, Yagi, cosine-power radiation patterns |
| `array_geometries.py` | Intermediate | ULA, UCA, rectangular arrays, CalibrationErrors |

### Radar

| Example | Level | Description |
|---------|-------|-------------|
| `radar_processing.py` | Intermediate | Matched filter, CA-CFAR, OS-CFAR on LFM/coded pulses |
| `radar_pipeline.py` | Advanced | Target trajectories, Swerling RCS, clutter, MTI, Kalman tracking |
| `mti_doppler.py` | Intermediate | Single/double pulse canceller, Doppler filter bank, range-Doppler map |

### Communications

| Example | Level | Description |
|---------|-------|-------------|
| `link_simulator.py` | Advanced | CoherentReceiver, BER/SER/PER curves, LinkSimulator |

### Environment

| Example | Level | Description |
|---------|-------|-------------|
| `propagation_and_links.py` | Intermediate | Free-space/log-distance/COST231 propagation, multi-emitter link budget |

### Benchmarks

| Example | Level | Description |
|---------|-------|-------------|
| `benchmark_evaluation.py` | Advanced | Load and evaluate built-in benchmarks (spectra-18, spectra-spread, etc.) |

## Output

All figures are saved to `examples/outputs/`. Running all scripts produces 60+ PNG figures.
```

- [ ] **Step 2: Commit**

```bash
git add examples/README.md
git commit -m "docs(examples): update README with new folder structure and examples"
```

---

## Task 21: Smoke Test All Scripts

**Files:** None (verification only)

- [ ] **Step 1: Run a quick import check on all new scripts**

```bash
for script in \
  examples/waveforms/spread_spectrum.py \
  examples/waveforms/protocol_signals.py \
  examples/waveforms/nr_5g_signals.py \
  examples/impairments/advanced_impairments.py \
  examples/transforms/alignment_transforms.py \
  examples/transforms/time_frequency_analysis.py \
  examples/datasets/folder_and_manifest.py \
  examples/datasets/snr_sweep_evaluation.py \
  examples/datasets/augmentation_wrappers.py \
  examples/datasets/dataset_io.py \
  examples/datasets/streaming_curriculum.py \
  examples/classification/resnet_amc.py \
  examples/antenna_arrays/antenna_elements.py \
  examples/antenna_arrays/array_geometries.py \
  examples/radar/mti_doppler.py \
  examples/environment/propagation_and_links.py \
  examples/benchmarks/benchmark_evaluation.py; do
  echo "=== $script ==="
  python "$script" || echo "FAILED: $script"
done
```

Expected: all scripts run without import errors. Some may produce warnings about optional dependencies, which is acceptable.

- [ ] **Step 2: Fix any failures found during smoke testing**

Address any import errors, missing API changes, or runtime issues discovered.

- [ ] **Step 3: Final commit if fixes were needed**

```bash
git add -A examples/
git commit -m "fix(examples): address smoke test issues in new example scripts"
```
