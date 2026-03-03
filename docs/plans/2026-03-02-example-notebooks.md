# Example Notebooks Implementation Plan

> **For Claude:** REQUIRED SUB-SKILL: Use superpowers:executing-plans to implement this plan task-by-task.

**Goal:** Create 6 example notebooks (+ matching .py scripts) progressing from novice to pro, covering waveform generation, impairments, transforms, datasets, wideband scenes, and full ML pipelines.

**Architecture:** Each example is a Jupyter notebook in `./examples/` with a companion `.py` file containing the same code. All matplotlib figures are saved to `./examples/outputs/`. Notebooks progress in difficulty: 3 novice/intermediate, 3 advanced/pro. Each notebook is self-contained with clear markdown explanations.

**Tech Stack:** SPECTRA (spectra), NumPy, Matplotlib, PyTorch (for datasets/DataLoader), Jupyter

---

## Directory Structure

```
examples/
  outputs/                          # All saved figures go here
  01_basic_waveforms.ipynb          # Novice
  01_basic_waveforms.py
  02_impairments_and_channels.ipynb # Intermediate
  02_impairments_and_channels.py
  03_transforms_and_spectrograms.ipynb  # Intermediate
  03_transforms_and_spectrograms.py
  04_narrowband_dataset.ipynb       # Advanced
  04_narrowband_dataset.py
  05_wideband_scenes.ipynb          # Pro
  05_wideband_scenes.py
  06_full_pipeline.ipynb            # Pro
  06_full_pipeline.py
```

---

### Task 1: Create directory structure and plotting helpers

**Files:**
- Create: `examples/outputs/.gitkeep`
- Create: `examples/plot_helpers.py`

**Step 1: Create directories and .gitkeep**

```bash
mkdir -p examples/outputs
touch examples/outputs/.gitkeep
```

**Step 2: Write shared plot helpers**

Create `examples/plot_helpers.py` with reusable plotting functions used across all notebooks:

```python
"""Shared plotting helpers for SPECTRA examples."""

import os

import matplotlib.pyplot as plt
import numpy as np

OUTPUT_DIR = os.path.join(os.path.dirname(__file__), "outputs")
os.makedirs(OUTPUT_DIR, exist_ok=True)


def savefig(name: str, dpi: int = 150) -> None:
    """Save current figure to outputs/ directory."""
    path = os.path.join(OUTPUT_DIR, name)
    plt.savefig(path, dpi=dpi, bbox_inches="tight")
    print(f"Saved: {path}")


def plot_iq_time(iq: np.ndarray, title: str = "", num_samples: int = 200) -> None:
    """Plot I and Q components vs sample index."""
    n = min(num_samples, len(iq))
    fig, axes = plt.subplots(2, 1, figsize=(10, 4), sharex=True)
    axes[0].plot(iq[:n].real, linewidth=0.8)
    axes[0].set_ylabel("In-Phase (I)")
    axes[0].set_title(title)
    axes[0].grid(True, alpha=0.3)
    axes[1].plot(iq[:n].imag, linewidth=0.8, color="tab:orange")
    axes[1].set_ylabel("Quadrature (Q)")
    axes[1].set_xlabel("Sample Index")
    axes[1].grid(True, alpha=0.3)
    fig.tight_layout()


def plot_constellation(iq: np.ndarray, title: str = "", max_pts: int = 2000) -> None:
    """Scatter plot of IQ constellation."""
    pts = iq[: min(max_pts, len(iq))]
    fig, ax = plt.subplots(figsize=(5, 5))
    ax.scatter(pts.real, pts.imag, s=2, alpha=0.5)
    ax.set_xlabel("In-Phase")
    ax.set_ylabel("Quadrature")
    ax.set_title(title)
    ax.set_aspect("equal")
    ax.grid(True, alpha=0.3)
    fig.tight_layout()


def plot_psd(
    iq: np.ndarray, sample_rate: float, title: str = "", nfft: int = 1024
) -> None:
    """Plot power spectral density."""
    fig, ax = plt.subplots(figsize=(10, 4))
    freqs = np.fft.fftshift(np.fft.fftfreq(nfft, d=1.0 / sample_rate))
    spectrum = np.fft.fftshift(np.fft.fft(iq[:nfft], n=nfft))
    psd = 10 * np.log10(np.abs(spectrum) ** 2 + 1e-12)
    ax.plot(freqs / 1e3, psd, linewidth=0.8)
    ax.set_xlabel("Frequency (kHz)")
    ax.set_ylabel("Power (dB)")
    ax.set_title(title)
    ax.grid(True, alpha=0.3)
    fig.tight_layout()


def plot_spectrogram_img(
    spec: np.ndarray, title: str = "", cmap: str = "viridis"
) -> None:
    """Plot a 2D spectrogram array (freq x time)."""
    fig, ax = plt.subplots(figsize=(10, 4))
    ax.imshow(
        spec,
        aspect="auto",
        origin="lower",
        cmap=cmap,
        interpolation="nearest",
    )
    ax.set_xlabel("Time Bin")
    ax.set_ylabel("Frequency Bin")
    ax.set_title(title)
    fig.tight_layout()
```

**Step 3: Commit**

```bash
git add examples/outputs/.gitkeep examples/plot_helpers.py
git commit -m "feat(examples): add output directory and shared plot helpers"
```

---

### Task 2: Example 01 — Basic Waveforms (Novice)

**Files:**
- Create: `examples/01_basic_waveforms.ipynb`
- Create: `examples/01_basic_waveforms.py`

**Skill level:** Novice. Introduces SPECTRA, generates simple waveforms, visualizes IQ time series, constellation diagrams, and power spectra.

**Step 1: Write the Python script**

Create `examples/01_basic_waveforms.py`:

```python
"""
SPECTRA Example 01: Basic Waveform Generation
==============================================
Level: Novice

Learn how to:
- Generate digital modulation waveforms (BPSK, QPSK, 16QAM, FSK)
- Visualize IQ time-domain signals
- Plot constellation diagrams
- View power spectral density (PSD)
"""

import sys
sys.path.insert(0, ".")

import numpy as np
import matplotlib.pyplot as plt
import spectra as sp
from plot_helpers import savefig, plot_iq_time, plot_constellation, plot_psd

# ── 1. Generate a QPSK signal ───────────────────────────────────────────────

waveform = sp.QPSK(samples_per_symbol=8, rolloff=0.35)
sample_rate = 1e6  # 1 MHz
iq = waveform.generate(num_symbols=512, sample_rate=sample_rate, seed=42)

print(f"Waveform: {waveform.label}")
print(f"IQ shape: {iq.shape}, dtype: {iq.dtype}")
print(f"Bandwidth: {waveform.bandwidth(sample_rate) / 1e3:.1f} kHz")

# ── 2. Visualize QPSK ───────────────────────────────────────────────────────

plot_iq_time(iq, title="QPSK — Time Domain")
savefig("01_qpsk_time.png")

plot_constellation(iq, title="QPSK — Constellation")
savefig("01_qpsk_constellation.png")

plot_psd(iq, sample_rate, title="QPSK — Power Spectral Density")
savefig("01_qpsk_psd.png")

# ── 3. Compare multiple modulation schemes ───────────────────────────────────

waveforms = [
    sp.BPSK(),
    sp.QPSK(),
    sp.QAM16(),
    sp.QAM64(),
    sp.PSK8(),
    sp.OOK(),
]

fig, axes = plt.subplots(2, 3, figsize=(14, 9))
for ax, wf in zip(axes.flat, waveforms):
    iq_i = wf.generate(num_symbols=512, sample_rate=sample_rate, seed=42)
    pts = iq_i[: min(2000, len(iq_i))]
    ax.scatter(pts.real, pts.imag, s=2, alpha=0.4)
    ax.set_title(wf.label)
    ax.set_aspect("equal")
    ax.grid(True, alpha=0.3)
fig.suptitle("Constellation Diagrams — Digital Modulation Comparison", fontsize=14)
fig.tight_layout()
savefig("01_constellation_grid.png")

# ── 4. Compare PSD of different waveforms ────────────────────────────────────

waveforms_psd = [
    ("BPSK", sp.BPSK()),
    ("QPSK", sp.QPSK()),
    ("16QAM", sp.QAM16()),
    ("OFDM-64", sp.OFDM()),
    ("FSK", sp.FSK()),
    ("GMSK", sp.GMSK()),
]

fig, axes = plt.subplots(2, 3, figsize=(15, 8))
nfft = 1024
freqs = np.fft.fftshift(np.fft.fftfreq(nfft, d=1.0 / sample_rate))
for ax, (name, wf) in zip(axes.flat, waveforms_psd):
    iq_i = wf.generate(num_symbols=512, sample_rate=sample_rate, seed=42)
    spectrum = np.fft.fftshift(np.fft.fft(iq_i[:nfft], n=nfft))
    psd = 10 * np.log10(np.abs(spectrum) ** 2 + 1e-12)
    ax.plot(freqs / 1e3, psd, linewidth=0.8)
    ax.set_title(name)
    ax.set_xlabel("Freq (kHz)")
    ax.set_ylabel("Power (dB)")
    ax.grid(True, alpha=0.3)
fig.suptitle("Power Spectral Density — Modulation Comparison", fontsize=14)
fig.tight_layout()
savefig("01_psd_grid.png")

# ── 5. Analog modulations: AM and FM ────────────────────────────────────────

analog_waveforms = [
    ("AM-DSB", sp.AMDSB()),
    ("AM-SSB (USB)", sp.AMUSB()),
    ("FM", sp.FM()),
    ("Tone", sp.Tone()),
]

fig, axes = plt.subplots(2, 2, figsize=(12, 8))
for ax, (name, wf) in zip(axes.flat, analog_waveforms):
    iq_i = wf.generate(num_symbols=1024, sample_rate=sample_rate, seed=42)
    spectrum = np.fft.fftshift(np.fft.fft(iq_i[:nfft], n=nfft))
    psd = 10 * np.log10(np.abs(spectrum) ** 2 + 1e-12)
    ax.plot(freqs / 1e3, psd, linewidth=0.8)
    ax.set_title(name)
    ax.set_xlabel("Freq (kHz)")
    ax.set_ylabel("Power (dB)")
    ax.grid(True, alpha=0.3)
fig.suptitle("Analog Modulation Spectra", fontsize=14)
fig.tight_layout()
savefig("01_analog_psd.png")

plt.show()
print("\nDone! Check examples/outputs/ for saved figures.")
```

**Step 2: Create the Jupyter notebook**

Create `examples/01_basic_waveforms.ipynb` — a Jupyter notebook with the same content split into cells with markdown explanations between them. Structure:

- **Cell 1 (markdown):** Title + overview (what you'll learn)
- **Cell 2 (code):** Imports
- **Cell 3 (markdown):** "1. Generate Your First Waveform"
- **Cell 4 (code):** QPSK generation + print info
- **Cell 5 (markdown):** "2. Visualize IQ Time-Domain"
- **Cell 6 (code):** `plot_iq_time` + `savefig`
- **Cell 7 (markdown):** "3. Constellation Diagram"
- **Cell 8 (code):** `plot_constellation` + `savefig`
- **Cell 9 (markdown):** "4. Power Spectral Density"
- **Cell 10 (code):** `plot_psd` + `savefig`
- **Cell 11 (markdown):** "5. Compare Multiple Modulation Schemes"
- **Cell 12 (code):** Constellation grid
- **Cell 13 (markdown):** "6. PSD Comparison"
- **Cell 14 (code):** PSD grid
- **Cell 15 (markdown):** "7. Analog Modulations"
- **Cell 16 (code):** AM/FM spectra
- **Cell 17 (markdown):** Summary + next steps

**Step 3: Run and verify**

```bash
cd examples && python 01_basic_waveforms.py
ls outputs/01_*.png
```

Expected: 5 PNG files in `outputs/`.

**Step 4: Commit**

```bash
git add examples/01_basic_waveforms.py examples/01_basic_waveforms.ipynb examples/outputs/01_*.png
git commit -m "feat(examples): add 01 basic waveforms notebook (novice)"
```

---

### Task 3: Example 02 — Impairments and Channel Effects (Intermediate)

**Files:**
- Create: `examples/02_impairments_and_channels.ipynb`
- Create: `examples/02_impairments_and_channels.py`

**Skill level:** Intermediate. Shows how to apply realistic channel impairments to clean signals.

**Step 1: Write the Python script**

Create `examples/02_impairments_and_channels.py`:

```python
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
sys.path.insert(0, ".")

import numpy as np
import matplotlib.pyplot as plt
import spectra as sp
from spectra.scene import SignalDescription
from plot_helpers import savefig, plot_constellation

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

plt.show()
print("\nDone! Check examples/outputs/ for saved figures.")
```

**Step 2: Create matching Jupyter notebook**

Same structure as Task 2: split into cells with markdown headers explaining each impairment.

**Step 3: Run and verify**

```bash
cd examples && python 02_impairments_and_channels.py
ls outputs/02_*.png
```

Expected: 6 PNG files.

**Step 4: Commit**

```bash
git add examples/02_impairments_and_channels.py examples/02_impairments_and_channels.ipynb examples/outputs/02_*.png
git commit -m "feat(examples): add 02 impairments and channels notebook (intermediate)"
```

---

### Task 4: Example 03 — Transforms and Spectrograms (Intermediate)

**Files:**
- Create: `examples/03_transforms_and_spectrograms.ipynb`
- Create: `examples/03_transforms_and_spectrograms.py`

**Skill level:** Intermediate. Shows signal transforms, spectrograms, and data augmentations.

**Step 1: Write the Python script**

Create `examples/03_transforms_and_spectrograms.py`:

```python
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
import matplotlib.pyplot as plt
import spectra as sp
from spectra.utils import dsp
from plot_helpers import savefig, plot_spectrogram_img

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

plt.show()
print("\nDone! Check examples/outputs/ for saved figures.")
```

**Step 2: Create matching Jupyter notebook**

Split into cells with markdown headers for each section.

**Step 3: Run and verify**

```bash
cd examples && python 03_transforms_and_spectrograms.py
ls outputs/03_*.png
```

Expected: 6 PNG files.

**Step 4: Commit**

```bash
git add examples/03_transforms_and_spectrograms.py examples/03_transforms_and_spectrograms.ipynb examples/outputs/03_*.png
git commit -m "feat(examples): add 03 transforms and spectrograms notebook (intermediate)"
```

---

### Task 5: Example 04 — Narrowband Classification Dataset (Advanced)

**Files:**
- Create: `examples/04_narrowband_dataset.ipynb`
- Create: `examples/04_narrowband_dataset.py`

**Skill level:** Advanced. Builds a PyTorch classification dataset, iterates with DataLoader, visualizes class distributions and spectrograms per class.

**Step 1: Write the Python script**

Create `examples/04_narrowband_dataset.py`:

```python
"""
SPECTRA Example 04: Narrowband Classification Dataset
=====================================================
Level: Advanced

Learn how to:
- Build a NarrowbandDataset for automatic modulation classification (AMC)
- Use PyTorch DataLoader for batched iteration
- Apply transforms and target transforms
- Visualize per-class spectrograms and IQ distributions
- Use ClassIndex and FamilyIndex target transforms
"""

import sys
sys.path.insert(0, ".")

import numpy as np
import matplotlib.pyplot as plt
import torch
from torch.utils.data import DataLoader
import spectra as sp
from plot_helpers import savefig

sample_rate = 1e6

# ── 1. Define waveform pool ─────────────────────────────────────────────────

waveform_pool = [
    sp.BPSK(),
    sp.QPSK(),
    sp.PSK8(),
    sp.QAM16(),
    sp.QAM64(),
    sp.FSK(),
    sp.GMSK(),
    sp.OFDM(),
]

class_names = [w.label for w in waveform_pool]
print(f"Classes ({len(class_names)}): {class_names}")

# ── 2. Create dataset with impairments and transforms ────────────────────────

impairments = sp.Compose([
    sp.AWGN(snr_range=(5, 25)),
    sp.FrequencyOffset(max_offset=1000),
])

target_transform = sp.ClassIndex(class_names)

dataset = sp.NarrowbandDataset(
    waveform_pool=waveform_pool,
    num_samples=800,             # 100 per class
    num_iq_samples=1024,
    sample_rate=sample_rate,
    impairments=impairments,
    transform=sp.ComplexTo2D(),
    target_transform=target_transform,
    seed=42,
)

print(f"Dataset size: {len(dataset)}")
data, label = dataset[0]
print(f"Sample shape: {data.shape}, label: {label} ({class_names[label]})")

# ── 3. Iterate with DataLoader ───────────────────────────────────────────────

loader = DataLoader(dataset, batch_size=32, shuffle=True)
all_labels = []

for batch_data, batch_labels in loader:
    all_labels.extend(batch_labels.numpy())

all_labels = np.array(all_labels)

fig, ax = plt.subplots(figsize=(10, 5))
counts = [np.sum(all_labels == i) for i in range(len(class_names))]
ax.bar(class_names, counts, color="steelblue")
ax.set_xlabel("Modulation Class")
ax.set_ylabel("Count")
ax.set_title("Class Distribution in Dataset")
ax.grid(True, alpha=0.3, axis="y")
fig.tight_layout()
savefig("04_class_distribution.png")

# ── 4. Visualize one sample per class ────────────────────────────────────────

# Create a dataset without ComplexTo2D for raw IQ visualization
raw_dataset = sp.NarrowbandDataset(
    waveform_pool=waveform_pool,
    num_samples=800,
    num_iq_samples=1024,
    sample_rate=sample_rate,
    impairments=impairments,
    target_transform=target_transform,
    seed=42,
)

fig, axes = plt.subplots(2, 4, figsize=(16, 8))
seen = set()
idx = 0
while len(seen) < len(class_names) and idx < len(raw_dataset):
    data, label = raw_dataset[idx]
    if label not in seen:
        seen.add(label)
        ax = axes.flat[label]
        iq = data.numpy() if isinstance(data, torch.Tensor) else data
        if np.iscomplexobj(iq):
            spec = np.abs(np.fft.fftshift(
                np.array([np.fft.fft(iq[i:i+256], n=256) for i in range(0, len(iq)-256, 64)]).T
            ))
        else:
            spec = np.abs(np.fft.fftshift(
                np.array([np.fft.fft((iq[0, i:i+256] + 1j * iq[1, i:i+256]), n=256) for i in range(0, iq.shape[1]-256, 64)]).T
            ))
        spec_db = 10 * np.log10(spec + 1e-12)
        ax.imshow(spec_db, aspect="auto", origin="lower", cmap="viridis")
        ax.set_title(class_names[label])
    idx += 1
fig.suptitle("Spectrogram per Class (with impairments)", fontsize=14)
fig.tight_layout()
savefig("04_per_class_spectrograms.png")

# ── 5. Family-level grouping ────────────────────────────────────────────────

family_transform = sp.FamilyName()
families = set()
for wf in waveform_pool:
    families.add(family_transform(wf.label))
print(f"Modulation families: {sorted(families)}")

# ── 6. Multiple SNR visualization ───────────────────────────────────────────

snr_levels = [0, 5, 10, 20]
wf = sp.QAM16()
fig, axes = plt.subplots(1, 4, figsize=(16, 4))
for ax, snr in zip(axes, snr_levels):
    ds = sp.NarrowbandDataset(
        waveform_pool=[wf],
        num_samples=1,
        num_iq_samples=1024,
        sample_rate=sample_rate,
        impairments=sp.Compose([sp.AWGN(snr=snr)]),
        seed=42,
    )
    data, _ = ds[0]
    iq = data.numpy() if isinstance(data, torch.Tensor) else data
    if np.iscomplexobj(iq):
        pts = iq[:500]
    else:
        pts = (iq[0, :500] + 1j * iq[1, :500])
    ax.scatter(pts.real, pts.imag, s=3, alpha=0.5)
    ax.set_title(f"SNR = {snr} dB")
    ax.set_aspect("equal")
    ax.grid(True, alpha=0.3)
    ax.set_xlim(-3, 3)
    ax.set_ylim(-3, 3)
fig.suptitle("16QAM at Different SNR Levels", fontsize=14)
fig.tight_layout()
savefig("04_snr_comparison.png")

plt.show()
print("\nDone! Check examples/outputs/ for saved figures.")
```

**Step 2: Create matching Jupyter notebook**

**Step 3: Run and verify**

```bash
cd examples && python 04_narrowband_dataset.py
ls outputs/04_*.png
```

Expected: 3 PNG files.

**Step 4: Commit**

```bash
git add examples/04_narrowband_dataset.py examples/04_narrowband_dataset.ipynb examples/outputs/04_*.png
git commit -m "feat(examples): add 04 narrowband classification dataset (advanced)"
```

---

### Task 6: Example 05 — Wideband Scene Composition (Pro)

**Files:**
- Create: `examples/05_wideband_scenes.ipynb`
- Create: `examples/05_wideband_scenes.py`

**Skill level:** Pro. Generates multi-signal wideband captures, visualizes spectrograms with bounding boxes, and produces COCO-format detection labels.

**Step 1: Write the Python script**

Create `examples/05_wideband_scenes.py`:

```python
"""
SPECTRA Example 05: Wideband Scene Composition
===============================================
Level: Pro

Learn how to:
- Configure wideband RF scenes with SceneConfig
- Generate composite captures with multiple signals
- Visualize spectrograms with overlaid bounding boxes
- Convert physical units to pixel-space COCO labels
- Use WidebandDataset with DataLoader
"""

import sys
sys.path.insert(0, ".")

import numpy as np
import matplotlib.pyplot as plt
import matplotlib.patches as patches
import spectra as sp
from spectra.scene import STFTParams
from spectra.utils import dsp
from plot_helpers import savefig

# ── 1. Configure wideband scene ─────────────────────────────────────────────

signal_pool = [
    sp.QPSK(),
    sp.QAM16(),
    sp.FSK(),
    sp.OFDM(),
    sp.GMSK(),
    sp.BPSK(),
]

config = sp.SceneConfig(
    capture_duration=1e-3,       # 1 ms capture
    capture_bandwidth=10e6,      # 10 MHz capture BW
    sample_rate=10e6,            # 10 MHz sample rate
    num_signals=(2, 5),          # 2 to 5 signals per scene
    signal_pool=signal_pool,
    snr_range=(5, 25),
    allow_overlap=True,
)

# ── 2. Generate a scene ─────────────────────────────────────────────────────

composer = sp.Composer(config)
iq, signal_descs = composer.generate(seed=42)

print(f"IQ shape: {iq.shape} ({len(iq)} samples)")
print(f"Number of signals: {len(signal_descs)}")
for i, desc in enumerate(signal_descs):
    print(f"  Signal {i}: {desc.label}, "
          f"f=[{desc.f_low/1e6:.2f}, {desc.f_high/1e6:.2f}] MHz, "
          f"t=[{desc.t_start*1e3:.3f}, {desc.t_stop*1e3:.3f}] ms, "
          f"SNR={desc.snr:.1f} dB")

# ── 3. Compute spectrogram and overlay bounding boxes ────────────────────────

nfft = 512
hop = 128

spec = dsp.compute_spectrogram(iq, nfft=nfft, hop=hop)
spec_db = 10 * np.log10(spec + 1e-12)

stft_params = STFTParams(
    nfft=nfft,
    hop_length=hop,
    sample_rate=config.sample_rate,
    num_samples=len(iq),
)

class_list = sorted(set(d.label for d in signal_descs))
targets = sp.to_coco(signal_descs, stft_params, class_list)

fig, ax = plt.subplots(figsize=(14, 6))
ax.imshow(spec_db, aspect="auto", origin="lower", cmap="viridis")

colors = plt.cm.Set1(np.linspace(0, 1, len(class_list)))
boxes = targets["boxes"].numpy()
labels = targets["labels"].numpy()

for box, label_idx in zip(boxes, labels):
    x1, y1, x2, y2 = box
    w, h = x2 - x1, y2 - y1
    color = colors[label_idx % len(colors)]
    rect = patches.Rectangle((x1, y1), w, h, linewidth=2,
                              edgecolor=color, facecolor="none")
    ax.add_patch(rect)
    ax.text(x1, y2 + 2, class_list[label_idx], color=color,
            fontsize=9, fontweight="bold",
            bbox=dict(boxstyle="round,pad=0.2", facecolor="black", alpha=0.5))

ax.set_xlabel("Time Bin")
ax.set_ylabel("Frequency Bin")
ax.set_title("Wideband Scene with COCO Bounding Boxes")
fig.tight_layout()
savefig("05_wideband_scene.png")

# ── 4. Generate multiple scenes ──────────────────────────────────────────────

fig, axes = plt.subplots(2, 3, figsize=(18, 10))
for idx, ax in enumerate(axes.flat):
    iq_i, descs_i = composer.generate(seed=idx * 7)
    spec_i = dsp.compute_spectrogram(iq_i, nfft=nfft, hop=hop)
    spec_db_i = 10 * np.log10(spec_i + 1e-12)
    ax.imshow(spec_db_i, aspect="auto", origin="lower", cmap="viridis")

    stft_p = STFTParams(nfft=nfft, hop_length=hop,
                        sample_rate=config.sample_rate, num_samples=len(iq_i))
    cls = sorted(set(d.label for d in descs_i))
    tgt = sp.to_coco(descs_i, stft_p, cls)
    for box, li in zip(tgt["boxes"].numpy(), tgt["labels"].numpy()):
        x1, y1, x2, y2 = box
        rect = patches.Rectangle((x1, y1), x2 - x1, y2 - y1,
                                  linewidth=1.5, edgecolor="red", facecolor="none")
        ax.add_patch(rect)
    signal_labels = [d.label for d in descs_i]
    ax.set_title(f"Scene {idx}: {', '.join(signal_labels)}", fontsize=9)

fig.suptitle("Multiple Wideband Scenes", fontsize=14)
fig.tight_layout()
savefig("05_multiple_scenes.png")

# ── 5. WidebandDataset with DataLoader ───────────────────────────────────────

wideband_ds = sp.WidebandDataset(
    scene_config=config,
    num_samples=16,
    transform=sp.STFT(nfft=512, hop_length=128),
    seed=42,
)

loader = torch.utils.data.DataLoader(
    wideband_ds,
    batch_size=4,
    collate_fn=sp.collate_fn,
)

for batch_data, batch_targets in loader:
    print(f"Batch shape: {batch_data.shape}")
    print(f"Num targets in first sample: {len(batch_targets[0]['boxes'])}")
    break

# Visualize batch
fig, axes = plt.subplots(1, 4, figsize=(18, 4))
batch_data_np = batch_data.numpy()
for i, ax in enumerate(axes):
    ax.imshow(batch_data_np[i, 0], aspect="auto", origin="lower", cmap="viridis")
    n_sigs = len(batch_targets[i]["boxes"])
    ax.set_title(f"Sample {i} ({n_sigs} signals)")
fig.suptitle("WidebandDataset — DataLoader Batch", fontsize=14)
fig.tight_layout()
savefig("05_wideband_batch.png")

plt.show()
print("\nDone! Check examples/outputs/ for saved figures.")
```

**Step 2: Create matching Jupyter notebook**

**Step 3: Run and verify**

```bash
cd examples && python 05_wideband_scenes.py
ls outputs/05_*.png
```

Expected: 3 PNG files.

**Step 4: Commit**

```bash
git add examples/05_wideband_scenes.py examples/05_wideband_scenes.ipynb examples/outputs/05_*.png
git commit -m "feat(examples): add 05 wideband scene composition (pro)"
```

---

### Task 7: Example 06 — Full Pipeline: Dataset to Classifier (Pro)

**Files:**
- Create: `examples/06_full_pipeline.ipynb`
- Create: `examples/06_full_pipeline.py`

**Skill level:** Pro. End-to-end workflow: generate dataset, apply augmentations, train a simple CNN classifier, evaluate, and persist to Zarr.

**Step 1: Write the Python script**

Create `examples/06_full_pipeline.py`:

```python
"""
SPECTRA Example 06: Full Pipeline — Dataset Generation to Classification
=========================================================================
Level: Pro

Learn how to:
- Generate a reproducible AMC dataset with diverse waveforms
- Chain transforms and augmentations
- Train a simple CNN classifier on spectrograms
- Evaluate accuracy per class and per SNR
- Save datasets to Zarr for reuse
- Use dataset metadata for reproducibility
"""

import sys
sys.path.insert(0, ".")

import numpy as np
import matplotlib.pyplot as plt
import torch
import torch.nn as nn
from torch.utils.data import DataLoader, random_split
import spectra as sp
from plot_helpers import savefig

# ── 1. Dataset configuration ────────────────────────────────────────────────

sample_rate = 1e6
num_iq_samples = 1024
nfft = 64

waveform_pool = [
    sp.BPSK(),
    sp.QPSK(),
    sp.PSK8(),
    sp.QAM16(),
    sp.QAM64(),
    sp.FSK(),
    sp.GMSK(),
    sp.OFDM(),
]

class_names = [w.label for w in waveform_pool]
num_classes = len(class_names)

impairments = sp.Compose([
    sp.AWGN(snr_range=(0, 20)),
    sp.FrequencyOffset(max_offset=1000),
    sp.PhaseNoise(noise_power_db=-35),
])

# ── 2. Build train/val datasets ─────────────────────────────────────────────

dataset = sp.NarrowbandDataset(
    waveform_pool=waveform_pool,
    num_samples=1600,  # 200 per class
    num_iq_samples=num_iq_samples,
    sample_rate=sample_rate,
    impairments=impairments,
    transform=sp.STFT(nfft=nfft, hop_length=nfft // 4),
    target_transform=sp.ClassIndex(class_names),
    seed=42,
)

train_size = int(0.8 * len(dataset))
val_size = len(dataset) - train_size
train_ds, val_ds = random_split(dataset, [train_size, val_size],
                                generator=torch.Generator().manual_seed(42))

print(f"Train: {len(train_ds)}, Val: {len(val_ds)}")
sample, label = train_ds[0]
print(f"Sample shape: {sample.shape}, label: {label} ({class_names[label]})")

train_loader = DataLoader(train_ds, batch_size=32, shuffle=True)
val_loader = DataLoader(val_ds, batch_size=32)

# ── 3. Simple CNN Classifier ────────────────────────────────────────────────

class SimpleCNN(nn.Module):
    def __init__(self, num_classes, input_channels=1):
        super().__init__()
        self.features = nn.Sequential(
            nn.Conv2d(input_channels, 32, 3, padding=1),
            nn.ReLU(),
            nn.AdaptiveAvgPool2d((8, 8)),
            nn.Conv2d(32, 64, 3, padding=1),
            nn.ReLU(),
            nn.AdaptiveAvgPool2d((4, 4)),
        )
        self.classifier = nn.Sequential(
            nn.Flatten(),
            nn.Linear(64 * 4 * 4, 128),
            nn.ReLU(),
            nn.Dropout(0.3),
            nn.Linear(128, num_classes),
        )

    def forward(self, x):
        return self.classifier(self.features(x))

model = SimpleCNN(num_classes)
optimizer = torch.optim.Adam(model.parameters(), lr=1e-3)
criterion = nn.CrossEntropyLoss()

# ── 4. Training loop ────────────────────────────────────────────────────────

num_epochs = 15
train_losses = []
val_accuracies = []

for epoch in range(num_epochs):
    model.train()
    epoch_loss = 0.0
    for batch_x, batch_y in train_loader:
        optimizer.zero_grad()
        out = model(batch_x)
        loss = criterion(out, batch_y)
        loss.backward()
        optimizer.step()
        epoch_loss += loss.item()
    train_losses.append(epoch_loss / len(train_loader))

    model.eval()
    correct = 0
    total = 0
    with torch.no_grad():
        for batch_x, batch_y in val_loader:
            preds = model(batch_x).argmax(dim=1)
            correct += (preds == batch_y).sum().item()
            total += len(batch_y)
    val_acc = correct / total
    val_accuracies.append(val_acc)

    if (epoch + 1) % 5 == 0:
        print(f"Epoch {epoch+1}/{num_epochs} — "
              f"Loss: {train_losses[-1]:.4f}, Val Acc: {val_acc:.2%}")

# ── 5. Plot training curves ─────────────────────────────────────────────────

fig, axes = plt.subplots(1, 2, figsize=(12, 4))
axes[0].plot(train_losses, linewidth=1.5)
axes[0].set_xlabel("Epoch")
axes[0].set_ylabel("Loss")
axes[0].set_title("Training Loss")
axes[0].grid(True, alpha=0.3)

axes[1].plot(val_accuracies, linewidth=1.5, color="tab:green")
axes[1].set_xlabel("Epoch")
axes[1].set_ylabel("Accuracy")
axes[1].set_title("Validation Accuracy")
axes[1].grid(True, alpha=0.3)
fig.tight_layout()
savefig("06_training_curves.png")

# ── 6. Per-class accuracy ───────────────────────────────────────────────────

model.eval()
class_correct = np.zeros(num_classes)
class_total = np.zeros(num_classes)

with torch.no_grad():
    for batch_x, batch_y in val_loader:
        preds = model(batch_x).argmax(dim=1)
        for pred, true in zip(preds, batch_y):
            class_total[true] += 1
            if pred == true:
                class_correct[true] += 1

class_acc = class_correct / np.maximum(class_total, 1)

fig, ax = plt.subplots(figsize=(10, 5))
bars = ax.bar(class_names, class_acc, color="steelblue")
ax.set_ylim(0, 1.05)
ax.set_xlabel("Modulation Class")
ax.set_ylabel("Accuracy")
ax.set_title("Per-Class Validation Accuracy")
ax.grid(True, alpha=0.3, axis="y")
for bar, acc in zip(bars, class_acc):
    ax.text(bar.get_x() + bar.get_width() / 2, bar.get_height() + 0.02,
            f"{acc:.0%}", ha="center", fontsize=9)
fig.tight_layout()
savefig("06_per_class_accuracy.png")

# ── 7. Confusion matrix ─────────────────────────────────────────────────────

confusion = np.zeros((num_classes, num_classes), dtype=int)
with torch.no_grad():
    for batch_x, batch_y in val_loader:
        preds = model(batch_x).argmax(dim=1)
        for pred, true in zip(preds, batch_y):
            confusion[true, pred] += 1

fig, ax = plt.subplots(figsize=(8, 7))
im = ax.imshow(confusion, cmap="Blues")
ax.set_xticks(range(num_classes))
ax.set_yticks(range(num_classes))
ax.set_xticklabels(class_names, rotation=45, ha="right")
ax.set_yticklabels(class_names)
ax.set_xlabel("Predicted")
ax.set_ylabel("True")
ax.set_title("Confusion Matrix")
for i in range(num_classes):
    for j in range(num_classes):
        ax.text(j, i, str(confusion[i, j]), ha="center", va="center",
                color="white" if confusion[i, j] > confusion.max() / 2 else "black")
fig.colorbar(im)
fig.tight_layout()
savefig("06_confusion_matrix.png")

# ── 8. Save dataset metadata ────────────────────────────────────────────────

metadata = sp.NarrowbandMetadata(
    name="amc_8class",
    num_samples=1600,
    sample_rate=sample_rate,
    seed=42,
    waveform_labels=class_names,
    num_iq_samples=num_iq_samples,
    snr_range=(0, 20),
)
metadata.to_yaml("outputs/06_dataset_metadata.yaml")
print(f"\nMetadata saved to outputs/06_dataset_metadata.yaml")

# Demonstrate reload
loaded = sp.NarrowbandMetadata.from_yaml("outputs/06_dataset_metadata.yaml")
print(f"Reloaded: {loaded.name}, {loaded.num_samples} samples, "
      f"{len(loaded.waveform_labels)} classes")

plt.show()
print("\nDone! Check examples/outputs/ for saved figures.")
```

**Step 2: Create matching Jupyter notebook**

**Step 3: Run and verify**

```bash
cd examples && python 06_full_pipeline.py
ls outputs/06_*.png outputs/06_*.yaml
```

Expected: 3 PNG files + 1 YAML file.

**Step 4: Commit**

```bash
git add examples/06_full_pipeline.py examples/06_full_pipeline.ipynb examples/outputs/06_*.png examples/outputs/06_*.yaml
git commit -m "feat(examples): add 06 full pipeline — dataset to classifier (pro)"
```

---

### Task 8: Final verification and cleanup

**Step 1: Run all examples end-to-end**

```bash
cd examples
for f in 01_basic_waveforms.py 02_impairments_and_channels.py 03_transforms_and_spectrograms.py 04_narrowband_dataset.py 05_wideband_scenes.py 06_full_pipeline.py; do
    echo "=== Running $f ==="
    python $f
    echo ""
done
```

All 6 scripts should run without error. Verify all expected PNGs exist in `outputs/`.

**Step 2: Run pytest to ensure examples didn't break anything**

```bash
pytest tests/ -v
```

All existing tests must still pass.

**Step 3: Final commit**

```bash
git add examples/
git commit -m "feat(examples): complete example suite — 6 notebooks from novice to pro"
```

---

## Summary

| # | Notebook | Level | Topics |
|---|----------|-------|--------|
| 01 | Basic Waveforms | Novice | Generate, IQ plots, constellation, PSD, analog vs digital |
| 02 | Impairments & Channels | Intermediate | AWGN, freq offset, phase noise, IQ imbalance, fading, Compose |
| 03 | Transforms & Spectrograms | Intermediate | STFT, Spectrogram, ComplexTo2D, Normalize, augmentations, DSP utils |
| 04 | Narrowband Dataset | Advanced | NarrowbandDataset, DataLoader, ClassIndex, FamilyIndex, per-class viz |
| 05 | Wideband Scenes | Pro | SceneConfig, Composer, COCO labels, bounding boxes, WidebandDataset |
| 06 | Full Pipeline | Pro | End-to-end AMC: dataset → CNN → confusion matrix → Zarr metadata |

**Figures saved:** ~22 PNGs + 1 YAML in `examples/outputs/`.
