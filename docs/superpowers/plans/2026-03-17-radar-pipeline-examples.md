# Radar Pipeline Examples Implementation Plan

> **For agentic workers:** REQUIRED: Use superpowers:subagent-driven-development (if subagents available) or superpowers:executing-plans to implement this plan. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Add Example 17 (Python script + Jupyter notebook) showcasing the full radar processing pipeline: trajectories, Swerling RCS, clutter, MTI, range-Doppler maps, CFAR, and Kalman tracking via `RadarPipelineDataset`.

**Architecture:** One example with 7 sections covering all new radar capabilities. Follows the existing pattern: config constants, numbered sections with comment separators, `plot_helpers.savefig()` for all figures. Notebook mirrors the script with explanatory markdown cells.

**Tech Stack:** Python, NumPy, matplotlib, PyTorch, SPECTRA radar modules.

---

## File Structure

| File | Responsibility |
|------|---------------|
| `examples/17_radar_pipeline.py` | Python script — full pipeline walkthrough |
| `examples/17_radar_pipeline.ipynb` | Jupyter notebook — same content with markdown explanations |
| `examples/README.md` | Modify: add example 17 entry + file structure line |

---

## Task 1: Python Script

**Files:**
- Create: `examples/17_radar_pipeline.py`

- [ ] **Step 1: Write the script**

```python
# examples/17_radar_pipeline.py
"""Example 17 — Radar Processing Pipeline
==========================================
Level: Advanced

This example shows how to:
  1. Define target trajectories (constant velocity + constant turn rate)
  2. Generate Swerling RCS amplitude fluctuations (cases 0-IV)
  3. Apply radar clutter with terrain-typed presets (ground, sea, weather)
  4. Use MTI pulse cancellers and Doppler filter banks
  5. Build a RadarPipelineDataset with end-to-end processing
  6. Visualise range-Doppler maps, CFAR detections, and Kalman tracks

Run:
    python examples/17_radar_pipeline.py
"""

import os
import sys
sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))
sys.path.insert(0, os.path.dirname(__file__))

import numpy as np
import matplotlib.pyplot as plt
import torch

from spectra.targets import ConstantVelocity, ConstantTurnRate
from spectra.targets import NonFluctuatingRCS, SwerlingRCS
from spectra.impairments import RadarClutter
from spectra.algorithms import (
    single_pulse_canceller,
    double_pulse_canceller,
    doppler_filter_bank,
)
from spectra.tracking import ConstantVelocityKF
from spectra.datasets import RadarPipelineDataset
from spectra.waveforms import LFM, BarkerCodedPulse

from plot_helpers import savefig, OUTPUT_DIR

# ── Configuration ──────────────────────────────────────────────────────────────

NUM_STEPS       = 50         # trajectory time steps
DT              = 0.5        # seconds per step
NUM_RANGE_BINS  = 256
SAMPLE_RATE     = 1e6        # Hz
CARRIER_FREQ    = 10e9       # 10 GHz
PRI             = 1e-3       # pulse repetition interval
PULSES_PER_CPI  = 32
SEQ_LEN         = 10         # CPIs per pipeline sample
N_SAMPLES       = 20
SEED            = 42

# ── 1. Target Trajectories ────────────────────────────────────────────────────

cv = ConstantVelocity(initial_range=100.0, velocity=2.0, dt=DT)
ct = ConstantTurnRate(initial_range=150.0, velocity=3.0, turn_rate=0.05, dt=DT)

cv_states = cv.states(NUM_STEPS)
ct_states = ct.states(NUM_STEPS)

fig, axes = plt.subplots(2, 1, figsize=(9, 6), sharex=True)
t = np.arange(NUM_STEPS) * DT

axes[0].plot(t, cv_states[:, 0], label="CV", color="steelblue")
axes[0].plot(t, ct_states[:, 0], label="CT", color="coral")
axes[0].set_ylabel("Range (bin units)")
axes[0].set_title("Target Range Trajectories")
axes[0].legend()
axes[0].grid(True, alpha=0.3)

axes[1].plot(t, cv_states[:, 1], label="CV", color="steelblue")
axes[1].plot(t, ct_states[:, 1], label="CT", color="coral")
axes[1].set_ylabel("Range rate (bins/s)")
axes[1].set_xlabel("Time (s)")
axes[1].set_title("Target Range Rate")
axes[1].legend()
axes[1].grid(True, alpha=0.3)

plt.tight_layout()
savefig("17_trajectories.png")
print("1. Trajectories plotted")

# ── 2. Swerling RCS Models ────────────────────────────────────────────────────

rng = np.random.default_rng(SEED)
num_dwells, num_pulses = 20, 16

fig, axes = plt.subplots(2, 2, figsize=(10, 6), sharex=True, sharey=True)
for ax, case in zip(axes.flat, [1, 2, 3, 4]):
    rcs = SwerlingRCS(case=case, sigma=1.0)
    amps = rcs.amplitudes(num_dwells, num_pulses, np.random.default_rng(SEED))
    im = ax.imshow(amps.T, aspect="auto", origin="lower", cmap="viridis",
                   extent=[0, num_dwells, 0, num_pulses])
    ax.set_title(f"Swerling {case}")
    ax.set_xlabel("Dwell")
    ax.set_ylabel("Pulse")

fig.suptitle("Swerling RCS Amplitude Patterns")
plt.tight_layout()
savefig("17_swerling_rcs.png")
print("2. Swerling RCS models plotted")

# ── 3. Radar Clutter Comparison ───────────────────────────────────────────────

fig, axes = plt.subplots(1, 3, figsize=(12, 4))
presets = [
    ("Ground (rural)", RadarClutter.ground(SAMPLE_RATE, terrain="rural")),
    ("Sea (state 4)",  RadarClutter.sea(SAMPLE_RATE, sea_state=4)),
    ("Weather (20 mm/hr)", RadarClutter.weather(SAMPLE_RATE, rain_rate_mmhr=20)),
]

for ax, (label, clutter) in zip(axes, presets):
    clean = np.zeros((64, 128), dtype=complex)
    cluttered = clutter(clean, np.random.default_rng(SEED))
    spec = np.abs(np.fft.fft(cluttered[:, 64], n=64)) ** 2
    freqs = np.fft.fftfreq(64, d=1.0 / SAMPLE_RATE)
    ax.plot(np.fft.fftshift(freqs) / 1e3, 10 * np.log10(np.fft.fftshift(spec) + 1e-30),
            color="steelblue")
    ax.set_title(label)
    ax.set_xlabel("Doppler (kHz)")
    ax.set_ylabel("Power (dB)")
    ax.grid(True, alpha=0.3)

fig.suptitle("Clutter Doppler Spectra (single range bin)")
plt.tight_layout()
savefig("17_clutter_spectra.png")
print("3. Clutter spectra plotted")

# ── 4. MTI and Range-Doppler Map ──────────────────────────────────────────────

# Build a pulse matrix with clutter + one moving target
pulse_matrix = np.ones((PULSES_PER_CPI, NUM_RANGE_BINS), dtype=complex) * 3.0
target_bin = 120
f_d = 300.0  # Hz
for n in range(PULSES_PER_CPI):
    pulse_matrix[n, target_bin] += 20.0 * np.exp(1j * 2 * np.pi * f_d * n * PRI)

# Add noise
pulse_matrix += np.sqrt(0.5) * (
    rng.standard_normal(pulse_matrix.shape) + 1j * rng.standard_normal(pulse_matrix.shape)
)

# Before MTI
rdm_before = doppler_filter_bank(pulse_matrix, window="hann")
# After single pulse canceller
cancelled = single_pulse_canceller(pulse_matrix)
rdm_after = doppler_filter_bank(cancelled, window="hann")

fig, axes = plt.subplots(1, 2, figsize=(12, 4))
for ax, rdm, title in zip(axes, [rdm_before, rdm_after],
                            ["Before MTI", "After Single Pulse Canceller"]):
    rdm_db = 10 * np.log10(rdm + 1e-30)
    im = ax.imshow(rdm_db, aspect="auto", origin="lower", cmap="inferno",
                   extent=[0, NUM_RANGE_BINS, 0, rdm.shape[0]])
    ax.axvline(target_bin, color="lime", linestyle="--", linewidth=1, alpha=0.7)
    ax.set_xlabel("Range bin")
    ax.set_ylabel("Doppler bin")
    ax.set_title(title)
    plt.colorbar(im, ax=ax, label="dB")

plt.tight_layout()
savefig("17_mti_rdm.png")
print("4. MTI range-Doppler maps plotted")

# ── 5. RadarPipelineDataset ───────────────────────────────────────────────────

ds = RadarPipelineDataset(
    waveform_pool=[LFM(), BarkerCodedPulse()],
    trajectory_pool=[
        ConstantVelocity(initial_range=100.0, velocity=0.5, dt=1.0),
        ConstantTurnRate(initial_range=120.0, velocity=0.3, turn_rate=0.02, dt=1.0),
    ],
    swerling_cases=[0, 1, 2],
    clutter_presets=[
        RadarClutter.ground(SAMPLE_RATE, terrain="rural"),
        RadarClutter.sea(SAMPLE_RATE, sea_state=3),
    ],
    num_range_bins=NUM_RANGE_BINS,
    sample_rate=SAMPLE_RATE,
    carrier_frequency=CARRIER_FREQ,
    pri=PRI,
    snr_range=(10.0, 25.0),
    num_targets_range=(1, 2),
    sequence_length=SEQ_LEN,
    pulses_per_cpi=PULSES_PER_CPI,
    apply_mti=True,
    cfar_type="ca",
    num_samples=N_SAMPLES,
    seed=SEED,
)

data, target = ds[0]
print(f"\n5. Pipeline dataset: {len(ds)} samples")
print(f"   Output shape: {data.shape}")
print(f"   Targets: {target.num_targets}, waveform: {target.waveform_label}")
print(f"   SNR: {target.snr_db:.1f} dB, clutter: {target.clutter_preset}")

# ── 6. Range Profile Sequence ─────────────────────────────────────────────────

fig, ax = plt.subplots(figsize=(10, 5))
im = ax.imshow(data.numpy(), aspect="auto", origin="lower", cmap="viridis",
               extent=[0, NUM_RANGE_BINS, 0, SEQ_LEN])
# Mark true target ranges
for k in range(target.num_targets):
    ax.plot(target.true_ranges[:, k], np.arange(SEQ_LEN), "r--", linewidth=1.5,
            label=f"Target {k} true range" if k == 0 else None)
# Mark CFAR detections
for frame, dets in enumerate(target.detections):
    if len(dets) > 0:
        ax.scatter(dets, [frame] * len(dets), c="yellow", s=10, zorder=5,
                   label="CFAR detections" if frame == 0 else None)
ax.set_xlabel("Range bin")
ax.set_ylabel("CPI frame")
ax.set_title("Range Profile Sequence with Targets and Detections")
ax.legend(fontsize=8, loc="upper right")
plt.colorbar(im, ax=ax, label="Normalised power (dB)")
plt.tight_layout()
savefig("17_range_sequence.png")
print("6. Range profile sequence plotted")

# ── 7. Kalman Tracking ────────────────────────────────────────────────────────

# Standalone KF demo: track target 0 from noisy measurements
kf = ConstantVelocityKF(
    dt=PRI * PULSES_PER_CPI,
    process_noise_std=1.0,
    measurement_noise_std=5.0,
    x0=np.array([target.true_ranges[0, 0], target.true_velocities[0, 0]]),
)
kf_range_est = []
for frame in range(SEQ_LEN):
    kf.predict()
    # Use true range + noise as a measurement (for demo clarity)
    noisy_meas = target.true_ranges[frame, 0] + rng.normal(0, 3)
    kf.update(np.array([noisy_meas]))
    kf_range_est.append(kf.state[0])

fig, ax = plt.subplots(figsize=(9, 5))
frames = np.arange(SEQ_LEN)

for k in range(target.num_targets):
    ax.plot(frames, target.true_ranges[:, k], "o-", markersize=4,
            label=f"Target {k} true", alpha=0.8)
    ax.plot(frames, target.kf_states[:, k, 0], "s--", markersize=5,
            label=f"Target {k} pipeline KF", alpha=0.7)

ax.plot(frames, kf_range_est, "x-", markersize=6, color="green",
        label="Standalone KF (target 0)")
ax.set_xlabel("CPI frame")
ax.set_ylabel("Range (bin units)")
ax.set_title("Kalman Filter Tracking vs Ground Truth")
ax.legend(fontsize=8)
ax.grid(True, alpha=0.3)
plt.tight_layout()
savefig("17_kalman_tracking.png")
print("7. Kalman tracking plotted")

print(f"\nAll figures saved to {OUTPUT_DIR}")
```

- [ ] **Step 2: Verify the script runs**

```bash
cd /Users/gditzler/git/SPECTRA && /Users/gditzler/.venvs/base/bin/python examples/17_radar_pipeline.py
```
Expected: prints section confirmations, saves 6 PNG figures to `examples/outputs/`.

- [ ] **Step 3: Commit**

```bash
git add examples/17_radar_pipeline.py
git commit -m "feat(examples): add radar pipeline example script (example 17)"
```

---

## Task 2: Jupyter Notebook

**Files:**
- Create: `examples/17_radar_pipeline.ipynb`

- [ ] **Step 1: Create the notebook**

The notebook has 14 cells mirroring the script with explanatory markdown cells. Use `nbformat` 4 / `nbformat_minor` 5.

| # | Type | Content |
|---|------|---------|
| 0 | markdown | Title, level, learning objectives |
| 1 | code | Imports + configuration constants |
| 2 | markdown | "## 1. Target Trajectories" — explain CV and CT models |
| 3 | code | Create trajectories, plot range + range rate |
| 4 | markdown | "## 2. Swerling RCS Models" — explain cases I-IV |
| 5 | code | Generate and visualise amplitude patterns |
| 6 | markdown | "## 3. Radar Clutter" — explain presets |
| 7 | code | Compare ground/sea/weather Doppler spectra |
| 8 | markdown | "## 4. MTI and Range-Doppler Maps" — explain pulse cancellation |
| 9 | code | Before/after MTI range-Doppler maps |
| 10 | markdown | "## 5. RadarPipelineDataset" — explain end-to-end pipeline |
| 11 | code | Build dataset, inspect sample, plot range profile sequence |
| 12 | markdown | "## 6. Kalman Tracking" — explain KF for range tracking |
| 13 | code | Plot KF estimates vs ground truth |

**Cell 0 (markdown):**
```markdown
# Example 17 — Radar Processing Pipeline

**Level:** Advanced

After working through this notebook you will know how to:

- Define target trajectories with `ConstantVelocity` and `ConstantTurnRate`
- Generate Swerling RCS fluctuations (cases 0–IV) with `SwerlingRCS`
- Apply radar clutter using terrain-typed presets (`RadarClutter.ground()`, `.sea()`, `.weather()`)
- Use MTI pulse cancellers and Doppler filter banks for clutter rejection
- Build a `RadarPipelineDataset` for end-to-end radar ML training
- Visualise Kalman filter tracking against ground truth
```

**Cell 1 (code):** Same imports and config as the script (with `sys.path.insert(0, ".")` and `%matplotlib inline`).

**Cell 2 (markdown):**
```markdown
## 1. Target Trajectories

`ConstantVelocity` models a target moving at fixed range rate:
`range(t) = initial_range + velocity × t`.

`ConstantTurnRate` projects a 2-D circular arc onto 1-D range,
producing sinusoidal range variation. Both satisfy the `Trajectory`
protocol and return state vectors `[range, range_rate]`.
```

**Cell 3 (code):** Trajectory creation + 2-panel plot (same as script section 1).

**Cell 4 (markdown):**
```markdown
## 2. Swerling RCS Models

Swerling models describe how a target's radar cross-section fluctuates:

| Case | Fluctuation | Distribution |
|------|------------|-------------|
| I | Constant within dwell | Chi-squared, 2 DoF |
| II | Pulse-to-pulse | Chi-squared, 2 DoF |
| III | Constant within dwell | Chi-squared, 4 DoF |
| IV | Pulse-to-pulse | Chi-squared, 4 DoF |

Cases I/III show horizontal bands (constant per dwell).
Cases II/IV show independent fluctuation at each pulse.
```

**Cell 5 (code):** 2x2 Swerling amplitude heatmaps (same as script section 2).

**Cell 6 (markdown):**
```markdown
## 3. Radar Clutter

`RadarClutter` generates Doppler-coloured Gaussian noise on a slow-time/fast-time
pulse matrix. Three preset factories set physically-motivated parameters:

- `.ground(terrain)` — low Doppler spread (wind-blown vegetation)
- `.sea(sea_state)` — moderate spread from wave motion
- `.weather(rain_rate_mmhr)` — nonzero Doppler center (moving precipitation)
```

**Cell 7 (code):** 3-panel clutter Doppler spectra (same as script section 3).

**Cell 8 (markdown):**
```markdown
## 4. MTI and Range-Doppler Maps

Moving Target Indication suppresses stationary (zero-Doppler) clutter:

- `single_pulse_canceller`: `y[n] = x[n+1] - x[n]` — nulls DC
- `double_pulse_canceller`: `y[n] = x[n+2] - 2x[n+1] + x[n]` — deeper null
- `doppler_filter_bank`: FFT along slow-time → range-Doppler power map

The left panel shows clutter dominating all range bins at DC. After the
single pulse canceller, the moving target emerges clearly.
```

**Cell 9 (code):** Before/after MTI range-Doppler maps (same as script section 4).

**Cell 10 (markdown):**
```markdown
## 5. RadarPipelineDataset

`RadarPipelineDataset` chains the full processing pipeline per sample:

1. Generate waveform template
2. For each CPI frame: inject targets → add noise + clutter → matched filter → MTI → Doppler filter bank → CFAR
3. Run Kalman filter over CFAR detections

Output: `(Tensor[sequence_length, num_range_bins], RadarPipelineTarget)`
```

**Cell 11 (code):** Build dataset + range profile sequence plot (script sections 5-6).

**Cell 12 (markdown):**
```markdown
## 6. Kalman Tracking

A `ConstantVelocityKF` (state: `[range, range_rate]`) tracks each target across
CPI frames using CFAR detections as measurements. Ground-truth-aided association
assigns each detection to the nearest predicted track.

We also demonstrate standalone `ConstantVelocityKF` usage: instantiate a filter,
feed it noisy range measurements, and compare its estimates to both ground truth
and the pipeline's internal tracker.
```

**Cell 13 (code):** Kalman tracking plot (script section 7).

- [ ] **Step 2: Verify the notebook executes**

```bash
cd /Users/gditzler/git/SPECTRA/examples && /Users/gditzler/.venvs/base/bin/jupyter nbconvert --to notebook --execute 17_radar_pipeline.ipynb --output /tmp/check_17.ipynb 2>&1 && echo "OK"
```
Expected: exits 0, creates `/tmp/check_17.ipynb`.

- [ ] **Step 3: Commit**

```bash
git add examples/17_radar_pipeline.ipynb
git commit -m "feat(examples): add radar pipeline Jupyter notebook (example 17)"
```

---

## Task 3: Update README

**Files:**
- Modify: `examples/README.md`

- [ ] **Step 1: Add example 17 entry**

Add after the example 16 entry (before the `## Output` section):

```markdown
### 17 — Radar Processing Pipeline (Advanced)

End-to-end radar simulation pipeline: define target trajectories, generate
Swerling RCS fluctuations, apply terrain-typed clutter, process with MTI
and Doppler filter banks, and track targets with a Kalman filter via
`RadarPipelineDataset`.

- Define **CV** and **CT** target trajectories with `ConstantVelocity` / `ConstantTurnRate`
- Generate **Swerling RCS** amplitude patterns (cases 0–IV)
- Apply **radar clutter** presets (ground, sea, weather) via `RadarClutter`
- Visualise **MTI** clutter suppression and **range-Doppler maps**
- Build a `RadarPipelineDataset` with waveform → channel → receiver → tracker
- Compare **Kalman filter** track estimates against ground truth

```bash
cd examples && python 17_radar_pipeline.py
```
```

- [ ] **Step 2: Add to file structure**

Add in the file structure tree:

```
  17_radar_pipeline.ipynb / .py       # Advanced
```

- [ ] **Step 3: Commit**

```bash
git add examples/README.md
git commit -m "docs(examples): add example 17 radar pipeline to README"
```

---

## Task 4: Full Verification

- [ ] **Step 1: Run full test suite**

```bash
/Users/gditzler/.venvs/base/bin/pytest tests/ -q --tb=short
```
Expected: all tests pass, no regressions.

- [ ] **Step 2: Run the example end-to-end**

```bash
/Users/gditzler/.venvs/base/bin/python examples/17_radar_pipeline.py
```
Expected: completes, prints section confirmations.

- [ ] **Step 3: Confirm output figures exist**

```bash
ls examples/outputs/17_*.png
```
Expected: 6 PNG files.
