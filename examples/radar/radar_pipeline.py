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

import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

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
from spectra.tracking import ConstantVelocityKF, RangeDopplerKF
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

# ── 8. Range+Doppler Tracking ─────────────────────────────────────────────────

# Build two datasets with the same seed — one range-only, one range+Doppler
common_args = dict(
    waveform_pool=[LFM()],
    trajectory_pool=[ConstantVelocity(initial_range=100.0, velocity=0.5, dt=1.0)],
    swerling_cases=[0],
    clutter_presets=[RadarClutter.ground(SAMPLE_RATE, terrain="rural")],
    num_range_bins=NUM_RANGE_BINS, sample_rate=SAMPLE_RATE,
    carrier_frequency=CARRIER_FREQ, pri=PRI,
    snr_range=(15.0, 25.0), num_targets_range=(1, 1),
    sequence_length=SEQ_LEN, pulses_per_cpi=PULSES_PER_CPI,
    apply_mti=True, cfar_type="ca", num_samples=5, seed=SEED,
)

ds_range = RadarPipelineDataset(**common_args, track_doppler=False)
ds_doppler = RadarPipelineDataset(**common_args, track_doppler=True)

_, tgt_r = ds_range[0]
_, tgt_d = ds_doppler[0]

fig, axes = plt.subplots(1, 2, figsize=(12, 5))

# Range estimate comparison
axes[0].plot(frames, tgt_r.true_ranges[:, 0], "o-", markersize=4, label="True range")
axes[0].plot(frames, tgt_r.kf_states[:, 0, 0], "s--", markersize=5, label="Range-only KF")
axes[0].plot(frames, tgt_d.kf_states[:, 0, 0], "D--", markersize=5, label="Range+Doppler KF")
axes[0].set_xlabel("CPI frame")
axes[0].set_ylabel("Range (bin units)")
axes[0].set_title("Range Estimate Comparison")
axes[0].legend(fontsize=8)
axes[0].grid(True, alpha=0.3)

# Velocity estimate comparison
axes[1].plot(frames, tgt_r.true_velocities[:, 0], "o-", markersize=4, label="True velocity")
axes[1].plot(frames, tgt_r.kf_states[:, 0, 1], "s--", markersize=5, label="Range-only KF")
axes[1].plot(frames, tgt_d.kf_states[:, 0, 1], "D--", markersize=5, label="Range+Doppler KF")
axes[1].set_xlabel("CPI frame")
axes[1].set_ylabel("Velocity (bin units/frame)")
axes[1].set_title("Velocity Estimate Comparison")
axes[1].legend(fontsize=8)
axes[1].grid(True, alpha=0.3)

plt.tight_layout()
savefig("17_range_doppler_tracking.png")
print("8. Range+Doppler tracking comparison plotted")

print(f"\nAll figures saved to {OUTPUT_DIR}")
