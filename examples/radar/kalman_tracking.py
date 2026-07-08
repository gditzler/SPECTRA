"""Kalman Filter Radar Tracking
================================
Level: Intermediate / Advanced

This example gives a focused walkthrough of SPECTRA's tracking module:

  1. **Standalone KF basics** — build a ``ConstantVelocityKF`` from scratch,
     run the predict/update loop manually, and inspect covariance evolution.
  2. **Filter tuning** — sweep the process-noise / measurement-noise ratio and
     visualise its effect on track smoothness vs. agility.
  3. **Multi-target tracking** — track two simultaneous targets (one constant-
     velocity, one constant-turn-rate) and compare estimates against truth.
  4. **Range-only vs Range+Doppler** — use ``RangeDopplerKF`` inside
     ``RadarPipelineDataset`` (``track_doppler=True/False``) and compare the
     RMSE improvement Doppler measurements provide.
  5. **Track quality** — plot per-frame RMSE curves for range and velocity for
     both tracker variants.

Run::

    python examples/radar/kalman_tracking.py

Output images are written to ``examples/outputs/``.
"""

from __future__ import annotations

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

import matplotlib
import numpy as np

matplotlib.use("Agg")
import matplotlib.pyplot as plt
from plot_helpers import savefig
from spectra.datasets import RadarPipelineDataset
from spectra.impairments import RadarClutter
from spectra.targets import ConstantTurnRate, ConstantVelocity, SwerlingRCS
from spectra.tracking import ConstantVelocityKF, RangeDopplerKF
from spectra.waveforms import LFM

# ── Global configuration ──────────────────────────────────────────────────────
SEED = 42
rng = np.random.default_rng(SEED)

SAMPLE_RATE = 1e6       # Hz
CARRIER_FREQ = 10e9     # Hz  (X-band)
PRI = 1e-3              # seconds
PULSES_PER_CPI = 16
NUM_RANGE_BINS = 256
SEQ_LEN = 40            # CPI frames to track over
DT = PRI * PULSES_PER_CPI   # effective time step per CPI frame  [s]

print("=" * 60)
print("SPECTRA — Kalman Filter Radar Tracking Example")
print("=" * 60)

# ─────────────────────────────────────────────────────────────────────────────
# § 1. Standalone Kalman Filter Basics
#      Build a ConstantVelocityKF by hand, step through predict/update, and
#      watch how the covariance shrinks as measurements accumulate.
# ─────────────────────────────────────────────────────────────────────────────
print("\n§ 1 — Standalone Kalman filter basics")

# True target: starts at bin 80, moves at +0.4 bins/frame
TRUE_RANGE_0 = 80.0
TRUE_VEL_0 = 0.4          # bins per second
MEAS_NOISE_STD = 4.0      # measurement noise σ (bins)

true_ranges = TRUE_RANGE_0 + TRUE_VEL_0 * DT * np.arange(SEQ_LEN)
# Noisy range-only measurements
meas_ranges = true_ranges + rng.normal(0, MEAS_NOISE_STD, size=SEQ_LEN)

kf = ConstantVelocityKF(
    dt=DT,
    process_noise_std=0.5,          # moderate manoeuvre uncertainty
    measurement_noise_std=MEAS_NOISE_STD,
    x0=np.array([TRUE_RANGE_0, TRUE_VEL_0]),
)

kf_ranges, kf_vels, kf_cov_traces = [], [], []
for z in meas_ranges:
    kf.predict()
    kf.update(np.array([z]))
    kf_ranges.append(kf.state[0])
    kf_vels.append(kf.state[1])
    kf_cov_traces.append(np.trace(kf.covariance))

frames = np.arange(SEQ_LEN)

fig, axes = plt.subplots(3, 1, figsize=(10, 9), sharex=True)

axes[0].plot(frames, true_ranges, "k-", lw=1.5, label="Truth")
axes[0].plot(frames, meas_ranges, ".", color="gray", markersize=5, alpha=0.7, label="Measurements")
axes[0].plot(frames, kf_ranges, "o-", color="steelblue", markersize=4, lw=1.5, label="KF estimate")
axes[0].set_ylabel("Range (bins)")
axes[0].set_title("§ 1 — ConstantVelocityKF: Range Tracking")
axes[0].legend(fontsize=9)
axes[0].grid(True, alpha=0.3)

axes[1].plot(frames, [TRUE_VEL_0] * SEQ_LEN, "k-", lw=1.5, label="Truth")
axes[1].plot(frames, kf_vels, "s-", color="coral", markersize=4, lw=1.5, label="KF velocity estimate")
axes[1].set_ylabel("Range rate (bins/s)")
axes[1].set_title("Velocity Estimate")
axes[1].legend(fontsize=9)
axes[1].grid(True, alpha=0.3)

axes[2].semilogy(frames, kf_cov_traces, "v-", color="purple", markersize=4)
axes[2].set_ylabel("trace(P)  [log scale]")
axes[2].set_xlabel("CPI frame")
axes[2].set_title("Covariance Trace (filter confidence)")
axes[2].grid(True, alpha=0.3)

plt.tight_layout()
savefig("kf_tracking_01_basics.png")
plt.close()
print("  → saved kf_tracking_01_basics.png")

# ─────────────────────────────────────────────────────────────────────────────
# § 2. Filter Tuning — Process Noise vs Measurement Noise
#      The Q/R ratio governs the trade-off between track smoothness (low Q/R)
#      and the ability to follow manoeuvres (high Q/R).
# ─────────────────────────────────────────────────────────────────────────────
print("\n§ 2 — Filter tuning: Q/R sweep")

# Sinusoidal range profile to expose tracking lag on a turning target
t_arr = np.arange(SEQ_LEN) * DT
true_ranges_manoeuvre = 80.0 + 20.0 * np.sin(2 * np.pi * t_arr / (SEQ_LEN * DT * 0.6))
meas_manoeuvre = true_ranges_manoeuvre + rng.normal(0, MEAS_NOISE_STD, size=SEQ_LEN)

process_noise_values = [0.05, 0.5, 3.0, 10.0]   # σ_q values to sweep
colors = ["#264653", "#2a9d8f", "#e9c46a", "#e76f51"]

fig, axes = plt.subplots(2, 1, figsize=(11, 7), sharex=True)

axes[0].plot(frames, true_ranges_manoeuvre, "k-", lw=2, label="Truth", zorder=5)
axes[0].plot(frames, meas_manoeuvre, ".", color="lightgray", markersize=5, alpha=0.9, label="Meas.")

rmse_records = {}
for q_std, col in zip(process_noise_values, colors):
    kf_q = ConstantVelocityKF(
        dt=DT, process_noise_std=q_std,
        measurement_noise_std=MEAS_NOISE_STD,
        x0=np.array([true_ranges_manoeuvre[0], 0.0]),
    )
    est = []
    for z in meas_manoeuvre:
        kf_q.predict()
        kf_q.update(np.array([z]))
        est.append(kf_q.state[0])
    est = np.array(est)
    rmse = np.sqrt(np.mean((est - true_ranges_manoeuvre) ** 2))
    rmse_records[q_std] = rmse
    lbl = f"σ_q={q_std:.2f}  (RMSE={rmse:.2f} bins)"
    axes[0].plot(frames, est, "-", color=col, lw=1.4, label=lbl, alpha=0.85)

axes[0].set_ylabel("Range (bins)")
axes[0].set_title("§ 2 — Effect of Process Noise σ_q on Track Quality")
axes[0].legend(fontsize=8, loc="upper right")
axes[0].grid(True, alpha=0.3)

q_vals = list(rmse_records.keys())
rmse_vals = [rmse_records[q] for q in q_vals]
axes[1].bar([str(q) for q in q_vals], rmse_vals, color=colors, alpha=0.85, edgecolor="k", lw=0.8)
axes[1].set_xlabel("Process noise σ_q")
axes[1].set_ylabel("RMSE (bins)")
axes[1].set_title("Range RMSE per σ_q (lower = better for this scenario)")
axes[1].grid(True, alpha=0.3, axis="y")

plt.tight_layout()
savefig("kf_tracking_02_tuning.png")
plt.close()
print("  → saved kf_tracking_02_tuning.png")

# ─────────────────────────────────────────────────────────────────────────────
# § 3. Multi-Target Tracking
#      Simultaneously track a constant-velocity target and a constant-turn-rate
#      target, each with independent ConstantVelocityKF instances.
# ─────────────────────────────────────────────────────────────────────────────
print("\n§ 3 — Multi-target tracking (CV + CTR trajectories)")

traj_cv = ConstantVelocity(initial_range=60.0, velocity=0.5, dt=DT)
traj_ct = ConstantTurnRate(initial_range=120.0, velocity=1.5, turn_rate=0.08, dt=DT)

states_cv = traj_cv.states(SEQ_LEN)   # (SEQ_LEN, 2)  [range, range_rate]
states_ct = traj_ct.states(SEQ_LEN)

MEAS_NOISE_MULTI = 3.0
meas_cv = states_cv[:, 0] + rng.normal(0, MEAS_NOISE_MULTI, SEQ_LEN)
meas_ct = states_ct[:, 0] + rng.normal(0, MEAS_NOISE_MULTI, SEQ_LEN)

kf_cv = ConstantVelocityKF(
    dt=DT, process_noise_std=0.5, measurement_noise_std=MEAS_NOISE_MULTI,
    x0=states_cv[0],
)
kf_ct = ConstantVelocityKF(
    dt=DT, process_noise_std=2.0,    # higher Q — target is manoeuvring
    measurement_noise_std=MEAS_NOISE_MULTI,
    x0=states_ct[0],
)

est_cv_r, est_cv_v = [], []
est_ct_r, est_ct_v = [], []
for i in range(SEQ_LEN):
    kf_cv.predict(); kf_cv.update(np.array([meas_cv[i]]))
    kf_ct.predict(); kf_ct.update(np.array([meas_ct[i]]))
    est_cv_r.append(kf_cv.state[0]);  est_cv_v.append(kf_cv.state[1])
    est_ct_r.append(kf_ct.state[0]);  est_ct_v.append(kf_ct.state[1])

fig, axes = plt.subplots(2, 2, figsize=(13, 8))

# Range
ax = axes[0, 0]
ax.plot(frames, states_cv[:, 0], "k-", lw=1.5, label="CV truth")
ax.plot(frames, meas_cv, ".", color="lightblue", markersize=5, alpha=0.8)
ax.plot(frames, est_cv_r, "o-", color="steelblue", markersize=3, lw=1.4, label="CV KF")
ax.set_ylabel("Range (bins)"); ax.set_title("Target 1 — Constant Velocity")
ax.legend(fontsize=9); ax.grid(True, alpha=0.3)

ax = axes[0, 1]
ax.plot(frames, states_ct[:, 0], "k-", lw=1.5, label="CT truth")
ax.plot(frames, meas_ct, ".", color="lightsalmon", markersize=5, alpha=0.8)
ax.plot(frames, est_ct_r, "o-", color="coral", markersize=3, lw=1.4, label="CT KF")
ax.set_ylabel("Range (bins)"); ax.set_title("Target 2 — Constant Turn Rate")
ax.legend(fontsize=9); ax.grid(True, alpha=0.3)

# Velocity
ax = axes[1, 0]
ax.plot(frames, states_cv[:, 1], "k-", lw=1.5, label="CV truth")
ax.plot(frames, est_cv_v, "s-", color="steelblue", markersize=3, lw=1.4, label="CV KF")
ax.set_xlabel("CPI frame"); ax.set_ylabel("Range rate (bins/s)")
ax.set_title("CV Velocity Estimate"); ax.legend(fontsize=9); ax.grid(True, alpha=0.3)

ax = axes[1, 1]
ax.plot(frames, states_ct[:, 1], "k-", lw=1.5, label="CT truth")
ax.plot(frames, est_ct_v, "s-", color="coral", markersize=3, lw=1.4, label="CT KF")
ax.set_xlabel("CPI frame"); ax.set_ylabel("Range rate (bins/s)")
ax.set_title("CTR Velocity Estimate"); ax.legend(fontsize=9); ax.grid(True, alpha=0.3)

fig.suptitle("§ 3 — Multi-Target Kalman Tracking", fontsize=13, fontweight="bold")
plt.tight_layout()
savefig("kf_tracking_03_multitarget.png")
plt.close()

rmse_cv_r = np.sqrt(np.mean((np.array(est_cv_r) - states_cv[:, 0]) ** 2))
rmse_ct_r = np.sqrt(np.mean((np.array(est_ct_r) - states_ct[:, 0]) ** 2))
print(f"  CV target  — range RMSE: {rmse_cv_r:.3f} bins")
print(f"  CTR target — range RMSE: {rmse_ct_r:.3f} bins")
print("  → saved kf_tracking_03_multitarget.png")

# ─────────────────────────────────────────────────────────────────────────────
# § 4. Range-Only vs Range+Doppler Tracking via RadarPipelineDataset
#      The RangeDopplerKF uses Doppler measurements to directly constrain the
#      velocity state — giving faster velocity convergence and tighter
#      covariance compared to ConstantVelocityKF which infers velocity
#      indirectly from successive range measurements.
#
#      We demonstrate this by comparing the velocity track estimates from both
#      KF variants and by running a standalone RangeDopplerKF with deliberate
#      velocity initialisation error to show recovery speed.
# ─────────────────────────────────────────────────────────────────────────────
print("\n§ 4 — Range-only vs Range+Doppler tracking (RadarPipelineDataset)")

common_kwargs = dict(
    waveform_pool=[LFM()],
    trajectory_pool=[ConstantVelocity(initial_range=100.0, velocity=0.5, dt=1.0)],
    swerling_cases=[0],
    clutter_presets=[RadarClutter.ground(SAMPLE_RATE, terrain="rural")],
    num_range_bins=NUM_RANGE_BINS,
    sample_rate=SAMPLE_RATE,
    carrier_frequency=CARRIER_FREQ,
    pri=PRI,
    snr_range=(15.0, 25.0),
    num_targets_range=(1, 1),
    sequence_length=SEQ_LEN,
    pulses_per_cpi=PULSES_PER_CPI,
    apply_mti=True,
    cfar_type="ca",
    num_samples=8,
    seed=SEED,
)

ds_range_only = RadarPipelineDataset(**common_kwargs, track_doppler=False)
ds_range_doppler = RadarPipelineDataset(**common_kwargs, track_doppler=True)

_, tgt_ro = ds_range_only[0]
_, tgt_rd = ds_range_doppler[0]

print(f"  Waveform: {tgt_ro.waveform_label},  SNR: {tgt_ro.snr_db:.1f} dB")
print(f"  Targets tracked: {tgt_ro.num_targets}")

# ── Velocity convergence: standalone ConstantVelocityKF vs RangeDopplerKF ──
# Start both filters with a deliberate velocity error to show how quickly
# each recovers using only range vs range+Doppler measurements.
TRUE_VEL_SIM = 0.5        # bins/s — matches ConstantVelocity(velocity=0.5, dt=1)
INIT_VEL_ERROR = 2.0       # introduce a 2 bins/s initialisation offset
WAVELEN = 3e8 / CARRIER_FREQ

true_r_sim = 100.0 + TRUE_VEL_SIM * DT * np.arange(SEQ_LEN)
meas_r_sim = true_r_sim + rng.normal(0, 4.0, SEQ_LEN)

# Simulate Doppler measurements: center Doppler index = vel * doppler_scale
doppler_scale = 2.0 * PRI * PULSES_PER_CPI / WAVELEN
meas_doppler = TRUE_VEL_SIM * doppler_scale + rng.normal(0, 2.0, SEQ_LEN)

# Range-only KF — wrong initial velocity
kf_ro_sim = ConstantVelocityKF(
    dt=DT, process_noise_std=0.5, measurement_noise_std=4.0,
    x0=np.array([true_r_sim[0], TRUE_VEL_SIM + INIT_VEL_ERROR]),
)
# Range+Doppler KF — same wrong initial velocity
kf_rd_sim = RangeDopplerKF(
    dt=DT,
    wavelength=WAVELEN,
    pri=PRI,
    pulses_per_cpi=PULSES_PER_CPI,
    process_noise_std=0.5,
    range_noise_std=4.0,
    doppler_noise_std=2.0,
    x0=np.array([true_r_sim[0], TRUE_VEL_SIM + INIT_VEL_ERROR]),
)

vel_ro_sim, vel_rd_sim = [], []
for i in range(SEQ_LEN):
    kf_ro_sim.predict()
    kf_ro_sim.update(np.array([meas_r_sim[i]]))
    vel_ro_sim.append(kf_ro_sim.state[1])

    kf_rd_sim.predict()
    kf_rd_sim.update(np.array([meas_r_sim[i], meas_doppler[i]]))
    vel_rd_sim.append(kf_rd_sim.state[1])

frames_s = np.arange(SEQ_LEN)
fig, axes = plt.subplots(1, 2, figsize=(13, 5))

ax = axes[0]
ax.axhline(TRUE_VEL_SIM, color="k", lw=1.5, ls="--", label=f"True vel ({TRUE_VEL_SIM} bins/s)")
ax.plot(frames_s, vel_ro_sim, "o-", color="steelblue", markersize=3, lw=1.4,
        label=f"Range-only KF (init +{INIT_VEL_ERROR} error)")
ax.plot(frames_s, vel_rd_sim, "s-", color="coral", markersize=3, lw=1.4,
        label=f"Range+Doppler KF (init +{INIT_VEL_ERROR} error)")
ax.set_xlabel("CPI frame")
ax.set_ylabel("Velocity estimate (bins/s)")
ax.set_title("§ 4 — Velocity Convergence from Bad Initialisation")
ax.legend(fontsize=9)
ax.grid(True, alpha=0.3)

# Annotate convergence point: first frame within 10% of true velocity
tol = 0.1 * TRUE_VEL_SIM
for est, col, label in [(vel_ro_sim, "steelblue", "RO"), (vel_rd_sim, "coral", "RD")]:
    for fi, v in enumerate(est):
        if abs(v - TRUE_VEL_SIM) < tol:
            ax.axvline(fi, color=col, alpha=0.4, ls=":")
            ax.text(fi + 0.3, ax.get_ylim()[0] + 0.05 * (ax.get_ylim()[1] - ax.get_ylim()[0]),
                    f"{label}\nframe {fi}", color=col, fontsize=8)
            break

# Pipeline velocity tracks from RadarPipelineDataset
ax = axes[1]
for k in range(tgt_ro.num_targets):
    ax.plot(frames_s, tgt_ro.true_velocities[:, k], "k-", lw=1.5,
            label="True vel" if k == 0 else None)
    ax.plot(frames_s, tgt_ro.kf_states[:, k, 1], "o-", color="steelblue",
            markersize=3, lw=1.2, label="Range-only KF" if k == 0 else None, alpha=0.8)
    ax.plot(frames_s, tgt_rd.kf_states[:, k, 1], "s-", color="coral",
            markersize=3, lw=1.2, label="Range+Doppler KF" if k == 0 else None, alpha=0.8)
ax.set_xlabel("CPI frame")
ax.set_ylabel("Velocity estimate (bins/s)")
ax.set_title("Pipeline KF Velocity Tracks (RadarPipelineDataset)")
ax.legend(fontsize=9)
ax.grid(True, alpha=0.3)

plt.tight_layout()
savefig("kf_tracking_04_doppler_velocity.png")
plt.close()
print("  → saved kf_tracking_04_doppler_velocity.png")

# Collect RMSE across several independent samples
n_eval = 6
range_rmse_ro, range_rmse_rd = [], []
vel_rmse_ro, vel_rmse_rd = [], []

for idx in range(n_eval):
    _, tgt_ro_i = ds_range_only[idx]
    _, tgt_rd_i = ds_range_doppler[idx]

    for k in range(tgt_ro_i.num_targets):
        range_rmse_ro.append(np.sqrt(np.mean(
            (tgt_ro_i.kf_states[:, k, 0] - tgt_ro_i.true_ranges[:, k]) ** 2)))
        range_rmse_rd.append(np.sqrt(np.mean(
            (tgt_rd_i.kf_states[:, k, 0] - tgt_rd_i.true_ranges[:, k]) ** 2)))
        vel_rmse_ro.append(np.sqrt(np.mean(
            (tgt_ro_i.kf_states[:, k, 1] - tgt_ro_i.true_velocities[:, k]) ** 2)))
        vel_rmse_rd.append(np.sqrt(np.mean(
            (tgt_rd_i.kf_states[:, k, 1] - tgt_rd_i.true_velocities[:, k]) ** 2)))

print(f"  Range-only   — range RMSE: {np.mean(range_rmse_ro):.3f} ± {np.std(range_rmse_ro):.3f} bins")
print(f"  Range+Doppler — range RMSE: {np.mean(range_rmse_rd):.3f} ± {np.std(range_rmse_rd):.3f} bins")
print(f"  Range-only   — vel RMSE: {np.mean(vel_rmse_ro):.6f} ± {np.std(vel_rmse_ro):.6f} bins/s")
print(f"  Range+Doppler — vel RMSE: {np.mean(vel_rmse_rd):.6f} ± {np.std(vel_rmse_rd):.6f} bins/s")
print("  Note: CFAR detection quality dominates range RMSE in cluttered scenes;")
print("  Doppler's primary benefit is velocity convergence speed (see left plot).")

# Re-bind tgt_ro / tgt_rd to sample 0 for use in §5
tgt_ro = ds_range_only[n_eval - 1][1]
tgt_rd = ds_range_doppler[n_eval - 1][1]

# ─────────────────────────────────────────────────────────────────────────────
# § 5. Track Quality — Per-Frame RMSE Over the CPI Sequence
#      Use the last sample from each dataset to show how KF error evolves
#      frame-by-frame as the filter warms up.
# ─────────────────────────────────────────────────────────────────────────────
print("\n§ 5 — Per-frame RMSE and bar comparison plots")

def per_frame_rmse(kf_states, true_vals, dim):
    """Root-mean-square error per frame across targets for state dimension `dim`."""
    err = kf_states[:, :, dim] - true_vals          # (frames, targets)
    return np.sqrt(np.mean(err ** 2, axis=1))        # (frames,)

frames_arr = np.arange(SEQ_LEN)

rmse_ro_range = per_frame_rmse(tgt_ro.kf_states, tgt_ro.true_ranges, 0)
rmse_rd_range = per_frame_rmse(tgt_rd.kf_states, tgt_rd.true_ranges, 0)
rmse_ro_vel = per_frame_rmse(tgt_ro.kf_states, tgt_ro.true_velocities, 1)
rmse_rd_vel = per_frame_rmse(tgt_rd.kf_states, tgt_rd.true_velocities, 1)

fig, axes = plt.subplots(1, 2, figsize=(13, 5))

ax = axes[0]
ax.plot(frames_arr, rmse_ro_range, "o-", color="steelblue", markersize=4, lw=1.4, label="Range-only KF")
ax.plot(frames_arr, rmse_rd_range, "s-", color="coral", markersize=4, lw=1.4, label="Range+Doppler KF")
ax.set_xlabel("CPI frame")
ax.set_ylabel("RMSE (bins)")
ax.set_title("§ 5 — Per-Frame Range RMSE")
ax.legend(fontsize=10)
ax.grid(True, alpha=0.3)

ax = axes[1]
ax.plot(frames_arr, rmse_ro_vel, "o-", color="steelblue", markersize=4, lw=1.4, label="Range-only KF")
ax.plot(frames_arr, rmse_rd_vel, "s-", color="coral", markersize=4, lw=1.4, label="Range+Doppler KF")
ax.set_xlabel("CPI frame")
ax.set_ylabel("RMSE (bins/s)")
ax.set_title("Per-Frame Velocity RMSE")
ax.legend(fontsize=10)
ax.grid(True, alpha=0.3)

plt.tight_layout()
savefig("kf_tracking_05_per_frame_rmse.png")
plt.close()
print("  → saved kf_tracking_05_per_frame_rmse.png")

# Bar chart summary
fig, axes = plt.subplots(1, 2, figsize=(9, 5))

labels = ["Range-only", "Range+Doppler"]
bar_colors = ["steelblue", "coral"]

ax = axes[0]
means_r = [np.mean(range_rmse_ro), np.mean(range_rmse_rd)]
stds_r = [np.std(range_rmse_ro), np.std(range_rmse_rd)]
bars = ax.bar(labels, means_r, yerr=stds_r, color=bar_colors, alpha=0.85,
              edgecolor="k", lw=0.8, capsize=6, width=0.5)
ax.set_ylabel("Mean Range RMSE (bins)")
ax.set_title("§ 4+5 — Range RMSE Comparison")
ax.grid(True, alpha=0.3, axis="y")
for bar, m in zip(bars, means_r):
    ax.text(bar.get_x() + bar.get_width() / 2, bar.get_height() + 0.05,
            f"{m:.2f}", ha="center", va="bottom", fontsize=10, fontweight="bold")

ax = axes[1]
means_v = [np.mean(vel_rmse_ro), np.mean(vel_rmse_rd)]
stds_v = [np.std(vel_rmse_ro), np.std(vel_rmse_rd)]
bars = ax.bar(labels, means_v, yerr=stds_v, color=bar_colors, alpha=0.85,
              edgecolor="k", lw=0.8, capsize=6, width=0.5)
ax.set_ylabel("Mean Velocity RMSE (bins/s)")
ax.set_title("Velocity RMSE Comparison")
ax.grid(True, alpha=0.3, axis="y")
for bar, m in zip(bars, means_v):
    ax.text(bar.get_x() + bar.get_width() / 2, bar.get_height() + 0.0002,
            f"{m:.4f}", ha="center", va="bottom", fontsize=10, fontweight="bold")

plt.tight_layout()
savefig("kf_tracking_04_rmse_bars.png")
plt.close()
print("  → saved kf_tracking_04_rmse_bars.png")

# ─────────────────────────────────────────────────────────────────────────────
# Final summary
# ─────────────────────────────────────────────────────────────────────────────
print("\n" + "=" * 60)
print("Done. Output images in examples/outputs/:")
print("  kf_tracking_01_basics.png       — filter mechanics & covariance")
print("  kf_tracking_02_tuning.png       — process-noise tuning sweep")
print("  kf_tracking_03_multitarget.png  — CV + CTR simultaneous tracks")
print("  kf_tracking_04_doppler_velocity.png — Doppler velocity convergence comparison")
print("  kf_tracking_04_rmse_bars.png    — range-only vs range+Doppler RMSE summary")
print("  kf_tracking_05_per_frame_rmse.png — per-frame RMSE evolution")
print("=" * 60)
