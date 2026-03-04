"""
SPECTRA Example 10: S3CA Paper Reproduction (Li et al. IEEE SPL 2015)
=====================================================================
Level: Advanced

Reproduces Figure 3 from "S3CA: A Sparse Strip Spectral Correlation Analyzer"
by Carol Jingyi Li et al.

The paper uses a DSSS-BPSK signal with:
  - Processing gain: 31  (length-31 m-sequence PN code)
  - Chip rate: 0.25       (4 samples per chip, fs normalised to 1)
  - Data rate: 0.25/31 ≈ 0.00806
  - SNR: 10 dB
  - N = 2^18 samples, Np = 64

Cycle frequencies appear at multiples of the data rate (≈ 0.00806).

Learn how to:
- Generate a DSSS-BPSK signal matching the paper's test conditions
- Compute the SCD via the S3CA with proper parameters
- Compute a reference SCD (S3CA with kappa=n_alpha ≈ full FFT)
- Visualise 3D SCD, alpha profiles, and residuals
"""

import sys
sys.path.insert(0, ".")

import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D  # noqa: F401

import spectra as sp
from plot_helpers import savefig


# -- 1. Generate DSSS-BPSK Signal -------------------------------------------

N = 2**18  # Total samples (paper uses 2^20; 2^18 is a practical compromise)
Np = 64    # Channelizer bands (paper value)
hop = Np // 4  # Hop size = 16
processing_gain = 31
samples_per_chip = 4  # chip_rate = 0.25 with fs=1
data_rate = 1.0 / (samples_per_chip * processing_gain)  # ≈ 0.00806
snr_db = 10.0

# Number of data symbols needed to fill N samples
num_chips = N // samples_per_chip
num_symbols = num_chips // processing_gain + 1

dsss = sp.DSSS_BPSK(processing_gain=processing_gain, samples_per_chip=samples_per_chip)
iq_clean = dsss.generate(num_symbols=num_symbols, sample_rate=1.0, seed=42)
iq_clean = iq_clean[:N]  # Trim to exactly N samples

# Add AWGN at 10 dB SNR
sig_power = np.mean(np.abs(iq_clean) ** 2)
noise_power = sig_power / (10 ** (snr_db / 10))
rng = np.random.RandomState(123)
noise = np.sqrt(noise_power / 2) * (
    rng.randn(N).astype(np.float32) + 1j * rng.randn(N).astype(np.float32)
)
iq = (iq_clean + noise).astype(np.complex64)

print(f"Signal: DSSS-BPSK, N={N}, Np={Np}, hop={hop}")
print(f"Processing gain={processing_gain}, chip_rate=0.25, data_rate={data_rate:.6f}")
print(f"SNR={snr_db} dB, {N} samples")


# -- 2. Compute SCD ---------------------------------------------------------

# Number of channelizer frames
n_frames = (N - Np) // hop + 1
# Pad n_alpha to next power of 2 for FFT efficiency
n_alpha = 1
while n_alpha < n_frames:
    n_alpha *= 2
print(f"n_frames={n_frames}, n_alpha={n_alpha}")

# Reference: S3CA with kappa = n_alpha (equivalent to full FFT → paper's SSCA)
kappa_ref = n_alpha
scd_ref = sp.SCD(
    nfft=Np, n_alpha=n_alpha, hop=hop,
    method="s3ca", output_format="magnitude",
    kappa=kappa_ref, seed=0,
)

# S3CA with kappa=80 (paper's sparse setting)
kappa_s3ca = 80
scd_s3ca = sp.SCD(
    nfft=Np, n_alpha=n_alpha, hop=hop,
    method="s3ca", output_format="magnitude",
    kappa=kappa_s3ca, seed=0,
)

print("Computing reference SCD (full FFT)...")
scd_ref_mag = scd_ref(iq).squeeze(0).numpy()
print("Computing S3CA SCD (kappa=80)...")
scd_s3ca_mag = scd_s3ca(iq).squeeze(0).numpy()

# Keep only top kappa*Np magnitudes for display (matching paper)
scd_ref_display = scd_ref_mag.copy()
scd_s3ca_display = scd_s3ca_mag.copy()


# -- 3. Alpha Profile -------------------------------------------------------

# Alpha profile: max magnitude over all spectral frequencies for each alpha
# Paper restricts display to alpha ∈ [0, 0.25] due to symmetry
alpha_axis = np.linspace(-1.0, 1.0, n_alpha, endpoint=False)

# Shift so alpha=0 is at center
alpha_profile_ref = np.max(scd_ref_display, axis=0)
alpha_profile_s3ca = np.max(scd_s3ca_display, axis=0)


# -- 4. Plot Figure 3 Reproduction ------------------------------------------

# Spectral frequency axis
f_axis = np.linspace(-0.5, 0.5, Np, endpoint=False)

# Restrict alpha display range to [0, 0.25] as in paper
alpha_mask = (alpha_axis >= 0) & (alpha_axis <= 0.25)
alpha_restricted = alpha_axis[alpha_mask]
profile_ref_restricted = alpha_profile_ref[alpha_mask]
profile_s3ca_restricted = alpha_profile_s3ca[alpha_mask]

fig = plt.figure(figsize=(18, 12))

# --- (a) SSCA reference 3D SCD ---
ax1 = fig.add_subplot(2, 3, 1, projection="3d")
A, F = np.meshgrid(alpha_restricted, f_axis)
scd_ref_3d = scd_ref_display[:, alpha_mask]
ax1.plot_surface(A, F, scd_ref_3d, cmap="viridis", linewidth=0, antialiased=False,
                 rcount=Np, ccount=min(200, alpha_restricted.size))
ax1.set_xlabel("cycle frequency (α)", fontsize=8)
ax1.set_ylabel("f", fontsize=8)
ax1.set_zlabel("magnitude", fontsize=8)
ax1.set_title(f"(a) SSCA (Sx_SSCA)", fontsize=10)
ax1.view_init(elev=30, azim=-60)

# --- (b) Alpha Profile — SSCA ---
ax2 = fig.add_subplot(2, 3, 4)
ax2.plot(alpha_restricted, profile_ref_restricted, linewidth=0.8)
ax2.set_xlabel("cycle frequency (α)")
ax2.set_ylabel("magnitude")
ax2.set_title("(b) Alpha Profile — SSCA", fontsize=10)
ax2.set_xlim([0, 0.25])
ax2.grid(True, alpha=0.3)

# --- (c) S3CA 3D SCD (kappa=80) ---
ax3 = fig.add_subplot(2, 3, 2, projection="3d")
scd_s3ca_3d = scd_s3ca_display[:, alpha_mask]
ax3.plot_surface(A, F, scd_s3ca_3d, cmap="viridis", linewidth=0, antialiased=False,
                 rcount=Np, ccount=min(200, alpha_restricted.size))
ax3.set_xlabel("cycle frequency (α)", fontsize=8)
ax3.set_ylabel("f", fontsize=8)
ax3.set_zlabel("magnitude", fontsize=8)
ax3.set_title(f"(c) S3CA (Sx_S3CA, κ={kappa_s3ca})", fontsize=10)
ax3.view_init(elev=30, azim=-60)

# --- (d) Alpha Profile — S3CA ---
ax4 = fig.add_subplot(2, 3, 5)
ax4.plot(alpha_restricted, profile_s3ca_restricted, linewidth=0.8)
ax4.set_xlabel("cycle frequency (α)")
ax4.set_ylabel("magnitude")
ax4.set_title(f"(d) Alpha Profile — S3CA (κ={kappa_s3ca})", fontsize=10)
ax4.set_xlim([0, 0.25])
ax4.grid(True, alpha=0.3)

# --- (e) Residual ---
ax5 = fig.add_subplot(2, 3, 3, projection="3d")
residual = scd_ref_display - scd_s3ca_display
residual_3d = np.abs(residual[:, alpha_mask])
ax5.plot_surface(A, F, residual_3d, cmap="viridis", linewidth=0, antialiased=False,
                 rcount=Np, ccount=min(200, alpha_restricted.size))
ax5.set_xlabel("cycle frequency (α)", fontsize=8)
ax5.set_ylabel("f", fontsize=8)
ax5.set_zlabel("magnitude", fontsize=8)
ax5.set_title(f"(e) Residual (κ={kappa_s3ca})", fontsize=10)
ax5.view_init(elev=30, azim=-60)

# --- (f) L1-norm of residual vs kappa ---
ax6 = fig.add_subplot(2, 3, 6)
kappas_sweep = [10, 20, 40, 60, 80, 100, 120]
l1_norms = []
for k in kappas_sweep:
    scd_k = sp.SCD(
        nfft=Np, n_alpha=n_alpha, hop=hop,
        method="s3ca", output_format="magnitude",
        kappa=k, seed=0,
    )
    scd_k_mag = scd_k(iq).squeeze(0).numpy()
    r = scd_ref_mag - scd_k_mag
    l1 = np.sum(np.abs(r)) / (k * Np)
    l1_norms.append(l1)
    print(f"  kappa={k:4d}: L1-norm = {l1:.4f}")

ax6.plot(kappas_sweep, l1_norms, "o-", linewidth=1.5)
ax6.set_xlabel("sparsity ratio (κ)")
ax6.set_ylabel("L1 norm")
ax6.set_title("(f) Error vs κ", fontsize=10)
ax6.grid(True, alpha=0.3)

fig.suptitle(
    f"S3CA Paper Reproduction — DSSS-BPSK (PG={processing_gain}, "
    f"chip_rate=0.25, N={N}, Np={Np})",
    fontsize=13,
)
fig.tight_layout()
savefig("10_s3ca_paper_reproduction.png", dpi=200)

plt.close("all")
print("\nDone! Check examples/outputs/ for saved figures.")
