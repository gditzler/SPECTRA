"""
SPECTRA Example 09: S3CA vs SSCA Comparison
============================================
Level: Intermediate

Learn how to:
- Compute the SCD using the Sparse Strip Spectral Correlation Analyzer (S3CA)
- Compare S3CA output against the full SSCA
- Visualize sparsity in S3CA results
- Tune the kappa parameter for accuracy vs sparsity trade-off
"""

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

import time

import matplotlib
import numpy as np

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import spectra as sp
from plot_helpers import savefig

# -- 1. Generate Test Signals ------------------------------------------------

sample_rate = 1e6
num_symbols = 512
num_iq = 8192

waveforms = {
    "BPSK": sp.BPSK(),
    "QPSK": sp.QPSK(),
}

signals = {}
for name, wf in waveforms.items():
    iq = wf.generate(num_symbols=num_symbols, sample_rate=sample_rate, seed=42)
    signals[name] = iq[:num_iq]

print(f"Generated {len(signals)} signals, each {num_iq} samples at {sample_rate/1e6:.0f} MHz")


# -- 2. SSCA vs S3CA Side-by-Side (BPSK) ------------------------------------

nfft = 64
n_alpha = 64
hop = 16

ssca = sp.SCD(nfft=nfft, n_alpha=n_alpha, hop=hop, method="ssca", output_format="magnitude")
s3ca = sp.SCD(nfft=nfft, n_alpha=n_alpha, hop=hop, method="s3ca", output_format="magnitude",
              kappa=8, seed=0)

iq_bpsk = signals["BPSK"]
ssca_bpsk = ssca(iq_bpsk).squeeze(0).numpy()
s3ca_bpsk = s3ca(iq_bpsk).squeeze(0).numpy()

fig, axes = plt.subplots(1, 2, figsize=(12, 5))
for ax, data, label in zip(axes, [ssca_bpsk, s3ca_bpsk], ["SSCA", "S3CA (kappa=8)"]):
    im = ax.imshow(
        10 * np.log10(data + 1e-12),
        aspect="auto",
        origin="lower",
        cmap="inferno",
        interpolation="nearest",
    )
    ax.set_title(f"SCD — BPSK — {label}")
    ax.set_xlabel("Cyclic Frequency Bin")
    ax.set_ylabel("Spectral Frequency Bin")
    fig.colorbar(im, ax=ax, label="dB")
fig.suptitle("SSCA vs S3CA — BPSK Signal", fontsize=14)
fig.tight_layout()
savefig("09_ssca_vs_s3ca_bpsk.png")


# -- 3. SSCA vs S3CA Side-by-Side (QPSK) ------------------------------------

iq_qpsk = signals["QPSK"]
ssca_qpsk = ssca(iq_qpsk).squeeze(0).numpy()
s3ca_qpsk = s3ca(iq_qpsk).squeeze(0).numpy()

fig, axes = plt.subplots(1, 2, figsize=(12, 5))
for ax, data, label in zip(axes, [ssca_qpsk, s3ca_qpsk], ["SSCA", "S3CA (kappa=8)"]):
    im = ax.imshow(
        10 * np.log10(data + 1e-12),
        aspect="auto",
        origin="lower",
        cmap="inferno",
        interpolation="nearest",
    )
    ax.set_title(f"SCD — QPSK — {label}")
    ax.set_xlabel("Cyclic Frequency Bin")
    ax.set_ylabel("Spectral Frequency Bin")
    fig.colorbar(im, ax=ax, label="dB")
fig.suptitle("SSCA vs S3CA — QPSK Signal", fontsize=14)
fig.tight_layout()
savefig("09_ssca_vs_s3ca_qpsk.png")


# -- 4. Sparsity Visualization -----------------------------------------------

# S3CA produces sparse output — most bins are zero
s3ca_sparse = sp.SCD(nfft=nfft, n_alpha=n_alpha, hop=hop, method="s3ca",
                      output_format="real_imag", kappa=4, seed=0)
s3ca_ri = s3ca_sparse(iq_bpsk)  # [2, nfft, n_alpha]
real_part = s3ca_ri[0].numpy()
nonzero_mask = np.abs(real_part) > 1e-10

total_bins = real_part.size
nonzero_count = np.count_nonzero(nonzero_mask)
sparsity = 1.0 - nonzero_count / total_bins

fig, axes = plt.subplots(1, 2, figsize=(12, 5))
axes[0].imshow(
    10 * np.log10(np.abs(real_part) + 1e-12),
    aspect="auto",
    origin="lower",
    cmap="inferno",
    interpolation="nearest",
)
axes[0].set_title("S3CA Real Part (kappa=4)")
axes[0].set_xlabel("Cyclic Frequency Bin")
axes[0].set_ylabel("Spectral Frequency Bin")

axes[1].imshow(
    nonzero_mask.astype(float),
    aspect="auto",
    origin="lower",
    cmap="gray_r",
    interpolation="nearest",
)
axes[1].set_title(f"Non-Zero Bins ({sparsity:.1%} sparse)")
axes[1].set_xlabel("Cyclic Frequency Bin")
axes[1].set_ylabel("Spectral Frequency Bin")

fig.suptitle("S3CA Sparsity — BPSK Signal", fontsize=14)
fig.tight_layout()
savefig("09_s3ca_sparsity.png")

print(f"\nSparsity: {nonzero_count}/{total_bins} bins non-zero ({sparsity:.1%} sparse)")


# -- 5. Kappa Sweep ----------------------------------------------------------

kappas = [2, 4, 8, 16]

fig, axes = plt.subplots(1, len(kappas), figsize=(16, 4))
for ax, k in zip(axes, kappas):
    scd_k = sp.SCD(nfft=nfft, n_alpha=n_alpha, hop=hop, method="s3ca",
                    output_format="magnitude", kappa=k, seed=0)
    result = scd_k(iq_bpsk).squeeze(0).numpy()
    im = ax.imshow(
        10 * np.log10(result + 1e-12),
        aspect="auto",
        origin="lower",
        cmap="inferno",
        interpolation="nearest",
    )
    nz = np.count_nonzero(result > 1e-10)
    ax.set_title(f"kappa={k} ({nz} bins)")
    ax.set_xlabel("Cyclic Freq Bin")
    if ax == axes[0]:
        ax.set_ylabel("Spectral Freq Bin")
fig.suptitle("S3CA — Effect of Kappa on BPSK SCD", fontsize=14)
fig.tight_layout()
savefig("09_s3ca_kappa_sweep.png")


# -- 6. Timing Comparison ----------------------------------------------------

n_trials = 5
ssca_times = []
s3ca_times = []

ssca_t = sp.SCD(nfft=nfft, n_alpha=n_alpha, hop=hop, method="ssca")
s3ca_t = sp.SCD(nfft=nfft, n_alpha=n_alpha, hop=hop, method="s3ca", kappa=8, seed=0)

for _ in range(n_trials):
    t0 = time.perf_counter()
    ssca_t(iq_bpsk)
    ssca_times.append(time.perf_counter() - t0)

    t0 = time.perf_counter()
    s3ca_t(iq_bpsk)
    s3ca_times.append(time.perf_counter() - t0)

ssca_avg = np.mean(ssca_times) * 1000
s3ca_avg = np.mean(s3ca_times) * 1000

fig, ax = plt.subplots(figsize=(6, 4))
bars = ax.bar(["SSCA", "S3CA (kappa=8)"], [ssca_avg, s3ca_avg], color=["tab:blue", "tab:orange"])
ax.set_ylabel("Time (ms)")
ax.set_title(f"Execution Time — {num_iq} samples, nfft={nfft}")
for bar, val in zip(bars, [ssca_avg, s3ca_avg]):
    ax.text(bar.get_x() + bar.get_width() / 2, bar.get_height() + 0.1,
            f"{val:.1f} ms", ha="center", va="bottom", fontsize=10)
ax.grid(True, alpha=0.3, axis="y")
fig.tight_layout()
savefig("09_timing_comparison.png")

print(f"SSCA: {ssca_avg:.1f} ms avg, S3CA: {s3ca_avg:.1f} ms avg")


plt.close("all")
print("\nDone! Check examples/outputs/ for saved figures.")
