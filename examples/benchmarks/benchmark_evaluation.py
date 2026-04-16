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
# (Predictions are intentionally simple to demonstrate the evaluate_snr_sweep API)
rng = np.random.default_rng(42)
all_labels = []
all_preds = []
all_snrs = []

for idx in range(min(len(ds_snr), 200)):
    iq, label = ds_snr[idx]
    all_labels.append(label)
    all_preds.append(label)  # placeholder "perfect" classifier
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
