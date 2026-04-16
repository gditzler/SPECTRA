"""
SPECTRA Example 08: CSP Classification
=======================================
Level: Advanced

Learn how to:
- Build a CyclostationaryDataset with multiple feature representations
- Visualize SCD, cumulant, and PSD representations for a single sample
- Train a CyclostationaryAMC classifier with random forests
- Evaluate with confusion matrices
- Compare feature sets (cumulants vs cyclic_peaks vs combined)
- Sweep accuracy vs SNR
"""

import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

import spectra as sp
from plot_helpers import savefig

try:
    from sklearn.model_selection import train_test_split
    from sklearn.metrics import confusion_matrix, accuracy_score
except ImportError:
    raise ImportError(
        "This example requires scikit-learn. Install it with:\n"
        "  pip install 'spectra[classifiers]'"
    )


# ── 1. Define Waveform Pool and Representations ──────────────────────────────

waveform_pool = [
    sp.BPSK(),
    sp.QPSK(),
    sp.PSK8(),
    sp.QAM16(),
    sp.QAM64(),
]
class_names = [wf.label for wf in waveform_pool]
print(f"Waveform pool: {class_names}")

representations = {
    "scd": sp.SCD(nfft=64, n_alpha=64, hop=16),
    "cum": sp.Cumulants(max_order=4),
    "psd": sp.PSD(nfft=256, overlap=128, db_scale=True),
}


# ── 2. Create CyclostationaryDataset ─────────────────────────────────────────

dataset = sp.CyclostationaryDataset(
    waveform_pool=waveform_pool,
    num_samples=500,
    num_iq_samples=4096,
    sample_rate=1e6,
    representations=representations,
    impairments=sp.Compose([sp.AWGN(snr_range=(10, 25))]),
    seed=42,
)

sample_data, sample_label = dataset[0]
print(f"\nDataset size: {len(dataset)}")
print(f"Representations per sample:")
for name, tensor in sample_data.items():
    print(f"  {name}: shape={list(tensor.shape)}, dtype={tensor.dtype}")
print(f"Label: {sample_label} ({class_names[sample_label]})")


# ── 3. Visualize One Sample's Representations ────────────────────────────────

fig, axes = plt.subplots(1, 3, figsize=(15, 4))

# SCD
scd_img = sample_data["scd"].squeeze(0).numpy()
im0 = axes[0].imshow(
    10 * np.log10(scd_img + 1e-12),
    aspect="auto",
    origin="lower",
    cmap="inferno",
    interpolation="nearest",
)
axes[0].set_title(f"SCD — {class_names[sample_label]}")
axes[0].set_xlabel("Cyclic Freq Bin")
axes[0].set_ylabel("Spectral Freq Bin")
fig.colorbar(im0, ax=axes[0], label="dB")

# Cumulants
cum_vals = sample_data["cum"].numpy()
cum_labels = ["|C20|", "|C21|", "|C40|", "|C41|", "|C42|"]
axes[1].bar(cum_labels, cum_vals, color="steelblue")
axes[1].set_title(f"Cumulants — {class_names[sample_label]}")
axes[1].set_ylabel("Magnitude")
axes[1].grid(True, alpha=0.3, axis="y")

# PSD
psd_vals = sample_data["psd"].squeeze(0).numpy()
freqs = np.linspace(-0.5, 0.5, len(psd_vals))
axes[2].plot(freqs, psd_vals, linewidth=0.8)
axes[2].set_title(f"PSD — {class_names[sample_label]}")
axes[2].set_xlabel("Normalized Frequency")
axes[2].set_ylabel("Power (dB)")
axes[2].grid(True, alpha=0.3)

fig.suptitle("Multi-Representation Sample View", fontsize=14)
fig.tight_layout()
savefig("08_sample_representations.png")


# ── 4. Train CyclostationaryAMC ──────────────────────────────────────────────

print("\nExtracting features and training classifier...")
amc = sp.CyclostationaryAMC(feature_set="cumulants", classifier="random_forest")

# Extract features for all samples
features_list = []
labels_list = []
for i in range(len(dataset)):
    iq = amc._regenerate_iq(dataset, i)
    features_list.append(amc.extract_features(iq))
    _, label = dataset[i]
    labels_list.append(label)

X = np.stack(features_list)
y = np.array(labels_list)

X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.3, random_state=42, stratify=y,
)

amc.fit(X_train, y_train)
y_pred = amc.predict(X_test)
acc = accuracy_score(y_test, y_pred)
print(f"Test accuracy: {acc:.1%}")


# ── 5. Confusion Matrix ──────────────────────────────────────────────────────

cm = confusion_matrix(y_test, y_pred)
fig, ax = plt.subplots(figsize=(7, 6))
im = ax.imshow(cm, cmap="Blues", interpolation="nearest")
ax.set_xticks(range(len(class_names)))
ax.set_yticks(range(len(class_names)))
ax.set_xticklabels(class_names, rotation=45, ha="right")
ax.set_yticklabels(class_names)
ax.set_xlabel("Predicted")
ax.set_ylabel("True")
ax.set_title(f"Confusion Matrix — Cumulants + Random Forest\nAccuracy: {acc:.1%}")
# Annotate cells
for i in range(len(class_names)):
    for j in range(len(class_names)):
        color = "white" if cm[i, j] > cm.max() / 2 else "black"
        ax.text(j, i, str(cm[i, j]), ha="center", va="center", color=color)
fig.colorbar(im, ax=ax)
fig.tight_layout()
savefig("08_confusion_matrix.png")


# ── 6. Compare Feature Sets ──────────────────────────────────────────────────

print("\nComparing feature sets...")
feature_sets = ["cumulants", "cyclic_peaks", "combined"]
accuracies = {}

for fs in feature_sets:
    amc_fs = sp.CyclostationaryAMC(feature_set=fs, classifier="random_forest")
    X_fs = np.stack([amc_fs.extract_features(amc._regenerate_iq(dataset, i))
                     for i in range(len(dataset))])
    X_tr, X_te, y_tr, y_te = train_test_split(
        X_fs, y, test_size=0.3, random_state=42, stratify=y,
    )
    amc_fs.fit(X_tr, y_tr)
    acc_fs = accuracy_score(y_te, amc_fs.predict(X_te))
    accuracies[fs] = acc_fs
    print(f"  {fs}: {acc_fs:.1%}")

fig, ax = plt.subplots(figsize=(8, 4))
bars = ax.bar(list(accuracies.keys()), list(accuracies.values()), color=["steelblue", "coral", "seagreen"])
ax.set_ylabel("Test Accuracy")
ax.set_title("Feature Set Comparison — Random Forest AMC")
ax.set_ylim(0, 1.05)
ax.grid(True, alpha=0.3, axis="y")
for bar, val in zip(bars, accuracies.values()):
    ax.text(bar.get_x() + bar.get_width() / 2, bar.get_height() + 0.01,
            f"{val:.1%}", ha="center", va="bottom", fontweight="bold")
fig.tight_layout()
savefig("08_feature_comparison.png")


# ── 7. Accuracy vs SNR Sweep ─────────────────────────────────────────────────

print("\nRunning accuracy vs SNR sweep...")
snr_values = [5, 10, 15, 20, 25, 30]
snr_accuracies = []

for snr in snr_values:
    ds_snr = sp.CyclostationaryDataset(
        waveform_pool=waveform_pool,
        num_samples=300,
        num_iq_samples=4096,
        sample_rate=1e6,
        representations={"cum": sp.Cumulants(max_order=4)},
        impairments=sp.Compose([sp.AWGN(snr=snr)]),
        seed=42,
    )
    amc_snr = sp.CyclostationaryAMC(feature_set="cumulants", classifier="random_forest")
    X_snr = np.stack([amc_snr.extract_features(amc_snr._regenerate_iq(ds_snr, i))
                      for i in range(len(ds_snr))])
    y_snr = np.array([ds_snr[i][1] for i in range(len(ds_snr))])
    X_tr, X_te, y_tr, y_te = train_test_split(
        X_snr, y_snr, test_size=0.3, random_state=42, stratify=y_snr,
    )
    amc_snr.fit(X_tr, y_tr)
    acc_snr = accuracy_score(y_te, amc_snr.predict(X_te))
    snr_accuracies.append(acc_snr)
    print(f"  SNR={snr:2d} dB: {acc_snr:.1%}")

fig, ax = plt.subplots(figsize=(8, 5))
ax.plot(snr_values, snr_accuracies, "o-", linewidth=2, markersize=8, color="steelblue")
ax.set_xlabel("SNR (dB)")
ax.set_ylabel("Test Accuracy")
ax.set_title("AMC Accuracy vs SNR — Cumulant Features + Random Forest")
ax.set_ylim(0, 1.05)
ax.grid(True, alpha=0.3)
fig.tight_layout()
savefig("08_accuracy_vs_snr.png")


plt.close("all")
print("\nDone! Check examples/outputs/ for saved figures.")
