"""Visualize benchmark results from speed and ML comparison JSON files."""
import argparse
import json
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np


def load_json(path):
    """Load JSON file, return None if missing."""
    p = Path(path)
    if not p.exists():
        return None
    with open(p) as f:
        return json.load(f)


# ---------------------------------------------------------------------------
# Speed visualization
# ---------------------------------------------------------------------------

def plot_speed(results, out_dir):
    """Bar chart of __getitem__ latency and DataLoader throughput."""
    sources = list(results.keys())
    if not sources:
        return

    fig, axes = plt.subplots(1, 2, figsize=(12, 5))

    # Latency
    ax = axes[0]
    means = [results[s]["getitem_mean_ms"] for s in sources]
    stds = [results[s]["getitem_std_ms"] for s in sources]
    bars = ax.bar(sources, means, yerr=stds, capsize=5, color=["#2196F3", "#FF9800"][:len(sources)])
    ax.set_ylabel("Latency (ms)")
    ax.set_title("__getitem__ Latency")
    for bar, val in zip(bars, means):
        ax.text(bar.get_x() + bar.get_width() / 2, bar.get_height(),
                f"{val:.2f}", ha="center", va="bottom", fontsize=10)

    # Throughput
    ax = axes[1]
    throughputs = [results[s]["dataloader_throughput_sps"] for s in sources]
    bars = ax.bar(sources, throughputs, color=["#2196F3", "#FF9800"][:len(sources)])
    ax.set_ylabel("Samples/second")
    ax.set_title("DataLoader Throughput")
    for bar, val in zip(bars, throughputs):
        ax.text(bar.get_x() + bar.get_width() / 2, bar.get_height(),
                f"{val:.0f}", ha="center", va="bottom", fontsize=10)

    fig.suptitle("Dataset Generation Speed Comparison", fontsize=14)
    fig.tight_layout()
    path = out_dir / "speed_comparison.png"
    fig.savefig(path, dpi=150)
    plt.close(fig)
    print(f"  Saved {path}")


# ---------------------------------------------------------------------------
# Confusion matrix visualization
# ---------------------------------------------------------------------------

def plot_confusion_matrices(ml_results, classifier, out_dir, class_names):
    """Plot a grid of confusion matrices for one classifier type."""
    data = ml_results.get(classifier, {})
    if not data:
        return

    keys = sorted(data.keys())
    n = len(keys)
    cols = min(n, 2)
    rows = (n + cols - 1) // cols

    fig, axes = plt.subplots(rows, cols, figsize=(6 * cols, 5 * rows), squeeze=False)

    for i, key in enumerate(keys):
        r, c = divmod(i, cols)
        ax = axes[r][c]
        cm = np.array(data[key]["confusion_matrix"])
        acc = data[key]["accuracy"]

        im = ax.imshow(cm, cmap="Blues", interpolation="nearest")
        ax.set_title(f"{key}\nacc={acc:.3f}", fontsize=9)
        ax.set_xlabel("Predicted")
        ax.set_ylabel("True")

        tick_marks = np.arange(len(class_names))
        ax.set_xticks(tick_marks)
        ax.set_xticklabels(class_names, rotation=45, ha="right", fontsize=7)
        ax.set_yticks(tick_marks)
        ax.set_yticklabels(class_names, fontsize=7)

        # Annotate cells
        thresh = cm.max() / 2.0
        for ii in range(cm.shape[0]):
            for jj in range(cm.shape[1]):
                ax.text(jj, ii, str(cm[ii, jj]), ha="center", va="center",
                        fontsize=6,
                        color="white" if cm[ii, jj] > thresh else "black")

    # Hide unused axes
    for i in range(len(keys), rows * cols):
        r, c = divmod(i, cols)
        axes[r][c].set_visible(False)

    fig.suptitle(f"{classifier.upper()} Confusion Matrices", fontsize=14)
    fig.tight_layout()
    path = out_dir / f"{classifier}_confusion_matrices.png"
    fig.savefig(path, dpi=150)
    plt.close(fig)
    print(f"  Saved {path}")


# ---------------------------------------------------------------------------
# Accuracy heatmap
# ---------------------------------------------------------------------------

def plot_accuracy_matrix(ml_results, out_dir):
    """Heatmap of accuracy for all (train, test, classifier) combos."""
    classifiers = [c for c in ["cnn", "csp"] if ml_results.get(c)]
    if not classifiers:
        return

    # Collect all source names
    all_sources = set()
    for clf in classifiers:
        for key in ml_results[clf]:
            parts = key.split("__")
            all_sources.add(parts[0].replace("train_", ""))
            all_sources.add(parts[1].replace("test_", ""))
    sources = sorted(all_sources)

    fig, axes = plt.subplots(1, len(classifiers), figsize=(6 * len(classifiers), 4),
                             squeeze=False)

    for ci, clf in enumerate(classifiers):
        ax = axes[0][ci]
        matrix = np.full((len(sources), len(sources)), np.nan)
        for key, res in ml_results[clf].items():
            parts = key.split("__")
            train_src = parts[0].replace("train_", "")
            test_src = parts[1].replace("test_", "")
            ri = sources.index(train_src)
            cj = sources.index(test_src)
            matrix[ri, cj] = res["accuracy"]

        im = ax.imshow(matrix, cmap="YlGn", vmin=0, vmax=1, interpolation="nearest")
        ax.set_title(f"{clf.upper()} Accuracy")
        ax.set_xlabel("Test Source")
        ax.set_ylabel("Train Source")
        ax.set_xticks(range(len(sources)))
        ax.set_xticklabels(sources)
        ax.set_yticks(range(len(sources)))
        ax.set_yticklabels(sources)

        for ii in range(len(sources)):
            for jj in range(len(sources)):
                if not np.isnan(matrix[ii, jj]):
                    ax.text(jj, ii, f"{matrix[ii, jj]:.3f}",
                            ha="center", va="center", fontsize=12, fontweight="bold")

        fig.colorbar(im, ax=ax, shrink=0.8)

    fig.suptitle("Cross-Framework Accuracy Matrix", fontsize=14)
    fig.tight_layout()
    path = out_dir / "accuracy_matrix.png"
    fig.savefig(path, dpi=150)
    plt.close(fig)
    print(f"  Saved {path}")


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def main():
    parser = argparse.ArgumentParser(description="Visualize benchmark results")
    parser.add_argument("--results-dir", default="benchmarks/comparison/results")
    parser.add_argument("--output-dir", default=None,
                        help="Output dir for plots (defaults to results-dir)")
    args = parser.parse_args()

    results_dir = Path(args.results_dir)
    out_dir = Path(args.output_dir) if args.output_dir else results_dir
    out_dir.mkdir(parents=True, exist_ok=True)

    from benchmarks.torchsig_compat.label_map import CANONICAL_CLASSES
    class_names = CANONICAL_CLASSES

    # Speed
    speed_results = load_json(results_dir / "speed_results.json")
    if speed_results:
        print("Plotting speed results ...")
        plot_speed(speed_results, out_dir)
    else:
        print("No speed_results.json found, skipping speed plots.")

    # ML
    ml_results = load_json(results_dir / "ml_results.json")
    if ml_results:
        print("Plotting ML results ...")
        for clf in ["cnn", "csp"]:
            plot_confusion_matrices(ml_results, clf, out_dir, class_names)
        plot_accuracy_matrix(ml_results, out_dir)
    else:
        print("No ml_results.json found, skipping ML plots.")

    print("Done.")


if __name__ == "__main__":
    main()
