"""
SPECTRA Example 11: Train a Narrowband AMC Classifier
======================================================
Level: Intermediate

Learn how to:
- Load the spectra-18 benchmark dataset
- Instantiate a CNNAMC reference model
- Train for N epochs with Adam optimizer
- Evaluate with confusion matrix
- Print per-class accuracy

Usage:
    python examples/11_train_narrowband_cnn.py
    python examples/11_train_narrowband_cnn.py --epochs 2 --train-samples 1000
"""

import argparse
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

import torch
import torch.nn as nn
from spectra.benchmarks import load_benchmark
from spectra.metrics import accuracy, classification_report, confusion_matrix
from spectra.models import CNNAMC
from torch.utils.data import DataLoader


def train_one_epoch(model, loader, optimizer, criterion, device):
    model.train()
    total_loss, total = 0.0, 0
    for x, y in loader:
        x, y = x.to(device), y.to(device)
        optimizer.zero_grad()
        loss = criterion(model(x), y)
        loss.backward()
        optimizer.step()
        total_loss += loss.item() * x.size(0)
        total += x.size(0)
    return total_loss / total


def evaluate_model(model, loader, device):
    model.eval()
    all_preds, all_labels = [], []
    with torch.no_grad():
        for x, y in loader:
            x = x.to(device)
            preds = model(x).argmax(dim=1).cpu()
            all_preds.append(preds)
            all_labels.append(y)
    return torch.cat(all_preds).numpy(), torch.cat(all_labels).numpy()


def main():
    parser = argparse.ArgumentParser(description="Train CNNAMC on spectra-18")
    parser.add_argument("--epochs", type=int, default=15)
    parser.add_argument("--batch-size", type=int, default=128)
    parser.add_argument("--lr", type=float, default=1e-3)
    parser.add_argument("--train-samples", type=int, default=None,
                        help="Override training set size (default: use benchmark config)")
    parser.add_argument("--num-workers", type=int, default=0)
    args = parser.parse_args()

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Device: {device}")

    # ── 1. Load the spectra-18 benchmark ──────────────────────────────────
    print("\nLoading spectra-18 benchmark...")
    train_ds = load_benchmark("spectra-18", split="train")
    test_ds = load_benchmark("spectra-18", split="test")

    if args.train_samples is not None:
        train_ds.num_samples = args.train_samples

    # Class names from the waveform pool
    class_names = [
        "BPSK", "QPSK", "8PSK", "16QAM", "64QAM", "256QAM",
        "2FSK", "4FSK", "MSK", "GMSK", "OFDM", "LFM",
        "Costas", "Frank", "P1", "AM-DSB-SC", "Barker-13", "Noise",
    ]
    num_classes = len(class_names)

    train_loader = DataLoader(train_ds, batch_size=args.batch_size,
                              shuffle=False, num_workers=args.num_workers)
    test_loader = DataLoader(test_ds, batch_size=args.batch_size,
                             shuffle=False, num_workers=args.num_workers)

    print(f"Training samples: {len(train_ds)}")
    print(f"Test samples:     {len(test_ds)}")
    print(f"Classes:          {num_classes}")

    # ── 2. Instantiate CNNAMC ─────────────────────────────────────────────
    model = CNNAMC(num_classes=num_classes, num_iq_samples=1024).to(device)
    optimizer = torch.optim.Adam(model.parameters(), lr=args.lr)
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=args.epochs)
    criterion = nn.CrossEntropyLoss()

    param_count = sum(p.numel() for p in model.parameters())
    print(f"\nModel: CNNAMC ({param_count:,} parameters)")

    # ── 3. Train ──────────────────────────────────────────────────────────
    print(f"\nTraining for {args.epochs} epochs...")
    for epoch in range(1, args.epochs + 1):
        loss = train_one_epoch(model, train_loader, optimizer, criterion, device)
        scheduler.step()
        if epoch % max(1, args.epochs // 5) == 0 or epoch == 1:
            print(f"  Epoch {epoch:3d}/{args.epochs}  loss={loss:.4f}")

    # ── 4. Evaluate on test set ───────────────────────────────────────────
    print("\nEvaluating on test set...")
    y_pred, y_true = evaluate_model(model, test_loader, device)
    test_acc = accuracy(y_true, y_pred)
    cm = confusion_matrix(y_true, y_pred, num_classes=num_classes)

    print(f"\nOverall Test Accuracy: {test_acc:.4f}")
    print(f"\nConfusion Matrix ({num_classes}x{num_classes}):")
    print(cm)

    # ── 5. Per-class report ─────────────────────────────────────────────
    report = classification_report(y_true, y_pred, class_names=class_names)
    print(f"\n{'Class':<15} {'Precision':>10} {'Recall':>10} {'F1':>10} {'Support':>10}")
    print("-" * 60)
    for name, metrics in report.items():
        print(
            f"{name:<15} {metrics['precision']:>10.3f} {metrics['recall']:>10.3f} "
            f"{metrics['f1']:>10.3f} {metrics['support']:>10d}"
        )

    print("\nDone!")


if __name__ == "__main__":
    main()
