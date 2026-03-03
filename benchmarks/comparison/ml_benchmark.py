"""Cross-framework ML comparison benchmark: SPECTRA vs TorchSig.

Runs a 2x2 matrix (train_source x test_source) for both CNN (ResNet-18
on spectrograms) and CSP (CyclostationaryAMC with cumulants) classifiers.
"""
import argparse
import json
import time
from pathlib import Path

import numpy as np
import torch
import torch.nn as nn
import yaml
from torch.utils.data import DataLoader, TensorDataset

import spectra as sp
from benchmarks.comparison.models import ResNetAMC
from benchmarks.torchsig_compat.label_map import (
    CANONICAL_CLASSES,
    spectra_waveform_pool,
)


# ---------------------------------------------------------------------------
# Dataset builders
# ---------------------------------------------------------------------------

def build_spectra_dataset(cfg, seed, split="train"):
    """Build a SPECTRA NarrowbandDataset from config."""
    pool = spectra_waveform_pool()
    impairments = sp.Compose([sp.AWGN(snr_range=tuple(cfg["snr_range"]))])
    return sp.NarrowbandDataset(
        waveform_pool=pool,
        num_samples=cfg["num_samples"][split],
        num_iq_samples=cfg["num_iq_samples"],
        sample_rate=cfg["sample_rate"],
        impairments=impairments,
        seed=seed,
    )


def build_torchsig_dataset(cfg, seed, split="train"):
    """Build a TorchSig dataset wrapped in the adapter."""
    from benchmarks.torchsig_compat.adapter import TorchSigAdapter
    from benchmarks.torchsig_compat.label_map import torchsig_class_names

    try:
        from torchsig.datasets import Sig53
    except ImportError:
        raise ImportError(
            "TorchSig not installed. Run: python benchmarks/torchsig_compat/install.py"
        )
    is_train = split == "train"
    ts_dataset = Sig53(
        root="./torchsig_data",
        train=is_train,
        impaired=True,
        class_list=torchsig_class_names(),
        num_iq_samples=cfg["num_iq_samples"],
        num_samples_per_class=cfg["num_samples"][split] // cfg["num_classes"],
    )
    return TorchSigAdapter(ts_dataset, class_list=CANONICAL_CLASSES)


# ---------------------------------------------------------------------------
# Spectrogram conversion
# ---------------------------------------------------------------------------

def iq_to_spectrogram(dataset, cfg, max_samples=None):
    """Convert dataset IQ samples to magnitude spectrograms.

    Returns a TensorDataset of (spectrogram[1, F, T], label).
    """
    stft = sp.Spectrogram(
        nfft=cfg["cnn"]["stft_nfft"], hop_length=cfg["cnn"]["stft_hop"]
    )
    specs, labels = [], []
    n = len(dataset) if max_samples is None else min(max_samples, len(dataset))
    for i in range(n):
        iq_tensor, label = dataset[i]
        # iq_tensor is [2, N] float; convert to complex for STFT
        iq_np = iq_tensor[0].numpy() + 1j * iq_tensor[1].numpy()
        spec = stft(iq_np)  # returns [1, F, T] tensor
        if not isinstance(spec, torch.Tensor):
            spec = torch.as_tensor(spec, dtype=torch.float32)
        specs.append(spec.float())
        labels.append(label)

    return TensorDataset(torch.stack(specs), torch.tensor(labels, dtype=torch.long))


# ---------------------------------------------------------------------------
# CNN training / evaluation
# ---------------------------------------------------------------------------

def train_cnn(model, train_loader, cfg, device):
    """Train CNN model."""
    optimizer = torch.optim.Adam(
        model.parameters(),
        lr=cfg["cnn"]["learning_rate"],
        weight_decay=cfg["cnn"]["weight_decay"],
    )
    criterion = nn.CrossEntropyLoss()
    model.train()
    for epoch in range(cfg["cnn"]["epochs"]):
        total_loss, correct, total = 0.0, 0, 0
        for batch_x, batch_y in train_loader:
            batch_x, batch_y = batch_x.to(device), batch_y.to(device)
            optimizer.zero_grad()
            logits = model(batch_x)
            loss = criterion(logits, batch_y)
            loss.backward()
            optimizer.step()
            total_loss += loss.item() * batch_x.size(0)
            correct += (logits.argmax(1) == batch_y).sum().item()
            total += batch_x.size(0)
        if (epoch + 1) % 10 == 0:
            acc = correct / total
            print(f"    Epoch {epoch + 1}/{cfg['cnn']['epochs']}  "
                  f"loss={total_loss / total:.4f}  acc={acc:.3f}")
    return model


def eval_cnn(model, test_loader, cfg, device):
    """Evaluate CNN model, return accuracy and confusion matrix."""
    model.eval()
    all_preds, all_labels = [], []
    with torch.no_grad():
        for batch_x, batch_y in test_loader:
            batch_x = batch_x.to(device)
            preds = model(batch_x).argmax(1).cpu()
            all_preds.append(preds)
            all_labels.append(batch_y)

    preds = torch.cat(all_preds).numpy()
    labels = torch.cat(all_labels).numpy()
    accuracy = float((preds == labels).mean())
    # Confusion matrix
    n_cls = cfg["num_classes"]
    cm = np.zeros((n_cls, n_cls), dtype=int)
    for t, p in zip(labels, preds):
        cm[t, p] += 1
    return accuracy, cm.tolist()


def run_cnn_experiment(train_ds, test_ds, cfg, device):
    """Full CNN experiment: spectrogram conversion, train, evaluate."""
    print("    Converting train IQ -> spectrograms ...")
    train_spec = iq_to_spectrogram(train_ds, cfg)
    print("    Converting test IQ -> spectrograms ...")
    test_spec = iq_to_spectrogram(test_ds, cfg)

    train_loader = DataLoader(
        train_spec, batch_size=cfg["cnn"]["batch_size"], shuffle=True
    )
    test_loader = DataLoader(
        test_spec, batch_size=cfg["cnn"]["batch_size"]
    )

    model = ResNetAMC(
        num_classes=cfg["num_classes"], input_channels=1
    ).to(device)

    print("    Training ResNet-18 ...")
    train_cnn(model, train_loader, cfg, device)

    accuracy, cm = eval_cnn(model, test_loader, cfg, device)
    return {"accuracy": accuracy, "confusion_matrix": cm}


# ---------------------------------------------------------------------------
# CSP training / evaluation
# ---------------------------------------------------------------------------

def run_csp_experiment(train_ds, test_ds, cfg):
    """Full CSP experiment: extract cumulant features, train RF, evaluate."""
    amc = sp.CyclostationaryAMC(
        feature_set=cfg["csp"]["feature_set"],
        classifier=cfg["csp"]["classifier"],
    )

    print("    Extracting train features ...")
    X_train, y_train = _extract_csp_features(amc, train_ds)
    print("    Extracting test features ...")
    X_test, y_test = _extract_csp_features(amc, test_ds)

    print("    Training classifier ...")
    amc.fit(X_train, y_train)

    preds = amc.predict(X_test)
    accuracy = float((preds == y_test).mean())

    n_cls = cfg["num_classes"]
    cm = np.zeros((n_cls, n_cls), dtype=int)
    for t, p in zip(y_test, preds):
        cm[int(t), int(p)] += 1

    return {"accuracy": accuracy, "confusion_matrix": cm.tolist()}


def _extract_csp_features(amc, dataset):
    """Extract CSP features from a dataset."""
    features, labels = [], []
    for i in range(len(dataset)):
        iq_tensor, label = dataset[i]
        iq_np = iq_tensor[0].numpy() + 1j * iq_tensor[1].numpy()
        feat = amc.extract_features(iq_np)
        features.append(feat)
        labels.append(label)
    return np.array(features), np.array(labels)


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def main():
    parser = argparse.ArgumentParser(
        description="ML comparison benchmark: SPECTRA vs TorchSig"
    )
    parser.add_argument("--config", default="benchmarks/comparison/config.yaml")
    parser.add_argument("--output-dir", default="benchmarks/comparison/results")
    parser.add_argument("--skip-torchsig", action="store_true")
    parser.add_argument("--skip-cnn", action="store_true")
    parser.add_argument("--skip-csp", action="store_true")
    args = parser.parse_args()

    with open(args.config) as f:
        cfg = yaml.safe_load(f)

    out_dir = Path(args.output_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Device: {device}")

    # Build datasets
    sources = {"spectra": build_spectra_dataset}
    if not args.skip_torchsig:
        sources["torchsig"] = build_torchsig_dataset

    datasets = {}
    for name, builder in sources.items():
        seed_key = f"{name}_train"
        print(f"Building {name} train dataset ...")
        datasets[(name, "train")] = builder(
            cfg, seed=cfg["seeds"][seed_key], split="train"
        )
        seed_key = f"{name}_test"
        print(f"Building {name} test dataset ...")
        datasets[(name, "test")] = builder(
            cfg, seed=cfg["seeds"][seed_key], split="test"
        )

    results = {"cnn": {}, "csp": {}}
    source_names = list(sources.keys())

    # CNN experiments
    if not args.skip_cnn:
        print("\n========== CNN (ResNet-18) Experiments ==========")
        for train_src in source_names:
            for test_src in source_names:
                key = f"train_{train_src}__test_{test_src}"
                print(f"\n  [{key}]")
                t0 = time.perf_counter()
                res = run_cnn_experiment(
                    datasets[(train_src, "train")],
                    datasets[(test_src, "test")],
                    cfg,
                    device,
                )
                res["elapsed_s"] = time.perf_counter() - t0
                results["cnn"][key] = res
                print(f"    Accuracy: {res['accuracy']:.4f}  "
                      f"({res['elapsed_s']:.1f}s)")

    # CSP experiments
    if not args.skip_csp:
        print("\n========== CSP (CyclostationaryAMC) Experiments ==========")
        for train_src in source_names:
            for test_src in source_names:
                key = f"train_{train_src}__test_{test_src}"
                print(f"\n  [{key}]")
                t0 = time.perf_counter()
                res = run_csp_experiment(
                    datasets[(train_src, "train")],
                    datasets[(test_src, "test")],
                    cfg,
                )
                res["elapsed_s"] = time.perf_counter() - t0
                results["csp"][key] = res
                print(f"    Accuracy: {res['accuracy']:.4f}  "
                      f"({res['elapsed_s']:.1f}s)")

    # Save
    results_path = out_dir / "ml_results.json"
    with open(results_path, "w") as f:
        json.dump(results, f, indent=2)
    print(f"\nResults saved to {results_path}")

    # Summary table
    print("\n========== Summary ==========")
    for classifier in ["cnn", "csp"]:
        if results[classifier]:
            print(f"\n{classifier.upper()}:")
            for key, res in results[classifier].items():
                print(f"  {key}: {res['accuracy']:.4f}")


if __name__ == "__main__":
    main()
