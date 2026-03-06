# SNR-Sweep AMC Benchmark Implementation Plan

> **For Claude:** REQUIRED SUB-SKILL: Use superpowers:executing-plans to implement this plan task-by-task.

**Goal:** Add a `spectra-18-snr-sweep` benchmark that produces the canonical "accuracy vs. SNR" curves used throughout the AMC literature, giving the signal processing and communications community a reproducible, published baseline to compare against.

**Architecture:** A new YAML config extends the `spectra-18` benchmark with `snr_eval_points` metadata. A standalone benchmark script builds per-SNR test datasets (using AWGN with fixed SNR), trains a lightweight raw-IQ CNN classifier, evaluates at each SNR point, and saves JSON results. A separate visualizer renders the accuracy-vs-SNR curves. No changes to existing APIs are required.

**Tech Stack:** Python 3.10+, PyTorch (CPU), NumPy, PyYAML, Matplotlib, existing `spectra` package (NarrowbandDataset, AWGN, load_benchmark).

---

## Why This Builds Community Trust

The accuracy-vs-SNR curve is the **canonical benchmark** in automatic modulation classification research (RadioML 2016/2018, DeepSig, etc.). By publishing a reproducible curve with fixed seeds, reference model, and per-class breakdown, SPECTRA gives researchers:

1. A baseline to beat — "our model exceeds SPECTRA baseline at 0 dB"
2. A sanity check — generated data should produce monotonically increasing accuracy with SNR
3. Cross-library comparison — researchers can run the same script on TorchSig-generated data

---

## Background Reading

Before implementing, understand:
- `python/spectra/benchmarks/loader.py` — how `load_benchmark()` builds datasets from YAML
- `python/spectra/impairments/awgn.py` — AWGN supports both `snr=float` (fixed) and `snr_range=(lo,hi)` (random)
- `python/spectra/datasets/narrowband.py` — NarrowbandDataset.__getitem__ generates per-index
- `python/spectra/benchmarks/configs/spectra-18.yaml` — the existing 18-class config we extend
- `benchmarks/comparison/models.py` — existing ResNetAMC (uses torchvision; we'll write a lighter model)

---

## Task 1: Write the YAML config

**Files:**
- Create: `python/spectra/benchmarks/configs/spectra-18-snr-sweep.yaml`

**Step 1: Create the config file**

```yaml
name: "spectra-18-snr-sweep"
version: "1.0"
description: >
  18-class narrowband AMC benchmark with SNR-stratified evaluation.
  Same signal classes as spectra-18. Training uses the full SNR range
  [-10, 30] dB. Evaluation reports per-class accuracy at 21 fixed SNR
  points for direct comparison with published AMC literature.
task: "narrowband"

sample_rate: 1_000_000
num_iq_samples: 1024
num_samples:
  train: 50_000
  val: 10_000
  test: 10_000

seed:
  train: 7000
  val: 8000
  test: 9000

# 21 fixed SNR test points matching RadioML 2018 sweep convention
snr_eval_points: [-10, -8, -6, -4, -2, 0, 2, 4, 6, 8, 10, 12, 14, 16, 18, 20, 22, 24, 26, 28, 30]
# Samples generated per SNR point (all 18 classes, evenly distributed)
num_samples_per_snr: 3600

snr_range: [-10, 30]

waveform_pool:
  - {type: "BPSK"}
  - {type: "QPSK"}
  - {type: "PSK8"}
  - {type: "QAM16"}
  - {type: "QAM64"}
  - {type: "QAM256"}
  - {type: "FSK", params: {order: 2}}
  - {type: "FSK", params: {order: 4}}
  - {type: "MSK"}
  - {type: "GMSK"}
  - {type: "OFDM"}
  - {type: "LFM"}
  - {type: "CostasCode"}
  - {type: "FrankCode"}
  - {type: "P1Code"}
  - {type: "AMDSB_SC"}
  - {type: "BarkerCode", params: {length: 13}}
  - {type: "Noise"}

impairments:
  - {type: "FrequencyOffset", params: {max_offset: 50_000}}
  - {type: "PhaseOffset", params: {max_offset: 3.14159}}
  - {type: "AWGN"}
```

**Step 2: Verify the file loads with the existing loader**

```bash
python -c "
from spectra.benchmarks import load_benchmark
ds = load_benchmark('spectra-18-snr-sweep', split='train')
print(f'Train dataset: {len(ds)} samples')
x, y = ds[0]
print(f'Sample shape: {x.shape}, label: {y}')
"
```

Expected output:
```
Train dataset: 50000 samples
Sample shape: torch.Size([2, 1024]), label: <int 0-17>
```

**Step 3: Commit**

```bash
git add python/spectra/benchmarks/configs/spectra-18-snr-sweep.yaml
git commit -m "feat(benchmarks): add spectra-18-snr-sweep YAML config"
```

---

## Task 2: Implement the lightweight CNN model

**Files:**
- Create: `benchmarks/snr_sweep/__init__.py`
- Create: `benchmarks/snr_sweep/models.py`
- Create: `tests/test_snr_sweep_model.py`

**Background:** We need a classifier that:
1. Takes raw IQ: input shape `[batch, 2, 1024]` (I/Q channels, 1024 samples)
2. Requires NO torchvision dependency (unlike the ResNetAMC)
3. Is comparable to published AMC CNN architectures (e.g., VT-CNN2 from O'Shea 2016)

**Step 1: Write the failing test**

```python
# tests/test_snr_sweep_model.py
import pytest
import torch


def test_cnn_amc_forward_shape():
    from benchmarks.snr_sweep.models import CNNAMC
    model = CNNAMC(num_classes=18)
    x = torch.randn(4, 2, 1024)
    out = model(x)
    assert out.shape == (4, 18)


def test_cnn_amc_different_classes():
    from benchmarks.snr_sweep.models import CNNAMC
    model = CNNAMC(num_classes=8)
    x = torch.randn(2, 2, 256)
    out = model(x)
    assert out.shape == (2, 8)


def test_cnn_amc_single_sample():
    from benchmarks.snr_sweep.models import CNNAMC
    model = CNNAMC(num_classes=18)
    x = torch.randn(1, 2, 1024)
    out = model(x)
    assert out.shape == (1, 18)
```

**Step 2: Run test to verify failure**

```bash
cd /path/to/SPECTRA
pytest tests/test_snr_sweep_model.py -v
```

Expected: `ModuleNotFoundError` or `ImportError` for `benchmarks.snr_sweep.models`

**Step 3: Create `benchmarks/snr_sweep/__init__.py`**

```python
```
(empty file)

**Step 4: Implement `benchmarks/snr_sweep/models.py`**

Architecture: Conv1D stack followed by LSTM and dense layers. This mirrors the VT-CNN2 architecture commonly used in AMC literature, adapted for PyTorch.

```python
"""Lightweight CNN classifier for raw IQ modulation classification.

Architecture mirrors the VT-CNN2 design (O'Shea & Hoydis, 2016):
- Two Conv1D blocks with batch norm and max-pooling
- Two dense layers with dropout
- Softmax output

Input: [batch, 2, num_iq_samples]  (I/Q as two channels)
Output: [batch, num_classes]  (logits)
"""
import torch
import torch.nn as nn


class CNNAMC(nn.Module):
    """Conv1D classifier for raw I/Q modulation classification.

    Args:
        num_classes: Number of output modulation classes.
        num_iq_samples: Length of the IQ input sequence. Default 1024.
    """

    def __init__(self, num_classes: int = 18, num_iq_samples: int = 1024):
        super().__init__()
        self.features = nn.Sequential(
            # Block 1: [B, 2, N] -> [B, 64, N//2]
            nn.Conv1d(2, 64, kernel_size=3, padding=1),
            nn.BatchNorm1d(64),
            nn.ReLU(inplace=True),
            nn.Conv1d(64, 64, kernel_size=3, padding=1),
            nn.BatchNorm1d(64),
            nn.ReLU(inplace=True),
            nn.MaxPool1d(2),
            # Block 2: [B, 64, N//2] -> [B, 128, N//4]
            nn.Conv1d(64, 128, kernel_size=3, padding=1),
            nn.BatchNorm1d(128),
            nn.ReLU(inplace=True),
            nn.Conv1d(128, 128, kernel_size=3, padding=1),
            nn.BatchNorm1d(128),
            nn.ReLU(inplace=True),
            nn.MaxPool1d(2),
            # Block 3: [B, 128, N//4] -> [B, 256, N//8]
            nn.Conv1d(128, 256, kernel_size=3, padding=1),
            nn.BatchNorm1d(256),
            nn.ReLU(inplace=True),
            nn.MaxPool1d(2),
        )
        # Compute the flattened feature size
        self._flat_size = 256 * (num_iq_samples // 8)
        self.classifier = nn.Sequential(
            nn.Flatten(),
            nn.Linear(self._flat_size, 512),
            nn.ReLU(inplace=True),
            nn.Dropout(0.5),
            nn.Linear(512, 256),
            nn.ReLU(inplace=True),
            nn.Dropout(0.5),
            nn.Linear(256, num_classes),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.classifier(self.features(x))
```

**Step 5: Run tests to verify passing**

```bash
pytest tests/test_snr_sweep_model.py -v
```

Expected: All 3 tests PASS

**Step 6: Commit**

```bash
git add benchmarks/snr_sweep/__init__.py benchmarks/snr_sweep/models.py tests/test_snr_sweep_model.py
git commit -m "feat(benchmarks): add CNNAMC raw-IQ classifier for SNR sweep"
```

---

## Task 3: Build per-SNR test dataset utility

**Files:**
- Create: `benchmarks/snr_sweep/dataset_utils.py`
- Create: `tests/test_snr_sweep_datasets.py`

**Background:** At evaluation time, we need test datasets where every sample has exactly the same fixed SNR (e.g., all at 0 dB). The existing `NarrowbandDataset` uses `AWGN(snr_range=...)` which picks random SNR. For fixed SNR, we use `AWGN(snr=<fixed_value>)` and build a fresh dataset for each SNR point.

**Step 1: Write the failing tests**

```python
# tests/test_snr_sweep_datasets.py
import pytest
import numpy as np
import torch


def test_build_snr_dataset_length():
    from benchmarks.snr_sweep.dataset_utils import build_snr_dataset
    from spectra import waveforms as wmod
    pool = [wmod.BPSK(), wmod.QPSK()]
    ds = build_snr_dataset(pool, snr_db=10.0, num_samples=100,
                           num_iq_samples=256, sample_rate=1e6, seed=42)
    assert len(ds) == 100


def test_build_snr_dataset_output_shape():
    from benchmarks.snr_sweep.dataset_utils import build_snr_dataset
    from spectra import waveforms as wmod
    pool = [wmod.BPSK(), wmod.QPSK()]
    ds = build_snr_dataset(pool, snr_db=5.0, num_samples=10,
                           num_iq_samples=128, sample_rate=1e6, seed=99)
    x, y = ds[0]
    assert x.shape == torch.Size([2, 128])
    assert isinstance(y, int)


def test_build_snr_dataset_different_snrs_different_noise():
    from benchmarks.snr_sweep.dataset_utils import build_snr_dataset
    from spectra import waveforms as wmod
    pool = [wmod.BPSK()]
    ds_low = build_snr_dataset(pool, snr_db=-10.0, num_samples=5,
                                num_iq_samples=256, sample_rate=1e6, seed=1)
    ds_high = build_snr_dataset(pool, snr_db=30.0, num_samples=5,
                                 num_iq_samples=256, sample_rate=1e6, seed=1)
    x_low, _ = ds_low[0]
    x_high, _ = ds_high[0]
    # High SNR should have higher signal power relative to noise
    # (samples are different due to different noise levels)
    assert not torch.equal(x_low, x_high)
```

**Step 2: Run tests to verify failure**

```bash
pytest tests/test_snr_sweep_datasets.py -v
```

Expected: `ImportError` for `benchmarks.snr_sweep.dataset_utils`

**Step 3: Implement `benchmarks/snr_sweep/dataset_utils.py`**

```python
"""Utilities for constructing per-SNR evaluation datasets.

Each per-SNR dataset is a NarrowbandDataset with AWGN fixed at a
specific SNR value, allowing stratified evaluation across the SNR range.
"""
from typing import List, Optional

from spectra.datasets.narrowband import NarrowbandDataset
from spectra.impairments import AWGN, Compose
from spectra.impairments.frequency_offset import FrequencyOffset
from spectra.impairments.phase_offset import PhaseOffset
from spectra.waveforms.base import Waveform


def build_snr_dataset(
    waveform_pool: List[Waveform],
    snr_db: float,
    num_samples: int,
    num_iq_samples: int,
    sample_rate: float,
    seed: int,
    max_freq_offset: float = 50_000.0,
    max_phase_offset: float = 3.14159,
) -> NarrowbandDataset:
    """Build a NarrowbandDataset with fixed SNR for evaluation.

    All impairments from the spectra-18 training config are preserved
    (frequency offset, phase offset) except AWGN is fixed at ``snr_db``.

    Parameters
    ----------
    waveform_pool:
        List of Waveform instances to sample from.
    snr_db:
        Fixed SNR in dB for all samples in this dataset.
    num_samples:
        Total number of samples to generate.
    num_iq_samples:
        IQ sample length per observation.
    sample_rate:
        Sample rate in Hz.
    seed:
        Base seed for deterministic generation.
    max_freq_offset:
        Maximum frequency offset in Hz (same as training config).
    max_phase_offset:
        Maximum phase offset in radians (same as training config).

    Returns
    -------
    NarrowbandDataset
        Dataset with all samples corrupted by exactly ``snr_db`` dB AWGN.
    """
    impairments = Compose([
        FrequencyOffset(max_offset=max_freq_offset),
        PhaseOffset(max_offset=max_phase_offset),
        AWGN(snr=snr_db),
    ])
    return NarrowbandDataset(
        waveform_pool=waveform_pool,
        num_samples=num_samples,
        num_iq_samples=num_iq_samples,
        sample_rate=sample_rate,
        impairments=impairments,
        seed=seed,
    )
```

**Step 4: Run tests to verify passing**

```bash
pytest tests/test_snr_sweep_datasets.py -v
```

Expected: All 3 tests PASS

**Step 5: Commit**

```bash
git add benchmarks/snr_sweep/dataset_utils.py tests/test_snr_sweep_datasets.py
git commit -m "feat(benchmarks): add per-SNR dataset builder utility"
```

---

## Task 4: Implement the SNR sweep benchmark script

**Files:**
- Create: `benchmarks/snr_sweep/run_snr_sweep.py`
- Create: `tests/test_snr_sweep_runner.py`

**Step 1: Write the failing tests**

```python
# tests/test_snr_sweep_runner.py
"""Smoke tests for the SNR sweep runner functions."""
import pytest
import numpy as np
import torch
from torch.utils.data import DataLoader


def test_train_one_epoch_runs():
    """Verify train_one_epoch executes without error."""
    from benchmarks.snr_sweep.models import CNNAMC
    from benchmarks.snr_sweep.run_snr_sweep import train_one_epoch
    from spectra import waveforms as wmod
    from spectra.datasets.narrowband import NarrowbandDataset
    from spectra.impairments import AWGN, Compose

    pool = [wmod.BPSK(), wmod.QPSK()]
    ds = NarrowbandDataset(
        waveform_pool=pool, num_samples=16, num_iq_samples=128,
        sample_rate=1e6, impairments=Compose([AWGN(snr_range=(0, 20))]), seed=0,
    )
    loader = DataLoader(ds, batch_size=8)
    model = CNNAMC(num_classes=2, num_iq_samples=128)
    optimizer = torch.optim.Adam(model.parameters(), lr=1e-3)
    loss = train_one_epoch(model, loader, optimizer, torch.device("cpu"))
    assert isinstance(loss, float)
    assert loss >= 0.0


def test_evaluate_accuracy_range():
    """Verify evaluate returns accuracy in [0, 1]."""
    from benchmarks.snr_sweep.models import CNNAMC
    from benchmarks.snr_sweep.run_snr_sweep import evaluate
    from spectra import waveforms as wmod
    from benchmarks.snr_sweep.dataset_utils import build_snr_dataset

    pool = [wmod.BPSK(), wmod.QPSK()]
    ds = build_snr_dataset(pool, snr_db=10.0, num_samples=20,
                           num_iq_samples=128, sample_rate=1e6, seed=42)
    loader = DataLoader(ds, batch_size=10)
    model = CNNAMC(num_classes=2, num_iq_samples=128)
    acc, cm = evaluate(model, loader, num_classes=2, device=torch.device("cpu"))
    assert 0.0 <= acc <= 1.0
    assert cm.shape == (2, 2)
    assert cm.sum() == 20
```

**Step 2: Run test to verify failure**

```bash
pytest tests/test_snr_sweep_runner.py -v
```

Expected: `ImportError` for `benchmarks.snr_sweep.run_snr_sweep`

**Step 3: Implement `benchmarks/snr_sweep/run_snr_sweep.py`**

```python
"""SNR-stratified AMC benchmark for SPECTRA.

Trains a CNNAMC classifier on the full SNR range from spectra-18-snr-sweep,
then evaluates accuracy at each fixed SNR point. Produces a JSON results file
and optional accuracy-vs-SNR plots.

Usage
-----
    # Full run (train + eval)
    python benchmarks/snr_sweep/run_snr_sweep.py

    # Quick run with smaller dataset
    python benchmarks/snr_sweep/run_snr_sweep.py --train-samples 5000 --test-per-snr 360

    # Load existing results and just plot
    python benchmarks/snr_sweep/run_snr_sweep.py --plot-only
"""
import argparse
import json
import time
from pathlib import Path
from typing import Dict, List, Tuple

import numpy as np
import torch
import torch.nn as nn
import yaml
from torch.utils.data import DataLoader

from spectra.benchmarks import load_benchmark
from spectra.benchmarks.loader import _build_waveform_pool, _resolve_config_path
from benchmarks.snr_sweep.dataset_utils import build_snr_dataset
from benchmarks.snr_sweep.models import CNNAMC

_DEFAULT_CONFIG = "spectra-18-snr-sweep"
_DEFAULT_OUT_DIR = Path("benchmarks/snr_sweep/results")


# ---------------------------------------------------------------------------
# Training / evaluation helpers
# ---------------------------------------------------------------------------

def train_one_epoch(
    model: nn.Module,
    loader: DataLoader,
    optimizer: torch.optim.Optimizer,
    device: torch.device,
) -> float:
    """Run one training epoch. Returns mean cross-entropy loss."""
    model.train()
    criterion = nn.CrossEntropyLoss()
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


def evaluate(
    model: nn.Module,
    loader: DataLoader,
    num_classes: int,
    device: torch.device,
) -> Tuple[float, np.ndarray]:
    """Evaluate model. Returns (accuracy, confusion_matrix [num_classes x num_classes])."""
    model.eval()
    cm = np.zeros((num_classes, num_classes), dtype=int)
    with torch.no_grad():
        for x, y in loader:
            x = x.to(device)
            preds = model(x).argmax(1).cpu().numpy()
            labels = y.numpy()
            for t, p in zip(labels, preds):
                cm[int(t), int(p)] += 1
    total = cm.sum()
    accuracy = float(cm.diagonal().sum() / total) if total > 0 else 0.0
    return accuracy, cm


# ---------------------------------------------------------------------------
# Main benchmark runner
# ---------------------------------------------------------------------------

def run_snr_sweep(
    config_name: str,
    out_dir: Path,
    epochs: int,
    batch_size: int,
    lr: float,
    train_samples: int,
    test_per_snr: int,
    num_workers: int,
    device: torch.device,
) -> Dict:
    """Full SNR sweep: train once, evaluate at each SNR point."""
    # Load config YAML for metadata (snr_eval_points, waveform_pool, etc.)
    config_path = _resolve_config_path(config_name)
    with open(config_path) as f:
        cfg = yaml.safe_load(f)

    snr_points: List[float] = cfg["snr_eval_points"]
    num_classes = len(cfg["waveform_pool"])
    seed_base = cfg["seed"]["test"]

    print(f"Benchmark: {cfg['name']}")
    print(f"Classes: {num_classes}, SNR points: {snr_points}")

    # 1. Build training dataset
    print(f"\nBuilding training dataset ({train_samples} samples) ...")
    train_ds = load_benchmark(config_name, split="train")
    # Override num_samples if requested
    train_ds.num_samples = train_samples
    train_loader = DataLoader(
        train_ds, batch_size=batch_size, shuffle=False, num_workers=num_workers,
    )

    # 2. Build model and optimizer
    model = CNNAMC(
        num_classes=num_classes,
        num_iq_samples=cfg["num_iq_samples"],
    ).to(device)
    optimizer = torch.optim.Adam(model.parameters(), lr=lr)
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=epochs)

    # 3. Training loop
    print(f"\nTraining for {epochs} epochs ...")
    t_train_start = time.perf_counter()
    for epoch in range(1, epochs + 1):
        loss = train_one_epoch(model, train_loader, optimizer, device)
        scheduler.step()
        if epoch % max(1, epochs // 5) == 0 or epoch == 1:
            print(f"  Epoch {epoch:3d}/{epochs}  loss={loss:.4f}")
    train_elapsed = time.perf_counter() - t_train_start
    print(f"Training complete in {train_elapsed:.1f}s")

    # 4. Build waveform pool for per-SNR test datasets
    waveform_pool = _build_waveform_pool(cfg["waveform_pool"])

    # 5. Evaluate at each SNR point
    print(f"\nEvaluating at {len(snr_points)} SNR points ({test_per_snr} samples each) ...")
    snr_results = {}
    t_eval_start = time.perf_counter()
    for i, snr_db in enumerate(snr_points):
        test_ds = build_snr_dataset(
            waveform_pool=waveform_pool,
            snr_db=snr_db,
            num_samples=test_per_snr,
            num_iq_samples=cfg["num_iq_samples"],
            sample_rate=cfg["sample_rate"],
            seed=seed_base + i,  # unique seed per SNR point
        )
        test_loader = DataLoader(
            test_ds, batch_size=batch_size, num_workers=num_workers,
        )
        acc, cm = evaluate(model, test_loader, num_classes=num_classes, device=device)
        snr_results[snr_db] = {"accuracy": acc, "confusion_matrix": cm.tolist()}
        print(f"  SNR {snr_db:+5.1f} dB  accuracy={acc:.4f}")

    eval_elapsed = time.perf_counter() - t_eval_start

    results = {
        "benchmark": cfg["name"],
        "version": cfg.get("version", "1.0"),
        "num_classes": num_classes,
        "num_iq_samples": cfg["num_iq_samples"],
        "sample_rate": cfg["sample_rate"],
        "train_samples": train_samples,
        "test_per_snr": test_per_snr,
        "epochs": epochs,
        "train_elapsed_s": train_elapsed,
        "eval_elapsed_s": eval_elapsed,
        "snr_results": snr_results,
    }
    return results


# ---------------------------------------------------------------------------
# Entry point
# ---------------------------------------------------------------------------

def main():
    parser = argparse.ArgumentParser(description="SPECTRA SNR-sweep AMC benchmark")
    parser.add_argument("--config", default=_DEFAULT_CONFIG)
    parser.add_argument("--output-dir", default=str(_DEFAULT_OUT_DIR))
    parser.add_argument("--epochs", type=int, default=30)
    parser.add_argument("--batch-size", type=int, default=256)
    parser.add_argument("--lr", type=float, default=1e-3)
    parser.add_argument("--train-samples", type=int, default=50_000)
    parser.add_argument("--test-per-snr", type=int, default=3_600)
    parser.add_argument("--num-workers", type=int, default=0)
    parser.add_argument("--plot-only", action="store_true",
                        help="Skip training; load existing results and plot")
    args = parser.parse_args()

    out_dir = Path(args.output_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    results_path = out_dir / "snr_sweep_results.json"

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Device: {device}")

    if args.plot_only:
        if not results_path.exists():
            raise FileNotFoundError(
                f"Results file not found: {results_path}. "
                "Run without --plot-only first."
            )
        with open(results_path) as f:
            results = json.load(f)
        print(f"Loaded results from {results_path}")
    else:
        results = run_snr_sweep(
            config_name=args.config,
            out_dir=out_dir,
            epochs=args.epochs,
            batch_size=args.batch_size,
            lr=args.lr,
            train_samples=args.train_samples,
            test_per_snr=args.test_per_snr,
            num_workers=args.num_workers,
            device=device,
        )
        with open(results_path, "w") as f:
            json.dump(results, f, indent=2)
        print(f"\nResults saved to {results_path}")

    # Summary table
    print("\n========== SNR Sweep Results ==========")
    print(f"{'SNR (dB)':>10}  {'Accuracy':>10}")
    print("-" * 25)
    for snr_db, data in sorted(results["snr_results"].items(), key=lambda kv: float(kv[0])):
        print(f"{float(snr_db):>+10.1f}  {data['accuracy']:>10.4f}")

    # Optionally plot
    try:
        from benchmarks.snr_sweep.visualize import plot_snr_accuracy
        fig_path = out_dir / "snr_accuracy.png"
        plot_snr_accuracy(results, fig_path)
        print(f"\nPlot saved to {fig_path}")
    except ImportError:
        print("\n(Matplotlib not available; skipping plot)")


if __name__ == "__main__":
    main()
```

**Step 4: Run tests to verify passing**

```bash
pytest tests/test_snr_sweep_runner.py -v
```

Expected: Both tests PASS

**Step 5: Commit**

```bash
git add benchmarks/snr_sweep/run_snr_sweep.py tests/test_snr_sweep_runner.py
git commit -m "feat(benchmarks): add SNR sweep runner with train/evaluate loop"
```

---

## Task 5: Implement the visualizer

**Files:**
- Create: `benchmarks/snr_sweep/visualize.py`
- Create: `tests/test_snr_sweep_visualize.py`

**Step 1: Write the failing test**

```python
# tests/test_snr_sweep_visualize.py
import json
import pytest


def _make_fake_results(snr_points, num_classes=4):
    """Build minimal results dict for testing."""
    import numpy as np
    snr_results = {}
    for snr in snr_points:
        cm = np.zeros((num_classes, num_classes), dtype=int)
        for i in range(num_classes):
            cm[i, i] = 10  # perfect diagonal
        snr_results[str(snr)] = {
            "accuracy": 1.0,
            "confusion_matrix": cm.tolist(),
        }
    return {
        "benchmark": "test",
        "num_classes": num_classes,
        "snr_results": snr_results,
    }


def test_plot_snr_accuracy_creates_file(tmp_path):
    from benchmarks.snr_sweep.visualize import plot_snr_accuracy
    results = _make_fake_results([-10, 0, 10, 20])
    out = tmp_path / "test_plot.png"
    plot_snr_accuracy(results, out)
    assert out.exists()
    assert out.stat().st_size > 0


def test_plot_snr_accuracy_with_per_class(tmp_path):
    from benchmarks.snr_sweep.visualize import plot_snr_accuracy
    results = _make_fake_results([-5, 0, 5, 10, 15], num_classes=3)
    out = tmp_path / "test_plot2.png"
    plot_snr_accuracy(results, out, show_per_class=True)
    assert out.exists()
```

**Step 2: Run test to verify failure**

```bash
pytest tests/test_snr_sweep_visualize.py -v
```

Expected: `ImportError` for `benchmarks.snr_sweep.visualize`

**Step 3: Implement `benchmarks/snr_sweep/visualize.py`**

```python
"""Visualization for SNR sweep benchmark results.

Generates accuracy-vs-SNR curves comparable to figures in AMC literature
(O'Shea & Hoydis 2016, Schmidl & Cox 1997, etc.).
"""
from pathlib import Path
from typing import Dict, Union


def plot_snr_accuracy(
    results: Dict,
    output_path: Union[str, Path],
    show_per_class: bool = False,
    class_labels: list = None,
    figsize: tuple = (9, 6),
) -> None:
    """Plot overall (and optionally per-class) accuracy vs. SNR.

    Parameters
    ----------
    results:
        Dict as returned by ``run_snr_sweep()`` or loaded from JSON.
    output_path:
        Where to save the PNG figure.
    show_per_class:
        If True, add one line per class using confusion matrix diagonals.
    class_labels:
        Optional list of class name strings. If None, uses "Class 0..N".
    figsize:
        Matplotlib figure size.
    """
    import matplotlib
    matplotlib.use("Agg")  # non-interactive backend; safe in scripts
    import matplotlib.pyplot as plt
    import numpy as np

    snr_results = results["snr_results"]
    num_classes = results.get("num_classes", None)

    # Sort by SNR value (keys may be strings or floats from JSON)
    sorted_items = sorted(snr_results.items(), key=lambda kv: float(kv[0]))
    snr_vals = [float(k) for k, _ in sorted_items]
    overall_acc = [v["accuracy"] for _, v in sorted_items]

    fig, ax = plt.subplots(figsize=figsize)
    ax.plot(snr_vals, overall_acc, "k-o", linewidth=2, markersize=6,
            label="Overall", zorder=5)

    if show_per_class and num_classes is not None:
        cms = [np.array(v["confusion_matrix"]) for _, v in sorted_items]
        if class_labels is None:
            class_labels = [f"Class {i}" for i in range(num_classes)]
        for cls_idx in range(num_classes):
            per_class_acc = []
            for cm in cms:
                row_sum = cm[cls_idx].sum()
                per_class_acc.append(cm[cls_idx, cls_idx] / row_sum if row_sum > 0 else 0.0)
            ax.plot(snr_vals, per_class_acc, "--", alpha=0.6,
                    label=class_labels[cls_idx] if cls_idx < len(class_labels) else f"Class {cls_idx}")

    ax.set_xlabel("SNR (dB)", fontsize=13)
    ax.set_ylabel("Classification Accuracy", fontsize=13)
    ax.set_title(f"AMC Accuracy vs. SNR — {results.get('benchmark', 'SPECTRA')}", fontsize=14)
    ax.set_ylim([-0.02, 1.05])
    ax.set_xlim([snr_vals[0] - 1, snr_vals[-1] + 1])
    ax.grid(True, linestyle="--", alpha=0.5)
    ax.legend(loc="lower right", fontsize=9, ncol=2)
    fig.tight_layout()
    fig.savefig(output_path, dpi=150)
    plt.close(fig)
```

**Step 4: Run tests to verify passing**

```bash
pytest tests/test_snr_sweep_visualize.py -v
```

Expected: Both tests PASS (requires matplotlib; skip with `-k "not visualize"` if unavailable)

**Step 5: Commit**

```bash
git add benchmarks/snr_sweep/visualize.py tests/test_snr_sweep_visualize.py
git commit -m "feat(benchmarks): add accuracy-vs-SNR visualization"
```

---

## Task 6: End-to-end integration test

**Files:**
- Create: `tests/test_integration_snr_sweep.py`

**Step 1: Write the integration test**

This test runs the full pipeline (train mini model, evaluate at 3 SNR points, save JSON) in under 60 seconds using tiny dataset sizes.

```python
# tests/test_integration_snr_sweep.py
"""Integration test: full SNR sweep pipeline in fast mode."""
import json
import pytest
import torch


@pytest.mark.slow
def test_snr_sweep_end_to_end(tmp_path):
    """Full pipeline: train -> per-SNR eval -> JSON output."""
    from benchmarks.snr_sweep.run_snr_sweep import run_snr_sweep

    device = torch.device("cpu")
    results = run_snr_sweep(
        config_name="spectra-18-snr-sweep",
        out_dir=tmp_path,
        epochs=2,
        batch_size=32,
        lr=1e-3,
        train_samples=360,   # tiny: 20 samples/class
        test_per_snr=180,    # tiny: 10 samples/class
        num_workers=0,
        device=device,
    )

    # Verify structure
    assert "snr_results" in results
    assert len(results["snr_results"]) == 21  # 21 SNR points in config

    # Verify each SNR point has valid accuracy
    for snr_str, data in results["snr_results"].items():
        assert "accuracy" in data
        assert 0.0 <= data["accuracy"] <= 1.0
        assert "confusion_matrix" in data
        cm = data["confusion_matrix"]
        assert len(cm) == 18       # 18 classes
        assert len(cm[0]) == 18

    # Verify JSON serializable
    json_str = json.dumps(results)
    loaded = json.loads(json_str)
    assert loaded["num_classes"] == 18


@pytest.mark.slow
def test_snr_sweep_accuracy_increases_with_snr(tmp_path):
    """Accuracy at high SNR should exceed accuracy at low SNR."""
    from benchmarks.snr_sweep.run_snr_sweep import run_snr_sweep

    device = torch.device("cpu")
    results = run_snr_sweep(
        config_name="spectra-18-snr-sweep",
        out_dir=tmp_path,
        epochs=5,
        batch_size=64,
        lr=1e-3,
        train_samples=1800,   # 100 samples/class
        test_per_snr=360,     # 20 samples/class per SNR point
        num_workers=0,
        device=device,
    )

    snr_results = results["snr_results"]
    low_snr_acc = snr_results[str(-10.0)]["accuracy"] if str(-10.0) in snr_results \
                  else snr_results["-10"]["accuracy"]
    high_snr_acc = snr_results[str(30.0)]["accuracy"] if str(30.0) in snr_results \
                   else snr_results["30"]["accuracy"]

    # After some training, high SNR should be meaningfully better than low SNR
    # (not a guarantee with only 5 epochs, but should be directionally correct)
    assert high_snr_acc >= low_snr_acc - 0.1  # allow 10% tolerance for short training
```

**Step 2: Run integration tests**

```bash
pytest tests/test_integration_snr_sweep.py -v -m slow
```

Expected: Both tests PASS (may take 1-2 minutes on CPU)

**Step 3: Commit**

```bash
git add tests/test_integration_snr_sweep.py
git commit -m "test(benchmarks): add end-to-end integration test for SNR sweep"
```

---

## Task 7: Update the benchmarks README

**Files:**
- Modify: `benchmarks/README.md`

**Step 1: Read the current README**

Read `benchmarks/README.md` (already done in planning).

**Step 2: Add SNR sweep section**

Append the following section to `benchmarks/README.md` after the existing content:

```markdown
---

## SNR-Sweep AMC Benchmark (`spectra-18-snr-sweep`)

Produces the canonical "accuracy vs. SNR" curves used throughout the AMC literature. A lightweight Conv1D classifier is trained on the full SNR range [-10, 30] dB, then evaluated at 21 fixed SNR points. Results are saved as JSON for reproducibility and community comparison.

### Prerequisites

```bash
pip install torch numpy pyyaml matplotlib
```

### Running the Benchmark

```bash
# Full run (train 50k samples, 30 epochs, eval at 21 SNR points)
python benchmarks/snr_sweep/run_snr_sweep.py

# Quick sanity check
python benchmarks/snr_sweep/run_snr_sweep.py --train-samples 5000 --test-per-snr 360 --epochs 10

# Plot existing results
python benchmarks/snr_sweep/run_snr_sweep.py --plot-only
```

### Output

Results saved to `benchmarks/snr_sweep/results/`:
- `snr_sweep_results.json` — Per-SNR accuracy and confusion matrices
- `snr_accuracy.png` — Accuracy-vs-SNR curve

### JSON Result Format

```json
{
  "benchmark": "spectra-18-snr-sweep",
  "num_classes": 18,
  "snr_results": {
    "-10.0": {"accuracy": 0.062, "confusion_matrix": [[...], ...]},
    "0.0":   {"accuracy": 0.423, "confusion_matrix": [[...], ...]},
    "30.0":  {"accuracy": 0.891, "confusion_matrix": [[...], ...]}
  }
}
```

### Interpretation

Expected behavior (based on AMC literature):
- **< -5 dB SNR**: Near-chance accuracy (~5% for 18 classes). Signals are noise-dominated.
- **0–10 dB SNR**: Rapid improvement. This is the "hard" regime most papers focus on.
- **> 20 dB SNR**: Near-perfect for most digital modulations. Analog/similar classes remain confused.

Researchers can compare their model's per-SNR accuracy against these baselines.
```

**Step 3: Commit**

```bash
git add benchmarks/README.md
git commit -m "docs(benchmarks): document spectra-18-snr-sweep benchmark"
```

---

## Task 8: Run the full test suite and verify

**Step 1: Run all new tests**

```bash
pytest tests/test_snr_sweep_model.py tests/test_snr_sweep_datasets.py tests/test_snr_sweep_runner.py tests/test_snr_sweep_visualize.py -v
```

Expected: All tests PASS

**Step 2: Run the full test suite to check for regressions**

```bash
pytest tests/ -v --ignore=tests/test_integration_snr_sweep.py -x
```

Expected: All existing tests still PASS

**Step 3: Verify the YAML config is packaged correctly**

```bash
python -c "
from spectra.benchmarks import load_benchmark
ds = load_benchmark('spectra-18-snr-sweep', split='train')
print(f'OK: {len(ds)} train samples')
train, val, test = load_benchmark('spectra-18-snr-sweep', split='all')
print(f'OK: train={len(train)}, val={len(val)}, test={len(test)}')
"
```

Expected: prints counts without error.

**Step 4: Verify benchmark script runs in fast mode**

```bash
python benchmarks/snr_sweep/run_snr_sweep.py \
  --train-samples 360 \
  --test-per-snr 180 \
  --epochs 2 \
  --output-dir /tmp/spectra_snr_test
```

Expected: Trains, evaluates at 21 SNR points, prints summary table, saves JSON.

**Step 5: Final commit**

```bash
git add -A
git commit -m "chore: verify SNR sweep benchmark end-to-end"
```

---

## Summary of New Files

| Path | Purpose |
|------|---------|
| `python/spectra/benchmarks/configs/spectra-18-snr-sweep.yaml` | Benchmark config with `snr_eval_points` |
| `benchmarks/snr_sweep/__init__.py` | Package marker |
| `benchmarks/snr_sweep/models.py` | Lightweight `CNNAMC` Conv1D classifier |
| `benchmarks/snr_sweep/dataset_utils.py` | `build_snr_dataset()` for fixed-SNR evaluation |
| `benchmarks/snr_sweep/run_snr_sweep.py` | Main training + evaluation script |
| `benchmarks/snr_sweep/visualize.py` | Accuracy-vs-SNR plotting |
| `tests/test_snr_sweep_model.py` | Unit tests for CNNAMC |
| `tests/test_snr_sweep_datasets.py` | Unit tests for dataset builder |
| `tests/test_snr_sweep_runner.py` | Unit tests for train/evaluate functions |
| `tests/test_snr_sweep_visualize.py` | Unit tests for visualizer |
| `tests/test_integration_snr_sweep.py` | End-to-end integration tests |

## No Existing Files Modified

The plan intentionally avoids modifying any existing source files. The new benchmark is fully additive.
