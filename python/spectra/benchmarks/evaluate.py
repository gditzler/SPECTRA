"""Evaluation utilities for SPECTRA benchmarks."""
from typing import Any, Callable, Dict

import torch
from torch.utils.data import DataLoader


def evaluate_snr_sweep(
    predict_fn: Callable[[torch.Tensor], torch.Tensor],
    dataset: Any,
    batch_size: int = 64,
) -> Dict[float, Dict[str, Any]]:
    """Evaluate a classifier over an SNRSweepDataset.

    Parameters
    ----------
    predict_fn : callable
        Accepts Tensor[B, 2, N], returns Tensor[B] of predicted class indices.
    dataset : SNRSweepDataset
    batch_size : int

    Returns
    -------
    dict  {snr_db: {"accuracy": float, "per_class": {class_idx: float}}}
    """
    from spectra.datasets.snr_sweep import SNRSweepDataset
    if not isinstance(dataset, SNRSweepDataset):
        raise TypeError(f"Expected SNRSweepDataset, got {type(dataset).__name__}")

    stats: Dict[float, Dict] = {}
    for batch in DataLoader(dataset, batch_size=batch_size, shuffle=False):
        data, labels, snrs = batch
        preds = predict_fn(data).cpu()
        labels, snrs = labels.cpu(), snrs.cpu()
        for pred, label, snr_val in zip(preds, labels, snrs):
            k = float(snr_val.item())
            c = int(label.item())
            if k not in stats:
                stats[k] = {"correct": {}, "total": {}}
            stats[k]["total"][c] = stats[k]["total"].get(c, 0) + 1
            if int(pred.item()) == c:
                stats[k]["correct"][c] = stats[k]["correct"].get(c, 0) + 1

    return {
        k: {
            "accuracy": sum(s["correct"].values()) / sum(s["total"].values()),
            "per_class": {c: s["correct"].get(c, 0) / s["total"][c] for c in s["total"]},
        }
        for k, s in stats.items()
    }


def evaluate_channel_conditions(
    predict_fn: Callable[[torch.Tensor], torch.Tensor],
    name: str,
    batch_size: int = 64,
) -> Dict[str, Dict[str, Any]]:
    """Evaluate a classifier over all conditions of a channel benchmark.

    Parameters
    ----------
    predict_fn : callable
        Accepts Tensor[B, 2, N], returns Tensor[B] of predicted class indices.
    name : str
        Built-in name or YAML path of a ``narrowband_channel`` config.
    batch_size : int

    Returns
    -------
    dict  {condition: {"accuracy": float, "per_class": {class_idx: float}}}
    """
    import yaml
    from spectra.benchmarks.loader import _resolve_config_path, load_channel_benchmark

    path = _resolve_config_path(name)
    with open(path) as f:
        config = yaml.safe_load(f)
    conditions = list(config.get("conditions", {}).keys())

    results: Dict[str, Dict] = {}
    for condition in conditions:
        dataset = load_channel_benchmark(name, condition=condition)
        correct_by_cls: Dict[int, int] = {}
        total_by_cls: Dict[int, int] = {}
        for batch in DataLoader(dataset, batch_size=batch_size, shuffle=False):
            data, labels = batch
            preds = predict_fn(data).cpu()
            for pred, label in zip(preds, labels.cpu()):
                c = int(label.item())
                total_by_cls[c] = total_by_cls.get(c, 0) + 1
                if int(pred.item()) == c:
                    correct_by_cls[c] = correct_by_cls.get(c, 0) + 1
        total = sum(total_by_cls.values())
        results[condition] = {
            "accuracy": sum(correct_by_cls.values()) / total if total else 0.0,
            "per_class": {c: correct_by_cls.get(c, 0) / total_by_cls[c] for c in total_by_cls},
        }
    return results
