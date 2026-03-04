"""Dataset generation speed benchmark: SPECTRA vs TorchSig."""
import argparse
import json
import time
from pathlib import Path

import numpy as np
import torch
import yaml
from torch.utils.data import DataLoader

import spectra as sp
from benchmarks.torchsig_compat.label_map import (
    CANONICAL_CLASSES,
    spectra_waveform_pool,
)


def build_spectra_dataset(cfg, seed):
    """Build a SPECTRA NarrowbandDataset from config."""
    pool = spectra_waveform_pool()
    impairments = sp.Compose([sp.AWGN(snr_range=tuple(cfg["snr_range"]))])
    return sp.NarrowbandDataset(
        waveform_pool=pool,
        num_samples=cfg["speed"]["num_samples"],
        num_iq_samples=cfg["num_iq_samples"],
        sample_rate=cfg["sample_rate"],
        impairments=impairments,
        seed=seed,
    )


def build_torchsig_dataset(cfg, seed):
    """Build a TorchSig v2.x dataset wrapped in the adapter."""
    from benchmarks.torchsig_compat.adapter import TorchSigAdapter
    from benchmarks.torchsig_compat.label_map import torchsig_class_names

    try:
        from torchsig.datasets.datasets import TorchSigIterableDataset
        from torchsig.utils.defaults import TorchSigDefaults
    except ImportError:
        raise ImportError(
            "TorchSig not installed. Run: python benchmarks/torchsig_compat/install.py"
        )

    n_iq = cfg["num_iq_samples"]
    sr = cfg["sample_rate"]
    defaults = TorchSigDefaults()
    metadata = defaults.default_dataset_metadata
    metadata.update({
        "num_iq_samples_dataset": n_iq,
        "sample_rate": sr,
        "signal_duration_in_samples_min": int(n_iq * 0.8),
        "signal_duration_in_samples_max": n_iq,
        "bandwidth_min": int(sr * 0.25),
        "bandwidth_max": int(sr * 0.33),
        "signal_center_freq_min": int(-sr * 0.25),
        "signal_center_freq_max": int(sr * 0.25) - 1,
        "frequency_min": int(-sr * 0.25),
        "frequency_max": int(sr * 0.25) - 1,
    })

    iterable_ds = TorchSigIterableDataset(
        signal_generators=torchsig_class_names(),
        target_labels=["class_name"],
        metadata=metadata,
    )
    return TorchSigAdapter(
        iterable_ds,
        num_samples=cfg["speed"]["num_samples"],
        class_list=CANONICAL_CLASSES,
    )


def time_getitem(dataset, n, warmup=100):
    """Time single __getitem__ calls."""
    for i in range(warmup):
        _ = dataset[i % len(dataset)]

    times = []
    for i in range(n):
        t0 = time.perf_counter()
        _ = dataset[i % len(dataset)]
        times.append(time.perf_counter() - t0)
    return np.array(times)


def time_dataloader(dataset, cfg):
    """Time full DataLoader throughput."""
    loader = DataLoader(
        dataset,
        batch_size=cfg["speed"]["batch_size"],
        num_workers=cfg["speed"]["num_workers"],
        pin_memory=True,
    )
    t0 = time.perf_counter()
    count = 0
    for batch in loader:
        count += batch[0].shape[0]
    elapsed = time.perf_counter() - t0
    return count, elapsed


def time_init(factory_fn):
    """Time construction/initialization of a dataset.

    Returns (dataset, elapsed_seconds).
    """
    t0 = time.perf_counter()
    dataset = factory_fn()
    elapsed = time.perf_counter() - t0
    return dataset, elapsed


def build_result_dict(init_time, getitem_times, dl_count, dl_elapsed):
    """Build a standardized result dict from timing measurements."""
    n = len(getitem_times)
    total_getitem = float(np.sum(getitem_times))
    return {
        "init_time_s": init_time,
        "getitem_mean_ms": float(np.mean(getitem_times) * 1000),
        "getitem_std_ms": float(np.std(getitem_times) * 1000),
        "getitem_median_ms": float(np.median(getitem_times) * 1000),
        "total_getitem_time_s": total_getitem,
        "num_samples_measured": n,
        "amortized_cost_ms": (init_time + total_getitem) / n * 1000,
        "dataloader_samples": dl_count,
        "dataloader_elapsed_s": dl_elapsed,
        "dataloader_throughput_sps": dl_count / dl_elapsed,
    }


def main():
    parser = argparse.ArgumentParser(
        description="Speed benchmark: SPECTRA vs TorchSig"
    )
    parser.add_argument("--config", default="benchmarks/comparison/config.yaml")
    parser.add_argument("--output-dir", default="benchmarks/comparison/results")
    parser.add_argument("--skip-torchsig", action="store_true")
    args = parser.parse_args()

    with open(args.config) as f:
        cfg = yaml.safe_load(f)

    out_dir = Path(args.output_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    results = {}

    # SPECTRA
    print("=== SPECTRA Speed Benchmark ===")
    ds_sp, init_sp = time_init(
        lambda: build_spectra_dataset(cfg, seed=cfg["seeds"]["spectra_train"])
    )
    times_sp = time_getitem(
        ds_sp, n=cfg["speed"]["num_samples"], warmup=cfg["speed"]["num_warmup"]
    )
    count_sp, elapsed_sp = time_dataloader(ds_sp, cfg)
    results["spectra"] = build_result_dict(init_sp, times_sp, count_sp, elapsed_sp)
    print(f"  Init time:       {init_sp:.3f} s")
    print(f"  __getitem__:     {results['spectra']['getitem_mean_ms']:.3f} ms (mean)")
    print(f"  Amortized cost:  {results['spectra']['amortized_cost_ms']:.3f} ms/sample")
    print(f"  DataLoader:      {results['spectra']['dataloader_throughput_sps']:.0f} samples/s")

    # TorchSig
    if not args.skip_torchsig:
        print("\n=== TorchSig Speed Benchmark ===")
        ds_ts, init_ts = time_init(
            lambda: build_torchsig_dataset(cfg, seed=cfg["seeds"]["torchsig_train"])
        )
        times_ts = time_getitem(
            ds_ts,
            n=cfg["speed"]["num_samples"],
            warmup=cfg["speed"]["num_warmup"],
        )
        count_ts, elapsed_ts = time_dataloader(ds_ts, cfg)
        results["torchsig"] = build_result_dict(init_ts, times_ts, count_ts, elapsed_ts)
        print(f"  Init time:       {init_ts:.3f} s")
        print(f"  __getitem__:     {results['torchsig']['getitem_mean_ms']:.3f} ms (mean)")
        print(f"  Amortized cost:  {results['torchsig']['amortized_cost_ms']:.3f} ms/sample")
        print(f"  DataLoader:      {results['torchsig']['dataloader_throughput_sps']:.0f} samples/s")

    results["methodology"] = (
        "TorchSig materializes all data during __init__(); "
        "SPECTRA generates on-the-fly in __getitem__(). "
        "amortized_cost_ms = (init_time + total_getitem_time) / num_samples "
        "is the fair per-sample comparison."
    )

    # Save
    results_path = out_dir / "speed_results.json"
    with open(results_path, "w") as f:
        json.dump(results, f, indent=2)
    print(f"\nResults saved to {results_path}")


if __name__ == "__main__":
    main()
