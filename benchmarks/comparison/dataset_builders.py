"""Shared dataset builder functions for benchmarks."""
import spectra as sp

from benchmarks.torchsig_compat.label_map import (
    CANONICAL_CLASSES,
    spectra_waveform_pool,
)


def build_spectra_dataset(cfg, seed, num_samples):
    """Build a SPECTRA NarrowbandDataset from config.

    Args:
        cfg: Benchmark config dict.
        seed: Random seed.
        num_samples: Number of dataset samples.
    """
    pool = spectra_waveform_pool()
    impairments = sp.Compose([sp.AWGN(snr_range=tuple(cfg["snr_range"]))])
    return sp.NarrowbandDataset(
        waveform_pool=pool,
        num_samples=num_samples,
        num_iq_samples=cfg["num_iq_samples"],
        sample_rate=cfg["sample_rate"],
        impairments=impairments,
        seed=seed,
    )


def build_torchsig_dataset(cfg, seed, num_samples):
    """Build a TorchSig v2.x dataset wrapped in the adapter.

    Args:
        cfg: Benchmark config dict.
        seed: Random seed.
        num_samples: Number of dataset samples.
    """
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
        num_samples=num_samples,
        class_list=CANONICAL_CLASSES,
    )
