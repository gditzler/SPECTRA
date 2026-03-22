import importlib.resources
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple, Union

import numpy as np
import yaml

from spectra.arrays.array import ula
from spectra.datasets import DirectionFindingDataset, NarrowbandDataset, WidebandDataset
from spectra.datasets.snr_sweep import SNRSweepDataset
from spectra.impairments import AWGN, Compose
from spectra.scene.composer import SceneConfig
from spectra.waveforms.base import Waveform

# Registry: name -> class. Lazy-populated on first use.
_WAVEFORM_REGISTRY: Optional[Dict[str, type]] = None
_IMPAIRMENT_REGISTRY: Optional[Dict[str, type]] = None


def _get_waveform_registry() -> Dict[str, type]:
    global _WAVEFORM_REGISTRY
    if _WAVEFORM_REGISTRY is None:
        from spectra import waveforms as wmod

        _WAVEFORM_REGISTRY = {}
        # Import all waveform classes from __all__
        for name in wmod.__all__:
            cls = getattr(wmod, name)
            _WAVEFORM_REGISTRY[name] = cls
    return _WAVEFORM_REGISTRY


def _get_impairment_registry() -> Dict[str, type]:
    global _IMPAIRMENT_REGISTRY
    if _IMPAIRMENT_REGISTRY is None:
        from spectra import impairments as imod

        _IMPAIRMENT_REGISTRY = {}
        for name in imod.__all__:
            cls = getattr(imod, name)
            if name != "Compose":
                _IMPAIRMENT_REGISTRY[name] = cls
    return _IMPAIRMENT_REGISTRY


def _build_waveform_pool(pool_config: List[Dict[str, Any]]) -> List[Waveform]:
    registry = _get_waveform_registry()
    pool = []
    for entry in pool_config:
        wtype = entry["type"]
        if wtype not in registry:
            raise ValueError(
                f"Unknown waveform type '{wtype}'. Available: {sorted(registry.keys())}"
            )
        params = entry.get("params", {})
        pool.append(registry[wtype](**params))
    return pool


def _build_impairments(imp_config: List[Dict[str, Any]]) -> Optional[Compose]:
    if not imp_config:
        return None
    registry = _get_impairment_registry()
    transforms = []
    for entry in imp_config:
        itype = entry["type"]
        if itype not in registry:
            raise ValueError(
                f"Unknown impairment type '{itype}'. Available: {sorted(registry.keys())}"
            )
        params = entry.get("params", {})
        # Skip bare AWGN — _build_narrowband adds it with snr_range from config
        if itype == "AWGN" and not params:
            continue
        transforms.append(registry[itype](**params))
    if not transforms:
        return None
    return Compose(transforms)


def _resolve_config_path(name: str) -> Path:
    """Resolve a benchmark name or file path to a Path object."""
    # If it looks like a file path, use it directly
    if name.endswith((".yaml", ".yml")) or "/" in name or "\\" in name:
        p = Path(name)
        if not p.exists():
            raise FileNotFoundError(f"Benchmark config not found: {name}")
        return p

    # Otherwise, look up built-in configs
    configs_pkg = "spectra.benchmarks.configs"
    try:
        ref = importlib.resources.files(configs_pkg) / f"{name}.yaml"
        with importlib.resources.as_file(ref) as path:
            if path.exists():
                return path
    except (ModuleNotFoundError, FileNotFoundError):
        pass

    raise FileNotFoundError(
        f"Benchmark '{name}' not found. Provide a file path or use a built-in name."
    )


def _build_narrowband(config: Dict[str, Any], split: str) -> NarrowbandDataset:
    pool = _build_waveform_pool(config["waveform_pool"])
    impairments = _build_impairments(config.get("impairments", []))

    # AWGN with snr_range if not already in impairments list
    snr_range = tuple(config["snr_range"])
    has_awgn = impairments is not None and any(
        isinstance(t, AWGN) for t in (impairments.transforms if impairments else [])
    )
    if has_awgn:
        # Replace AWGN with one using the config's snr_range
        new_transforms = []
        for t in impairments.transforms:
            if isinstance(t, AWGN):
                new_transforms.append(AWGN(snr_range=snr_range))
            else:
                new_transforms.append(t)
        impairments = Compose(new_transforms)
    else:
        # Add AWGN with the config's snr_range
        existing = impairments.transforms if impairments else []
        impairments = Compose([*existing, AWGN(snr_range=snr_range)])

    return NarrowbandDataset(
        waveform_pool=pool,
        num_samples=config["num_samples"][split],
        num_iq_samples=config["num_iq_samples"],
        sample_rate=config["sample_rate"],
        impairments=impairments,
        seed=config["seed"][split],
    )


def _build_wideband(config: Dict[str, Any], split: str) -> WidebandDataset:
    pool = _build_waveform_pool(config["waveform_pool"])
    impairments = _build_impairments(config.get("impairments", []))

    scene_cfg = config.get("scene", {})
    num_signals = scene_cfg.get("num_signals", (1, 3))
    if isinstance(num_signals, list):
        num_signals = tuple(num_signals)

    sc = SceneConfig(
        capture_duration=scene_cfg.get(
            "capture_duration",
            config["num_iq_samples"] / config["sample_rate"],
        ),
        capture_bandwidth=scene_cfg.get("capture_bandwidth", config["sample_rate"] / 2),
        sample_rate=config["sample_rate"],
        num_signals=num_signals,
        signal_pool=pool,
        snr_range=tuple(config["snr_range"]),
        allow_overlap=scene_cfg.get("allow_overlap", True),
    )

    return WidebandDataset(
        scene_config=sc,
        num_samples=config["num_samples"][split],
        impairments=impairments,
        seed=config["seed"][split],
    )


def _build_direction_finding(config: Dict[str, Any], split: str) -> DirectionFindingDataset:
    """Build a :class:`~spectra.datasets.DirectionFindingDataset` from YAML."""
    pool = _build_waveform_pool(config["signal_pool"])
    impairments = _build_impairments(config.get("impairments", []))

    arr_cfg = config["array"]
    if arr_cfg.get("type", "ula") != "ula":
        raise ValueError(
            f"Only array type 'ula' is supported in benchmark configs, got {arr_cfg.get('type')!r}"
        )
    array = ula(
        num_elements=arr_cfg["num_elements"],
        spacing=arr_cfg.get("spacing", 0.5),
        frequency=arr_cfg.get("frequency", 1e9),
    )

    ns = config["num_signals"]
    num_signals: Union[int, Tuple[int, int]] = tuple(ns) if isinstance(ns, list) else int(ns)

    az_deg = config["azimuth_range_deg"]
    el_deg = config["elevation_range_deg"]
    azimuth_range = (np.deg2rad(az_deg[0]), np.deg2rad(az_deg[1]))
    elevation_range = (np.deg2rad(el_deg[0]), np.deg2rad(el_deg[1]))

    min_sep = config.get("min_angular_separation_deg")
    min_angular_separation = None if min_sep is None else float(np.deg2rad(min_sep))

    return DirectionFindingDataset(
        array=array,
        signal_pool=pool,
        num_signals=num_signals,
        num_snapshots=config["num_snapshots"],
        sample_rate=config["sample_rate"],
        snr_range=tuple(config["snr_range"]),
        azimuth_range=azimuth_range,
        elevation_range=elevation_range,
        min_angular_separation=min_angular_separation,
        impairments=impairments,
        num_samples=config["num_samples"][split],
        seed=config["seed"][split],
    )


def _build_channel(config: dict, condition: str) -> NarrowbandDataset:
    conditions = config.get("conditions", {})
    if condition not in conditions:
        raise ValueError(
            f"condition='{condition}' not found. Available: {sorted(conditions.keys())}"
        )
    pool = _build_waveform_pool(config["waveform_pool"])
    # _build_impairments handles AWGN with explicit params (snr=10) correctly
    impairments = _build_impairments(conditions[condition]["impairments"])
    return NarrowbandDataset(
        waveform_pool=pool,
        num_samples=config["num_samples_per_condition"],
        num_iq_samples=config["num_iq_samples"],
        sample_rate=config["sample_rate"],
        impairments=impairments,
        seed=config["seed"][condition],
    )


def load_channel_benchmark(name: str, condition: str) -> NarrowbandDataset:
    """Load one impairment condition from a channel-robustness benchmark."""
    path = _resolve_config_path(name)
    with open(path, "r") as f:
        config = yaml.safe_load(f)

    task = config.get("task", "")
    if task != "narrowband_channel":
        raise ValueError(f"Config '{name}' has task='{task}', expected 'narrowband_channel'.")
    return _build_channel(config, condition)


def _build_snr_sweep(config: dict, split: str) -> SNRSweepDataset:
    pool = _build_waveform_pool(config["waveform_pool"])

    slc = config["snr_levels"]
    snr_levels = [
        float(v) for v in range(int(slc["start"]), int(slc["stop"]) + 1, int(slc["step"]))
    ]

    # Extra impairments (FrequencyOffset, PhaseOffset, etc.) — no AWGN here
    extra = _build_impairments(
        [e for e in config.get("impairments", []) if e.get("type") != "AWGN"]
    )

    def impairments_fn(snr_db: float) -> Compose:
        awgn = AWGN(snr=snr_db)
        base = list(extra.transforms) if extra is not None else []
        return Compose([*base, awgn])

    return SNRSweepDataset(
        waveform_pool=pool,
        snr_levels=snr_levels,
        samples_per_cell=config["samples_per_cell"][split],
        num_iq_samples=config["num_iq_samples"],
        sample_rate=config["sample_rate"],
        impairments_fn=impairments_fn,
        seed=config["seed"][split],
    )


def load_snr_sweep(name: str, split: str = "test") -> SNRSweepDataset:
    """Load an SNR-sweep benchmark. split ∈ {"train", "val", "test"} (no "all")."""
    if split not in {"train", "val", "test"}:
        raise ValueError(f"split must be 'train', 'val', or 'test', got '{split}'")

    path = _resolve_config_path(name)
    with open(path, "r") as f:
        config = yaml.safe_load(f)

    task = config.get("task", "")
    if task != "narrowband_snr_sweep":
        raise ValueError(f"Config '{name}' has task='{task}', expected 'narrowband_snr_sweep'.")
    return _build_snr_sweep(config, split)


def load_benchmark(
    name: str, split: str = "train"
) -> Union[
    NarrowbandDataset,
    WidebandDataset,
    DirectionFindingDataset,
    Tuple[NarrowbandDataset, NarrowbandDataset, NarrowbandDataset],
    Tuple[WidebandDataset, WidebandDataset, WidebandDataset],
    Tuple[
        DirectionFindingDataset,
        DirectionFindingDataset,
        DirectionFindingDataset,
    ],
]:
    """Load a benchmark dataset from a YAML config.

    Parameters
    ----------
    name : str
        Built-in benchmark name (e.g., ``"spectra-18"``) or path to a
        ``.yaml`` file.
    split : str
        ``"train"``, ``"val"``, ``"test"``, or ``"all"`` (returns a 3-tuple).

    Returns
    -------
    Dataset or tuple of Datasets
        Configured dataset(s) ready for use with PyTorch DataLoader.
    """
    valid_splits = {"train", "val", "test", "all"}
    if split not in valid_splits:
        raise ValueError(f"split must be one of {valid_splits}, got '{split}'")

    path = _resolve_config_path(name)
    with open(path, "r") as f:
        config = yaml.safe_load(f)

    task = config.get("task", "narrowband")
    if task == "narrowband":
        builder = _build_narrowband
    elif task == "wideband":
        builder = _build_wideband
    elif task == "direction_finding":
        builder = _build_direction_finding
    else:
        raise ValueError(
            f"Unsupported task '{task}' for load_benchmark(). "
            "Use load_channel_benchmark() for narrowband_channel, "
            "or load_snr_sweep() for narrowband_snr_sweep."
        )

    if split == "all":
        return builder(config, "train"), builder(config, "val"), builder(config, "test")
    return builder(config, split)
