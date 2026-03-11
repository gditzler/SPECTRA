"""Interactive Signal Builder CLI for SPECTRA.

Walks users through configuring and exporting a SPECTRA dataset
via interactive prompts. Output is a benchmark-compatible YAML config.
"""

import sys
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

from spectra.cli.config_builder import (
    IMPAIRMENT_PRESETS,
    WAVEFORM_CATEGORIES,
    build_config,
    default_num_samples,
    default_seeds,
    get_impairment_registry,
    get_waveform_registry,
    serialize_config,
    validate_config,
)


# ---------------------------------------------------------------------------
# Prompt helpers
# ---------------------------------------------------------------------------


def _print_header(title: str) -> None:
    print(f"\n{'=' * 60}")
    print(f"  {title}")
    print(f"{'=' * 60}\n")


def _choose_one(prompt: str, options: List[str]) -> int:
    """Numbered menu, returns 0-based index."""
    for i, opt in enumerate(options, 1):
        print(f"  {i}. {opt}")
    while True:
        raw = input(f"\n{prompt} [1-{len(options)}]: ").strip()
        try:
            idx = int(raw) - 1
            if 0 <= idx < len(options):
                return idx
        except ValueError:
            pass
        print(f"  Please enter a number between 1 and {len(options)}.")


def _choose_many(prompt: str, options: List[str]) -> List[int]:
    """Comma-separated indices or 'a' for all. Returns list of 0-based indices."""
    for i, opt in enumerate(options, 1):
        print(f"  {i}. {opt}")
    while True:
        raw = input(f"\n{prompt} (comma-separated, or 'a' for all): ").strip()
        if raw.lower() == "a":
            return list(range(len(options)))
        try:
            indices = [int(x.strip()) - 1 for x in raw.split(",")]
            if all(0 <= i < len(options) for i in indices) and indices:
                return indices
        except ValueError:
            pass
        print(f"  Enter numbers 1-{len(options)} separated by commas, or 'a' for all.")


def _input_int(prompt: str, default: int) -> int:
    raw = input(f"{prompt} [{default}]: ").strip()
    if not raw:
        return default
    try:
        return int(raw)
    except ValueError:
        print(f"  Invalid input, using default: {default}")
        return default


def _input_float(prompt: str, default: float) -> float:
    raw = input(f"{prompt} [{default}]: ").strip()
    if not raw:
        return default
    try:
        return float(raw)
    except ValueError:
        print(f"  Invalid input, using default: {default}")
        return default


def _input_range(prompt: str, default: Tuple[float, float]) -> Tuple[float, float]:
    raw = input(f"{prompt} [{default[0]}, {default[1]}]: ").strip()
    if not raw:
        return default
    try:
        parts = [float(x.strip()) for x in raw.split(",")]
        if len(parts) == 2:
            return (parts[0], parts[1])
    except ValueError:
        pass
    print(f"  Invalid input, using default: {default}")
    return default


def _input_str(prompt: str, default: str) -> str:
    raw = input(f"{prompt} [{default}]: ").strip()
    return raw if raw else default


def _confirm(prompt: str, default: bool = True) -> bool:
    suffix = "Y/n" if default else "y/N"
    raw = input(f"{prompt} [{suffix}]: ").strip().lower()
    if not raw:
        return default
    return raw in ("y", "yes")


def _parse_params(raw: str) -> Dict[str, Any]:
    """Parse 'key=value, key=value' into a typed dict."""
    params: Dict[str, Any] = {}
    if not raw.strip():
        return params
    for pair in raw.split(","):
        pair = pair.strip()
        if "=" not in pair:
            continue
        key, val = pair.split("=", 1)
        key = key.strip()
        val = val.strip()
        # Try int, float, bool, then string
        try:
            params[key] = int(val)
            continue
        except ValueError:
            pass
        try:
            params[key] = float(val)
            continue
        except ValueError:
            pass
        if val.lower() in ("true", "false"):
            params[key] = val.lower() == "true"
            continue
        params[key] = val
    return params


# ---------------------------------------------------------------------------
# Main interactive flow
# ---------------------------------------------------------------------------


def run() -> None:
    """Run the interactive Signal Builder."""
    _print_header("SPECTRA Signal Builder")
    print("  Build a dataset config interactively.\n")

    # 1. Config name
    name = _input_str("Config name", "my-dataset")

    # 2. Task type
    _print_header("Task Type")
    task_idx = _choose_one(
        "Select task type",
        ["narrowband (single signal classification)", "wideband (multi-signal detection)"],
    )
    task = "narrowband" if task_idx == 0 else "wideband"

    # 3. Waveform selection
    _print_header("Waveform Selection")
    categories = list(WAVEFORM_CATEGORIES.keys())
    cat_indices = _choose_many("Select waveform categories", categories)

    waveform_pool: List[Dict[str, Any]] = []
    for ci in cat_indices:
        cat_name = categories[ci]
        waveforms_in_cat = WAVEFORM_CATEGORIES[cat_name]
        print(f"\n  --- {cat_name} ---")
        wf_indices = _choose_many(f"Select {cat_name} waveforms", waveforms_in_cat)
        for wi in wf_indices:
            wf_name = waveforms_in_cat[wi]
            entry: Dict[str, Any] = {"type": wf_name}
            raw_params = input(f"  Params for {wf_name} (key=val,...) or Enter to skip: ").strip()
            if raw_params:
                params = _parse_params(raw_params)
                if params:
                    entry["params"] = params
            waveform_pool.append(entry)

    if not waveform_pool:
        print("\n  No waveforms selected. Adding BPSK as default.")
        waveform_pool = [{"type": "BPSK"}]

    print(f"\n  Selected {len(waveform_pool)} waveform(s).")

    # 4. Impairments
    _print_header("Impairments")
    preset_names = list(IMPAIRMENT_PRESETS.keys()) + ["custom"]
    imp_idx = _choose_one("Select impairment preset", preset_names)

    if imp_idx < len(IMPAIRMENT_PRESETS):
        preset_name = list(IMPAIRMENT_PRESETS.keys())[imp_idx]
        impairments = list(IMPAIRMENT_PRESETS[preset_name])
        print(f"\n  Using '{preset_name}' preset:")
        for entry in impairments:
            print(f"    - {entry['type']}")
    else:
        # Custom impairment selection
        imp_reg = get_impairment_registry()
        imp_names = sorted(imp_reg.keys())
        imp_indices = _choose_many("Select impairments", imp_names)
        impairments = []
        has_awgn = False
        for ii in imp_indices:
            imp_name = imp_names[ii]
            entry: Dict[str, Any] = {"type": imp_name}
            raw_params = input(
                f"  Params for {imp_name} (key=val,...) or Enter to skip: "
            ).strip()
            if raw_params:
                params = _parse_params(raw_params)
                if params:
                    entry["params"] = params
            impairments.append(entry)
            if imp_name == "AWGN":
                has_awgn = True
        if not has_awgn:
            impairments.append({"type": "AWGN"})
            print("  (AWGN appended automatically)")

    # 5. Dataset parameters
    _print_header("Dataset Parameters")
    sample_rate = _input_float("Sample rate (Hz)", 1_000_000)
    num_iq_samples = _input_int("IQ samples per example", 1024)
    snr_range = _input_range("SNR range (dB)", (-10.0, 30.0))

    print("\n  Samples per split:")
    ns = default_num_samples()
    ns["train"] = _input_int("  Train samples", ns["train"])
    ns["val"] = _input_int("  Validation samples", ns["val"])
    ns["test"] = _input_int("  Test samples", ns["test"])

    print("\n  Seeds per split:")
    seeds = default_seeds()
    seeds["train"] = _input_int("  Train seed", seeds["train"])
    seeds["val"] = _input_int("  Validation seed", seeds["val"])
    seeds["test"] = _input_int("  Test seed", seeds["test"])

    # 6. Wideband scene config
    scene: Optional[Dict[str, Any]] = None
    if task == "wideband":
        _print_header("Wideband Scene Configuration")
        capture_bw = _input_float("Capture bandwidth (Hz)", sample_rate / 2)
        capture_dur = _input_float("Capture duration (s)", num_iq_samples / sample_rate)
        sig_min = _input_int("Min signals per scene", 1)
        sig_max = _input_int("Max signals per scene", 5)
        scene = {
            "capture_bandwidth": capture_bw,
            "capture_duration": capture_dur,
            "num_signals": [sig_min, sig_max],
            "allow_overlap": True,
        }

    # 7. Build config
    cfg = build_config(
        name=name,
        task=task,
        waveform_pool=waveform_pool,
        impairments=impairments,
        snr_range=snr_range,
        sample_rate=sample_rate,
        num_iq_samples=num_iq_samples,
        num_samples=ns,
        seed=seeds,
        scene=scene,
    )

    # 8. Validate
    errors = validate_config(cfg)
    if errors:
        print("\n  Validation errors:")
        for e in errors:
            print(f"    - {e}")
        print("\n  Please fix the issues and try again.")
        return

    # 9. Review
    _print_header("Generated Config")
    yaml_str = serialize_config(cfg)
    print(yaml_str)

    if not _confirm("Proceed with this config?"):
        print("  Aborted.")
        return

    # 10. Output
    _print_header("Output")
    output_options = ["Save YAML config only", "Save YAML and generate dataset (Zarr)", "Print only (no save)"]
    out_idx = _choose_one("Select output", output_options)

    if out_idx == 2:
        print("\n  Config printed above. Done!")
        return

    yaml_path = _input_str("Output YAML path", f"{name}.yaml")
    Path(yaml_path).write_text(yaml_str)
    print(f"\n  Saved config to: {yaml_path}")

    if out_idx == 1:
        # Generate dataset
        zarr_path = _input_str("Output Zarr path", f"{name}.zarr")
        print(f"\n  Generating dataset to: {zarr_path}")

        from spectra.benchmarks.loader import load_benchmark
        from spectra.utils.writer import DatasetWriter

        for split in ("train", "val", "test"):
            ds = load_benchmark(yaml_path, split)
            writer = DatasetWriter(ds, f"{zarr_path}/{split}")
            writer.write(progress=True)
            writer.finalize()
            print(f"    {split}: {len(ds)} samples written")

    # 11. Done
    _print_header("Done!")
    print(f"  Load your dataset with:")
    print(f'    from spectra import load_benchmark')
    print(f'    ds = load_benchmark("{yaml_path}", "train")')
    print()


def main() -> None:
    """Entry point with KeyboardInterrupt handling."""
    try:
        run()
    except KeyboardInterrupt:
        print("\n\n  Interrupted. Goodbye!")
        sys.exit(0)
