# SPECTRA Studio Implementation Plan

> **For agentic workers:** REQUIRED: Use superpowers:subagent-driven-development (if subagents available) or superpowers:executing-plans to implement this plan. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Add a Gradio-based UI ("SPECTRA Studio") for interactive RF waveform generation, visualization, and SigMF export, plus `spectra generate`, `spectra viz`, and `spectra studio` CLI subcommands.

**Architecture:** The UI lives in `spectra/studio/` as an optional dependency (`spectra[ui]`). It reuses existing waveform/impairment registries from `config_builder.py`, the benchmark YAML schema, and `SigMFWriter` for export. A new `spectra` CLI entry point dispatches to subcommands. Testable logic (plotting, parameter registry, CLI) is separated from Gradio UI code.

**Tech Stack:** Python 3.10+, Gradio 4+, matplotlib, NumPy. Optional: `spectra[ui]` adds gradio.

**Spec:** `docs/superpowers/specs/2026-03-18-spectra-studio-design.md`

---

## File Structure

| File | Responsibility |
|------|---------------|
| `python/spectra/studio/__init__.py` | `launch()` entry point |
| `python/spectra/studio/app.py` | Gradio Blocks app (3-tab assembly) |
| `python/spectra/studio/generate_tab.py` | Generate tab UI + callbacks |
| `python/spectra/studio/visualize_tab.py` | Visualize tab UI + callbacks |
| `python/spectra/studio/export_tab.py` | Export tab UI + callbacks |
| `python/spectra/studio/plotting.py` | 7 plot functions (testable, no Gradio) |
| `python/spectra/studio/params.py` | Waveform parameter registry (testable) |
| `python/spectra/studio/theme.py` | Custom Gradio theme |
| `python/spectra/cli/main.py` | Unified CLI: studio, generate, viz, build |
| `python/spectra/cli/__main__.py` | Modify: point to new dispatcher |
| `pyproject.toml` | Modify: add `spectra[ui]`, `spectra` entry point |
| `tests/test_studio_plotting.py` | Tests for plot functions |
| `tests/test_studio_params.py` | Tests for parameter registry |
| `tests/test_cli_main.py` | Tests for CLI subcommands |

---

## Task 1: Scaffolding, Dependencies, Theme

**Files:**
- Create: `python/spectra/studio/__init__.py`
- Create: `python/spectra/studio/theme.py`
- Modify: `pyproject.toml`

- [ ] **Step 1: Add `ui` optional dependency to `pyproject.toml`**

Add to `[project.optional-dependencies]`:
```toml
ui = ["gradio>=4.0", "scipy>=1.10"]
```

Add `"gradio>=4.0"` to the `all` list.

Add to `[project.scripts]`:
```toml
spectra = "spectra.cli.main:main"
```

(Keep existing `spectra-build` entry unchanged.)

- [ ] **Step 2: Create `studio/__init__.py`**

```python
# python/spectra/studio/__init__.py
"""SPECTRA Studio — interactive RF dataset generation and visualization."""


def launch(port: int = 7860, share: bool = False, dark: bool = False) -> None:
    """Launch the SPECTRA Studio Gradio app.

    Args:
        port: Local port for the Gradio server.
        share: Create a public Gradio share link.
        dark: Start in dark mode.
    """
    try:
        import gradio  # noqa: F401
    except ImportError:
        raise ImportError(
            "SPECTRA Studio requires gradio. Install with: pip install spectra[ui]"
        )

    from spectra.studio.app import create_app

    app = create_app(dark=dark)
    app.launch(server_port=port, share=share)
```

- [ ] **Step 3: Create `studio/theme.py`**

```python
# python/spectra/studio/theme.py
"""Custom Gradio theme for SPECTRA Studio."""

from __future__ import annotations

import gradio as gr


def spectra_theme(dark: bool = False) -> gr.themes.Base:
    """Create the SPECTRA Studio theme.

    Args:
        dark: Whether to default to dark mode.
    """
    theme = gr.themes.Soft(
        primary_hue=gr.themes.Color(
            c50="#e8f0fe", c100="#c5d9f7", c200="#9ebfef",
            c300="#7aa5e7", c400="#5a8edf", c500="#4a90d9",
            c600="#3b73ae", c700="#2d5783", c800="#1e3a58",
            c900="#101e2d", c950="#0a1119",
        ),
        secondary_hue=gr.themes.colors.green,
        neutral_hue=gr.themes.colors.slate,
    ).set(
        body_background_fill_dark="#0d1117",
        block_background_fill_dark="#1a1a2e",
        button_primary_background_fill="#4a90d9",
        button_primary_background_fill_hover="#5a9ee3",
        button_secondary_background_fill="#50b87a",
        button_secondary_background_fill_hover="#5fc888",
    )
    return theme
```

- [ ] **Step 4: Commit**

```bash
git add python/spectra/studio/__init__.py python/spectra/studio/theme.py pyproject.toml
git commit -m "feat(studio): scaffold SPECTRA Studio package with theme and dependencies"
```

---

## Task 2: Plotting Module (Testable)

**Files:**
- Create: `python/spectra/studio/plotting.py`
- Create: `tests/test_studio_plotting.py`

- [ ] **Step 1: Write the failing tests**

```python
# tests/test_studio_plotting.py
"""Tests for SPECTRA Studio plotting functions."""
import numpy as np
import pytest
import matplotlib
matplotlib.use("Agg")  # non-interactive backend for CI


def _make_iq(n=1024, seed=42):
    rng = np.random.default_rng(seed)
    return (rng.standard_normal(n) + 1j * rng.standard_normal(n)).astype(np.complex64)


def test_plot_iq_returns_figure():
    from spectra.studio.plotting import plot_iq
    fig = plot_iq(_make_iq(), sample_rate=1e6)
    assert fig is not None
    assert hasattr(fig, "savefig")  # matplotlib Figure


def test_plot_fft_returns_figure():
    from spectra.studio.plotting import plot_fft
    fig = plot_fft(_make_iq(), sample_rate=1e6)
    assert fig is not None


def test_plot_waterfall_returns_figure():
    from spectra.studio.plotting import plot_waterfall
    fig = plot_waterfall(_make_iq(4096), sample_rate=1e6)
    assert fig is not None


def test_plot_constellation_returns_figure():
    from spectra.studio.plotting import plot_constellation
    fig = plot_constellation(_make_iq())
    assert fig is not None


def test_plot_scd_returns_figure():
    from spectra.studio.plotting import plot_scd
    fig = plot_scd(_make_iq(2048), sample_rate=1e6)
    assert fig is not None


def test_plot_ambiguity_returns_figure():
    from spectra.studio.plotting import plot_ambiguity
    fig = plot_ambiguity(_make_iq(512))
    assert fig is not None


def test_plot_eye_returns_figure():
    from spectra.studio.plotting import plot_eye
    fig = plot_eye(_make_iq(1024), samples_per_symbol=8)
    assert fig is not None


def test_plot_eye_incompatible_returns_none():
    from spectra.studio.plotting import plot_eye
    # samples_per_symbol=0 or None should return None (incompatible)
    result = plot_eye(_make_iq(), samples_per_symbol=0)
    assert result is None
```

- [ ] **Step 2: Run tests to verify they fail**

```bash
/Users/gditzler/.venvs/base/bin/pytest tests/test_studio_plotting.py -v
```
Expected: `ModuleNotFoundError`

- [ ] **Step 3: Write the implementation**

```python
# python/spectra/studio/plotting.py
"""Plot functions for SPECTRA Studio.

All functions accept IQ data (complex64 ndarray) and return a matplotlib
Figure. They are Gradio-agnostic and independently testable.
"""

from __future__ import annotations

from typing import Optional

import matplotlib
import matplotlib.pyplot as plt
import numpy as np


def _dark_style() -> dict:
    """Base style dict for dark-themed plots."""
    return {
        "figure.facecolor": "#0d1117",
        "axes.facecolor": "#161b22",
        "axes.edgecolor": "#30363d",
        "axes.labelcolor": "#c9d1d9",
        "text.color": "#c9d1d9",
        "xtick.color": "#8b949e",
        "ytick.color": "#8b949e",
        "grid.color": "#21262d",
        "grid.alpha": 0.5,
    }


def plot_iq(
    iq: np.ndarray,
    sample_rate: float = 1e6,
    start: int = 0,
    num_samples: int = 500,
    dark: bool = True,
) -> plt.Figure:
    """Plot IQ time-domain (I and Q channels vs sample index)."""
    with plt.rc_context(_dark_style() if dark else {}):
        fig, axes = plt.subplots(2, 1, figsize=(8, 4), sharex=True)
        seg = iq[start : start + num_samples]
        t = np.arange(len(seg)) / sample_rate * 1e6  # microseconds
        axes[0].plot(t, seg.real, color="#4a90d9", linewidth=0.8)
        axes[0].set_ylabel("I")
        axes[0].grid(True, alpha=0.3)
        axes[1].plot(t, seg.imag, color="#50b87a", linewidth=0.8)
        axes[1].set_ylabel("Q")
        axes[1].set_xlabel("Time (us)")
        axes[1].grid(True, alpha=0.3)
        fig.suptitle("IQ Time Domain", fontsize=11)
        fig.tight_layout()
    return fig


def plot_fft(
    iq: np.ndarray,
    sample_rate: float = 1e6,
    nfft: int = 1024,
    dark: bool = True,
) -> plt.Figure:
    """Plot FFT / power spectral density (Welch-style)."""
    with plt.rc_context(_dark_style() if dark else {}):
        fig, ax = plt.subplots(figsize=(8, 4))
        from scipy.signal import welch

        freqs, psd = welch(iq, fs=sample_rate, nperseg=min(nfft, len(iq)),
                           return_onesided=False)
        freqs = np.fft.fftshift(freqs)
        psd = np.fft.fftshift(psd)
        ax.plot(freqs / 1e3, 10 * np.log10(psd + 1e-30), color="#4a90d9", linewidth=0.8)
        ax.set_xlabel("Frequency (kHz)")
        ax.set_ylabel("PSD (dB/Hz)")
        ax.set_title("Power Spectral Density")
        ax.grid(True, alpha=0.3)
        fig.tight_layout()
    return fig


def plot_waterfall(
    iq: np.ndarray,
    sample_rate: float = 1e6,
    nfft: int = 256,
    hop: int = 64,
    dark: bool = True,
) -> plt.Figure:
    """Plot waterfall / spectrogram."""
    with plt.rc_context(_dark_style() if dark else {}):
        fig, ax = plt.subplots(figsize=(8, 5))
        n_frames = max(1, (len(iq) - nfft) // hop)
        spec = np.zeros((nfft, n_frames), dtype=complex)
        for i in range(n_frames):
            seg = iq[i * hop : i * hop + nfft]
            spec[:, i] = np.fft.fftshift(np.fft.fft(seg * np.hanning(nfft)))
        spec_db = 10 * np.log10(np.abs(spec) ** 2 + 1e-30)
        freqs = np.fft.fftshift(np.fft.fftfreq(nfft, d=1.0 / sample_rate))
        # Transpose so time is on Y, frequency on X (per spec)
        ax.imshow(spec_db.T, aspect="auto", origin="lower", cmap="viridis",
                  extent=[freqs[0] / 1e3, freqs[-1] / 1e3, 0, n_frames])
        ax.set_xlabel("Frequency (kHz)")
        ax.set_ylabel("Time (frame)")
        ax.set_title("Spectrogram")
        fig.tight_layout()
    return fig


def plot_constellation(
    iq: np.ndarray,
    max_points: int = 5000,
    dark: bool = True,
) -> plt.Figure:
    """Plot IQ constellation diagram."""
    with plt.rc_context(_dark_style() if dark else {}):
        fig, ax = plt.subplots(figsize=(5, 5))
        pts = iq[:max_points]
        ax.scatter(pts.real, pts.imag, s=2, alpha=0.5, color="#4a90d9")
        ax.set_xlabel("I")
        ax.set_ylabel("Q")
        ax.set_title("Constellation")
        ax.set_aspect("equal")
        ax.grid(True, alpha=0.3)
        fig.tight_layout()
    return fig


def plot_scd(
    iq: np.ndarray,
    sample_rate: float = 1e6,
    dark: bool = True,
) -> plt.Figure:
    """Plot Spectral Correlation Density."""
    from spectra.transforms import SCD
    with plt.rc_context(_dark_style() if dark else {}):
        fig, ax = plt.subplots(figsize=(8, 5))
        scd_transform = SCD(nfft=128)
        scd_out = scd_transform(iq)
        if hasattr(scd_out, "numpy"):
            scd_out = scd_out.numpy()
        ax.imshow(np.abs(scd_out), aspect="auto", origin="lower", cmap="viridis")
        ax.set_xlabel("Spectral Frequency")
        ax.set_ylabel("Cyclic Frequency")
        ax.set_title("Spectral Correlation Density")
        fig.tight_layout()
    return fig


def plot_ambiguity(
    iq: np.ndarray,
    dark: bool = True,
) -> plt.Figure:
    """Plot ambiguity function surface."""
    from spectra.transforms import AmbiguityFunction
    with plt.rc_context(_dark_style() if dark else {}):
        fig, ax = plt.subplots(figsize=(8, 5))
        af = AmbiguityFunction()
        af_out = af(iq)
        if hasattr(af_out, "numpy"):
            af_out = af_out.numpy()
        ax.imshow(np.abs(af_out), aspect="auto", origin="lower", cmap="inferno")
        ax.set_xlabel("Delay")
        ax.set_ylabel("Doppler")
        ax.set_title("Ambiguity Function")
        fig.tight_layout()
    return fig


def plot_eye(
    iq: np.ndarray,
    samples_per_symbol: int = 8,
    num_traces: int = 100,
    dark: bool = True,
) -> Optional[plt.Figure]:
    """Plot eye diagram. Returns None if samples_per_symbol is invalid."""
    if samples_per_symbol < 2:
        return None
    sps = samples_per_symbol
    trace_len = 2 * sps  # two symbol periods
    n_available = len(iq) // sps
    n_traces = min(num_traces, max(1, n_available - 2))

    with plt.rc_context(_dark_style() if dark else {}):
        fig, ax = plt.subplots(figsize=(6, 4))
        t = np.arange(trace_len)
        for i in range(n_traces):
            start = i * sps
            if start + trace_len > len(iq):
                break
            ax.plot(t, iq[start : start + trace_len].real, color="#4a90d9",
                    alpha=0.15, linewidth=0.5)
        ax.set_xlabel("Sample (within 2 symbol periods)")
        ax.set_ylabel("Amplitude (I)")
        ax.set_title("Eye Diagram")
        ax.grid(True, alpha=0.3)
        fig.tight_layout()
    return fig
```

- [ ] **Step 4: Run tests**

```bash
/Users/gditzler/.venvs/base/bin/pytest tests/test_studio_plotting.py -v
```
Expected: 8/8 PASS

- [ ] **Step 5: Commit**

```bash
git add python/spectra/studio/plotting.py tests/test_studio_plotting.py
git commit -m "feat(studio): add 7 plot functions (IQ, FFT, waterfall, constellation, SCD, ambiguity, eye)"
```

---

## Task 3: Parameter Registry (Testable)

**Files:**
- Create: `python/spectra/studio/params.py`
- Create: `tests/test_studio_params.py`

- [ ] **Step 1: Write the failing tests**

```python
# tests/test_studio_params.py
"""Tests for waveform parameter registry."""
import pytest


def test_get_waveform_params_bpsk():
    from spectra.studio.params import get_waveform_params
    params = get_waveform_params("BPSK")
    assert isinstance(params, list)
    # RRC waveforms should have samples_per_symbol
    names = [p["name"] for p in params]
    assert "samples_per_symbol" in names


def test_get_waveform_params_lfm():
    from spectra.studio.params import get_waveform_params
    params = get_waveform_params("LFM")
    assert isinstance(params, list)
    assert len(params) > 0


def test_get_waveform_params_pulsed_radar():
    from spectra.studio.params import get_waveform_params
    params = get_waveform_params("PulsedRadar")
    names = [p["name"] for p in params]
    assert "pri_samples" in names
    assert "num_pulses" in names
    assert "pulse_shape" in names


def test_get_waveform_params_unknown():
    from spectra.studio.params import get_waveform_params
    params = get_waveform_params("NONEXISTENT")
    assert params == []


def test_get_all_categories():
    from spectra.studio.params import get_all_categories
    cats = get_all_categories()
    assert "PSK" in cats
    assert "Radar" in cats
    assert len(cats) >= 10


def test_get_waveforms_for_category():
    from spectra.studio.params import get_waveforms_for_category
    psk = get_waveforms_for_category("PSK")
    assert "BPSK" in psk
    assert "QPSK" in psk
```

- [ ] **Step 2: Run tests to verify they fail**

```bash
/Users/gditzler/.venvs/base/bin/pytest tests/test_studio_params.py -v
```
Expected: `ModuleNotFoundError`

- [ ] **Step 3: Write the implementation**

```python
# python/spectra/studio/params.py
"""Waveform parameter registry for SPECTRA Studio.

Auto-discovers constructor parameters from waveform classes via
``inspect.signature()``, providing the UI with type, default, and
label information for dynamic parameter rendering.
"""

from __future__ import annotations

import inspect
from typing import Any, Dict, List, Optional

from spectra.cli.config_builder import WAVEFORM_CATEGORIES, get_waveform_registry

# Parameters to exclude from UI (internal/inherited)
_EXCLUDE_PARAMS = {"self", "args", "kwargs"}

# Choice-type overrides for specific parameters
_CHOICE_OVERRIDES: Dict[str, List[str]] = {
    "pulse_shape": ["rect", "hamming", "hann"],
    "code_type": ["frank", "p1", "p2", "p3", "p4"],
    "sweep_type": ["sawtooth", "triangle", "tandem_hooked", "s_curve"],
    "prf_mode": ["low", "medium", "high"],
}


def get_waveform_params(waveform_name: str) -> List[Dict[str, Any]]:
    """Get UI-renderable parameter descriptors for a waveform class.

    Returns a list of dicts, each with keys:
    ``name``, ``type`` (``"int"``, ``"float"``, ``"choice"``),
    ``default``, ``label``, and optionally ``choices``.

    Returns empty list for unknown waveform names.
    """
    registry = get_waveform_registry()
    cls = registry.get(waveform_name)
    if cls is None:
        return []

    try:
        sig = inspect.signature(cls.__init__)
    except (ValueError, TypeError):
        return []

    params = []
    for name, param in sig.parameters.items():
        if name in _EXCLUDE_PARAMS:
            continue

        default = param.default if param.default is not inspect.Parameter.empty else None
        annotation = param.annotation

        # Determine UI type
        if name in _CHOICE_OVERRIDES:
            p_type = "choice"
            choices = _CHOICE_OVERRIDES[name]
        elif annotation == int or isinstance(default, int):
            p_type = "int"
            choices = None
        elif annotation == float or isinstance(default, (float, int)):
            p_type = "float"
            choices = None
        elif isinstance(default, str):
            p_type = "choice"
            choices = _CHOICE_OVERRIDES.get(name, [str(default)])
        else:
            p_type = "float"
            choices = None

        label = name.replace("_", " ").title()

        entry: Dict[str, Any] = {
            "name": name,
            "type": p_type,
            "default": default,
            "label": label,
        }
        if choices:
            entry["choices"] = choices

        params.append(entry)

    return params


def get_all_categories() -> List[str]:
    """Return all waveform category names."""
    return list(WAVEFORM_CATEGORIES.keys())


def get_waveforms_for_category(category: str) -> List[str]:
    """Return waveform names for a given category."""
    return WAVEFORM_CATEGORIES.get(category, [])
```

- [ ] **Step 4: Run tests**

```bash
/Users/gditzler/.venvs/base/bin/pytest tests/test_studio_params.py -v
```
Expected: 6/6 PASS

- [ ] **Step 5: Commit**

```bash
git add python/spectra/studio/params.py tests/test_studio_params.py
git commit -m "feat(studio): add waveform parameter registry with auto-discovery"
```

---

## Task 4: CLI Subcommands

**Files:**
- Create: `python/spectra/cli/main.py`
- Modify: `python/spectra/cli/__main__.py`
- Create: `tests/test_cli_main.py`

- [ ] **Step 1: Write the failing tests**

```python
# tests/test_cli_main.py
"""Tests for unified CLI entry point."""
import subprocess
import sys
import pytest


def test_spectra_help():
    result = subprocess.run(
        [sys.executable, "-m", "spectra.cli", "--help"],
        capture_output=True, text=True, timeout=10,
    )
    assert result.returncode == 0
    assert "studio" in result.stdout
    assert "generate" in result.stdout
    assert "viz" in result.stdout
    assert "build" in result.stdout


def test_spectra_generate_no_config():
    result = subprocess.run(
        [sys.executable, "-m", "spectra.cli", "generate"],
        capture_output=True, text=True, timeout=10,
    )
    assert result.returncode != 0  # missing required --config


def test_spectra_viz_no_file():
    result = subprocess.run(
        [sys.executable, "-m", "spectra.cli", "viz"],
        capture_output=True, text=True, timeout=10,
    )
    assert result.returncode != 0  # missing required file arg


def test_spectra_studio_missing_gradio():
    """Studio should fail gracefully if gradio not installed (may pass if installed)."""
    result = subprocess.run(
        [sys.executable, "-m", "spectra.cli", "studio", "--help"],
        capture_output=True, text=True, timeout=10,
    )
    # Either shows help (gradio installed) or error message
    assert result.returncode == 0
```

- [ ] **Step 2: Run tests to verify they fail**

```bash
/Users/gditzler/.venvs/base/bin/pytest tests/test_cli_main.py -v
```
Expected: failures (main.py doesn't exist)

- [ ] **Step 3: Write `cli/main.py`**

```python
# python/spectra/cli/main.py
"""Unified CLI for SPECTRA: studio, generate, viz, build."""

from __future__ import annotations

import argparse
import sys


def _cmd_studio(args: argparse.Namespace) -> None:
    """Launch SPECTRA Studio (Gradio UI)."""
    from spectra.studio import launch

    launch(port=args.port, share=args.share, dark=args.dark)


def _cmd_generate(args: argparse.Namespace) -> None:
    """Headless batch generation from YAML config."""
    from spectra.benchmarks.loader import load_benchmark
    from spectra.utils.file_handlers.sigmf_writer import SigMFWriter

    dataset = load_benchmark(args.config, split=args.split)

    if isinstance(dataset, tuple):
        # "all" split returns a 3-tuple
        for split_name, ds in zip(["train", "val", "test"], dataset):
            output_dir = f"{args.output}/{split_name}"
            print(f"Exporting {split_name} split ({len(ds)} samples) to {output_dir}...")
            SigMFWriter.write_from_dataset(
                ds, output_dir, sample_rate=args.sample_rate or 1e6
            )
    else:
        print(f"Exporting {args.split} split ({len(dataset)} samples) to {args.output}...")
        SigMFWriter.write_from_dataset(
            dataset, args.output, sample_rate=args.sample_rate or 1e6
        )
    print("Done.")


def _cmd_viz(args: argparse.Namespace) -> None:
    """Quick visualization of an IQ file."""
    import matplotlib
    if args.save:
        matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    from spectra.utils.file_handlers import get_reader

    reader = get_reader(args.file)
    iq, metadata = reader.read(args.file)
    sr = metadata.sample_rate or 1e6

    from spectra.studio.plotting import plot_fft, plot_iq, plot_waterfall, plot_constellation

    plot_funcs = {
        "fft": lambda: plot_fft(iq, sample_rate=sr, dark=False),
        "iq": lambda: plot_iq(iq, sample_rate=sr, dark=False),
        "waterfall": lambda: plot_waterfall(iq, sample_rate=sr, dark=False),
        "constellation": lambda: plot_constellation(iq, dark=False),
    }

    fig = plot_funcs[args.plot]()
    if args.save:
        fig.savefig(args.save, dpi=150)
        print(f"Saved to {args.save}")
    else:
        plt.show()


def _cmd_build(args: argparse.Namespace) -> None:
    """Run the interactive signal builder wizard."""
    from spectra.cli.signal_builder import main as builder_main

    builder_main()


def main() -> None:
    """SPECTRA CLI entry point."""
    parser = argparse.ArgumentParser(
        prog="spectra",
        description="SPECTRA — RF waveform generation toolkit",
    )
    subparsers = parser.add_subparsers(dest="command", help="Available commands")

    # studio
    sp_studio = subparsers.add_parser("studio", help="Launch SPECTRA Studio UI")
    sp_studio.add_argument("--port", type=int, default=7860)
    sp_studio.add_argument("--share", action="store_true")
    sp_studio.add_argument("--dark", action="store_true")

    # generate
    sp_gen = subparsers.add_parser("generate", help="Headless batch generation")
    sp_gen.add_argument("--config", required=True, help="YAML config path")
    sp_gen.add_argument("--output", default="./spectra_output", help="Output directory")
    sp_gen.add_argument("--split", default="all", choices=["train", "val", "test", "all"])
    sp_gen.add_argument("--sample-rate", type=float, default=None)

    # viz (requires spectra[ui] for plotting functions)
    sp_viz = subparsers.add_parser("viz", help="Quick IQ file visualization")
    sp_viz.add_argument("file", help="Path to IQ file (.sigmf-meta, .cf32, .npy)")
    sp_viz.add_argument("--plot", default="fft", choices=["fft", "iq", "waterfall", "constellation"])
    sp_viz.add_argument("--save", default=None, help="Save plot to file instead of showing")

    # build
    subparsers.add_parser("build", help="Interactive signal builder wizard")

    args = parser.parse_args()

    if args.command is None:
        parser.print_help()
        sys.exit(0)

    cmds = {
        "studio": _cmd_studio,
        "generate": _cmd_generate,
        "viz": _cmd_viz,
        "build": _cmd_build,
    }
    cmds[args.command](args)


if __name__ == "__main__":
    main()
```

- [ ] **Step 4: Update `cli/__main__.py`**

Replace contents with:
```python
from spectra.cli.main import main

main()
```

- [ ] **Step 5: Run tests**

```bash
/Users/gditzler/.venvs/base/bin/pytest tests/test_cli_main.py -v
```
Expected: 4/4 PASS

- [ ] **Step 6: Commit**

```bash
git add python/spectra/cli/main.py python/spectra/cli/__main__.py tests/test_cli_main.py
git commit -m "feat(cli): add unified spectra CLI with studio, generate, viz, build subcommands"
```

---

## Task 5: Gradio App Assembly (Generate + Visualize + Export Tabs)

**Files:**
- Create: `python/spectra/studio/app.py`
- Create: `python/spectra/studio/generate_tab.py`
- Create: `python/spectra/studio/visualize_tab.py`
- Create: `python/spectra/studio/export_tab.py`

This task creates the Gradio UI. Due to the interactive nature of Gradio apps, verification is done via manual launch rather than unit tests.

- [ ] **Step 1: Write `generate_tab.py`**

```python
# python/spectra/studio/generate_tab.py
"""Generate tab for SPECTRA Studio."""

from __future__ import annotations

import numpy as np
import gradio as gr

from spectra.cli.config_builder import (
    IMPAIRMENT_PRESETS,
    get_waveform_registry,
    get_impairment_registry,
    build_config,
)
from spectra.studio.params import get_all_categories, get_waveforms_for_category, get_waveform_params
from spectra.studio.plotting import plot_constellation, plot_fft


def _generate_signal(category, waveform_name, sample_rate, num_samples, snr_db, seed,
                     preset, wideband, capture_bw, num_signals, *param_values):
    """Callback: generate IQ, return (iq, meta, config, constellation_fig, psd_fig)."""
    registry = get_waveform_registry()
    cls = registry.get(waveform_name)
    if cls is None:
        return None, {}, {}, None, None

    # Build kwargs from dynamic params
    params = get_waveform_params(waveform_name)
    kwargs = {}
    for i, p in enumerate(params[:10]):  # max 10 param slots
        if i < len(param_values) and param_values[i] is not None:
            val = param_values[i]
            if p["type"] == "int":
                val = int(val)
            elif p["type"] == "float":
                val = float(val)
            kwargs[p["name"]] = val

    waveform = cls(**kwargs)

    if wideband:
        from spectra.scene.composer import Composer, SceneConfig
        config = SceneConfig(
            capture_duration=num_samples / sample_rate,
            capture_bandwidth=capture_bw,
            sample_rate=sample_rate,
            num_signals=num_signals,
            signal_pool=[waveform],
            snr_range=(snr_db, snr_db + 10),
        )
        composer = Composer(config)
        iq, descs = composer.generate(seed=seed)
        label = f"Scene ({len(descs)} signals)"
    else:
        sps = getattr(waveform, "samples_per_symbol", 8)
        n_sym = max(1, num_samples // sps)
        iq = waveform.generate(num_symbols=n_sym, sample_rate=sample_rate, seed=seed)
        iq = iq[:num_samples]
        label = waveform.label

    # Apply impairments
    if preset != "clean" and preset in IMPAIRMENT_PRESETS:
        from spectra.impairments import Compose, AWGN
        from spectra.scene.signal_desc import SignalDescription
        imp_reg = get_impairment_registry()
        chain = []
        for entry in IMPAIRMENT_PRESETS[preset]:
            imp_cls = imp_reg.get(entry["type"])
            if imp_cls:
                chain.append(imp_cls(**entry.get("params", {})))
        if chain:
            desc = SignalDescription(0, len(iq)/sample_rate, -sample_rate/2, sample_rate/2, label, snr_db)
            iq, _ = Compose(chain)(iq, desc, sample_rate=sample_rate)

    meta = {"waveform_label": label, "sample_rate": sample_rate, "num_samples": len(iq), "snr_db": snr_db}
    fig_const = plot_constellation(iq)
    fig_psd = plot_fft(iq, sample_rate=sample_rate)
    return iq, meta, {}, fig_const, fig_psd


def build_generate_tab(iq_state, meta_state, config_state):
    """Build the Generate tab UI components and wire callbacks."""
    categories = get_all_categories()

    with gr.Row():
        with gr.Column(scale=2):
            category = gr.Dropdown(choices=categories, value=categories[0], label="Category")
            waveform = gr.Dropdown(choices=get_waveforms_for_category(categories[0]), label="Waveform")
            sample_rate = gr.Number(value=1e6, label="Sample Rate (Hz)")
            num_samples = gr.Number(value=1024, label="Number of IQ Samples", precision=0)
            snr_db = gr.Slider(-10, 40, value=20, label="SNR (dB)")
            seed = gr.Number(value=42, label="Seed", precision=0)
            preset = gr.Dropdown(choices=["clean", "mild", "realistic"], value="clean", label="Impairment Preset")

            wideband = gr.Checkbox(value=False, label="Wideband Scene Mode")
            capture_bw = gr.Number(value=10e6, label="Capture Bandwidth (Hz)", visible=False)
            num_signals = gr.Number(value=3, label="Number of Signals", visible=False, precision=0)
            wideband.change(lambda w: (gr.update(visible=w), gr.update(visible=w)),
                           inputs=[wideband], outputs=[capture_bw, num_signals])

            # Dynamic param slots (10 generic components)
            param_slots = []
            for i in range(10):
                slot = gr.Number(value=0, label=f"Param {i}", visible=False)
                param_slots.append(slot)

            def update_waveform_list(cat):
                wfs = get_waveforms_for_category(cat)
                return gr.update(choices=wfs, value=wfs[0] if wfs else None)

            def update_params(wf_name):
                params = get_waveform_params(wf_name)
                updates = []
                for i in range(10):
                    if i < len(params):
                        p = params[i]
                        updates.append(gr.update(visible=True, label=p["label"],
                                                  value=p["default"] if p["default"] is not None else 0))
                    else:
                        updates.append(gr.update(visible=False))
                return updates

            category.change(update_waveform_list, inputs=[category], outputs=[waveform])
            waveform.change(update_params, inputs=[waveform], outputs=param_slots)

            gen_btn = gr.Button("Generate", variant="primary")

        with gr.Column(scale=3):
            const_plot = gr.Plot(label="Constellation")
            psd_plot = gr.Plot(label="Power Spectral Density")
            info = gr.Markdown("*Click Generate to preview*")

    gen_btn.click(
        _generate_signal,
        inputs=[category, waveform, sample_rate, num_samples, snr_db, seed,
                preset, wideband, capture_bw, num_signals] + param_slots,
        outputs=[iq_state, meta_state, config_state, const_plot, psd_plot],
    )
```

- [ ] **Step 2: Write `visualize_tab.py`**

```python
# python/spectra/studio/visualize_tab.py
"""Visualize tab for SPECTRA Studio."""

from __future__ import annotations

import numpy as np
import gradio as gr

from spectra.studio.plotting import (
    plot_iq, plot_fft, plot_waterfall, plot_constellation,
    plot_scd, plot_ambiguity, plot_eye,
)


def _load_file(file_obj):
    """Load IQ from uploaded file, return (iq, sample_rate)."""
    if file_obj is None:
        return None, 1e6
    from spectra.utils.file_handlers import get_reader, SigMFReader
    path = file_obj.name
    if path.endswith(".sigmf-meta"):
        reader = SigMFReader()
    else:
        reader = get_reader(path)
    iq, meta = reader.read(path)
    sr = meta.sample_rate or 1e6
    return iq, sr


def build_visualize_tab(iq_state, meta_state):
    """Build the Visualize tab UI."""
    source = gr.Radio(["Use generated signal", "Load file"], value="Use generated signal", label="Data Source")
    file_upload = gr.File(label="Upload IQ file", file_types=[".sigmf-meta", ".cf32", ".npy", ".raw"], visible=False)
    sr_input = gr.Number(value=1e6, label="Sample Rate (Hz)")

    source.change(lambda s: gr.update(visible=(s == "Load file")), inputs=[source], outputs=[file_upload])

    with gr.Tabs():
        for name, plot_fn, extra_inputs in [
            ("IQ", plot_iq, []),
            ("FFT", plot_fft, []),
            ("Waterfall", plot_waterfall, []),
            ("Constellation", plot_constellation, []),
            ("SCD", plot_scd, []),
            ("Ambiguity", plot_ambiguity, []),
            ("Eye Diagram", plot_eye, []),
        ]:
            with gr.Tab(name):
                plot_output = gr.Plot(label=name)
                sps_input = gr.Number(value=8, label="Samples per Symbol", visible=(name == "Eye Diagram"))
                plot_btn = gr.Button(f"Plot {name}")

                def make_callback(fn, tab_name):
                    def cb(iq_data, meta, src, file_obj, sr, sps):
                        if src == "Load file":
                            iq, sr_loaded = _load_file(file_obj)
                            if iq is None:
                                return None
                            sr = sr_loaded
                        else:
                            iq = iq_data
                        if iq is None:
                            return None
                        if tab_name == "Eye Diagram":
                            return fn(iq, samples_per_symbol=int(sps))
                        elif tab_name in ("Constellation", "Ambiguity"):
                            return fn(iq)
                        else:
                            return fn(iq, sample_rate=sr)
                    return cb

                plot_btn.click(
                    make_callback(plot_fn, name),
                    inputs=[iq_state, meta_state, source, file_upload, sr_input, sps_input],
                    outputs=[plot_output],
                )
```

- [ ] **Step 3: Write `export_tab.py`**

```python
# python/spectra/studio/export_tab.py
"""Export tab for SPECTRA Studio."""

from __future__ import annotations

import os
import numpy as np
import gradio as gr

from spectra.cli.config_builder import build_config, serialize_config


def _export_single(iq_data, meta, output_path, fmt, center_freq):
    """Export current signal as a single file."""
    if iq_data is None:
        return "No signal generated. Go to Generate tab first."
    sr = meta.get("sample_rate", 1e6)
    label = meta.get("waveform_label", "unknown")

    os.makedirs(os.path.dirname(output_path) or ".", exist_ok=True)

    if fmt == "SigMF":
        from spectra.utils.file_handlers.sigmf_writer import SigMFWriter
        writer = SigMFWriter(output_path, sample_rate=sr, center_frequency=center_freq)
        writer.write(iq_data)
        return f"Exported SigMF to {output_path}.sigmf-meta ({len(iq_data)} samples)"
    elif fmt == "NumPy":
        np.save(f"{output_path}.npy", iq_data)
        return f"Exported NumPy to {output_path}.npy"
    elif fmt == "Raw IQ":
        iq_data.astype(np.complex64).tofile(f"{output_path}.cf32")
        return f"Exported raw IQ to {output_path}.cf32"
    return "Unknown format"


def _save_config_only(config_state):
    """Serialize config to YAML string for download."""
    if not config_state:
        return "No config available. Generate a signal first."
    return serialize_config(config_state)


def build_export_tab(iq_state, meta_state, config_state):
    """Build the Export tab UI."""
    mode = gr.Radio(["Single file", "Dataset"], value="Single file", label="Export Mode")
    fmt = gr.Dropdown(choices=["SigMF", "NumPy", "Raw IQ"], value="SigMF", label="Format")
    output_path = gr.Textbox(value="./spectra_output/signal", label="Output Path (base name)")
    center_freq = gr.Number(value=0.0, label="Center Frequency (Hz)")

    # Dataset mode controls (hidden by default)
    with gr.Group(visible=False) as dataset_group:
        n_train = gr.Number(value=50000, label="Train Samples", precision=0)
        n_val = gr.Number(value=10000, label="Val Samples", precision=0)
        n_test = gr.Number(value=10000, label="Test Samples", precision=0)
        snr_min = gr.Slider(-10, 40, value=-10, label="SNR Min (dB)")
        snr_max = gr.Slider(-10, 40, value=30, label="SNR Max (dB)")

    mode.change(lambda m: gr.update(visible=(m == "Dataset")), inputs=[mode], outputs=[dataset_group])

    export_btn = gr.Button("Export", variant="primary")
    save_yaml_btn = gr.Button("Save Config Only", variant="secondary")
    status = gr.Textbox(label="Status", interactive=False)
    yaml_output = gr.Code(label="YAML Config", language="yaml", visible=False)

    export_btn.click(
        _export_single,
        inputs=[iq_state, meta_state, output_path, fmt, center_freq],
        outputs=[status],
    )

    def save_yaml(cfg):
        yaml_str = _save_config_only(cfg)
        return gr.update(visible=True, value=yaml_str)

    save_yaml_btn.click(save_yaml, inputs=[config_state], outputs=[yaml_output])
```

Key imports: `SigMFWriter`; `build_config`, `serialize_config` from `config_builder.py`.

- [ ] **Step 4: Write `app.py`**

```python
# python/spectra/studio/app.py
"""SPECTRA Studio — Gradio application assembly."""

from __future__ import annotations

import gradio as gr
import numpy as np

from spectra.studio.theme import spectra_theme


def create_app(dark: bool = False) -> gr.Blocks:
    """Create and return the SPECTRA Studio Gradio app."""
    theme = spectra_theme(dark=dark)

    with gr.Blocks(
        theme=theme,
        title="SPECTRA Studio",
        css=".gradio-container { max-width: 1200px; margin: auto; }",
    ) as app:
        gr.Markdown("# SPECTRA Studio\nInteractive RF waveform generation, visualization, and export.")

        # Shared state
        iq_state = gr.State(value=None)
        meta_state = gr.State(value={})
        config_state = gr.State(value={})

        with gr.Tabs():
            with gr.Tab("Generate"):
                from spectra.studio.generate_tab import build_generate_tab
                build_generate_tab(iq_state, meta_state, config_state)

            with gr.Tab("Visualize"):
                from spectra.studio.visualize_tab import build_visualize_tab
                build_visualize_tab(iq_state, meta_state)

            with gr.Tab("Export"):
                from spectra.studio.export_tab import build_export_tab
                build_export_tab(iq_state, meta_state, config_state)

    return app
```

- [ ] **Step 5: Verify by launching**

```bash
/Users/gditzler/.venvs/base/bin/pip install gradio>=4.0
/Users/gditzler/.venvs/base/bin/python -c "from spectra.studio import launch; launch(port=7861)"
```

Expected: Gradio app launches at http://localhost:7861 with 3 tabs, theme applied.

- [ ] **Step 6: Commit**

```bash
git add python/spectra/studio/app.py python/spectra/studio/generate_tab.py python/spectra/studio/visualize_tab.py python/spectra/studio/export_tab.py
git commit -m "feat(studio): add Gradio app with Generate, Visualize, and Export tabs"
```

---

## Task 6: Full Verification

- [ ] **Step 1: Run all tests (no regressions)**

```bash
/Users/gditzler/.venvs/base/bin/pytest tests/ -q --tb=short
```
Expected: all tests pass.

- [ ] **Step 2: Verify CLI commands**

```bash
# Help
/Users/gditzler/.venvs/base/bin/python -m spectra.cli --help

# Viz (requires a SigMF file — create one first)
/Users/gditzler/.venvs/base/bin/python -c "
from spectra.waveforms import QPSK
from spectra.utils.file_handlers.sigmf_writer import SigMFWriter
iq = QPSK().generate(1024, 1e6, seed=42)
w = SigMFWriter('/tmp/test_viz', sample_rate=1e6)
w.write(iq)
print('SigMF written')
"
/Users/gditzler/.venvs/base/bin/python -m spectra.cli viz /tmp/test_viz.sigmf-meta --save /tmp/test_plot.png
ls /tmp/test_plot.png

# Studio launch (manual check)
/Users/gditzler/.venvs/base/bin/python -m spectra.cli studio --help
```

- [ ] **Step 3: Verify imports**

```bash
/Users/gditzler/.venvs/base/bin/python -c "
from spectra.studio import launch
from spectra.studio.plotting import plot_iq, plot_fft, plot_waterfall, plot_constellation, plot_scd, plot_ambiguity, plot_eye
from spectra.studio.params import get_waveform_params, get_all_categories, get_waveforms_for_category
print('All imports OK')
"
```
