# SPECTRA Studio — Design Spec

**Goal:** Add a Gradio-based UI ("SPECTRA Studio") for interactive RF waveform generation, visualization, and SigMF export, plus CLI subcommands for headless generation and quick visualization.

**Scope:** v1 delivers a 3-tab Gradio app (Generate, Visualize, Export), a custom dark/light theme, 7 plot types, and three new CLI subcommands (`spectra studio`, `spectra generate`, `spectra viz`). The UI lives in `spectra/studio/` as an optional dependency (`spectra[ui]`), completely isolated from core package functionality.

---

## Architecture

```
spectra studio  ──→  spectra/studio/app.py  ──→  Gradio Blocks (3 tabs)
                                                    ├── Generate Tab
                                                    ├── Visualize Tab
                                                    └── Export Tab

spectra generate ──→  spectra/cli/main.py  ──→  load_benchmark() → SigMFWriter
spectra viz      ──→  spectra/cli/main.py  ──→  FileReader → matplotlib
spectra build    ──→  spectra/cli/signal_builder.py  (unchanged)
```

The Studio imports SPECTRA's existing waveform/impairment registries, scene composer, benchmark loader, and SigMF writer. It adds no new DSP logic — only UI orchestration and plotting.

---

## Sub-project 1: File Structure & Dependencies

### New Files

```
python/spectra/
  studio/
    __init__.py          # launch() entry point
    app.py               # Gradio Blocks app (3-tab layout)
    generate_tab.py      # Generate tab UI + callbacks
    visualize_tab.py     # Visualize tab UI + callbacks
    export_tab.py        # Export tab UI + callbacks
    plotting.py          # 7 plot functions (IQ, FFT, waterfall, constellation, SCD, ambiguity, eye)
    params.py            # Waveform parameter registry (auto-generated from inspect.signature)
    theme.py             # Custom Gradio theme (dark/light)
  cli/
    main.py              # Unified CLI: studio, generate, viz subcommands
```

### Modified Files

- `pyproject.toml` — add `spectra[ui]` optional dep, add `spectra` CLI entry point

### Optional Dependency

```toml
[project.optional-dependencies]
ui = ["gradio>=4.0"]
```

Added to the `all` group as well. The `spectra studio` command checks for gradio at runtime and prints `pip install spectra[ui]` if missing.

### CLI Entry Points

```toml
[project.scripts]
spectra = "spectra.cli.main:main"
spectra-build = "spectra.cli.signal_builder:main"  # unchanged
```

---

## Sub-project 2: Generate Tab (`studio/generate_tab.py`)

### Layout

Left column (40%) — configuration. Right column (60%) — live preview.

### Left Column — Configuration

**1. Waveform selector** — cascading dropdowns:
- Category dropdown: PSK, QAM, FSK, ASK, OFDM, Analog, Radar, 5G NR, Spread Spectrum, Protocol
- Waveform type dropdown: populated from `WAVEFORM_CATEGORIES` in `config_builder.py`
- Dynamic parameter panel: renders sliders/dropdowns/number inputs based on waveform constructor signature

**Waveform parameter registry** (`studio/params.py` — new file):
A `_WAVEFORM_PARAMS` dict maps each waveform class name to a list of parameter descriptors `(name, type, default, label)`. This is auto-generated from `inspect.signature()` on each class in `get_waveform_registry()` (from `config_builder.py`), with manual overrides for display labels and UI widget types (slider vs dropdown). The registry is built lazily on first access. For the ~86 waveforms, most share the same base params (RRC waveforms all have `samples_per_symbol`, `rolloff`, `filter_span`), so the registry is compact. The UI renders params using a fixed pool of ~10 generic Gradio components (`gr.Slider`, `gr.Dropdown`, `gr.Number`) that are shown/hidden via `gr.update(visible=...)` based on the selected waveform — no per-waveform component pre-building needed. Examples:
- **RRC waveforms (BPSK, QPSK, QAM, etc.):** `samples_per_symbol` (int, 8), `rolloff` (float, 0.35), `filter_span` (int, 10)
- **PulsedRadar:** `pulse_width_samples` (int, 64), `pri_samples` (int, 512), `num_pulses` (int, 16), `pulse_shape` (choice: rect/hamming/hann)
- **BarkerCodedPulse:** `barker_length` (choice: 2-13), `samples_per_chip` (int, 4), `pri_samples` (int, 512), `num_pulses` (int, 16)
- **PolyphaseCodedPulse:** `code_type` (choice: frank/p1/p2/p3/p4), `code_order` (int, 4), `samples_per_chip` (int, 4), `pri_samples` (int, 512), `num_pulses` (int, 16)
- **FMCW:** `sweep_bandwidth_fraction` (float, 0.8), `sweep_samples` (int, 256), `idle_samples` (int, 64), `num_sweeps` (int, 8), `sweep_type` (choice: sawtooth/triangle)
- **LFM:** `bandwidth_fraction` (float, 0.8), `samples_per_pulse` (int, 256)
- **PulseDoppler:** `prf_mode` (choice: low/medium/high), `num_pulses_per_cpi` (int, 32), `pulse_width_samples` (int, 16), `num_cpis` (int, 4)
- **NonlinearFM:** `sweep_type` (choice: tandem_hooked/s_curve), `bandwidth_fraction` (float, 0.8), `num_samples` (int, 256)
- **SteppedFrequency:** `num_steps` (int, 8), `samples_per_step` (int, 64), `freq_step_fraction` (float, 0.1), `num_bursts` (int, 4)

**2. Impairment chain builder:**
- Presets dropdown: "Clean" (none), "Mild", "Realistic", "Custom"
- Custom mode: checkboxes per impairment with expandable parameter sliders
- Reuses `IMPAIRMENT_PRESETS` from `config_builder.py`

**3. Signal parameters:**
- Sample rate (Hz), number of IQ samples, SNR (dB), seed

**4. Wideband toggle:**
When enabled, replaces single-waveform selector with:
- Capture bandwidth (Hz), capture duration (s), num signals range (min, max)
- Signal pool builder: add/remove waveforms to the pool
- Allow overlap checkbox
- Uses `SceneConfig` + `Composer` under the hood

**5. "Generate" button** — runs generation, updates preview plots

### Right Column — Live Preview

- Constellation diagram (top)
- PSD/FFT plot (bottom)
- Signal info text: waveform label, bandwidth, sample count, SNR

---

## Sub-project 3: Visualize Tab (`studio/visualize_tab.py`)

### Input Section

- Radio button: "Use generated signal" (carries IQ from Generate tab) or "Load file"
- File upload: supports `.sigmf-meta`, `.cf32`, `.cs16`, `.raw`, `.npy`
- Sample rate input (auto-populated from metadata when available)

### Plot Sub-Tabs (7 types)

1. **IQ Time-Domain** — I and Q channels vs sample index. Sample range slider for zooming.

2. **FFT / PSD** — Welch's PSD. X-axis in Hz (centered at DC), Y-axis in dB. NFFT selector (256, 512, 1024, 2048).

3. **Waterfall / Spectrogram** — STFT magnitude heatmap. Time on Y, frequency on X. Colormap selector. NFFT and hop size controls.

4. **Constellation** — I vs Q scatter. Max points slider.

5. **SCD** — Spectral Correlation Density heatmap. Uses `spectra.transforms.SCD`. Resolution parameter.

6. **Ambiguity Function** — Delay-Doppler surface. Uses `spectra.transforms.AmbiguityFunction`.

7. **Eye Diagram** — Overlaid symbol-period traces. Requires `samples_per_symbol` input.

### Plot Infrastructure (`studio/plotting.py`)

All plots are matplotlib `Figure` objects rendered via Gradio's `gr.Plot`. A theme-aware style dict auto-detects dark/light mode for consistent appearance. Each plot function has a `save_path` option for the download button.

---

## Sub-project 4: Export Tab (`studio/export_tab.py`)

### Mode Selector

- **Single file:** export current signal as one SigMF file pair
- **Dataset:** batch generation with train/val/test splits

### Single File Mode

- Output format: SigMF (default), raw IQ (`.cf32`), NumPy (`.npy`)
- Output path text input
- Center frequency field (for SigMF metadata)
- "Export" button → writes file, shows status

### Dataset Mode

- Number of samples per split (train, val, test)
- SNR range slider
- Per-split seeds
- Waveform pool (carry from Generate tab or build new)
- Impairment chain (carry or configure)
- Output directory picker
- "Save Config Only" button — exports YAML (compatible with `spectra generate` and `load_benchmark()`)
- "Export Dataset" button — generates and writes all files with progress bar

### Output

- Uses `SigMFWriter` for SigMF, `RawIQWriter` for `.cf32`, `NumpyWriter` for `.npy`
- YAML config follows existing benchmark schema
- Progress bar via Gradio's `gr.Progress`

---

## Sub-project 5: CLI Subcommands (`cli/main.py`)

### `spectra studio`

```bash
spectra studio [--port 7860] [--share] [--dark]
```

Launches Gradio app. Guards `import gradio` with helpful error message.

### `spectra generate`

```bash
spectra generate --config config.yaml --output ./data [--format sigmf] [--split train|val|test|all]
```

Headless batch generation. Parses YAML via `load_benchmark()`, iterates dataset, writes SigMF files via `SigMFWriter.write_from_dataset()` with `tqdm` progress. v1 supports SigMF output only (the existing `write_from_dataset` static method handles the full pipeline). Future versions can add `npy` and `cf32` formats by extending the writer classes.

### `spectra viz`

```bash
spectra viz recording.sigmf-meta [--plot fft|iq|waterfall|constellation] [--save plot.png]
```

Quick single-file visualization. Auto-detects format via `FileReader` registry. Opens matplotlib window or saves to file. Default plot is FFT/PSD.

### `spectra build`

Unchanged — alias to existing `signal_builder:main`.

### Implementation

`argparse` with `add_subparsers()`. Each subcommand is a function in `main.py` that imports the relevant module lazily (avoids loading Gradio for non-UI commands).

---

## Sub-project 6: Theme & Styling (`studio/theme.py`)

### Gradio Theme

Extends `gr.themes.Soft()`:
- Primary color: steel blue (#4a90d9)
- Accent: sea green (#50b87a)
- Dark backgrounds: #0d1117 / #1a1a2e
- Native light/dark toggle (Gradio built-in)
- Monospace font for parameter values
- App title: "SPECTRA Studio" with version

### Plot Styling

- Theme-aware matplotlib style dict (dark bg + light lines, or white bg + gray lines)
- Consistent colormaps: viridis for spectrograms, inferno for RDMs
- Uniform axis labels with units

---

## Cross-Tab State Management

The Gradio app uses `gr.State` to share data between tabs:

- **`iq_state`** (`gr.State`): the generated/loaded complex64 IQ array. Set by the Generate tab's "Generate" button or the Visualize tab's file upload. Consumed by all three tabs.
- **`meta_state`** (`gr.State`): a dict with `sample_rate`, `waveform_label`, `num_samples`, `snr`, and waveform constructor params. Carries configuration context to Visualize and Export tabs.
- **`config_state`** (`gr.State`): the serialized YAML config dict. Built by Generate tab, used by Export tab's "Save Config Only" button.

**Memory note:** IQ arrays up to ~1M samples (8 MB) are manageable in `gr.State`. For larger arrays, the state stores a file path reference instead and plots read from disk. The Generate tab caps preview at 1M samples; the Export tab handles arbitrarily large datasets via streaming iteration (never holds the full dataset in memory).

---

## Build Order

```
Sub-project 1: File structure + deps       (scaffolding)
Sub-project 2: Generate tab                (core feature)
Sub-project 3: Visualize tab               (standalone from Generate)
Sub-project 4: Export tab                   (depends on Generate for carry-over)
Sub-project 5: CLI subcommands             (depends on 1, independent of UI)
Sub-project 6: Theme                        (can be done anytime, applied last)
```

Sub-projects 2, 3, and 5 are independent and can be built in parallel. Sub-project 4 depends on 2 (for signal carry-over). Sub-project 6 is applied as polish.

---

## Isolation Guarantees

- **Studio is a leaf package** — `studio/` imports from `spectra.*` but nothing in core imports from `studio/`
- **CLI additions are minimal** — `cli/main.py` is a new file in the core CLI package. It adds `spectra generate` and `spectra viz` subcommands that use only existing SPECTRA APIs (registries, `load_benchmark`, file handlers). The existing `cli/__main__.py` is updated to point to the new dispatcher so `python -m spectra.cli` routes through it. No existing CLI behavior changes.
- **Optional dependency** — `gradio` is only in `spectra[ui]`, not base deps
- **Lazy imports** — CLI commands import Gradio only when `spectra studio` is called; `generate` and `viz` never import Gradio
- **No test regressions** — studio tests are in a separate test file, skipped when gradio is not installed

---

## Future Work

- **Real-time SDR preview** — integrate with `LiveCapture` for live IQ visualization
- **Benchmark browser** — load and preview built-in benchmark configs
- **Hosted demo** — deploy on Hugging Face Spaces as a public SPECTRA playground
- **Plugin system** — user-defined waveforms/impairments discoverable by the UI
