# Studio (Interactive UI)

SPECTRA Studio is a Gradio-based browser UI for interactive waveform generation,
visualization, and dataset export. It is the fastest way to explore the waveform,
impairment, and scene parameter spaces without writing any code.

---

## Installation

Studio requires the `[ui]` optional dependency group:

```
uv pip install -e ".[ui]"
```

This installs:

- `gradio>=4.0`
- `scipy>=1.10`

---

## Launching

### From the command line

```bash
spectra studio                     # default port 7860, light mode
spectra studio --port 8080         # custom port
spectra studio --dark              # start in dark mode
spectra studio --share             # create a public Gradio tunnel link
```

### Programmatically

```python
from spectra.studio import launch

launch(port=7860, share=False, dark=True)
```

`launch()` accepts three keyword arguments:

| Argument | Type  | Default | Description                          |
|----------|-------|---------|--------------------------------------|
| `port`   | `int` | `7860`  | Local port for the Gradio server     |
| `share`  | `bool`| `False` | Create a public Gradio share link    |
| `dark`   | `bool`| `False` | Start in dark mode                   |

Opening the browser to `http://localhost:7860` (or the printed URL) loads the app.

---

## Tabs

The app has three tabs: **Generate**, **Visualize**, and **Export**. IQ data and
metadata generated in the Generate tab flow automatically to the other tabs via
shared Gradio state.

### Generate

The Generate tab is where waveforms are configured and rendered. Controls on the
left panel let you:

- Select a **waveform category** (e.g. PSK, QAM, Radar, Spread Spectrum) and a
  specific waveform within that category.
- Set **sample rate**, **number of IQ samples**, **SNR (dB)**, and a **random
  seed** for reproducibility.
- Choose an **impairment preset** (`clean`, `mild`, or `realistic`) to optionally
  apply a pre-wired impairment chain.
- Toggle **Wideband Scene Mode** to place multiple signals within a shared capture
  bandwidth, with configurable capture bandwidth (Hz) and number of signals.
- Adjust up to ten waveform-specific **dynamic parameters** (e.g. modulation order,
  bandwidth, pulse width) that update automatically when the waveform changes.

Clicking **Generate** renders the signal and displays a **constellation diagram**
and a **power spectral density (PSD)** plot on the right panel.

### Visualize

The Visualize tab provides seven different plot views of any IQ signal. The data
source can be either the signal generated in the Generate tab or an IQ file
uploaded directly (supported formats: `.sigmf-meta`, `.cf32`, `.npy`, `.raw`).

Each view lives in its own sub-tab and is rendered on demand by clicking a
**Plot** button:

| Sub-tab         | Description                                                        |
|-----------------|--------------------------------------------------------------------|
| **IQ**          | Time-domain I and Q traces                                         |
| **FFT**         | Single-sided power spectrum                                        |
| **Waterfall**   | Short-time Fourier transform spectrogram (time × frequency)        |
| **Constellation** | Complex I/Q scatter plot                                         |
| **SCD**         | Spectral Correlation Density (cyclostationary feature)             |
| **Ambiguity**   | Ambiguity function (delay × Doppler)                               |
| **Eye Diagram** | Overlaid symbol-interval segments; configurable samples-per-symbol |

When loading a file with a SigMF `.sigmf-meta` metadata sidecar, the sample rate
is read automatically from the metadata.

### Export

The Export tab saves the current signal to disk and optionally serializes the
generation config for headless reproduction.

**Single-file export** writes one signal in the chosen format:

| Format    | Extension       | Notes                                   |
|-----------|-----------------|-----------------------------------------|
| SigMF     | `.sigmf-meta` + `.sigmf-data` | Writes IQ data with SigMF metadata sidecar |
| NumPy     | `.npy`          | Standard NumPy complex64 array          |
| Raw IQ    | `.cf32`         | Interleaved float32 complex samples     |

Center frequency (Hz) and output path are configurable. Switching to **Dataset**
mode reveals train/val/test sample count fields and an SNR sweep range, for
generating multi-split datasets.

The **Save Config Only** button serializes the current generation parameters to a
YAML snippet displayed inline. This YAML can be passed directly to `spectra
generate` for headless reproduction (see below).

---

## CLI complement

The `spectra` command exposes the same primitives without a browser UI.

```
spectra studio                                              # launch the UI
spectra generate --config <yaml> --output <dir>            # headless batch export
spectra viz <iq-file> --plot <view> [--save plot.png]      # quick IQ file plot
spectra build                                              # interactive config wizard
```

### `spectra studio`

Launches the Gradio UI (equivalent to `launch()`). Accepts `--port`, `--share`,
and `--dark`.

### `spectra generate`

Headless batch export. Loads a benchmark config YAML, generates the dataset, and
writes SigMF files to an output directory. Useful for reproducing a Studio session
at scale from the YAML config exported in the Export tab.

```bash
spectra generate \
    --config my_config.yaml \
    --output ./dataset \
    --split all \
    --sample-rate 2e6
```

`--split` can be `train`, `val`, `test`, or `all` (default). When `all` is chosen,
three subdirectories are created automatically.

### `spectra viz`

Quick visualization of an IQ file without launching the full UI. Uses matplotlib;
pass `--save` to write the figure to disk instead of opening an interactive window.

```bash
spectra viz signal.sigmf-meta --plot waterfall
spectra viz signal.cf32 --plot constellation --save constellation.png
```

Available `--plot` choices: `fft`, `iq`, `waterfall`, `constellation`.

### `spectra build`

Launches an interactive terminal wizard (`spectra.cli.signal_builder`) that guides
you through building a dataset config step-by-step, outputting a YAML file
compatible with `spectra generate`. Also available as the standalone entry point
`spectra-build`.

---

## See also

- [Datasets](datasets.md) — `NarrowbandDataset`, `WidebandDataset`, and other
  dataset classes used for full-scale generation
- [Benchmarks](benchmarks.md) — pre-built YAML configs loadable by `spectra generate`
- [File I/O](file-io.md) — readers and writers used by the Export tab and `spectra viz`
- [Transforms](transforms.md) — signal transforms underlying the Visualize tab plots
