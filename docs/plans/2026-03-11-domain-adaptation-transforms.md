# Domain Adaptation Transforms — Design Spec

**Date:** 2026-03-11
**Status:** Approved
**Goal:** Add composable signal alignment transforms to SPECTRA for normalizing IQ data across different capture sources, SDR hardware, and gain settings.

---

## Motivation

When training on synthetic SPECTRA data and deploying against real captures — or when combining data from different SDR hardware (RTL-SDR vs USRP vs Pluto) — signals differ in sample rate, gain/power level, DC offset, IQ imbalance, and spectral coloring. These differences degrade model transfer performance.

Domain adaptation transforms normalize these differences so that a model trained on one source generalizes to another. They operate on raw IQ signals in the existing `Transform` pipeline, composable via `Compose`.

**Primary pain points addressed:**
- Sample rate mismatches between training data and deployment captures
- Different gain settings / AGC behaviors across hardware

---

## Architecture

### Where it lives

- **New file:** `python/spectra/transforms/alignment.py`
- **New optional dep group:** `spectra[alignment]` (adds `scipy>=1.10`)
- **New test file:** `tests/test_transforms_alignment.py`
- **Export:** Register in `python/spectra/transforms/__init__.py`

### Transform ABC compliance

All transforms follow the existing `Transform` signature:

```python
class SomeTransform(Transform):
    def __call__(self, iq: np.ndarray, desc: SignalDescription, **kwargs) -> tuple[np.ndarray, SignalDescription]:
        # transform iq, optionally update desc
        return iq_out, desc
```

Composable via `Compose([DCRemove(), PowerNormalize(-20), SpectralWhitening()])`.

### Dependency strategy

Only `Resample` requires scipy. All other transforms use pure NumPy plus existing Rust CSP functions (e.g., `compute_psd_welch` for spectral whitening). The `scipy` import is lazy — guarded inside `Resample.__init__` with a clear error message if missing.

---

## Transform Catalog

### Tier 1: Statistical Alignment (Core)

These address the most common cross-source differences: sample rate, power level, DC offset, and gain normalization.

#### `Resample`

Rational resampling to a target sample rate using `scipy.signal.resample_poly`.

| Parameter | Type | Default | Description |
|-----------|------|---------|-------------|
| `target_sample_rate` | `float` | (required) | Target sample rate in Hz |

- Computes rational approximation of `target_sample_rate / current_sample_rate` using `fractions.Fraction` with a denominator limit of 1000
- Applies `scipy.signal.resample_poly(iq, up, down)`
- Updates `desc` sample rate metadata if present
- Output length: `len(iq) * up // down`

#### `PowerNormalize`

Scale IQ signal to a target RMS power level in dBFS.

| Parameter | Type | Default | Description |
|-----------|------|---------|-------------|
| `target_power_dbfs` | `float` | `-20.0` | Target RMS power in dB relative to full scale |

- Computes current RMS: `rms = sqrt(mean(|iq|^2))`
- Computes target linear: `target = 10^(target_power_dbfs / 20)`
- Scales: `iq_out = iq * (target / rms)`
- Handles zero-power signals gracefully (returns unchanged)

#### `AGCNormalize`

Normalize gain to undo differences in hardware AGC settings.

| Parameter | Type | Default | Description |
|-----------|------|---------|-------------|
| `method` | `str` | `"rms"` | `"rms"` (unit power) or `"peak"` (bounded to [-1, 1]) |
| `target_level` | `float` | `1.0` | Target normalization level |

- `"rms"` mode: `iq_out = iq * (target_level / rms)`
- `"peak"` mode: `iq_out = iq * (target_level / max(|iq|))`
- Handles zero-amplitude signals gracefully

#### `DCRemove`

Remove DC offset via mean subtraction.

No parameters.

- `iq_out = iq - mean(iq)`
- Simple but critical — many SDRs introduce DC spurs

#### `ClipNormalize`

Clip outlier samples beyond N standard deviations, then scale to [-1, 1].

| Parameter | Type | Default | Description |
|-----------|------|---------|-------------|
| `clip_sigma` | `float` | `3.0` | Clip threshold in standard deviations |

- Computes clip threshold: `thresh = clip_sigma * std(iq)`
- Clips real and imaginary parts independently
- Scales to [-1, 1] by dividing by `max(|iq_clipped|)`

### Tier 2: Spectral Alignment

These address receiver-specific frequency response differences and noise floor variations.

#### `SpectralWhitening`

Flatten the power spectral density by dividing by the smoothed spectral envelope. Removes receiver-specific frequency coloring without needing a reference measurement.

| Parameter | Type | Default | Description |
|-----------|------|---------|-------------|
| `smoothing_window` | `int` | `64` | Moving average window size for PSD smoothing |

- Estimates PSD via `np.fft.fft` (or Rust `compute_psd_welch` for longer signals)
- Smooths PSD magnitude with a moving average of `smoothing_window` bins
- Divides signal FFT by smoothed envelope (with floor to avoid division by zero)
- IFFTs back to time domain
- Preserves signal phase, flattens amplitude spectrum

#### `NoiseFloorMatch`

Estimate the noise floor of a signal and scale to match a target noise level. Useful when combining captures with different noise figures.

| Parameter | Type | Default | Description |
|-----------|------|---------|-------------|
| `target_noise_floor_db` | `float` | `-40.0` | Target noise floor in dB |
| `estimation_method` | `str` | `"median"` | `"median"` (robust) or `"minimum"` (lower bound) |

- Estimates noise floor from PSD: median or minimum of PSD bins (in dB)
- Computes scale factor: `10^((target_noise_floor_db - estimated_floor_db) / 20)`
- Scales entire signal by this factor

#### `BandpassAlign`

Frequency-shift and filter a signal to align its energy to a target center frequency and bandwidth.

| Parameter | Type | Default | Description |
|-----------|------|---------|-------------|
| `center_freq` | `float` | `0.0` | Target center frequency (relative to sample rate) |
| `bandwidth` | `float` | (required) | Target bandwidth as fraction of sample rate |

- Estimates current signal center of mass in frequency domain
- Frequency-shifts to align center: `iq * exp(j * 2pi * shift * t)`
- Applies rectangular bandpass filter in frequency domain
- Updates `desc` frequency bounds if present

### Tier 3: Reference-Based (Stubs)

These are research-grade transforms that require real captured reference data. They ship as stubs with `NotImplementedError` and docstrings describing intended behavior and interface.

#### `NoiseProfileTransfer`

Replace synthetic AWGN with noise characteristics sampled from a real capture.

| Parameter | Type | Default | Description |
|-----------|------|---------|-------------|
| `noise_source` | `str` or `np.ndarray` | (required) | Path to noise capture file, or raw noise array |

**Intended approach (documented in docstring):**
1. Estimate and subtract signal component from `noise_source` (or use a known noise-only capture)
2. Estimate noise PSD profile from the reference
3. Generate colored noise matching the reference PSD
4. Replace the synthetic noise component in the input signal

#### `ReceiverEQ`

Equalize receiver frequency response using a reference PSD profile (e.g., from a known flat-spectrum calibration signal).

| Parameter | Type | Default | Description |
|-----------|------|---------|-------------|
| `reference_psd` | `str` or `np.ndarray` | (required) | Reference PSD array or path to file |

**Intended approach (documented in docstring):**
1. Load or accept reference PSD (from a calibration capture of known-flat signal)
2. Compute ratio: `current_psd / reference_psd`
3. Apply inverse filter to equalize

---

## SignalDescription Updates

Transforms that change signal properties update `desc` accordingly:

| Transform | Fields updated |
|-----------|---------------|
| `Resample` | Sample rate metadata (if `desc` carries it) |
| `BandpassAlign` | `f_low`, `f_high` (if present) |
| All others | No `desc` changes (normalize without changing semantic meaning) |

---

## Optional Dependency

```toml
# pyproject.toml addition
[project.optional-dependencies]
alignment = ["scipy>=1.10"]
```

Only `Resample` requires scipy. All other transforms are pure NumPy. The scipy import is lazy:

```python
class Resample(Transform):
    def __init__(self, target_sample_rate: float):
        try:
            from scipy.signal import resample_poly  # noqa: F401
        except ImportError:
            raise ImportError(
                "Resample requires scipy. Install with: pip install spectra[alignment]"
            )
        self._target_sample_rate = target_sample_rate
```

---

## Test Plan

**New file:** `tests/test_transforms_alignment.py`

### Per-transform tests

| Transform | Tests |
|-----------|-------|
| `Resample` | Output length matches expected ratio; round-trip (2x up then 2x down) ≈ original within tolerance; dtype preserved; desc sample rate updated |
| `PowerNormalize` | Output RMS matches target (within 0.1 dB); zero-power input returns unchanged; dtype preserved |
| `AGCNormalize` | RMS mode → unit power; peak mode → max(abs) = target_level; zero-amplitude input safe |
| `DCRemove` | Output mean ≈ 0 (within 1e-6); preserves signal shape |
| `ClipNormalize` | Output bounded to [-1, 1]; outliers removed; dtype preserved |
| `SpectralWhitening` | Output PSD variance < input PSD variance (flatter); signal energy preserved within tolerance |
| `NoiseFloorMatch` | Noise floor of output ≈ target (within 2 dB) |
| `BandpassAlign` | Energy concentrated at target center frequency; out-of-band energy suppressed |
| `NoiseProfileTransfer` | Raises `NotImplementedError` |
| `ReceiverEQ` | Raises `NotImplementedError` |

### Integration tests

- `Compose` chain: `DCRemove → AGCNormalize → SpectralWhitening` produces valid IQ
- All transforms accept `complex64` and return `complex64`
- All transforms are deterministic (no internal randomness)

---

## Scope Exclusions

- **Evaluation toolkit / DuckDB / TUI** — scoped to pt-model-lib
- **External dataset adapters** (RadioML, DeepSig) — separate future effort
- **MIMO** — separate future effort
- **Model export** (ONNX/TorchScript) — belongs in pt-model-lib
