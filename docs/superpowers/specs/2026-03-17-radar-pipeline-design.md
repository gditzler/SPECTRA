# Radar Processing Pipeline — Design Spec

**Goal:** Add an end-to-end radar processing pipeline to SPECTRA: waveform → channel (clutter + target RCS) → receiver (matched filter, CFAR, MTI) → tracker (Kalman filter), with a configurable `RadarPipelineDataset` that produces training data for radar ML tasks.

**Scope:** v1 targets range-only processing with a linear Kalman filter. The tracker and dataset are designed with arbitrary state dimensions so extending to range+azimuth (v2) requires only changing the state/measurement matrices.

---

## Architecture

The pipeline is decomposed into 5 independent sub-projects built in order:

```
Waveform → Target RCS → Clutter → Matched Filter → MTI → CFAR → Kalman Tracker
(existing)  (new)        (new)     (existing)        (new)  (existing) (new)
```

### Package Layout (Hybrid)

| Package | Responsibility |
|---------|---------------|
| `spectra/targets/` (new) | Target kinematics (trajectories) and RCS fluctuation models |
| `spectra/impairments/clutter.py` (new file) | Radar clutter as a composable `Transform` |
| `spectra/algorithms/mti.py` (new file) | MTI pulse cancellers and Doppler filter bank |
| `spectra/tracking/` (new) | General-purpose Kalman filter + radar convenience presets |
| `spectra/datasets/radar_pipeline.py` (new file) | End-to-end pipeline dataset |

**Rationale:** Clutter is a channel effect and belongs with other impairments (composable via `Compose`). Target kinematics and RCS model scene objects, not channel corruption, so they get their own module. The tracker is a new domain (state estimation) that doesn't fit in algorithms or datasets.

---

## Sub-project 1: Target Kinematics & RCS (`spectra/targets/`)

### Files

- `spectra/targets/__init__.py`
- `spectra/targets/trajectory.py`
- `spectra/targets/rcs.py`

### `trajectory.py` — Motion Models

Two motion model classes sharing a common interface:

**`ConstantVelocity(initial_range, velocity, dt)`**
- State: `[range, range_rate]`
- Propagation: `range += velocity * dt`
- Methods:
  - `.state_at(step: int) -> np.ndarray` — ground-truth state at time step
  - `.states(num_steps: int) -> np.ndarray` — full trajectory, shape `(num_steps, 2)`
  - `.range_at(step: int) -> float` — convenience accessor

**`ConstantTurnRate(initial_range, velocity, turn_rate, dt)`**
- State: `[range, range_rate]` with sinusoidal range evolution from turning geometry
- Same interface as `ConstantVelocity`

### `rcs.py` — Swerling RCS Models

**`NonFluctuatingRCS(sigma)`** — Swerling 0/V. Constant cross-section.

**`SwerlingRCS(case, sigma, num_pulses_per_dwell)`** — Cases I–IV:

| Case | Fluctuation | Distribution |
|------|------------|-------------|
| I | Scan-to-scan (constant within dwell) | Chi-squared, 2 DoF (exponential) |
| II | Pulse-to-pulse | Chi-squared, 2 DoF |
| III | Scan-to-scan (constant within dwell) | Chi-squared, 4 DoF |
| IV | Pulse-to-pulse | Chi-squared, 4 DoF |

Common method: `.amplitudes(num_dwells, rng) -> np.ndarray` — returns amplitude sequence of appropriate shape.

---

## Sub-project 2: Radar Clutter (`spectra/impairments/clutter.py`)

### Files

- `spectra/impairments/clutter.py` (new)
- `spectra/impairments/__init__.py` (modify — add export)

### `RadarClutter` Class

Follows the existing `Transform` interface: `__call__(iq, desc, **kwargs) -> (iq, desc)`.

**Core parameters:**
- `cnr: float` — clutter-to-noise ratio in dB
- `doppler_spread: float` — clutter Doppler spectral width in Hz
- `doppler_center: float` — center Doppler frequency (default 0.0 for ground clutter)
- `range_extent: Optional[Tuple[int, int]]` — range bin span (default: full extent)
- `spectral_shape: str` — `"gaussian"` (default) or `"exponential"`

**Mechanism:** Generates colored complex Gaussian noise with the specified Doppler power spectrum. For each range bin in `range_extent`, draws noise filtered to match `(doppler_center, doppler_spread)` spectral shape, scaled to `cnr` relative to the thermal noise floor.

**Presets (class methods):**
- `.ground(terrain="rural")` — low Doppler spread. Terrain types: `"rural"`, `"urban"`, `"forest"`, `"desert"`.
- `.sea(sea_state=3)` — moderate Doppler spread from wave motion. Sea states 1–6.
- `.weather(rain_rate_mmhr=10)` — nonzero Doppler center (moving precipitation), spread and CNR from rain rate.

Presets set physically-motivated defaults; users can override individual parameters.

---

## Sub-project 3: MTI Processing (`spectra/algorithms/mti.py`)

### Files

- `spectra/algorithms/mti.py` (new)
- `spectra/algorithms/__init__.py` (modify — add exports)

Three stateless functions, matching the pattern of `matched_filter`, `ca_cfar`, `os_cfar`:

**`single_pulse_canceller(pulses: np.ndarray) -> np.ndarray`**
- Input: `(num_pulses, num_range_bins)` complex
- Output: `(num_pulses - 1, num_range_bins)` — `y[n] = x[n+1] - x[n]`
- Nulls zero-Doppler clutter

**`double_pulse_canceller(pulses: np.ndarray) -> np.ndarray`**
- Input: `(num_pulses, num_range_bins)` complex
- Output: `(num_pulses - 2, num_range_bins)` — `y[n] = x[n+2] - 2*x[n+1] + x[n]`
- Deeper null at zero Doppler

**`doppler_filter_bank(pulses: np.ndarray, num_doppler_bins: Optional[int] = None, window: str = "hann") -> np.ndarray`**
- Input: `(num_pulses, num_range_bins)` complex
- Output: `(num_doppler_bins, num_range_bins)` — range-Doppler map (magnitude squared)
- FFT along pulse dimension (slow-time) with optional windowing
- `num_doppler_bins` defaults to `num_pulses`; larger values zero-pad

---

## Sub-project 4: Kalman Tracker (`spectra/tracking/`)

### Files

- `spectra/tracking/__init__.py`
- `spectra/tracking/kalman.py`

### `KalmanFilter(F, H, Q, R, x0=None, P0=None)` — Generic Linear KF

State dimension is arbitrary. No radar-specific logic.

**Constructor:**
- `F`: state transition `(n, n)`
- `H`: measurement matrix `(m, n)`
- `Q`: process noise covariance `(n, n)`
- `R`: measurement noise covariance `(m, m)`
- `x0`: initial state `(n,)`, defaults to zeros
- `P0`: initial covariance `(n, n)`, defaults to identity

**Methods:**
- `.predict() -> np.ndarray` — propagate state + covariance, return predicted state
- `.update(z: np.ndarray) -> np.ndarray` — incorporate measurement, return updated state
- `.step(z: np.ndarray) -> np.ndarray` — predict + update
- `.run(measurements: np.ndarray) -> np.ndarray` — batch process `(T, m)` → `(T, n)` state history
- `.state` / `.covariance` — property accessors for current estimate

### `ConstantVelocityKF(dt, process_noise_std, measurement_noise_std)` — Convenience Factory

Returns a configured `KalmanFilter` instance (function, not subclass) with:
- `F = [[1, dt], [0, 1]]`
- `H = [[1, 0]]` (observe range only)
- `Q` from discrete white noise acceleration model scaled by `process_noise_std`
- `R = [[measurement_noise_std**2]]`

---

## Sub-project 5: Radar Pipeline Dataset (`spectra/datasets/radar_pipeline.py`)

### Files

- `spectra/datasets/radar_pipeline.py` (new)
- `spectra/datasets/__init__.py` (modify — add exports)

### `RadarPipelineTarget` Dataclass

```
true_ranges:     np.ndarray  # (sequence_length, num_targets)
true_velocities: np.ndarray  # (sequence_length, num_targets)
rcs_amplitudes:  np.ndarray  # (sequence_length, num_targets)
detections:      List[np.ndarray]  # per-frame CFAR detection bin indices
kf_states:       np.ndarray  # (sequence_length, state_dim)
num_targets:     int
waveform_label:  str
snr_db:          float       # nominal target SNR before clutter
clutter_preset:  str         # e.g. "sea_state_3"
```

### `RadarPipelineDataset` Constructor

| Parameter | Type | Description |
|-----------|------|-------------|
| `waveform_pool` | `List[Waveform]` | Radar waveforms to draw from |
| `trajectory_pool` | `List` | CV and/or CT trajectory configs |
| `swerling_cases` | `List[int]` | RCS cases to sample from (default `[0,1,2,3,4]`) |
| `clutter_presets` | `List[RadarClutter]` | Clutter configurations to sample from |
| `num_range_bins` | `int` | Range profile length |
| `sample_rate` | `float` | Receiver sample rate |
| `pri` | `float` | Pulse repetition interval in seconds |
| `snr_range` | `Tuple[float, float]` | Target SNR range in dB |
| `num_targets_range` | `Tuple[int, int]` | Min/max targets per sample |
| `sequence_length` | `int` | CPIs per sample (1 = single-frame mode) |
| `pulses_per_cpi` | `int` | Pulses per coherent processing interval |
| `apply_mti` | `bool` | Apply pulse cancellation before CFAR (default `True`) |
| `cfar_type` | `str` | `"ca"` or `"os"` (default `"ca"`) |
| `num_samples` | `int` | Dataset size |
| `seed` | `int` | Base seed |

### `__getitem__` Pipeline

For each sample (deterministically seeded as `(seed, idx)`):

1. **Sample configuration:** Pick waveform, trajectory, Swerling case, clutter preset
2. **Per-CPI loop** (for each frame in `sequence_length`):
   a. Generate pulse train from waveform
   b. Inject target returns at ground-truth range bins with Swerling RCS amplitude
   c. Apply clutter via `RadarClutter` transform
   d. Matched filter each pulse
   e. Apply MTI (if `apply_mti`) to get clutter-suppressed pulses
   f. Run `doppler_filter_bank` → range-Doppler map
   g. Collapse to range profile (max over Doppler), apply CFAR
3. **Track:** Run `KalmanFilter` over the sequence of CFAR detections (nearest-to-prediction association)
4. **Return:** `(Tensor[sequence_length, num_range_bins], RadarPipelineTarget)`

### Output Modes

- `sequence_length=1`: single-frame mode for detection/classification tasks
- `sequence_length>1`: sequence mode for tracking/prediction tasks (temporal ML)

Requires custom `collate_fn` (same pattern as `RadarDataset`).

---

## Build Order & Dependencies

```
Sub-project 1: targets/         (no dependencies)
Sub-project 2: impairments/     (no dependencies — clutter is standalone)
Sub-project 3: algorithms/mti   (no dependencies)
Sub-project 4: tracking/        (no dependencies)
Sub-project 5: datasets/        (depends on 1–4)
```

Sub-projects 1–4 are independent and can be built in any order (or in parallel). Sub-project 5 integrates everything.

---

## Future Work (out of scope for v1)

- **Range + Azimuth tracker (v2):** Extend `KalmanFilter` state to `[range, range_rate, azimuth, azimuth_rate]`, add array processing to the pipeline
- **STAP:** Space-Time Adaptive Processing for joint spatial-temporal clutter suppression
- **Particle filter:** Nonlinear tracker for maneuvering targets
- **Multi-target association:** Hungarian algorithm or JPDA for crossing targets
- **Extended Kalman Filter:** For constant-turn-rate segments where the state transition is nonlinear
