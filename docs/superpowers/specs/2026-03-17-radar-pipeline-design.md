# Radar Processing Pipeline — Design Spec

**Goal:** Add an end-to-end radar processing pipeline to SPECTRA: waveform → channel (clutter + target RCS) → receiver (matched filter, CFAR, MTI) → tracker (Kalman filter), with a configurable `RadarPipelineDataset` that produces training data for radar ML tasks.

**Scope:** v1 targets range-only processing with a linear Kalman filter. The tracker and dataset are designed with arbitrary state dimensions so extending to range+azimuth (v2) requires only changing the state/measurement matrices.

---

## Architecture

The pipeline is decomposed into 5 independent sub-projects built in order:

```
Waveform → Target RCS → AWGN + Clutter → Matched Filter → MTI → CFAR → Kalman Tracker
(existing)  (new)        (new)            (existing)        (new)  (existing) (new)
```

### Package Layout (Hybrid)

| Package | Responsibility |
|---------|---------------|
| `spectra/targets/` (new) | Target kinematics (trajectories) and RCS fluctuation models |
| `spectra/impairments/clutter.py` (new file) | Radar clutter as a standalone callable (not a `Transform` subclass) |
| `spectra/algorithms/mti.py` (new file) | MTI pulse cancellers and Doppler filter bank |
| `spectra/tracking/` (new) | General-purpose Kalman filter + radar convenience presets |
| `spectra/datasets/radar_pipeline.py` (new file) | End-to-end pipeline dataset |

**Rationale:** Clutter is a channel effect and lives alongside other impairments, but it operates on 2-D slow-time/fast-time matrices rather than 1-D IQ streams, so it does not subclass the `Transform` ABC. Target kinematics and RCS model scene objects, not channel corruption, so they get their own module. The tracker is a new domain (state estimation) that doesn't fit in algorithms or datasets.

---

## Sub-project 1: Target Kinematics & RCS (`spectra/targets/`)

### Files

- `spectra/targets/__init__.py`
- `spectra/targets/trajectory.py`
- `spectra/targets/rcs.py`

### `trajectory.py` — Motion Models

A `Trajectory` protocol (runtime-checkable) defines the common interface:

```python
class Trajectory(Protocol):
    def state_at(self, step: int) -> np.ndarray: ...
    def states(self, num_steps: int) -> np.ndarray: ...
    def range_at(self, step: int) -> float: ...
```

Two implementations:

**`ConstantVelocity(initial_range, velocity, dt)`**
- State: `[range, range_rate]`
- Propagation: `range(t) = initial_range + velocity * t`
- `range_rate` is constant at `velocity`

**`ConstantTurnRate(initial_range, velocity, turn_rate, dt)`**
- Models a target at constant speed on a circular arc, observed from a fixed radar
- State: `[range, range_rate]`
- Propagation equations (1-D projection of 2-D turning motion):
  ```
  x(t) = x0 + (v / omega) * sin(omega * t)
  y(t) = y0 + (v / omega) * (1 - cos(omega * t))
  range(t) = sqrt(x(t)^2 + y(t)^2)
  range_rate(t) = d/dt range(t)
  ```
  where `omega = turn_rate` in rad/s, `v = velocity`, and `(x0, y0)` is derived from `initial_range` by placing the target at `(initial_range, 0)`.
- This produces a time-varying range with sinusoidal character, physically meaningful in 1-D range-only scenarios.

### `rcs.py` — Swerling RCS Models

**`NonFluctuatingRCS(sigma)`** — Swerling 0/V. Constant cross-section.
- `.amplitudes(num_dwells, num_pulses_per_dwell, rng) -> np.ndarray` — returns `(num_dwells, num_pulses_per_dwell)` filled with `sqrt(sigma)`.

**`SwerlingRCS(case, sigma)`** — Cases I–IV:

| Case | Fluctuation | Distribution |
|------|------------|-------------|
| I | Scan-to-scan (constant within dwell) | Chi-squared, 2 DoF (exponential) |
| II | Pulse-to-pulse | Chi-squared, 2 DoF |
| III | Scan-to-scan (constant within dwell) | Chi-squared, 4 DoF |
| IV | Pulse-to-pulse | Chi-squared, 4 DoF |

Method: `.amplitudes(num_dwells, num_pulses_per_dwell, rng) -> np.ndarray`
- Always returns shape `(num_dwells, num_pulses_per_dwell)`.
- Cases I/III: draw one value per dwell, broadcast across the pulse dimension.
- Cases II/IV: draw independently for each `(dwell, pulse)` entry.
- Values are amplitude scale factors: `sqrt(rcs_draw / sigma)` where `rcs_draw` is from the appropriate chi-squared distribution with mean `sigma`.

Both classes share the same method signature via duck typing (both satisfy a common `RCSModel` protocol).

---

## Sub-project 2: Radar Clutter (`spectra/impairments/clutter.py`)

### Files

- `spectra/impairments/clutter.py` (new)
- `spectra/impairments/__init__.py` (modify — add export)

### `RadarClutter` Class

**Not a `Transform` subclass.** Radar clutter operates on 2-D slow-time/fast-time matrices `(num_pulses, num_range_bins)`, which is incompatible with the 1-D `Transform.__call__(iq, desc)` interface. Instead, `RadarClutter` is a standalone callable:

```python
def __call__(self, pulse_matrix: np.ndarray, rng: np.random.Generator) -> np.ndarray:
    """Add clutter to a (num_pulses, num_range_bins) complex matrix. Returns same shape."""
```

The `rng` parameter ensures deterministic seeding when called from the pipeline dataset.

**Core parameters:**
- `cnr: float` — clutter-to-noise ratio in dB (relative to thermal noise power)
- `doppler_spread: float` — clutter Doppler spectral width in Hz
- `doppler_center: float` — center Doppler frequency (default 0.0 for ground clutter)
- `sample_rate: float` — needed to convert Doppler Hz to normalized frequency
- `range_extent: Optional[Tuple[int, int]]` — range bin span (default: full extent)
- `spectral_shape: str` — `"gaussian"` (default) or `"exponential"`

**Mechanism:** For each range bin in `range_extent`, generates complex Gaussian noise shaped by the specified Doppler power spectral density. The shaping is applied along the pulse (slow-time) dimension via FFT-domain filtering. The clutter power is scaled so that the total clutter power equals `cnr` dB above unit noise power.

**Presets (class methods):**
- `.ground(sample_rate, terrain="rural")` — low Doppler spread. Terrain types: `"rural"`, `"urban"`, `"forest"`, `"desert"`.
- `.sea(sample_rate, sea_state=3)` — moderate Doppler spread from wave motion. Sea states 1–6.
- `.weather(sample_rate, rain_rate_mmhr=10)` — nonzero Doppler center (moving precipitation), spread and CNR from rain rate.

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
detections:      List[np.ndarray]  # per-frame CFAR detection bin indices, len=sequence_length
kf_states:       np.ndarray  # (sequence_length, num_targets, state_dim) — per-target KF tracks
num_targets:     int
waveform_label:  str
snr_db:          float       # nominal target SNR relative to thermal noise (before clutter)
clutter_preset:  str         # e.g. "sea_state_3"
```

### `RadarPipelineDataset` Constructor

| Parameter | Type | Description |
|-----------|------|-------------|
| `waveform_pool` | `List[Waveform]` | Radar waveforms to draw from |
| `trajectory_pool` | `List[Trajectory]` | CV and/or CT trajectory configurations |
| `swerling_cases` | `List[int]` | RCS cases to sample from (default `[0,1,2,3,4]`) |
| `clutter_presets` | `List[RadarClutter]` | Clutter configurations to sample from |
| `num_range_bins` | `int` | Range profile length |
| `sample_rate` | `float` | Receiver sample rate |
| `pri` | `float` | Pulse repetition interval in seconds |
| `snr_range` | `Tuple[float, float]` | Target SNR range in dB (relative to thermal noise) |
| `num_targets_range` | `Tuple[int, int]` | Min/max targets per sample |
| `sequence_length` | `int` | CPIs per sample (1 = single-frame mode) |
| `pulses_per_cpi` | `int` | Pulses per coherent processing interval |
| `apply_mti` | `bool` | Apply pulse cancellation before CFAR (default `True`) |
| `cfar_type` | `str` | `"ca"` or `"os"` (default `"ca"`) |
| `num_samples` | `int` | Dataset size |
| `seed` | `int` | Base seed |

### `__getitem__` Pipeline — Data Flow with Shapes

For each sample (deterministically seeded as `(seed, idx)`):

**Step 1 — Sample configuration:**
Pick waveform, trajectory(ies), Swerling case, clutter preset. One trajectory per target, each with randomised initial conditions.

**Step 2 — Per-CPI loop** (for each frame `f` in `sequence_length`):

a. **Generate pulse train** from waveform: `(pulses_per_cpi, pulse_length)` complex. Each pulse is a replica of the transmitted waveform.

b. **Form received signal matrix:** Reshape into `(pulses_per_cpi, num_range_bins)` by truncating/padding each pulse to `num_range_bins` samples. This is the fast-time/slow-time matrix.

c. **Inject target returns:** For each target `k`, place a delayed copy of the transmitted pulse at range bin `range_bins[k]` (from `trajectory.range_at(f)`), scaled by the Swerling RCS amplitude for this `(dwell, pulse)`. Apply Doppler phase progression: `exp(j * 2 * pi * f_d_k * n * PRI)` on pulse `n`, where `f_d_k = 2 * velocity_k / wavelength`.

d. **Add thermal noise (AWGN):** Add complex Gaussian noise at unit power to the full `(pulses_per_cpi, num_range_bins)` matrix. Target `snr_db` is defined relative to this thermal noise floor.

e. **Add clutter:** Apply `RadarClutter.__call__(pulse_matrix, rng)` to add Doppler-colored clutter. Clutter CNR is relative to the thermal noise power added in step d.

f. **Matched filter:** Convolve each pulse (each row) with the conjugate-reversed transmitted waveform via `matched_filter()`. Output: `(pulses_per_cpi, num_range_bins)` (trimmed to `num_range_bins`).

g. **MTI** (if `apply_mti`): Apply `single_pulse_canceller()` to the matched-filtered matrix. Output: `(pulses_per_cpi - 1, num_range_bins)`.

h. **Doppler filter bank:** Apply `doppler_filter_bank()` to produce range-Doppler map. Output: `(num_doppler_bins, num_range_bins)` power values.

i. **Range profile + CFAR:** Collapse to 1-D range profile via `np.max(rdm, axis=0)` → shape `(num_range_bins,)`. Apply `ca_cfar` or `os_cfar` → boolean detection mask `(num_range_bins,)`. Record detection bin indices.

**Step 3 — Tracking (ground-truth-aided association):**
Run one `KalmanFilter` per target. For each frame, associate the nearest CFAR detection to each target's predicted range (nearest-to-prediction). If no detection is within a gating threshold, use prediction only (no update). This is ground-truth-aided in the sense that we know the number of targets and initialise each KF at the true starting range. Full multi-target association (Hungarian, JPDA) is deferred to v2.

**Step 4 — Return:**
`(Tensor[sequence_length, num_range_bins], RadarPipelineTarget)`

The range profile tensor contains the max-over-Doppler power profile (log-magnitude, normalised to [0, 1]) for each CPI frame.

### Output Modes

- `sequence_length=1`: single-frame mode for detection/classification tasks
- `sequence_length>1`: sequence mode for tracking/prediction tasks (temporal ML)

### Collation

Requires custom `collate_fn`. The existing `spectra.datasets.collate_fn` works: it stacks tensors and collects targets into a list. This is compatible because `RadarPipelineTarget` is a dataclass (not a dict), and the collate function does not introspect target internals — it simply gathers them as `List[RadarPipelineTarget]`.

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
- **Multi-target association:** Hungarian algorithm or JPDA for crossing targets (replace ground-truth-aided association)
- **Extended Kalman Filter:** For constant-turn-rate segments where the state transition is nonlinear
