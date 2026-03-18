# Range+Doppler 2D Tracker — Design Spec

**Goal:** Extend the existing `RadarPipelineDataset` and Kalman tracker to use both range and Doppler measurements, improving velocity estimation by exploiting the range-Doppler map that is already computed but currently collapsed to 1D.

**Scope:** Add a `RangeDopplerKF` factory function and a `track_doppler` flag to `RadarPipelineDataset`. No new files — only modifications to existing modules. Backward compatible (default behavior unchanged).

---

## Architecture

The existing pipeline computes a range-Doppler map (RDM) via `doppler_filter_bank()` but immediately collapses it to a 1D range profile via `np.max(rdm, axis=0)`. When `track_doppler=True`:

1. CFAR still runs on the 1D range profile (reuses existing `ca_cfar`/`os_cfar`)
2. For each detected range bin, extract the Doppler bin with maximum power (argmax over Doppler dimension)
3. Feed `[range_bin, doppler_bin]` as a 2D measurement to `RangeDopplerKF`

The Doppler measurement directly constrains `range_rate` through the physics: `doppler_bin = range_rate * (2 * pri * pulses_per_cpi / wavelength)`.

---

## Sub-project 1: `RangeDopplerKF` Factory

### Files

- `python/spectra/tracking/kalman.py` (modify — add factory)
- `python/spectra/tracking/__init__.py` (modify — add export)

### `RangeDopplerKF` Function

```python
def RangeDopplerKF(
    dt: float,
    wavelength: float,
    pri: float,
    pulses_per_cpi: int,
    process_noise_std: float,
    range_noise_std: float,
    doppler_noise_std: float,
    x0: Optional[np.ndarray] = None,
    P0: Optional[np.ndarray] = None,
) -> KalmanFilter:
```

**State:** `[range, range_rate]` (n=2) — same as `ConstantVelocityKF`.

**Measurement:** `[range_bin, doppler_bin]` (m=2).

**Matrices:**
- `F = [[1, dt], [0, 1]]` — constant-velocity dynamics (same as CV)
- `H = [[1, 0], [0, doppler_scale]]` where `doppler_scale = 2 * pri * pulses_per_cpi / wavelength`
- `Q = process_noise_std^2 * [[dt^4/4, dt^3/2], [dt^3/2, dt^2]]` — discrete white noise acceleration
- `R = diag(range_noise_std^2, doppler_noise_std^2)`

The key insight: Doppler bin and range_rate are physically coupled via `doppler_scale`, so the `H` matrix encodes this relationship. Two measurements constraining the same 2-state system yields faster convergence and better velocity estimates than range-only tracking.

---

## Sub-project 2: Pipeline Modifications

### Files

- `python/spectra/datasets/radar_pipeline.py` (modify)

### New Constructor Parameter

```python
track_doppler: bool = False
```

When `False` (default), pipeline behavior is completely unchanged.

### New `RadarPipelineTarget` Field

```python
doppler_detections: Optional[List[np.ndarray]] = None
```

Per-frame Doppler bin indices, parallel to `detections` (range bins). `None` when `track_doppler=False`.

### Modified `__getitem__` (only when `track_doppler=True`)

**Detection extraction:** After CFAR produces `det_bins` from the range profile, extract the corresponding Doppler bin for each detection:

```python
doppler_bins = np.array([np.argmax(rdm[:, rb]) for rb in det_bins])
```

**Tracker selection:** Use `RangeDopplerKF` instead of `ConstantVelocityKF`:

```python
kf = RangeDopplerKF(
    dt=self.pri * self.pulses_per_cpi,
    wavelength=wavelength,
    pri=self.pri,
    pulses_per_cpi=self.pulses_per_cpi,
    process_noise_std=1.0,
    range_noise_std=5.0,
    doppler_noise_std=2.0,
    x0=np.array([true_ranges[0, k], true_velocities[0, k]]),
)
```

**2D association:** For each frame, find the nearest detection using Euclidean distance in `(range, doppler)` space:

```python
distances = np.sqrt((dets_range - pred_range)**2 + (dets_doppler - pred_doppler)**2)
nearest_idx = np.argmin(distances)
z = np.array([dets_range[nearest_idx], dets_doppler[nearest_idx]])
kf.update(z)
```

Where `pred_doppler = H[1, 1] * predicted_state[1]` (predicted Doppler from predicted range_rate).

### Output

- `kf_states` shape remains `(sequence_length, num_targets, 2)` — same state dimension
- `doppler_detections` is a new `List[np.ndarray]` parallel to `detections`
- All other fields unchanged

---

## Build Order

```
Sub-project 1: RangeDopplerKF factory    (no dependencies beyond existing KalmanFilter)
Sub-project 2: Pipeline modifications    (depends on 1)
```

---

## Future Work (Feature 2 — separate spec)

- **Range + Azimuth tracker:** state `[range, range_rate, azimuth, azimuth_rate]` (4-state), requires fusing antenna array processing with radar range processing in a new `ArrayRadarPipelineDataset`
- **2D CA-CFAR:** proper 2D adaptive threshold on the RDM (replaces sequential approach)
- **Extended Kalman Filter:** for nonlinear azimuth state evolution in the range+azimuth tracker
