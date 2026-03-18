# Range+Doppler 2D Tracker Implementation Plan

> **For agentic workers:** REQUIRED: Use superpowers:subagent-driven-development (if subagents available) or superpowers:executing-plans to implement this plan. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Extend the radar pipeline's Kalman tracker to use both range and Doppler measurements from the range-Doppler map, improving velocity estimation.

**Architecture:** Add a `RangeDopplerKF` factory function (2-state, 2-measurement) and a `track_doppler` flag to `RadarPipelineDataset`. When enabled, extracts `(range_bin, centered_doppler_bin)` pairs from the RDM and uses Mahalanobis-gated 2D association. No new files — only modifications to existing modules. Fully backward compatible.

**Tech Stack:** Python 3.10+, NumPy, pytest. No new dependencies.

**Spec:** `docs/superpowers/specs/2026-03-18-range-doppler-tracker-design.md`

---

## File Structure

| File | Change |
|------|--------|
| `python/spectra/tracking/kalman.py` | Add `RangeDopplerKF` factory function after `ConstantVelocityKF` |
| `python/spectra/tracking/__init__.py` | Add `RangeDopplerKF` to exports |
| `python/spectra/datasets/radar_pipeline.py` | Add `track_doppler` param, `doppler_detections` field, 2D detection + Mahalanobis tracking |
| `tests/test_kalman.py` | Add tests for `RangeDopplerKF` |
| `tests/test_radar_pipeline.py` | Add tests for `track_doppler=True` |

---

## Task 1: `RangeDopplerKF` Factory

**Files:**
- Modify: `python/spectra/tracking/kalman.py`
- Modify: `python/spectra/tracking/__init__.py`
- Modify: `tests/test_kalman.py`

- [ ] **Step 1: Write the failing tests**

Append to `tests/test_kalman.py`:

```python
def test_range_doppler_kf_factory():
    from spectra.tracking.kalman import RangeDopplerKF, KalmanFilter
    kf = RangeDopplerKF(
        dt=0.016, wavelength=0.03, pri=1e-3, pulses_per_cpi=16,
        process_noise_std=1.0, range_noise_std=5.0, doppler_noise_std=2.0,
    )
    assert isinstance(kf, KalmanFilter)
    assert kf.state.shape == (2,)
    # H should be (2, 2): range + doppler measurements
    assert kf.measurement_matrix.shape == (2, 2)


def test_range_doppler_kf_2d_measurement():
    from spectra.tracking.kalman import RangeDopplerKF
    kf = RangeDopplerKF(
        dt=0.016, wavelength=0.03, pri=1e-3, pulses_per_cpi=16,
        process_noise_std=1.0, range_noise_std=5.0, doppler_noise_std=2.0,
        x0=np.array([100.0, 5.0]),
    )
    pred = kf.predict()
    assert pred.shape == (2,)
    # Update with 2D measurement [range_bin, doppler_bin]
    updated = kf.update(np.array([101.0, 3.0]))
    assert updated.shape == (2,)
    # Should move toward the measurement
    assert updated[0] > 100.0


def test_range_doppler_kf_velocity_converges():
    """With both range and Doppler measurements, velocity should converge faster."""
    from spectra.tracking.kalman import RangeDopplerKF
    wavelength = 0.03  # 10 GHz
    pri = 1e-3
    pulses_per_cpi = 16
    dt = pri * pulses_per_cpi
    doppler_scale = 2 * pri * pulses_per_cpi / wavelength
    true_velocity = 5.0
    true_doppler = true_velocity * doppler_scale

    kf = RangeDopplerKF(
        dt=dt, wavelength=wavelength, pri=pri, pulses_per_cpi=pulses_per_cpi,
        process_noise_std=0.5, range_noise_std=3.0, doppler_noise_std=1.0,
        x0=np.array([100.0, 0.0]),  # start with wrong velocity
    )

    rng = np.random.default_rng(42)
    for t in range(1, 30):
        true_range = 100.0 + true_velocity * t * dt
        z = np.array([
            true_range + rng.normal(0, 3),
            true_doppler + rng.normal(0, 1),
        ])
        kf.step(z)

    # Velocity estimate should be close to truth
    assert abs(kf.state[1] - true_velocity) < 2.0


def test_range_doppler_kf_doppler_scale():
    """Verify the H matrix correctly maps velocity to Doppler bin."""
    from spectra.tracking.kalman import RangeDopplerKF
    wavelength = 0.03
    pri = 1e-3
    pulses_per_cpi = 32
    kf = RangeDopplerKF(
        dt=pri * pulses_per_cpi, wavelength=wavelength, pri=pri,
        pulses_per_cpi=pulses_per_cpi,
        process_noise_std=1.0, range_noise_std=5.0, doppler_noise_std=2.0,
        x0=np.array([0.0, 10.0]),  # velocity = 10
    )
    # Predicted measurement should be [range, velocity * doppler_scale]
    pred = kf.predict()
    expected_doppler = 10.0 * (2 * pri * pulses_per_cpi / wavelength)
    pred_measurement = kf.measurement_matrix @ pred
    assert pred_measurement[1] == pytest.approx(expected_doppler, rel=1e-6)
```

- [ ] **Step 2: Run tests to verify they fail**

```bash
/Users/gditzler/.venvs/base/bin/pytest tests/test_kalman.py::test_range_doppler_kf_factory -v
```
Expected: `ImportError` (RangeDopplerKF doesn't exist)

- [ ] **Step 3: Add public properties to `KalmanFilter`**

Add two read-only properties to the `KalmanFilter` class (after the existing `covariance` property at line 51):

```python
    @property
    def measurement_matrix(self) -> np.ndarray:
        """Measurement matrix H, shape ``(m, n)``."""
        return self._H.copy()

    @property
    def measurement_noise(self) -> np.ndarray:
        """Measurement noise covariance R, shape ``(m, m)``."""
        return self._R.copy()
```

- [ ] **Step 4: Write the `RangeDopplerKF` factory**

Add to `python/spectra/tracking/kalman.py` after the `ConstantVelocityKF` function:

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
    """Create a range+Doppler Kalman filter for 2D radar tracking.

    State: ``[range, range_rate]`` (same as CV).
    Measurement: ``[range_bin, centered_doppler_idx]``.

    The Doppler measurement constrains ``range_rate`` via the physics:
    ``centered_doppler_idx = range_rate * doppler_scale`` where
    ``doppler_scale = 2 * pri * pulses_per_cpi / wavelength``.

    Args:
        dt: Time step between measurements in seconds.
        wavelength: Carrier wavelength in metres (``3e8 / carrier_freq``).
        pri: Pulse repetition interval in seconds.
        pulses_per_cpi: Pulses per coherent processing interval.
        process_noise_std: Acceleration process noise standard deviation.
        range_noise_std: Range measurement noise standard deviation (bins).
        doppler_noise_std: Doppler measurement noise standard deviation (bins).
        x0: Initial state ``(2,)``. Defaults to zeros.
        P0: Initial covariance ``(2, 2)``. Defaults to identity.

    Returns:
        Configured :class:`KalmanFilter` instance.
    """
    F = np.array([[1.0, dt], [0.0, 1.0]])

    doppler_scale = 2.0 * pri * pulses_per_cpi / wavelength
    H = np.array([[1.0, 0.0], [0.0, doppler_scale]])

    q = process_noise_std**2
    Q = q * np.array([
        [dt**4 / 4, dt**3 / 2],
        [dt**3 / 2, dt**2],
    ])

    R = np.array([
        [range_noise_std**2, 0.0],
        [0.0, doppler_noise_std**2],
    ])

    return KalmanFilter(F, H, Q, R, x0=x0, P0=P0)
```

- [ ] **Step 5: Update `tracking/__init__.py`**

Change to:
```python
from spectra.tracking.kalman import ConstantVelocityKF, KalmanFilter, RangeDopplerKF

__all__ = ["ConstantVelocityKF", "KalmanFilter", "RangeDopplerKF"]
```

- [ ] **Step 6: Run tests to verify they pass**

```bash
/Users/gditzler/.venvs/base/bin/pytest tests/test_kalman.py -v
```
Expected: 12/12 PASS (8 existing + 4 new)

- [ ] **Step 7: Commit**

```bash
git add python/spectra/tracking/kalman.py python/spectra/tracking/__init__.py tests/test_kalman.py
git commit -m "feat(tracking): add RangeDopplerKF factory for 2D range+Doppler tracking"
```

---

## Task 2: Pipeline Modifications

**Files:**
- Modify: `python/spectra/datasets/radar_pipeline.py`
- Modify: `tests/test_radar_pipeline.py`

- [ ] **Step 1: Write the failing tests**

Append to `tests/test_radar_pipeline.py`:

```python
def test_pipeline_track_doppler_output():
    ds = _make_pipeline_ds(track_doppler=True, sequence_length=5)
    data, target = ds[0]
    assert isinstance(data, torch.Tensor)
    assert target.doppler_detections is not None
    assert len(target.doppler_detections) == 5
    # kf_states shape unchanged: (seq_len, n_targets, 2)
    assert target.kf_states.shape[0] == 5
    assert target.kf_states.shape[2] == 2


def test_pipeline_track_doppler_false_backward_compat():
    ds = _make_pipeline_ds(track_doppler=False)
    _, target = ds[0]
    assert target.doppler_detections is None


def test_pipeline_track_doppler_default_is_false():
    ds = _make_pipeline_ds()
    _, target = ds[0]
    assert target.doppler_detections is None


def test_pipeline_track_doppler_deterministic():
    ds = _make_pipeline_ds(track_doppler=True)
    d1, t1 = ds[3]
    d2, t2 = ds[3]
    assert torch.allclose(d1, d2)
    assert np.allclose(t1.kf_states, t2.kf_states)
    for dd1, dd2 in zip(t1.doppler_detections, t2.doppler_detections):
        assert np.array_equal(dd1, dd2)
```

- [ ] **Step 2: Run tests to verify they fail**

```bash
/Users/gditzler/.venvs/base/bin/pytest tests/test_radar_pipeline.py::test_pipeline_track_doppler_output -v
```
Expected: `TypeError` (unexpected keyword `track_doppler`)

- [ ] **Step 3: Modify `RadarPipelineTarget` dataclass**

Add the new field to the `RadarPipelineTarget` dataclass (after `clutter_preset`):

```python
    doppler_detections: Optional[List[np.ndarray]] = None
```

- [ ] **Step 4: Modify `RadarPipelineDataset.__init__`**

Add `track_doppler` parameter after `cfar_type`:

```python
        track_doppler: bool = False,
```

And store it:
```python
        self.track_doppler = track_doppler
```

Also add the import at the top of the file:
```python
from spectra.tracking.kalman import ConstantVelocityKF, RangeDopplerKF
```
(Replace the existing `from spectra.tracking.kalman import ConstantVelocityKF` line.)

- [ ] **Step 5: Modify `__getitem__` — detection extraction**

After line 218 (`det_bins = np.where(det_mask)[0]`), add:

```python
            if self.track_doppler and len(det_bins) > 0:
                N_doppler = rdm.shape[0]
                raw_doppler = np.array([np.argmax(rdm[:, rb]) for rb in det_bins], dtype=int)
                centered_doppler = np.where(raw_doppler < N_doppler // 2, raw_doppler, raw_doppler - N_doppler)
                all_doppler_detections.append(centered_doppler)
            elif self.track_doppler:
                all_doppler_detections.append(np.array([], dtype=int))
```

Also initialise `all_doppler_detections = []` alongside `all_detections = []` (before the CPI loop), and store `rdm` for Doppler extraction (it's already computed at line 209).

- [ ] **Step 6: Modify `__getitem__` — tracker selection and 2D association**

Replace the tracking block (lines 222-242) with:

```python
        state_dim = 2
        kf_states = np.zeros((self.sequence_length, n_targets, state_dim))

        for k in range(n_targets):
            if self.track_doppler:
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
            else:
                kf = ConstantVelocityKF(
                    dt=self.pri * self.pulses_per_cpi,
                    process_noise_std=1.0,
                    measurement_noise_std=5.0,
                    x0=np.array([true_ranges[0, k], true_velocities[0, k]]),
                )

            for frame in range(self.sequence_length):
                predicted = kf.predict()
                dets_range = all_detections[frame]

                if len(dets_range) > 0:
                    if self.track_doppler:
                        dets_doppler = all_doppler_detections[frame]
                        # Mahalanobis-gated 2D association
                        H = kf.measurement_matrix
                        R = kf.measurement_noise
                        pred_z = H @ predicted
                        S = H @ kf.covariance @ H.T + R
                        S_inv = np.linalg.inv(S)
                        best_dist = float("inf")
                        best_idx = -1
                        for j in range(len(dets_range)):
                            innov = np.array([dets_range[j], dets_doppler[j]]) - pred_z
                            d = float(innov @ S_inv @ innov)
                            if d < best_dist:
                                best_dist = d
                                best_idx = j
                        gate_threshold = 9.21  # chi-squared, 2 DoF, 99%
                        if best_dist < gate_threshold and best_idx >= 0:
                            kf.update(np.array([dets_range[best_idx], dets_doppler[best_idx]]))
                    else:
                        pred_range = predicted[0]
                        nearest_idx = np.argmin(np.abs(dets_range - pred_range))
                        nearest_det = float(dets_range[nearest_idx])
                        gate = 20.0
                        if abs(nearest_det - pred_range) < gate:
                            kf.update(np.array([nearest_det]))

                kf_states[frame, k] = kf.state
```

- [ ] **Step 7: Modify `__getitem__` — output construction**

In the `RadarPipelineTarget` construction, add:

```python
            doppler_detections=all_doppler_detections if self.track_doppler else None,
```

- [ ] **Step 8: Run tests to verify they pass**

```bash
/Users/gditzler/.venvs/base/bin/pytest tests/test_radar_pipeline.py -v
```
Expected: 13/13 PASS (9 existing + 4 new)

- [ ] **Step 9: Commit**

```bash
git add python/spectra/datasets/radar_pipeline.py tests/test_radar_pipeline.py
git commit -m "feat(datasets): add track_doppler mode to RadarPipelineDataset"
```

---

## Task 3: Full Verification

- [ ] **Step 1: Run all tests**

```bash
/Users/gditzler/.venvs/base/bin/pytest tests/ -q --tb=short
```
Expected: all tests pass, no regressions.

- [ ] **Step 2: Verify imports**

```bash
/Users/gditzler/.venvs/base/bin/python -c "
from spectra.tracking import RangeDopplerKF, KalmanFilter, ConstantVelocityKF
print('RangeDopplerKF imported OK')

# Quick smoke test
import numpy as np
kf = RangeDopplerKF(dt=0.016, wavelength=0.03, pri=1e-3, pulses_per_cpi=16,
                     process_noise_std=1.0, range_noise_std=5.0, doppler_noise_std=2.0,
                     x0=np.array([100.0, 5.0]))
pred = kf.predict()
updated = kf.update(np.array([101.0, 3.0]))
print(f'Predict: {pred}, Update: {updated}')
print('All OK')
"
```

- [ ] **Step 3: Verify backward compatibility**

```bash
/Users/gditzler/.venvs/base/bin/python -c "
from spectra.datasets import RadarPipelineDataset
from spectra.targets import ConstantVelocity
from spectra.impairments import RadarClutter
from spectra.waveforms import LFM
import numpy as np

ds = RadarPipelineDataset(
    waveform_pool=[LFM()],
    trajectory_pool=[ConstantVelocity(initial_range=100.0, velocity=0.5, dt=1.0)],
    swerling_cases=[0],
    clutter_presets=[RadarClutter.ground(sample_rate=1e6, terrain='rural')],
    num_range_bins=256, sample_rate=1e6, carrier_frequency=10e9,
    pri=1e-3, snr_range=(10.0, 20.0), num_targets_range=(1, 1),
    sequence_length=3, pulses_per_cpi=16, num_samples=5, seed=42,
)
_, target = ds[0]
assert target.doppler_detections is None, 'Default should be None'
print('Backward compatibility OK')

# Now with track_doppler=True
ds2 = RadarPipelineDataset(
    waveform_pool=[LFM()],
    trajectory_pool=[ConstantVelocity(initial_range=100.0, velocity=0.5, dt=1.0)],
    swerling_cases=[0],
    clutter_presets=[RadarClutter.ground(sample_rate=1e6, terrain='rural')],
    num_range_bins=256, sample_rate=1e6, carrier_frequency=10e9,
    pri=1e-3, snr_range=(10.0, 20.0), num_targets_range=(1, 1),
    sequence_length=3, pulses_per_cpi=16, num_samples=5, seed=42,
    track_doppler=True,
)
_, target2 = ds2[0]
assert target2.doppler_detections is not None
print(f'Doppler tracking: {len(target2.doppler_detections)} frames with detections')
print('All OK')
"
```
