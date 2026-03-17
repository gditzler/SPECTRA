# Beamforming Module Implementation Plan

> **For agentic workers:** REQUIRED: Use superpowers:subagent-driven-development (if subagents available) or superpowers:executing-plans to implement this plan. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Add `delay_and_sum`, `mvdr`, and `lcmv` beamformers to `spectra.algorithms.beamforming`, with tests and a runnable example (example 14).

**Architecture:** A single `beamforming.py` module in the existing `algorithms/` package. All three functions share the same interface: `(X, array, ...) → np.ndarray` (beamformed time-series of shape `(T,)`). MVDR builds on the same covariance machinery as Capon. LCMV generalises MVDR to multiple simultaneous constraints. The module is exported from `spectra.algorithms.__init__`.

**Tech Stack:** NumPy, PyTorch (for DataLoader in example), pytest, matplotlib.

---

## File Structure

| File | Action | Responsibility |
|------|--------|----------------|
| `python/spectra/algorithms/beamforming.py` | Create | `delay_and_sum`, `mvdr`, `lcmv`, `compute_beam_pattern` |
| `python/spectra/algorithms/__init__.py` | Modify | Export new beamforming functions |
| `tests/test_beamforming.py` | Create | Unit tests for all three beamformers |
| `examples/14_beamforming.py` | Create | Runnable script |
| `examples/14_beamforming.ipynb` | Create | Jupyter notebook |
| `examples/README.md` | Modify | Add example 14 entry |

---

## Task 1: delay_and_sum

**Files:**
- Create: `python/spectra/algorithms/beamforming.py`
- Create: `tests/test_beamforming.py`

- [ ] **Step 1: Write the failing tests**

```python
# tests/test_beamforming.py
"""Tests for delay-and-sum, MVDR, and LCMV beamformers."""
import numpy as np
import pytest
from spectra.arrays.array import ula


def _snapshot(azimuth_rad: float, num_elements: int = 8, snr_db: float = 20.0,
              num_snapshots: int = 512, seed: int = 0) -> np.ndarray:
    """Single-source snapshot matrix (N_elem, T)."""
    rng = np.random.default_rng(seed)
    arr = ula(num_elements=num_elements, spacing=0.5, frequency=1e9)
    sv = arr.steering_vector(azimuth=azimuth_rad, elevation=0.0)
    signal = rng.standard_normal(num_snapshots) + 1j * rng.standard_normal(num_snapshots)
    snr_linear = 10 ** (snr_db / 10.0)
    scale = np.sqrt(snr_linear / np.mean(np.abs(signal) ** 2))
    noise = np.sqrt(0.5) * (rng.standard_normal((num_elements, num_snapshots))
                            + 1j * rng.standard_normal((num_elements, num_snapshots)))
    return sv[:, np.newaxis] * (signal * scale)[np.newaxis, :] + noise


def test_das_output_shape():
    from spectra.algorithms.beamforming import delay_and_sum
    arr = ula(num_elements=8, spacing=0.5, frequency=1e9)
    X = _snapshot(np.deg2rad(45.0))
    out = delay_and_sum(X, arr, target_az=np.deg2rad(45.0))
    assert out.shape == (512,)
    assert out.dtype == complex


def test_das_gain_at_target():
    """DAS output power should be higher when steered to true source than away from it."""
    from spectra.algorithms.beamforming import delay_and_sum
    arr = ula(num_elements=8, spacing=0.5, frequency=1e9)
    X = _snapshot(np.deg2rad(60.0), snr_db=15.0)
    out_on  = delay_and_sum(X, arr, target_az=np.deg2rad(60.0))
    out_off = delay_and_sum(X, arr, target_az=np.deg2rad(120.0))
    assert np.mean(np.abs(out_on)**2) > np.mean(np.abs(out_off)**2)


def test_mvdr_output_shape():
    from spectra.algorithms.beamforming import mvdr
    arr = ula(num_elements=8, spacing=0.5, frequency=1e9)
    X = _snapshot(np.deg2rad(45.0))
    out = mvdr(X, arr, target_az=np.deg2rad(45.0))
    assert out.shape == (512,)
    assert out.dtype == complex


def test_mvdr_distortionless_constraint():
    """MVDR weights must satisfy |w^H a| ≈ 1 at the target direction."""
    from spectra.algorithms.beamforming import _mvdr_weights
    arr = ula(num_elements=8, spacing=0.5, frequency=1e9)
    X = _snapshot(np.deg2rad(45.0))
    w = _mvdr_weights(X, arr, target_az=np.deg2rad(45.0))
    a = arr.steering_vector(azimuth=np.deg2rad(45.0), elevation=0.0)
    response = float(np.abs(w.conj() @ a))
    assert abs(response - 1.0) < 0.05, f"MVDR distortionless constraint violated: |w^H a| = {response:.4f}"


def test_lcmv_output_shape():
    from spectra.algorithms.beamforming import lcmv
    arr = ula(num_elements=8, spacing=0.5, frequency=1e9)
    X = _snapshot(np.deg2rad(45.0))
    out = lcmv(X, arr, constraints=[(np.deg2rad(45.0), 0.0)], responses=[1.0+0j])
    assert out.shape == (512,)


def test_lcmv_null_steering():
    """LCMV with desired=1 at target and null at interference should suppress interference."""
    from spectra.algorithms.beamforming import lcmv
    arr = ula(num_elements=8, spacing=0.5, frequency=1e9)
    az_sig = np.deg2rad(50.0)
    az_int = np.deg2rad(100.0)
    X_sig = _snapshot(az_sig, snr_db=10.0, seed=1)
    X_int = _snapshot(az_int, snr_db=20.0, seed=2)  # strong interferer
    X = X_sig + X_int
    constraints = [(az_sig, 0.0), (az_int, 0.0)]
    responses = [1.0 + 0j, 0.0 + 0j]
    w_lcmv = lcmv(X, arr, constraints=constraints, responses=responses, return_weights=True)
    a_int = arr.steering_vector(azimuth=az_int, elevation=0.0)
    null_depth = float(np.abs(w_lcmv.conj() @ a_int))
    assert null_depth < 0.1, f"LCMV null depth {null_depth:.4f} should be < 0.1"


def test_compute_beam_pattern():
    from spectra.algorithms.beamforming import compute_beam_pattern
    arr = ula(num_elements=8, spacing=0.5, frequency=1e9)
    scan = np.linspace(0, np.pi, 181)
    weights = np.ones(8, dtype=complex) / 8
    pattern = compute_beam_pattern(weights, arr, scan)
    assert pattern.shape == (181,)
    assert pattern.dtype == float
    assert 0.0 <= pattern.max() <= 1.0 + 1e-6
```

- [ ] **Step 2: Run tests to verify they fail**

```bash
/Users/gditzler/.venvs/base/bin/pytest tests/test_beamforming.py::test_das_output_shape -v
```
Expected: `ModuleNotFoundError: No module named 'spectra.algorithms.beamforming'`

- [ ] **Step 3: Create `python/spectra/algorithms/beamforming.py`**

```python
# python/spectra/algorithms/beamforming.py
"""Beamforming algorithms: delay-and-sum, MVDR, and LCMV.

All functions accept a complex snapshot matrix ``X`` of shape
``(N_elements, T)`` and return a beamformed time-series of shape ``(T,)``,
except where noted.
"""

from typing import List, Optional, Tuple, Union

import numpy as np

from spectra.arrays.array import AntennaArray


def delay_and_sum(
    X: np.ndarray,
    array: AntennaArray,
    target_az: float,
    elevation: float = 0.0,
) -> np.ndarray:
    """Conventional delay-and-sum (phase-shift) beamformer.

    Applies conjugate steering-vector weights normalised by the number of
    elements.  No covariance estimate is needed, so this works even for very
    short snapshots.

    Args:
        X: Complex snapshot matrix, shape ``(N_elements, T)``.
        array: :class:`~spectra.arrays.array.AntennaArray` geometry.
        target_az: Target azimuth in radians.
        elevation: Target elevation in radians (default 0).

    Returns:
        Beamformed complex signal, shape ``(T,)``.
    """
    a = array.steering_vector(azimuth=target_az, elevation=elevation)  # (N,)
    w = a.conj() / array.num_elements
    return w @ X  # (T,)


def _mvdr_weights(
    X: np.ndarray,
    array: AntennaArray,
    target_az: float,
    elevation: float = 0.0,
    diagonal_loading: float = 1e-6,
) -> np.ndarray:
    """Compute MVDR weight vector (internal helper).

    Returns:
        Weight vector, shape ``(N_elements,)``.
    """
    N, T = X.shape
    R = (X @ X.conj().T) / T
    R_reg = R + diagonal_loading * np.eye(N)
    R_inv = np.linalg.inv(R_reg)
    a = array.steering_vector(azimuth=target_az, elevation=elevation)
    R_inv_a = R_inv @ a
    denom = float(np.real(a.conj() @ R_inv_a))
    return R_inv_a / (denom + 1e-30)


def mvdr(
    X: np.ndarray,
    array: AntennaArray,
    target_az: float,
    elevation: float = 0.0,
    diagonal_loading: float = 1e-6,
) -> np.ndarray:
    """Minimum Variance Distortionless Response (MVDR / Capon) beamformer.

    Minimises output power subject to a unit-gain constraint at
    ``target_az``.  Suppresses interference and noise more effectively than
    delay-and-sum when the interference direction differs from the target.

    Args:
        X: Complex snapshot matrix, shape ``(N_elements, T)``.
        array: :class:`~spectra.arrays.array.AntennaArray` geometry.
        target_az: Target azimuth in radians.
        elevation: Target elevation in radians (default 0).
        diagonal_loading: Regularisation for covariance inversion (1e-6).

    Returns:
        Beamformed complex signal, shape ``(T,)``.
    """
    w = _mvdr_weights(X, array, target_az, elevation, diagonal_loading)
    return w.conj() @ X  # (T,)


def lcmv(
    X: np.ndarray,
    array: AntennaArray,
    constraints: List[Tuple[float, float]],
    responses: List[complex],
    diagonal_loading: float = 1e-6,
    return_weights: bool = False,
) -> np.ndarray:
    """Linearly-Constrained Minimum Variance (LCMV) beamformer.

    Generalises MVDR to multiple simultaneous linear constraints.  A common
    use-case is steering a unit-gain beam toward the desired source while
    placing nulls (response = 0) at known interference directions.

    Args:
        X: Complex snapshot matrix, shape ``(N_elements, T)``.
        array: :class:`~spectra.arrays.array.AntennaArray` geometry.
        constraints: List of ``(azimuth_rad, elevation_rad)`` constraint
            directions.  Must have at least one entry.
        responses: Desired complex response at each constraint direction.
            Length must match ``constraints``.  Use ``1+0j`` for unit gain
            and ``0+0j`` for a null.
        diagonal_loading: Regularisation for covariance inversion (1e-6).
        return_weights: If ``True``, return weight vector instead of
            beamformed signal. Useful for pattern analysis or null-depth
            verification.

    Returns:
        Beamformed complex signal shape ``(T,)`` unless ``return_weights=True``,
        in which case returns weight vector shape ``(N_elements,)``.

    Raises:
        ValueError: If ``constraints`` and ``responses`` have different lengths
            or if the constraint matrix is singular.
    """
    if len(constraints) != len(responses):
        raise ValueError(
            f"constraints ({len(constraints)}) and responses ({len(responses)}) must have the same length"
        )

    N, T = X.shape
    R = (X @ X.conj().T) / T
    R_reg = R + diagonal_loading * np.eye(N)
    R_inv = np.linalg.inv(R_reg)

    # Constraint matrix C: columns are steering vectors (N, K)
    C = np.column_stack([
        array.steering_vector(azimuth=az, elevation=el)
        for az, el in constraints
    ])
    g = np.asarray(responses, dtype=complex)  # (K,)

    # LCMV solution: w = R^{-1} C (C^H R^{-1} C)^{-1} g
    R_inv_C = R_inv @ C               # (N, K)
    M = C.conj().T @ R_inv_C          # (K, K)
    w = R_inv_C @ np.linalg.solve(M, g)  # (N,)

    if return_weights:
        return w
    return w.conj() @ X              # (T,)


def compute_beam_pattern(
    weights: np.ndarray,
    array: AntennaArray,
    scan_angles: np.ndarray,
    elevation: float = 0.0,
) -> np.ndarray:
    """Evaluate the normalised beam pattern for a fixed weight vector.

    Args:
        weights: Complex weight vector, shape ``(N_elements,)``.
        array: :class:`~spectra.arrays.array.AntennaArray` geometry.
        scan_angles: 1-D array of azimuth angles in radians.
        elevation: Fixed elevation angle in radians.

    Returns:
        Normalised power pattern in ``[0, 1]``, shape ``(len(scan_angles),)``.
    """
    responses = np.array([
        float(np.abs(weights.conj() @ array.steering_vector(azimuth=az, elevation=elevation)))
        for az in scan_angles
    ])
    peak = responses.max()
    if peak > 0:
        responses = responses / peak
    return responses
```

- [ ] **Step 4: Run tests to verify they pass**

```bash
/Users/gditzler/.venvs/base/bin/pytest tests/test_beamforming.py -v
```
Expected: 7/7 PASS

- [ ] **Step 5: Update `python/spectra/algorithms/__init__.py`**

```python
from spectra.algorithms.beamforming import (
    compute_beam_pattern,
    delay_and_sum,
    lcmv,
    mvdr,
)
from spectra.algorithms.doa import capon, esprit, find_peaks_doa, music, root_music

__all__ = [
    "capon",
    "compute_beam_pattern",
    "delay_and_sum",
    "esprit",
    "find_peaks_doa",
    "lcmv",
    "music",
    "mvdr",
    "root_music",
]
```

- [ ] **Step 6: Commit**

```bash
git add python/spectra/algorithms/beamforming.py python/spectra/algorithms/__init__.py tests/test_beamforming.py
git commit -m "feat(algorithms): add delay-and-sum, MVDR, and LCMV beamformers"
```

---

## Task 2: Python Example Script

**Files:**
- Create: `examples/14_beamforming.py`

- [ ] **Step 1: Create the script**

```python
# examples/14_beamforming.py
"""Example 14 — Beamforming with a Uniform Linear Array
=======================================================
Level: Intermediate

This example shows how to:
  1. Build a DirectionFindingDataset with a desired source and an interferer
  2. Apply delay-and-sum, MVDR, and LCMV beamformers to the snapshot matrix
  3. Visualise the output SNR of each beamformer
  4. Compare beam patterns (DAS vs MVDR vs LCMV with null)

Run:
    python examples/14_beamforming.py
"""

import os
import sys
sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))
sys.path.insert(0, os.path.dirname(__file__))

import numpy as np
import matplotlib.pyplot as plt

from spectra.arrays import ula
from spectra.datasets import DirectionFindingDataset
from spectra.waveforms import QPSK
from spectra.algorithms import delay_and_sum, mvdr, lcmv, compute_beam_pattern

from plot_helpers import savefig, OUTPUT_DIR

# ── Configuration ─────────────────────────────────────────────────────────────

FREQ_HZ     = 2.4e9
N_ELEMENTS  = 8
SPACING     = 0.5
N_SNAPSHOTS = 512
SAMPLE_RATE = 1e6
AZ_SOURCE   = np.deg2rad(50.0)   # desired source
AZ_INTER    = np.deg2rad(110.0)  # interferer
SNR_SOURCE  = 10.0               # dB
SNR_INTER   = 20.0               # dB (strong interferer)
SEED        = 0

SCAN_DEG = np.linspace(1, 179, 512)
SCAN_RAD = np.deg2rad(SCAN_DEG)

# ── 1. Build Dataset and Get One Sample ───────────────────────────────────────

arr = ula(num_elements=N_ELEMENTS, spacing=SPACING, frequency=FREQ_HZ)

# Two-source dataset: [source, interferer] at fixed azimuths with fixed SNRs
ds = DirectionFindingDataset(
    array=arr,
    signal_pool=[QPSK(samples_per_symbol=4)],
    num_signals=2,
    num_snapshots=N_SNAPSHOTS,
    sample_rate=SAMPLE_RATE,
    snr_range=(SNR_SOURCE, SNR_INTER),
    azimuth_range=(np.deg2rad(10), np.deg2rad(170)),
    elevation_range=(0.0, 0.0),
    min_angular_separation=np.deg2rad(30),
    num_samples=50,
    seed=SEED,
)

data, target = ds[0]
X = data[:, 0, :].numpy() + 1j * data[:, 1, :].numpy()  # (N, T)
print(f"Sources: az = {np.rad2deg(target.azimuths)} degrees")
print(f"SNRs:    {target.snrs} dB\n")

# Use first source as desired, second as interferer
az_desired    = float(target.azimuths[0])
az_interferer = float(target.azimuths[1])

# ── 2. Apply Beamformers ───────────────────────────────────────────────────────

y_das  = delay_and_sum(X, arr, target_az=az_desired)
y_mvdr = mvdr(X, arr, target_az=az_desired)
y_lcmv = lcmv(
    X, arr,
    constraints=[(az_desired, 0.0), (az_interferer, 0.0)],
    responses=[1.0 + 0j, 0.0 + 0j],
)

print("Output power comparison:")
for name, y in [("DAS", y_das), ("MVDR", y_mvdr), ("LCMV", y_lcmv)]:
    print(f"  {name}: {10*np.log10(np.mean(np.abs(y)**2)):.1f} dB")

# ── 3. Beam Pattern Comparison ────────────────────────────────────────────────

# Compute weights for each beamformer
from spectra.algorithms.beamforming import _mvdr_weights, lcmv

w_das = arr.steering_vector(azimuth=az_desired, elevation=0.0).conj() / N_ELEMENTS
w_mvdr = _mvdr_weights(X, arr, target_az=az_desired)
w_lcmv = lcmv(
    X, arr,
    constraints=[(az_desired, 0.0), (az_interferer, 0.0)],
    responses=[1.0 + 0j, 0.0 + 0j],
    return_weights=True,
)

fig, ax = plt.subplots(figsize=(9, 4))
for name, w, color in [("DAS", w_das, "steelblue"), ("MVDR", w_mvdr, "seagreen"), ("LCMV", w_lcmv, "darkorange")]:
    pattern = compute_beam_pattern(w, arr, SCAN_RAD)
    ax.plot(SCAN_DEG, 10 * np.log10(pattern + 1e-12), label=name, color=color, linewidth=1.3)

ax.axvline(np.rad2deg(az_desired), color="crimson", linestyle="--", linewidth=1.2, label=f"Desired {np.rad2deg(az_desired):.0f}°")
ax.axvline(np.rad2deg(az_interferer), color="black", linestyle=":", linewidth=1.2, label=f"Interferer {np.rad2deg(az_interferer):.0f}°")
ax.set_xlabel("Azimuth (degrees)")
ax.set_ylabel("Normalised beam pattern (dB)")
ax.set_title(f"Beam Pattern Comparison — {N_ELEMENTS}-element ULA")
ax.set_ylim(-50, 5)
ax.legend(fontsize=8)
ax.grid(True, alpha=0.3)
plt.tight_layout()
savefig("14_beam_patterns.png")

# ── 4. Output Spectra ─────────────────────────────────────────────────────────

freqs = np.fft.fftshift(np.fft.fftfreq(N_SNAPSHOTS, d=1.0 / SAMPLE_RATE))
fig, axes = plt.subplots(3, 1, figsize=(10, 9), sharex=True)
for ax, name, y in zip(axes, ["DAS", "MVDR", "LCMV"], [y_das, y_mvdr, y_lcmv]):
    psd = np.abs(np.fft.fftshift(np.fft.fft(y))) ** 2
    ax.plot(freqs / 1e3, 10 * np.log10(psd + 1e-30), linewidth=0.8)
    ax.set_ylabel(f"{name}\nPower (dB)")
    ax.grid(True, alpha=0.3)
axes[-1].set_xlabel("Frequency (kHz)")
fig.suptitle("Output Spectrum per Beamformer")
plt.tight_layout()
savefig("14_output_spectra.png")

print(f"\nFigures saved to {OUTPUT_DIR}")
```

- [ ] **Step 2: Verify the script runs**

```bash
cd /Users/gditzler/git/SPECTRA && /Users/gditzler/.venvs/base/bin/python examples/14_beamforming.py
```
Expected: prints source angles and power comparison, saves 2 figures.

- [ ] **Step 3: Commit**

```bash
git add examples/14_beamforming.py
git commit -m "feat(examples): add beamforming example script (DAS, MVDR, LCMV)"
```

---

## Task 3: Jupyter Notebook

**Files:**
- Create: `examples/14_beamforming.ipynb`

- [ ] **Step 1: Create `examples/14_beamforming.ipynb`**

Write as nbformat 4 / nbformat_minor 5 JSON with the following cell structure:

| # | Type | Content |
|---|------|---------|
| 0 | markdown | Title + learning objectives |
| 1 | code | Imports + config |
| 2 | markdown | "## 1. Array and Dataset Setup" |
| 3 | code | `arr = ula(...)`, `ds = DirectionFindingDataset(...)`, `data, target = ds[0]` |
| 4 | markdown | "## 2. Delay-and-Sum" — explain uniform weights, no covariance |
| 5 | code | `y_das = delay_and_sum(...)`, plot output PSD |
| 6 | markdown | "## 3. MVDR Beamformer" — explain distortionless constraint, power minimisation |
| 7 | code | `y_mvdr = mvdr(...)`, compare output power vs DAS |
| 8 | markdown | "## 4. LCMV with Null Steering" — explain multiple constraints, null placement |
| 9 | code | `y_lcmv = lcmv(...)`, verify null depth |
| 10 | markdown | "## 5. Beam Pattern Comparison" |
| 11 | code | `compute_beam_pattern` for all three, plot |

Cell 0 content:
```markdown
# Example 14 — Beamforming with a ULA

**Level:** Intermediate

After working through this notebook you will know how to:

- Apply **Delay-and-Sum** (DAS) conventional beamforming
- Apply **MVDR** (Capon) minimum-variance distortionless-response beamforming
- Apply **LCMV** beamforming with simultaneous gain and null constraints
- Visualise and compare normalised beam patterns
- Use `compute_beam_pattern()` to evaluate spatial filter response
```

Cell 2 explanation (3 sentences max):
```markdown
## 1. Array and Dataset Setup

We use the same `DirectionFindingDataset` from example 13, but configure it
with two sources: a desired signal and a strong interferer. The spatial mixing
inside the dataset produces a multi-element snapshot matrix from which we
can extract `X = I + jQ` for each element.
```

- [ ] **Step 2: Execute notebook to verify**

```bash
cd /Users/gditzler/git/SPECTRA/examples && /Users/gditzler/.venvs/base/bin/jupyter nbconvert --to notebook --execute 14_beamforming.ipynb --output /tmp/check_14.ipynb 2>&1 && echo "OK"
```
Expected: `OK`

- [ ] **Step 3: Commit**

```bash
git add examples/14_beamforming.ipynb
git commit -m "feat(examples): add beamforming Jupyter notebook"
```

---

## Task 4: Update examples/README.md

- [ ] **Step 1: Add entry after example 13**

```markdown
### 14 — Beamforming with a ULA (Intermediate)

Apply delay-and-sum, MVDR, and LCMV beamformers to a multi-antenna snapshot
from `DirectionFindingDataset`. Visualises beam patterns and compares output
SNR when a strong interferer is present.

```bash
cd examples && python 14_beamforming.py
```
```

Also add to the File Structure section:
```
  14_beamforming.ipynb / .py            # Intermediate
```

- [ ] **Step 2: Verify**

```bash
grep -A4 "### 14" examples/README.md
```

- [ ] **Step 3: Commit**

```bash
git add examples/README.md
git commit -m "docs(examples): add example 14 to README"
```

---

## Task 5: Full Verification

- [ ] **Step 1: Run full test suite**

```bash
cd /Users/gditzler/git/SPECTRA && /Users/gditzler/.venvs/base/bin/pytest -q --tb=short 2>&1 | tail -5
```
Expected: all pass.

- [ ] **Step 2: Confirm imports**

```bash
/Users/gditzler/.venvs/base/bin/python -c "
from spectra.algorithms import delay_and_sum, mvdr, lcmv, compute_beam_pattern
print('Beamforming imports OK')
"
```
