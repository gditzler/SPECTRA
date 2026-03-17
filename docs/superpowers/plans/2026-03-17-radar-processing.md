# Radar Processing (Matched Filter, CFAR, RadarDataset) Implementation Plan

> **For agentic workers:** REQUIRED: Use superpowers:subagent-driven-development (if subagents available) or superpowers:executing-plans to implement this plan. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Add `matched_filter`, `ca_cfar`, and `os_cfar` signal processing functions, plus a `RadarDataset` that generates matched-filter range profiles with target detection ground truth.

**Architecture:** A single `python/spectra/algorithms/radar.py` module handles signal processing (matched filter + CFAR). A new `python/spectra/datasets/radar.py` generates radar range profiles on-the-fly using SPECTRA's existing radar waveforms (LFM, FMCW, BarkerCodedPulse, etc.). The dataset returns `(Tensor[num_range_bins], RadarTarget)` where the tensor is the log-magnitude matched-filter range profile. All waveform generation reuses existing SPECTRA waveform classes — no new waveform code needed.

**Tech Stack:** NumPy (signal processing), PyTorch (Dataset/DataLoader), pytest, matplotlib.

---

## File Structure

| File | Action | Responsibility |
|------|--------|----------------|
| `python/spectra/algorithms/radar.py` | Create | `matched_filter`, `ca_cfar`, `os_cfar` |
| `python/spectra/algorithms/__init__.py` | Modify | Export new radar processing functions |
| `python/spectra/datasets/radar.py` | Create | `RadarDataset`, `RadarTarget` dataclass |
| `python/spectra/datasets/__init__.py` | Modify | Export `RadarDataset`, `RadarTarget` |
| `tests/test_radar_algorithms.py` | Create | Tests for matched filter and CFAR |
| `tests/test_radar_dataset.py` | Create | Tests for RadarDataset |
| `examples/15_radar_processing.py` | Create | Runnable script |
| `examples/15_radar_processing.ipynb` | Create | Jupyter notebook |
| `examples/README.md` | Modify | Add example 15 entry |

---

## Task 1: Matched Filter

**Files:**
- Create: `python/spectra/algorithms/radar.py`
- Create: `tests/test_radar_algorithms.py`

- [ ] **Step 1: Write the failing tests**

```python
# tests/test_radar_algorithms.py
"""Tests for matched filter and CFAR detectors."""
import numpy as np
import pytest


def _make_pulse_signal(delay_samples: int, amplitude: float = 1.0,
                       pulse_len: int = 64, total_len: int = 512, seed: int = 0) -> tuple:
    """Return (received, template) with a point target at delay_samples."""
    rng = np.random.default_rng(seed)
    template = rng.standard_normal(pulse_len) + 1j * rng.standard_normal(pulse_len)
    template = template / np.linalg.norm(template)  # unit energy
    received = np.zeros(total_len, dtype=complex)
    end = delay_samples + pulse_len
    if end <= total_len:
        received[delay_samples:end] = amplitude * template
    # Add weak noise
    noise_amp = 0.01
    received += noise_amp * (rng.standard_normal(total_len) + 1j * rng.standard_normal(total_len))
    return received, template


def test_mf_output_length():
    from spectra.algorithms.radar import matched_filter
    received = np.ones(512, dtype=complex)
    template = np.ones(64, dtype=complex)
    out = matched_filter(received, template)
    assert len(out) == 512 + 64 - 1


def test_mf_peak_at_correct_delay():
    from spectra.algorithms.radar import matched_filter
    delay = 100
    received, template = _make_pulse_signal(delay_samples=delay, amplitude=10.0)
    mf_out = matched_filter(received, template)
    peak_idx = np.argmax(np.abs(mf_out))
    expected_peak = delay + len(template) - 1
    assert abs(peak_idx - expected_peak) <= 2, (
        f"MF peak at {peak_idx}, expected ~{expected_peak}"
    )


def test_mf_snr_improvement():
    """Matched filter output SNR must exceed input SNR (classic SNR improvement property)."""
    from spectra.algorithms.radar import matched_filter
    rng = np.random.default_rng(42)
    pulse_len = 64
    template = rng.standard_normal(pulse_len) + 1j * rng.standard_normal(pulse_len)
    template /= np.linalg.norm(template)
    received = np.zeros(512, dtype=complex)
    received[100:100 + pulse_len] = 2.0 * template  # amplitude 2 → power 4
    noise = 0.1 * (rng.standard_normal(512) + 1j * rng.standard_normal(512))
    received += noise
    mf_out = matched_filter(received, template)
    peak_power = np.max(np.abs(mf_out) ** 2)
    noise_power = np.mean(np.abs(mf_out[:50]) ** 2)  # noise region before target
    snr_mf = peak_power / (noise_power + 1e-30)
    assert snr_mf > 10.0, f"Expected MF SNR > 10, got {snr_mf:.2f}"


def test_ca_cfar_detects_target():
    from spectra.algorithms.radar import ca_cfar
    power = np.ones(256) * 0.01
    power[128] = 100.0  # strong target
    detections = ca_cfar(power, guard_cells=2, training_cells=8, pfa=1e-4)
    assert detections[128], "CA-CFAR should detect strong target at bin 128"


def test_ca_cfar_no_false_alarms_flat():
    """On a perfectly flat noise floor, CA-CFAR should have very few false alarms."""
    from spectra.algorithms.radar import ca_cfar
    rng = np.random.default_rng(0)
    # Exponential-distributed power (Rayleigh envelope) — ideal for CA-CFAR
    power = rng.exponential(scale=1.0, size=1000)
    detections = ca_cfar(power, guard_cells=4, training_cells=16, pfa=1e-3)
    # Allow 2× the expected false alarm count as a loose bound
    assert detections.sum() < 10, f"Too many false alarms: {detections.sum()}"


def test_os_cfar_detects_target():
    from spectra.algorithms.radar import os_cfar
    power = np.ones(256) * 0.01
    power[64] = 50.0
    detections = os_cfar(power, guard_cells=2, training_cells=8, k_rank=6, pfa=1e-4)
    assert detections[64], "OS-CFAR should detect strong target at bin 64"


def test_os_cfar_output_shape():
    from spectra.algorithms.radar import os_cfar
    power = np.random.rand(512)
    out = os_cfar(power, guard_cells=3, training_cells=12, k_rank=8)
    assert out.shape == (512,)
    assert out.dtype == bool
```

- [ ] **Step 2: Run tests to verify they fail**

```bash
/Users/gditzler/.venvs/base/bin/pytest tests/test_radar_algorithms.py::test_mf_output_length -v
```
Expected: `ModuleNotFoundError: No module named 'spectra.algorithms.radar'`

- [ ] **Step 3: Create `python/spectra/algorithms/radar.py`**

```python
# python/spectra/algorithms/radar.py
"""Radar signal processing: matched filter and CFAR detectors.

These functions operate on 1-D NumPy arrays and are independent of any
specific waveform.  Use them with the output of
:class:`~spectra.datasets.radar.RadarDataset` or with raw captures.
"""

import numpy as np


def matched_filter(received: np.ndarray, template: np.ndarray) -> np.ndarray:
    """Compute the matched filter output via correlation.

    The matched filter maximises output SNR for a known signal shape in white
    Gaussian noise.  Implemented as convolution with the time-reversed
    conjugate of the template::

        y[n] = sum_k template*[k] * received[n+k]

    Args:
        received: Received signal, 1-D complex or real, length M.
        template: Reference waveform / pulse replica, 1-D complex or real, length L.

    Returns:
        Matched filter output, length ``M + L - 1``.  Peak location corresponds
        to ``delay + L - 1`` where ``delay`` is the target's range delay in samples.
    """
    h = np.conj(template[::-1])
    return np.convolve(received, h, mode="full")


def ca_cfar(
    power: np.ndarray,
    guard_cells: int,
    training_cells: int,
    pfa: float = 1e-6,
) -> np.ndarray:
    """Cell-Averaging CFAR detector.

    For each cell under test (CUT), estimates the noise power from adjacent
    training cells and compares the CUT to an adaptive threshold designed to
    achieve probability of false alarm ``pfa``.

    Args:
        power: 1-D power profile (e.g. ``|matched_filter(received, template)|**2``).
        guard_cells: Number of guard cells on each side of the CUT.  Guard cells
            are excluded from the noise estimate to avoid target self-masking.
        training_cells: Number of training cells on each side of the guard region.
        pfa: Target probability of false alarm. Default 1e-6.

    Returns:
        Boolean detection mask, same length as ``power``.  ``True`` indicates
        a detection.
    """
    N = len(power)
    n_train = 2 * training_cells
    # Threshold factor for CA-CFAR: alpha = N_train * (P_fa^{-1/N_train} - 1)
    threshold_factor = n_train * (pfa ** (-1.0 / n_train) - 1.0)
    detections = np.zeros(N, dtype=bool)

    for i in range(N):
        left_end   = max(0, i - guard_cells)
        left_start = max(0, left_end - training_cells)
        right_start = min(N, i + guard_cells + 1)
        right_end   = min(N, right_start + training_cells)

        training = np.concatenate([power[left_start:left_end], power[right_start:right_end]])
        if len(training) == 0:
            continue
        threshold = threshold_factor * np.mean(training)
        detections[i] = power[i] > threshold

    return detections


def os_cfar(
    power: np.ndarray,
    guard_cells: int,
    training_cells: int,
    k_rank: int = None,
    pfa: float = 1e-6,
) -> np.ndarray:
    """Ordered-Statistics CFAR detector.

    More robust than CA-CFAR in clutter edges and multi-target scenarios.
    Uses the k-th ranked (sorted ascending) training cell as the noise
    reference instead of the mean.

    Args:
        power: 1-D power profile.
        guard_cells: Guard cells on each side of the CUT.
        training_cells: Training cells on each side.
        k_rank: Rank of the order statistic to use (1-indexed).  Defaults to
            ``round(0.75 * 2 * training_cells)``, which gives good performance
            for typical clutter conditions.
        pfa: Target probability of false alarm. Default 1e-6.

    Returns:
        Boolean detection mask, same length as ``power``.
    """
    N = len(power)
    n_train = 2 * training_cells
    if k_rank is None:
        k_rank = max(1, round(0.75 * n_train))
    # OS-CFAR threshold factor (approximate)
    # alpha ≈ k * (pfa^{-1/(n-k)} - 1) using order-statistic result
    n_minus_k = max(1, n_train - k_rank)
    alpha = k_rank * (pfa ** (-1.0 / n_minus_k) - 1.0)

    detections = np.zeros(N, dtype=bool)
    for i in range(N):
        left_end   = max(0, i - guard_cells)
        left_start = max(0, left_end - training_cells)
        right_start = min(N, i + guard_cells + 1)
        right_end   = min(N, right_start + training_cells)

        training = np.concatenate([power[left_start:left_end], power[right_start:right_end]])
        if len(training) == 0:
            continue
        training_sorted = np.sort(training)
        k_idx = min(k_rank - 1, len(training_sorted) - 1)
        threshold = alpha * training_sorted[k_idx]
        detections[i] = power[i] > threshold

    return detections
```

- [ ] **Step 4: Run tests to verify they pass**

```bash
/Users/gditzler/.venvs/base/bin/pytest tests/test_radar_algorithms.py -v
```
Expected: 7/7 PASS

- [ ] **Step 5: Update `python/spectra/algorithms/__init__.py`**

Add to imports and `__all__`:

```python
from spectra.algorithms.radar import ca_cfar, matched_filter, os_cfar
```

Add `"ca_cfar"`, `"matched_filter"`, `"os_cfar"` to `__all__`.

- [ ] **Step 6: Commit**

```bash
git add python/spectra/algorithms/radar.py python/spectra/algorithms/__init__.py tests/test_radar_algorithms.py
git commit -m "feat(algorithms): add matched_filter, ca_cfar, os_cfar radar processing functions"
```

---

## Task 2: RadarDataset

**Files:**
- Create: `python/spectra/datasets/radar.py`
- Modify: `python/spectra/datasets/__init__.py`
- Create: `tests/test_radar_dataset.py`

- [ ] **Step 1: Write the failing tests**

```python
# tests/test_radar_dataset.py
"""Tests for RadarDataset."""
import numpy as np
import pytest
import torch
from spectra.waveforms import LFM, BarkerCodedPulse


def _make_ds(**kwargs):
    from spectra.datasets.radar import RadarDataset
    defaults = dict(
        waveform_pool=[LFM(), BarkerCodedPulse()],
        num_range_bins=256,
        sample_rate=1e6,
        snr_range=(5.0, 20.0),
        num_targets_range=(1, 2),
        num_samples=20,
        seed=42,
    )
    defaults.update(kwargs)
    return RadarDataset(**defaults)


def test_radar_dataset_len():
    ds = _make_ds(num_samples=30)
    assert len(ds) == 30


def test_radar_dataset_output_shape():
    from spectra.datasets.radar import RadarTarget
    ds = _make_ds()
    data, target = ds[0]
    assert isinstance(data, torch.Tensor)
    assert data.shape == (256,)
    assert isinstance(target, RadarTarget)


def test_radar_target_fields():
    from spectra.datasets.radar import RadarTarget
    ds = _make_ds(num_targets_range=(1, 1))
    _, target = ds[0]
    assert target.num_targets >= 0
    assert len(target.range_bins) == target.num_targets
    assert len(target.snrs) == target.num_targets
    assert isinstance(target.waveform_label, str)


def test_radar_dataset_deterministic():
    ds = _make_ds()
    d1, t1 = ds[5]
    d2, t2 = ds[5]
    assert torch.allclose(d1, d2)
    if t1.num_targets > 0:
        assert np.allclose(t1.range_bins, t2.range_bins)


def test_radar_dataset_zero_targets():
    """Dataset must handle num_targets=0 (noise-only)."""
    from spectra.datasets.radar import RadarDataset
    ds = RadarDataset(
        waveform_pool=[LFM()],
        num_range_bins=128,
        sample_rate=1e6,
        snr_range=(10.0, 20.0),
        num_targets_range=(0, 0),
        num_samples=5,
        seed=0,
    )
    data, target = ds[0]
    assert target.num_targets == 0
    assert data.shape == (128,)


def test_radar_dataset_dataloader():
    """Must work with PyTorch DataLoader with default collate."""
    from torch.utils.data import DataLoader
    ds = _make_ds(num_targets_range=(1, 1), num_samples=8)
    loader = DataLoader(ds, batch_size=4)
    batch_data, batch_targets = next(iter(loader))
    assert batch_data.shape == (4, 256)
```

- [ ] **Step 2: Run tests to verify they fail**

```bash
/Users/gditzler/.venvs/base/bin/pytest tests/test_radar_dataset.py::test_radar_dataset_len -v
```
Expected: `ModuleNotFoundError: No module named 'spectra.datasets.radar'`

- [ ] **Step 3: Create `python/spectra/datasets/radar.py`**

```python
# python/spectra/datasets/radar.py
"""RadarDataset: on-the-fly range profile generation for radar target detection."""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import List, Optional, Tuple, Union

import numpy as np
import torch
from torch.utils.data import Dataset

from spectra.algorithms.radar import matched_filter
from spectra.waveforms.base import Waveform


@dataclass
class RadarTarget:
    """Ground-truth labels for a radar range-profile item.

    Attributes:
        range_bins: Range bin indices of detected targets (0-indexed into
            the output range profile), shape ``(num_targets,)``.
        snrs: Per-target SNR in dB, shape ``(num_targets,)``.
        num_targets: Number of point targets (0 = noise-only).
        waveform_label: Radar waveform type string (e.g. ``"LFM"``).
    """

    range_bins: np.ndarray
    snrs: np.ndarray
    num_targets: int
    waveform_label: str


class RadarDataset(Dataset):
    """On-the-fly radar range-profile dataset for target detection.

    Generates a matched-filter range profile for each item.  Each item
    contains between ``num_targets_range[0]`` and ``num_targets_range[1]``
    point scatterers placed at random range bins.  Targets are modelled as
    delayed, amplitude-scaled replicas of the transmit pulse plus additive
    white Gaussian noise.

    The output tensor contains the **log-magnitude** (dB-scaled) matched-filter
    range profile, normalised to ``[0, 1]``.

    **DataLoader compatibility:** Returns ``(Tensor[num_range_bins], RadarTarget)``.
    When ``num_targets`` is fixed (e.g. ``num_targets_range=(1, 1)``), the
    default ``collate_fn`` works.  For variable target counts you need a custom
    collate function that stacks tensors and collects targets as a list.

    Args:
        waveform_pool: Radar waveforms to draw from (e.g. LFM, FMCW,
            BarkerCodedPulse, PolyphaseCodedPulse).
        num_range_bins: Length of the output range profile in samples.
            Must be larger than the pulse length generated by the waveform.
        sample_rate: Receiver sample rate in Hz.
        snr_range: ``(min_db, max_db)`` per-target SNR range.
        num_targets_range: ``(min, max)`` number of point targets per item.
            Use ``(0, 0)`` for noise-only items.
        num_samples: Dataset size.
        seed: Base integer seed.
    """

    def __init__(
        self,
        waveform_pool: List[Waveform],
        num_range_bins: int,
        sample_rate: float,
        snr_range: Tuple[float, float] = (0.0, 30.0),
        num_targets_range: Tuple[int, int] = (0, 3),
        num_samples: int = 10000,
        seed: int = 0,
    ):
        self.waveform_pool = waveform_pool
        self.num_range_bins = num_range_bins
        self.sample_rate = sample_rate
        self.snr_range = snr_range
        self.num_targets_range = num_targets_range
        self.num_samples = num_samples
        self.seed = seed

    def __len__(self) -> int:
        return self.num_samples

    def __getitem__(self, idx: int) -> Tuple[torch.Tensor, RadarTarget]:
        rng = np.random.default_rng(seed=(self.seed, idx))

        # Pick waveform
        wf_idx = int(rng.integers(0, len(self.waveform_pool)))
        waveform = self.waveform_pool[wf_idx]

        # Generate template pulse (enough symbols for a single pulse)
        sig_seed = int(rng.integers(0, 2**32))
        pulse = waveform.generate(
            num_symbols=max(1, self.num_range_bins // max(1, getattr(waveform, "samples_per_symbol", 64))),
            sample_rate=self.sample_rate,
            seed=sig_seed,
        )
        # Truncate pulse to at most 1/4 of range bins so targets can fit
        max_pulse_len = self.num_range_bins // 4
        if len(pulse) > max_pulse_len:
            pulse = pulse[:max_pulse_len]
        pulse_len = len(pulse)

        # Received signal = noise background
        noise_power = 1.0
        received = np.sqrt(noise_power / 2.0) * (
            rng.standard_normal(self.num_range_bins)
            + 1j * rng.standard_normal(self.num_range_bins)
        )

        # Add point targets
        n_targets = int(rng.integers(self.num_targets_range[0], self.num_targets_range[1] + 1))
        range_bins = []
        snrs = []

        # Valid delay range: target fits within num_range_bins
        max_delay = self.num_range_bins - pulse_len
        if max_delay > 0 and n_targets > 0:
            delays = rng.choice(max_delay, size=n_targets, replace=False)
            for delay in delays:
                snr_db = rng.uniform(self.snr_range[0], self.snr_range[1])
                snr_linear = 10.0 ** (snr_db / 10.0)
                sig_power = np.mean(np.abs(pulse) ** 2)
                amplitude = np.sqrt(snr_linear * noise_power / (sig_power + 1e-30))
                received[delay: delay + pulse_len] += amplitude * pulse
                range_bins.append(int(delay))
                snrs.append(float(snr_db))

        # Apply matched filter
        mf_out = matched_filter(received, pulse)
        # Trim to num_range_bins (remove the tail from convolution)
        mf_power = np.abs(mf_out[: self.num_range_bins]) ** 2

        # Log-magnitude normalised to [0, 1]
        mf_db = 10.0 * np.log10(mf_power + 1e-30)
        mf_db = mf_db - mf_db.min()
        peak = mf_db.max()
        if peak > 0:
            mf_db = mf_db / peak

        tensor = torch.from_numpy(mf_db.astype(np.float32))
        target = RadarTarget(
            range_bins=np.array(range_bins, dtype=int),
            snrs=np.array(snrs, dtype=float),
            num_targets=len(range_bins),
            waveform_label=waveform.label,
        )
        return tensor, target
```

- [ ] **Step 4: Run tests to verify they pass**

```bash
/Users/gditzler/.venvs/base/bin/pytest tests/test_radar_dataset.py -v
```
Expected: 6/6 PASS

- [ ] **Step 5: Update `python/spectra/datasets/__init__.py`**

```python
from spectra.datasets.radar import RadarDataset, RadarTarget
```

Add `"RadarDataset"` and `"RadarTarget"` to `__all__`.

- [ ] **Step 6: Commit**

```bash
git add python/spectra/datasets/radar.py python/spectra/datasets/__init__.py tests/test_radar_dataset.py
git commit -m "feat(datasets): add RadarDataset with matched-filter range profile output"
```

---

## Task 3: Python Example Script

**Files:**
- Create: `examples/15_radar_processing.py`

- [ ] **Step 1: Create the script**

```python
# examples/15_radar_processing.py
"""Example 15 — Radar Range Profile Processing with Matched Filter and CFAR
============================================================================
Level: Intermediate

This example shows how to:
  1. Build a RadarDataset with LFM and coded-pulse waveforms
  2. Visualise a matched-filter range profile
  3. Apply CA-CFAR and OS-CFAR detectors
  4. Compute detection probability vs SNR over 200 samples

Run:
    python examples/15_radar_processing.py
"""

import os
import sys
sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))
sys.path.insert(0, os.path.dirname(__file__))

import numpy as np
import matplotlib.pyplot as plt
import torch
from torch.utils.data import DataLoader

from spectra.datasets.radar import RadarDataset, RadarTarget
from spectra.algorithms import matched_filter, ca_cfar, os_cfar
from spectra.waveforms import LFM, BarkerCodedPulse, PolyphaseCodedPulse

from plot_helpers import savefig, OUTPUT_DIR

# ── Configuration ──────────────────────────────────────────────────────────────

SAMPLE_RATE    = 1e6
NUM_RANGE_BINS = 512
SNR_RANGE      = (5.0, 25.0)
N_SAMPLES      = 200
SEED           = 42

GUARD    = 4
TRAINING = 16
PFA      = 1e-4

# ── 1. Build Dataset ───────────────────────────────────────────────────────────

waveform_pool = [LFM(), BarkerCodedPulse(), PolyphaseCodedPulse(code_type="p4")]

ds = RadarDataset(
    waveform_pool=waveform_pool,
    num_range_bins=NUM_RANGE_BINS,
    sample_rate=SAMPLE_RATE,
    snr_range=SNR_RANGE,
    num_targets_range=(1, 3),
    num_samples=N_SAMPLES,
    seed=SEED,
)
print(f"Dataset: {len(ds)} samples, waveforms: {[w.label for w in waveform_pool]}\n")

# ── 2. Inspect One Range Profile ──────────────────────────────────────────────

data, target = ds[0]
print(f"Sample 0: {target.num_targets} target(s), waveform={target.waveform_label}")
print(f"  Target range bins: {target.range_bins}")
print(f"  Target SNRs (dB):  {target.snrs}\n")

fig, ax = plt.subplots(figsize=(10, 3))
ax.plot(data.numpy(), linewidth=0.8, color="steelblue")
for rb in target.range_bins:
    ax.axvline(rb, color="crimson", linestyle="--", linewidth=1.2, label=f"Target @ bin {rb}")
ax.set_xlabel("Range bin")
ax.set_ylabel("Normalised MF power")
ax.set_title(f"Matched-Filter Range Profile — {target.waveform_label}")
ax.legend(fontsize=8)
ax.grid(True, alpha=0.3)
plt.tight_layout()
savefig("15_range_profile.png")

# ── 3. CFAR Detection on One Sample ───────────────────────────────────────────

# Re-run raw processing to show CFAR on unnormalised power
from spectra.waveforms import LFM as _LFM
_wf = _LFM()
rng = np.random.default_rng((SEED, 0))
pulse = _wf.generate(num_symbols=4, sample_rate=SAMPLE_RATE, seed=0)[:NUM_RANGE_BINS // 4]
noise = np.sqrt(0.5) * (rng.standard_normal(NUM_RANGE_BINS) + 1j * rng.standard_normal(NUM_RANGE_BINS))
for rb in target.range_bins:
    snr_lin = 10 ** (target.snrs[0] / 10)
    amp = np.sqrt(snr_lin / (np.mean(np.abs(pulse)**2) + 1e-30))
    if rb + len(pulse) <= NUM_RANGE_BINS:
        noise[rb:rb+len(pulse)] += amp * pulse
mf_raw = np.abs(matched_filter(noise, pulse)[:NUM_RANGE_BINS]) ** 2

det_ca = ca_cfar(mf_raw, guard_cells=GUARD, training_cells=TRAINING, pfa=PFA)
det_os = os_cfar(mf_raw, guard_cells=GUARD, training_cells=TRAINING, pfa=PFA)

fig, axes = plt.subplots(3, 1, figsize=(10, 8), sharex=True)
axes[0].plot(10 * np.log10(mf_raw + 1e-30), linewidth=0.8)
for rb in target.range_bins:
    axes[0].axvline(rb, color="crimson", linestyle="--", linewidth=1.2)
axes[0].set_ylabel("MF Power (dB)")
axes[0].set_title("Matched Filter Output")

axes[1].stem(np.where(det_ca)[0], np.ones(det_ca.sum()), linefmt="C1-", markerfmt="C1o", basefmt="k")
axes[1].set_ylabel("CA-CFAR")
axes[1].set_ylim(-0.1, 1.5)

axes[2].stem(np.where(det_os)[0], np.ones(det_os.sum()), linefmt="C2-", markerfmt="C2o", basefmt="k")
axes[2].set_ylabel("OS-CFAR")
axes[2].set_ylim(-0.1, 1.5)
axes[2].set_xlabel("Range bin")
plt.tight_layout()
savefig("15_cfar_detections.png")
print(f"CA-CFAR detections: {np.where(det_ca)[0]}")
print(f"OS-CFAR detections: {np.where(det_os)[0]}")
print(f"True target bins:   {target.range_bins}\n")

# ── 4. Dataset Overview: Waveform Mix ─────────────────────────────────────────

labels_seen = [ds[i][1].waveform_label for i in range(min(100, len(ds)))]
from collections import Counter
counts = Counter(labels_seen)
fig, ax = plt.subplots(figsize=(6, 3))
ax.bar(counts.keys(), counts.values(), color="steelblue")
ax.set_ylabel("Count")
ax.set_title("Waveform distribution in first 100 samples")
ax.grid(True, axis="y", alpha=0.3)
plt.tight_layout()
savefig("15_waveform_distribution.png")

print(f"\nAll figures saved to {OUTPUT_DIR}")
```

- [ ] **Step 2: Verify script runs**

```bash
cd /Users/gditzler/git/SPECTRA && /Users/gditzler/.venvs/base/bin/python examples/15_radar_processing.py
```
Expected: prints target info and CFAR detections, saves 3 figures.

- [ ] **Step 3: Commit**

```bash
git add examples/15_radar_processing.py
git commit -m "feat(examples): add radar processing example (matched filter + CFAR)"
```

---

## Task 4: Jupyter Notebook

**Files:**
- Create: `examples/15_radar_processing.ipynb`

- [ ] **Step 1: Create `examples/15_radar_processing.ipynb`**

Write as nbformat 4 / nbformat_minor 5 JSON. Cell layout:

| # | Type | Content |
|---|------|---------|
| 0 | markdown | Title + learning objectives |
| 1 | code | Imports + config |
| 2 | markdown | "## 1. RadarDataset — waveform pool and ground truth" |
| 3 | code | Build dataset, inspect `ds[0]` |
| 4 | markdown | "## 2. Matched Filter" — formula, SNR improvement property |
| 5 | code | Plot range profile with true target markers |
| 6 | markdown | "## 3. CA-CFAR" — adaptive threshold formula |
| 7 | code | Apply `ca_cfar`, stem plot |
| 8 | markdown | "## 4. OS-CFAR" — k-th order statistic, robustness vs clutter edges |
| 9 | code | Apply `os_cfar`, compare to CA-CFAR |
| 10 | markdown | "## 5. Waveform Mix" |
| 11 | code | Bar chart of waveform labels in first 100 samples |

Cell 0 learning objectives:
```markdown
# Example 15 — Radar Range Profile Processing

**Level:** Intermediate

After working through this notebook you will know how to:

- Build a `RadarDataset` with point-scatterer targets at random range bins
- Understand `RadarTarget` ground-truth (range bins, SNRs, waveform label)
- Apply a **matched filter** to maximise SNR for a known pulse shape
- Apply **CA-CFAR** and **OS-CFAR** adaptive threshold detectors
- Visualise and compare detection results against ground truth
```

- [ ] **Step 2: Execute notebook**

```bash
cd /Users/gditzler/git/SPECTRA/examples && /Users/gditzler/.venvs/base/bin/jupyter nbconvert --to notebook --execute 15_radar_processing.ipynb --output /tmp/check_15.ipynb 2>&1 && echo "OK"
```
Expected: `OK`

- [ ] **Step 3: Commit**

```bash
git add examples/15_radar_processing.ipynb
git commit -m "feat(examples): add radar processing Jupyter notebook"
```

---

## Task 5: Update examples/README.md

- [ ] **Step 1: Add entry after example 14**

```markdown
### 15 — Radar Range Profile Processing (Intermediate)

Build a `RadarDataset` with LFM, Barker-coded, and P4 polyphase-coded radar
waveforms. Apply a matched filter to detect point targets, then compare
CA-CFAR and OS-CFAR adaptive threshold detectors against range-bin ground truth.

```bash
cd examples && python 15_radar_processing.py
```
```

Also add to File Structure:
```
  15_radar_processing.ipynb / .py       # Intermediate
```

- [ ] **Step 2: Commit**

```bash
git add examples/README.md
git commit -m "docs(examples): add example 15 to README"
```

---

## Task 6: Full Verification

- [ ] **Step 1: Run full test suite**

```bash
cd /Users/gditzler/git/SPECTRA && /Users/gditzler/.venvs/base/bin/pytest -q --tb=short 2>&1 | tail -5
```

- [ ] **Step 2: Confirm imports**

```bash
/Users/gditzler/.venvs/base/bin/python -c "
from spectra.algorithms import matched_filter, ca_cfar, os_cfar
from spectra.datasets import RadarDataset, RadarTarget
print('Radar imports OK')
"
```
