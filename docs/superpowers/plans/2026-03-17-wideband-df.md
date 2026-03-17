# WidebandDirectionFindingDataset Implementation Plan

> **For agentic workers:** REQUIRED: Use superpowers:subagent-driven-development (if subagents available) or superpowers:executing-plans to implement this plan. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Add `WidebandDirectionFindingDataset` — a multi-antenna dataset that generates wideband captures with multiple signals at different center frequencies, each with its own DoA angle, enabling joint spectrum + direction-finding tasks.

**Architecture:** A single new dataset class in `python/spectra/datasets/wideband_df.py`. Each item places `N_signals` sources at random center frequencies (within the capture bandwidth) and random azimuths. Sources are frequency-shifted to their center frequencies before spatial mixing, so the per-element IQ captures a wideband snapshot where each signal occupies a distinct spectral band. The steering vector for each source uses the per-signal carrier frequency for the phase argument, making the spatial signature frequency-dependent. Ground truth is a `WidebandDFTarget` dataclass with per-signal azimuth, elevation, center frequency, SNR, and label.

**Key design decisions:**
- Spatial mixing uses `array.steering_vector(azimuth, elevation)` — this is already frequency-aware through the array's `reference_frequency`; for wideband we pass frequency-scaled positions explicitly using a per-signal steering vector computed via `np.exp(j * 2π * d/λ_k * cos(az))`.
- No new Rust needed; all computation is in NumPy.
- Output tensor shape: `[N_elements, 2, num_snapshots]` (same as `DirectionFindingDataset`) so downstream MUSIC/ESPRIT code can be reused per sub-band.

**Tech Stack:** NumPy, PyTorch, pytest, matplotlib.

---

## File Structure

| File | Action | Responsibility |
|------|--------|----------------|
| `python/spectra/datasets/wideband_df.py` | Create | `WidebandDirectionFindingDataset`, `WidebandDFTarget` |
| `python/spectra/datasets/__init__.py` | Modify | Export new dataset and target |
| `tests/test_wideband_df.py` | Create | Unit tests |
| `examples/16_wideband_direction_finding.py` | Create | Runnable script |
| `examples/16_wideband_direction_finding.ipynb` | Create | Jupyter notebook |
| `examples/README.md` | Modify | Add example 16 entry |

---

## Architecture Detail: Wideband Spatial Mixing

For a narrowband signal at frequency `f_k`:
- Wavelength: `λ_k = c / f_k`
- Element positions in wavelengths at `f_k`: `d_k = d_meters * f_k / c`
- Phase argument: `φ_i = 2π · d_k[i] · cos(el) · cos(az)` (x-axis element only for ULA)

SPECTRA's `AntennaArray` stores positions in wavelengths at `reference_frequency`. For wideband mixing we rescale:
```python
freq_scale = center_freq_k / array.reference_frequency
# Scale positions to wavelengths at this signal's frequency
positions_scaled = array.positions * freq_scale
phase_arg = positions_scaled[:, 0] * np.cos(elevation) * np.cos(azimuth)
           + positions_scaled[:, 1] * np.cos(elevation) * np.sin(azimuth)
sv_k = element_pattern * np.exp(1j * 2 * np.pi * phase_arg)
```

This is implemented in a private `_wideband_steering_vector()` helper inside the dataset.

---

## Task 1: WidebandDFTarget and WidebandDirectionFindingDataset

**Files:**
- Create: `python/spectra/datasets/wideband_df.py`
- Create: `tests/test_wideband_df.py`

- [ ] **Step 1: Write the failing tests**

```python
# tests/test_wideband_df.py
"""Tests for WidebandDirectionFindingDataset."""
import numpy as np
import pytest
import torch
from spectra.arrays.array import ula
from spectra.waveforms import BPSK, QPSK, QAM16


def _make_ds(**kwargs):
    from spectra.datasets.wideband_df import WidebandDirectionFindingDataset
    arr = ula(num_elements=4, spacing=0.5, frequency=2.4e9)
    defaults = dict(
        array=arr,
        signal_pool=[BPSK(samples_per_symbol=4), QPSK(samples_per_symbol=4)],
        num_signals=2,
        num_snapshots=128,
        sample_rate=10e6,           # 10 MHz wideband capture
        capture_bandwidth=8e6,      # ±4 MHz around DC
        snr_range=(10.0, 20.0),
        azimuth_range=(np.deg2rad(20), np.deg2rad(160)),
        elevation_range=(0.0, 0.0),
        min_freq_separation=1e6,    # 1 MHz min separation
        min_angular_separation=np.deg2rad(15),
        num_samples=20,
        seed=0,
    )
    defaults.update(kwargs)
    return WidebandDirectionFindingDataset(**defaults)


def test_wbdf_len():
    ds = _make_ds(num_samples=25)
    assert len(ds) == 25


def test_wbdf_output_shape():
    from spectra.datasets.wideband_df import WidebandDFTarget
    ds = _make_ds()
    data, target = ds[0]
    assert isinstance(data, torch.Tensor)
    assert data.shape == (4, 2, 128)   # (N_elem, 2, T)
    assert isinstance(target, WidebandDFTarget)


def test_wbdf_target_fields():
    from spectra.datasets.wideband_df import WidebandDFTarget
    ds = _make_ds(num_signals=2)
    _, target = ds[0]
    assert target.num_signals == 2
    assert len(target.azimuths) == 2
    assert len(target.center_freqs) == 2
    assert len(target.snrs) == 2
    assert len(target.labels) == 2


def test_wbdf_deterministic():
    ds = _make_ds()
    d1, t1 = ds[3]
    d2, t2 = ds[3]
    assert torch.allclose(d1, d2)
    assert np.allclose(t1.azimuths, t2.azimuths)
    assert np.allclose(t1.center_freqs, t2.center_freqs)


def test_wbdf_center_freqs_in_range():
    """All center frequencies must lie within ±capture_bandwidth/2."""
    bw = 8e6
    ds = _make_ds(capture_bandwidth=bw, num_samples=30)
    for i in range(len(ds)):
        _, target = ds[i]
        for f in target.center_freqs:
            assert abs(f) <= bw / 2, f"Center freq {f/1e6:.2f} MHz out of ±{bw/2/1e6:.0f} MHz"


def test_wbdf_freq_separation():
    """Signals must be at least min_freq_separation apart."""
    ds = _make_ds(min_freq_separation=1.5e6, num_signals=2, num_samples=20)
    for i in range(len(ds)):
        _, target = ds[i]
        if target.num_signals >= 2:
            freqs = sorted(target.center_freqs)
            sep = freqs[1] - freqs[0]
            assert sep >= 1.5e6 - 1.0, f"Freq separation {sep/1e6:.3f} MHz < 1.5 MHz"


def test_wbdf_variable_num_signals():
    from spectra.datasets.wideband_df import WidebandDirectionFindingDataset
    arr = ula(num_elements=4, spacing=0.5, frequency=2.4e9)
    ds = WidebandDirectionFindingDataset(
        array=arr,
        signal_pool=[BPSK(samples_per_symbol=4), QPSK(samples_per_symbol=4)],
        num_signals=(1, 3),
        num_snapshots=64,
        sample_rate=10e6,
        capture_bandwidth=8e6,
        snr_range=(10.0, 20.0),
        num_samples=30,
        seed=42,
    )
    counts = {ds[i][1].num_signals for i in range(30)}
    assert len(counts) > 1, "With variable num_signals, should see different counts"
```

- [ ] **Step 2: Run tests to verify they fail**

```bash
/Users/gditzler/.venvs/base/bin/pytest tests/test_wideband_df.py::test_wbdf_len -v
```
Expected: `ModuleNotFoundError: No module named 'spectra.datasets.wideband_df'`

- [ ] **Step 3: Create `python/spectra/datasets/wideband_df.py`**

```python
# python/spectra/datasets/wideband_df.py
"""WidebandDirectionFindingDataset: joint wideband spectrum + DoA dataset."""

from __future__ import annotations

import warnings
from dataclasses import dataclass, field
from typing import Callable, List, Optional, Tuple, Union

import numpy as np
import torch
from torch.utils.data import Dataset

from spectra.arrays.array import AntennaArray
from spectra.datasets.iq_utils import truncate_pad
from spectra.scene.signal_desc import SignalDescription
from spectra.waveforms.base import Waveform


@dataclass
class WidebandDFTarget:
    """Ground-truth labels for a wideband direction-finding item.

    Attributes:
        azimuths: Source azimuth angles in radians, shape ``(num_signals,)``.
        elevations: Source elevation angles in radians, shape ``(num_signals,)``.
        center_freqs: Per-source center frequencies in Hz relative to DC,
            shape ``(num_signals,)``. Negative values are below DC.
        snrs: Per-source SNR in dB, shape ``(num_signals,)``.
        num_signals: Number of active signals.
        labels: Modulation label string per source.
        signal_descs: Full :class:`~spectra.scene.signal_desc.SignalDescription`
            per source.
    """

    azimuths: np.ndarray
    elevations: np.ndarray
    center_freqs: np.ndarray
    snrs: np.ndarray
    num_signals: int
    labels: List[str]
    signal_descs: List[SignalDescription] = field(default_factory=list)


class WidebandDirectionFindingDataset(Dataset):
    """On-the-fly wideband direction-finding dataset.

    Generates multi-antenna wideband IQ captures with ``num_signals`` co-channel
    sources.  Each source occupies a distinct sub-band (separated by at least
    ``min_freq_separation`` Hz) and arrives from a distinct spatial direction
    (separated by at least ``min_angular_separation`` radians when specified).

    The received wideband signal at element ``n`` is::

        x_n[t] = sum_k a_n(az_k, el_k, f_k) * s_k[t] * exp(j*2*pi*f_k*t/fs) + w_n[t]

    where ``a_n(az, el, f)`` is the frequency-dependent element response and
    ``s_k[t]`` is the baseband signal of source ``k``.

    **Frequency-dependent steering:** Positions are stored in wavelengths at
    ``array.reference_frequency``. For source ``k`` at frequency ``f_k``, the
    phase shifts are rescaled by ``f_k / reference_frequency``.

    **Output:** ``(Tensor[N_elements, 2, num_snapshots], WidebandDFTarget)``

    .. note::
        Use a custom ``collate_fn`` with DataLoader::

            def collate_fn(batch):
                return torch.stack([x for x, _ in batch]), [t for _, t in batch]

    Args:
        array: :class:`~spectra.arrays.array.AntennaArray` geometry.
        signal_pool: Waveforms to draw from for each source.
        num_signals: Fixed source count (int) or ``(min, max)`` range.
        num_snapshots: IQ samples per antenna element.
        sample_rate: Wideband receiver sample rate in Hz.
        capture_bandwidth: Usable bandwidth in Hz. Center frequencies are
            sampled uniformly from ``(-capture_bandwidth/2, +capture_bandwidth/2)``.
        snr_range: ``(min_db, max_db)`` per-source SNR.
        azimuth_range: ``(min_rad, max_rad)`` azimuth range.
        elevation_range: ``(min_rad, max_rad)`` elevation range. Default (0, 0).
        min_freq_separation: Minimum Hz between source center frequencies.
            If ``None``, no constraint is applied.
        min_angular_separation: Minimum radians between source angles.
            If ``None``, no constraint is applied.
        transform: Optional callable on the output tensor.
        num_samples: Dataset size.
        seed: Base integer seed.
    """

    def __init__(
        self,
        array: AntennaArray,
        signal_pool: List[Waveform],
        num_signals: Union[int, Tuple[int, int]],
        num_snapshots: int,
        sample_rate: float,
        capture_bandwidth: float,
        snr_range: Tuple[float, float],
        azimuth_range: Tuple[float, float] = (0.0, 2 * np.pi),
        elevation_range: Tuple[float, float] = (0.0, 0.0),
        min_freq_separation: Optional[float] = None,
        min_angular_separation: Optional[float] = None,
        transform: Optional[Callable] = None,
        num_samples: int = 10000,
        seed: int = 0,
    ):
        if num_snapshots <= 0:
            raise ValueError(f"num_snapshots must be positive, got {num_snapshots}")
        self.array = array
        self.signal_pool = signal_pool
        self.num_signals = num_signals
        self.num_snapshots = num_snapshots
        self.sample_rate = sample_rate
        self.capture_bandwidth = capture_bandwidth
        self.snr_range = snr_range
        self.azimuth_range = azimuth_range
        self.elevation_range = elevation_range
        self.min_freq_separation = min_freq_separation
        self.min_angular_separation = min_angular_separation
        self.transform = transform
        self.num_samples = num_samples
        self.seed = seed

    def __len__(self) -> int:
        return self.num_samples

    def __getitem__(self, idx: int) -> Tuple[torch.Tensor, WidebandDFTarget]:
        rng = np.random.default_rng(seed=(self.seed, idx))

        # Number of sources
        if isinstance(self.num_signals, tuple):
            n_src = int(rng.integers(self.num_signals[0], self.num_signals[1] + 1))
        else:
            n_src = int(self.num_signals)

        # Sample center frequencies within capture bandwidth
        center_freqs = self._sample_freqs(rng, n_src)

        # Sample azimuths/elevations
        azimuths, elevations = self._sample_angles(rng, n_src)

        # Sample SNRs
        snrs_db = rng.uniform(self.snr_range[0], self.snr_range[1], size=n_src)

        # Generate baseband signals and apply frequency shift
        source_iq = []
        labels = []
        signal_descs = []

        for k in range(n_src):
            wf_idx = int(rng.integers(0, len(self.signal_pool)))
            waveform = self.signal_pool[wf_idx]
            sps = getattr(waveform, "samples_per_symbol", 8)
            num_symbols = self.num_snapshots // sps + 1
            sig_seed = int(rng.integers(0, 2**32))

            # Generate baseband IQ
            iq = waveform.generate(
                num_symbols=num_symbols,
                sample_rate=self.sample_rate,
                seed=sig_seed,
            )
            iq = truncate_pad(iq, self.num_snapshots)

            # Frequency-shift to center_freqs[k]
            t = np.arange(self.num_snapshots) / self.sample_rate
            iq = iq * np.exp(1j * 2 * np.pi * center_freqs[k] * t)

            bw = waveform.bandwidth(self.sample_rate)
            desc = SignalDescription(
                t_start=0.0,
                t_stop=self.num_snapshots / self.sample_rate,
                f_low=center_freqs[k] - bw / 2,
                f_high=center_freqs[k] + bw / 2,
                label=waveform.label,
                snr=float(snrs_db[k]),
                modulation_params={
                    "doa": {
                        "azimuth_rad": float(azimuths[k]),
                        "elevation_rad": float(elevations[k]),
                    },
                    "center_freq_hz": float(center_freqs[k]),
                },
            )
            source_iq.append(iq)
            labels.append(waveform.label)
            signal_descs.append(desc)

        # Spatial mixing with frequency-dependent steering vectors
        X = self._wideband_spatial_mix(source_iq, azimuths, elevations, center_freqs, snrs_db, rng)

        # Convert (N, T) complex → (N, 2, T) float32
        tensor = torch.from_numpy(
            np.stack([X.real, X.imag], axis=1).astype(np.float32)
        )

        if self.transform is not None:
            tensor = self.transform(tensor)

        target = WidebandDFTarget(
            azimuths=azimuths,
            elevations=elevations,
            center_freqs=center_freqs,
            snrs=snrs_db,
            num_signals=n_src,
            labels=labels,
            signal_descs=signal_descs,
        )
        return tensor, target

    # ── private helpers ────────────────────────────────────────────────────────

    def _sample_freqs(self, rng: np.random.Generator, n_src: int) -> np.ndarray:
        """Sample center frequencies within ±capture_bandwidth/2."""
        f_min = -self.capture_bandwidth / 2.0
        f_max = self.capture_bandwidth / 2.0

        if self.min_freq_separation is None or n_src == 1:
            return rng.uniform(f_min, f_max, size=n_src)

        max_attempts = 1000
        for _ in range(max_attempts):
            freqs = rng.uniform(f_min, f_max, size=n_src)
            freqs_sorted = np.sort(freqs)
            if np.all(np.diff(freqs_sorted) >= self.min_freq_separation):
                return freqs
        warnings.warn(
            f"Could not satisfy min_freq_separation={self.min_freq_separation:.0f} Hz "
            f"for {n_src} signals after {max_attempts} attempts.",
            UserWarning,
            stacklevel=2,
        )
        return freqs

    def _sample_angles(
        self, rng: np.random.Generator, n_src: int
    ) -> Tuple[np.ndarray, np.ndarray]:
        """Sample azimuth/elevation angles with optional separation constraint."""
        az_min, az_max = self.azimuth_range
        el_min, el_max = self.elevation_range

        if self.min_angular_separation is None or n_src == 1:
            return rng.uniform(az_min, az_max, n_src), rng.uniform(el_min, el_max, n_src)

        max_attempts = 1000
        for _ in range(max_attempts):
            azimuths = rng.uniform(az_min, az_max, n_src)
            elevations = rng.uniform(el_min, el_max, n_src)
            ok = True
            for i in range(n_src):
                for j in range(i + 1, n_src):
                    cos_sep = (
                        np.sin(elevations[i]) * np.sin(elevations[j])
                        + np.cos(elevations[i]) * np.cos(elevations[j])
                        * np.cos(azimuths[i] - azimuths[j])
                    )
                    sep = float(np.arccos(np.clip(cos_sep, -1.0, 1.0)))
                    if sep < self.min_angular_separation:
                        ok = False
                        break
                if not ok:
                    break
            if ok:
                return azimuths, elevations

        warnings.warn(
            f"Could not satisfy min_angular_separation for {n_src} sources.",
            UserWarning,
            stacklevel=2,
        )
        return azimuths, elevations

    def _wideband_steering_vector(
        self, azimuth: float, elevation: float, center_freq: float
    ) -> np.ndarray:
        """Compute frequency-scaled steering vector for a single direction.

        Scales the stored element positions (in wavelengths at reference_frequency)
        to wavelengths at ``center_freq``, then computes phase shifts.

        Returns:
            Complex array, shape ``(N_elements,)``.
        """
        freq_scale = (
            (self.array.reference_frequency + center_freq) / self.array.reference_frequency
        )
        x = self.array.positions[:, 0] * freq_scale
        y = self.array.positions[:, 1] * freq_scale
        cos_el = np.cos(elevation)
        cos_az = np.cos(azimuth)
        sin_az = np.sin(azimuth)
        phase_arg = x * cos_el * cos_az + y * cos_el * sin_az
        phase = np.exp(1j * 2 * np.pi * phase_arg)

        # Apply per-element patterns
        pattern = np.zeros(self.array.num_elements, dtype=complex)
        az_arr = np.array([azimuth])
        el_arr = np.array([elevation])
        for i, elem in enumerate(self.array.elements):
            pattern[i] = elem.pattern(az_arr, el_arr)[0]

        return pattern * phase

    def _wideband_spatial_mix(
        self,
        source_iq: List[np.ndarray],
        azimuths: np.ndarray,
        elevations: np.ndarray,
        center_freqs: np.ndarray,
        snrs_db: np.ndarray,
        rng: np.random.Generator,
    ) -> np.ndarray:
        """Sum frequency-shifted signals with frequency-dependent steering + noise."""
        n_elem = self.array.num_elements
        n_snap = self.num_snapshots
        noise_power = 1.0
        noise = np.sqrt(noise_power / 2.0) * (
            rng.standard_normal((n_elem, n_snap))
            + 1j * rng.standard_normal((n_elem, n_snap))
        )
        X = np.zeros((n_elem, n_snap), dtype=complex)
        for iq, az, el, f_k, snr_db in zip(source_iq, azimuths, elevations, center_freqs, snrs_db):
            sv = self._wideband_steering_vector(az, el, f_k)
            sig_power = np.mean(np.abs(iq) ** 2)
            if sig_power > 0:
                scale = np.sqrt((10.0 ** (snr_db / 10.0)) * noise_power / sig_power)
                iq = iq * scale
            X += sv[:, np.newaxis] * iq[np.newaxis, :]
        return X + noise
```

- [ ] **Step 4: Run tests to verify they pass**

```bash
/Users/gditzler/.venvs/base/bin/pytest tests/test_wideband_df.py -v
```
Expected: 6/6 PASS

- [ ] **Step 5: Update `python/spectra/datasets/__init__.py`**

```python
from spectra.datasets.wideband_df import WidebandDirectionFindingDataset, WidebandDFTarget
```

Add `"WidebandDirectionFindingDataset"` and `"WidebandDFTarget"` to `__all__`.

- [ ] **Step 6: Commit**

```bash
git add python/spectra/datasets/wideband_df.py python/spectra/datasets/__init__.py tests/test_wideband_df.py
git commit -m "feat(datasets): add WidebandDirectionFindingDataset with frequency-dependent steering"
```

---

## Task 2: Python Example Script

**Files:**
- Create: `examples/16_wideband_direction_finding.py`

- [ ] **Step 1: Create the script**

```python
# examples/16_wideband_direction_finding.py
"""Example 16 — Wideband Direction-Finding Dataset
===================================================
Level: Intermediate-Advanced

This example shows how to:
  1. Build a WidebandDirectionFindingDataset with 3 co-channel sources
  2. Visualise the multi-antenna wideband spectrogram
  3. Apply MUSIC sub-band processing at each source's center frequency
  4. Compare per-frequency estimated azimuths to ground truth

Run:
    python examples/16_wideband_direction_finding.py
"""

import os
import sys
sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))
sys.path.insert(0, os.path.dirname(__file__))

import numpy as np
import matplotlib.pyplot as plt
import torch
from torch.utils.data import DataLoader

from spectra.arrays import ula
from spectra.datasets import WidebandDirectionFindingDataset
from spectra.algorithms import music, find_peaks_doa
from spectra.waveforms import BPSK, QPSK, QAM16

from plot_helpers import savefig, OUTPUT_DIR

# ── Configuration ──────────────────────────────────────────────────────────────

FREQ_HZ       = 2.4e9
N_ELEMENTS    = 8
SPACING       = 0.5
N_SNAPSHOTS   = 512
SAMPLE_RATE   = 20e6       # 20 MHz wideband
CAPTURE_BW    = 16e6       # usable ±8 MHz
N_SOURCES     = 3
SNR_RANGE     = (15.0, 25.0)
N_SAMPLES     = 100
SEED          = 42
MIN_FREQ_SEP  = 3e6        # 3 MHz minimum spacing
MIN_ANG_SEP   = np.deg2rad(15)

SCAN_DEG = np.linspace(5, 175, 512)
SCAN_RAD = np.deg2rad(SCAN_DEG)

# ── 1. Build Dataset ───────────────────────────────────────────────────────────

arr = ula(num_elements=N_ELEMENTS, spacing=SPACING, frequency=FREQ_HZ)
signal_pool = [BPSK(samples_per_symbol=2), QPSK(samples_per_symbol=2), QAM16(samples_per_symbol=2)]

ds = WidebandDirectionFindingDataset(
    array=arr,
    signal_pool=signal_pool,
    num_signals=N_SOURCES,
    num_snapshots=N_SNAPSHOTS,
    sample_rate=SAMPLE_RATE,
    capture_bandwidth=CAPTURE_BW,
    snr_range=SNR_RANGE,
    azimuth_range=(np.deg2rad(10), np.deg2rad(170)),
    elevation_range=(0.0, 0.0),
    min_freq_separation=MIN_FREQ_SEP,
    min_angular_separation=MIN_ANG_SEP,
    num_samples=N_SAMPLES,
    seed=SEED,
)
print(f"Dataset: {len(ds)} wideband captures, {N_SOURCES} sources each")

# ── 2. Inspect One Sample ──────────────────────────────────────────────────────

data, target = ds[0]
print("\nSample 0 ground truth:")
for k in range(target.num_signals):
    print(f"  Source {k}: az={np.rad2deg(target.azimuths[k]):.1f}°, "
          f"f_c={target.center_freqs[k]/1e6:+.1f} MHz, "
          f"SNR={target.snrs[k]:.1f} dB, label={target.labels[k]}")

# ── 3. Multi-Antenna Spectrogram (element 0) ──────────────────────────────────

iq0 = data[0, 0, :].numpy() + 1j * data[0, 1, :].numpy()
NFFT = 128
hop = 32
freqs_stft = np.fft.fftshift(np.fft.fftfreq(NFFT, d=1.0 / SAMPLE_RATE))
n_frames = (N_SNAPSHOTS - NFFT) // hop + 1
spec = np.zeros((NFFT, n_frames), dtype=complex)
for fi in range(n_frames):
    seg = iq0[fi * hop: fi * hop + NFFT]
    spec[:, fi] = np.fft.fftshift(np.fft.fft(seg * np.hanning(NFFT)))
spec_db = 10 * np.log10(np.abs(spec) ** 2 + 1e-30)

fig, ax = plt.subplots(figsize=(10, 4))
ax.imshow(
    spec_db,
    aspect="auto", origin="lower",
    extent=[0, n_frames, freqs_stft[0] / 1e6, freqs_stft[-1] / 1e6],
    cmap="viridis",
)
for fc in target.center_freqs:
    ax.axhline(fc / 1e6, color="crimson", linestyle="--", linewidth=1.2)
ax.set_xlabel("Frame")
ax.set_ylabel("Frequency (MHz)")
ax.set_title(f"Wideband Spectrogram — element 0 (sample 0)")
plt.tight_layout()
savefig("16_wideband_spectrogram.png")

# ── 4. Sub-Band MUSIC at Each Source Frequency ────────────────────────────────

X_full = data[:, 0, :].numpy() + 1j * data[:, 1, :].numpy()  # (N, T)

fig, axes = plt.subplots(N_SOURCES, 1, figsize=(9, 3 * N_SOURCES), sharex=True)
for k, ax in enumerate(axes):
    fc = target.center_freqs[k]
    true_az = target.azimuths[k]

    # Narrow-band filter: frequency-shift down then low-pass via truncated FFT
    t = np.arange(N_SNAPSHOTS) / SAMPLE_RATE
    X_shift = X_full * np.exp(-1j * 2 * np.pi * fc * t)[np.newaxis, :]
    # Keep central BW_sub samples in frequency domain = 2 MHz sub-band
    BW_sub = 2e6
    fft_len = N_SNAPSHOTS
    sub_bins = int(BW_sub / SAMPLE_RATE * fft_len)
    X_fft = np.fft.fft(X_shift, axis=1)
    X_fft[:, sub_bins // 2: fft_len - sub_bins // 2] = 0
    X_sub = np.fft.ifft(X_fft, axis=1)

    # Build a frequency-adjusted array for MUSIC
    # Pass the reference array; steering vectors will be slightly off at ≠ reference freq
    # (good enough for visualisation at close-to-carrier frequencies)
    spectrum = music(X_sub, num_sources=1, array=arr, scan_angles=SCAN_RAD)
    peaks = find_peaks_doa(spectrum, SCAN_RAD, num_peaks=1)

    ax.plot(SCAN_DEG, 10 * np.log10(spectrum / spectrum.max()), color="steelblue", linewidth=1.0)
    ax.axvline(np.rad2deg(true_az), color="crimson", linestyle="--", linewidth=1.5,
               label=f"True {np.rad2deg(true_az):.1f}°")
    ax.axvline(np.rad2deg(peaks[0]), color="orange", linestyle=":", linewidth=1.5,
               label=f"Est. {np.rad2deg(peaks[0]):.1f}°")
    ax.set_ylabel(f"Source {k}\n{fc/1e6:+.1f} MHz")
    ax.legend(fontsize=8)
    ax.grid(True, alpha=0.3)
axes[-1].set_xlabel("Azimuth (degrees)")
fig.suptitle("Sub-Band MUSIC per Source")
plt.tight_layout()
savefig("16_subband_music.png")

# ── 5. Wideband Capture Statistics ────────────────────────────────────────────

all_freqs_mhz = []
for i in range(len(ds)):
    _, t = ds[i]
    all_freqs_mhz.extend([f / 1e6 for f in t.center_freqs])

fig, ax = plt.subplots(figsize=(7, 3))
ax.hist(all_freqs_mhz, bins=40, color="steelblue", edgecolor="white")
ax.set_xlabel("Center frequency (MHz relative to DC)")
ax.set_ylabel("Count")
ax.set_title(f"Source frequency distribution ({N_SAMPLES} samples × {N_SOURCES} sources)")
ax.grid(True, axis="y", alpha=0.3)
plt.tight_layout()
savefig("16_freq_distribution.png")

print(f"\nAll figures saved to {OUTPUT_DIR}")
```

- [ ] **Step 2: Verify script runs**

```bash
cd /Users/gditzler/git/SPECTRA && /Users/gditzler/.venvs/base/bin/python examples/16_wideband_direction_finding.py
```
Expected: prints ground truth for sample 0, saves 3 figures.

- [ ] **Step 3: Commit**

```bash
git add examples/16_wideband_direction_finding.py
git commit -m "feat(examples): add wideband direction-finding example script"
```

---

## Task 3: Jupyter Notebook

**Files:**
- Create: `examples/16_wideband_direction_finding.ipynb`

- [ ] **Step 1: Create `examples/16_wideband_direction_finding.ipynb`**

nbformat 4 / nbformat_minor 5. Cell layout:

| # | Type | Content |
|---|------|---------|
| 0 | markdown | Title + learning objectives |
| 1 | code | Imports + constants |
| 2 | markdown | "## 1. Dataset Configuration" — explain wideband vs narrowband DF |
| 3 | code | Build `arr`, build `ds`, print sample 0 target |
| 4 | markdown | "## 2. Wideband Spectrogram" — explain STFT, frequency-shifted sources |
| 5 | code | STFT of element 0 IQ, imshow with frequency markers |
| 6 | markdown | "## 3. Sub-Band MUSIC" — explain frequency downconversion → narrow-band array processing |
| 7 | code | Sub-band MUSIC for each source, subplot |
| 8 | markdown | "## 4. Frequency Distribution" |
| 9 | code | Histogram of center frequencies across all samples |

Cell 0 content:
```markdown
# Example 16 — Wideband Direction-Finding Dataset

**Level:** Intermediate–Advanced

After working through this notebook you will know how to:

- Configure a `WidebandDirectionFindingDataset` with multiple co-channel sources at different center frequencies
- Understand `WidebandDFTarget` (per-signal azimuth, frequency, SNR, label)
- Visualise multi-antenna wideband spectrograms
- Apply sub-band frequency downconversion to isolate each source
- Use MUSIC on the narrowband sub-band captures for per-source DoA
```

Cell 2 explanation:
```markdown
## 1. Dataset Configuration

In `DirectionFindingDataset` all sources share the same carrier frequency (the array's reference frequency). In `WidebandDirectionFindingDataset` each source occupies a distinct sub-band within `capture_bandwidth`. The frequency-dependent steering vector rescales element positions to wavelengths at each source's carrier, so spatially-coloured interference from co-frequency signals is physically accurate. The `min_freq_separation` parameter enforces a guard band between sources.
```

Cell 6 explanation:
```markdown
## 3. Sub-Band MUSIC

Standard narrowband MUSIC assumes all sources share one carrier. For wideband captures we isolate each source by: (1) frequency-shifting the wideband snapshot to put source *k* at DC, (2) low-pass filtering to keep a narrow sub-band, (3) applying MUSIC to the filtered sub-band. This approximation is accurate when the source bandwidth is small relative to the array's spatial resolution.
```

- [ ] **Step 2: Execute notebook**

```bash
cd /Users/gditzler/git/SPECTRA/examples && /Users/gditzler/.venvs/base/bin/jupyter nbconvert --to notebook --execute 16_wideband_direction_finding.ipynb --output /tmp/check_16.ipynb 2>&1 && echo "OK"
```
Expected: `OK`

- [ ] **Step 3: Commit**

```bash
git add examples/16_wideband_direction_finding.ipynb
git commit -m "feat(examples): add wideband direction-finding Jupyter notebook"
```

---

## Task 4: Update examples/README.md

- [ ] **Step 1: Add entry after example 15**

```markdown
### 16 — Wideband Direction-Finding Dataset (Intermediate–Advanced)

Build a `WidebandDirectionFindingDataset` with multiple co-channel sources at
different center frequencies and azimuths. Visualise multi-antenna wideband
spectrograms, apply sub-band frequency downconversion, and run MUSIC on each
sub-band to estimate per-source DoA.

```bash
cd examples && python 16_wideband_direction_finding.py
```
```

Also add to File Structure:
```
  16_wideband_direction_finding.ipynb / .py  # Intermediate-Advanced
```

- [ ] **Step 2: Commit**

```bash
git add examples/README.md
git commit -m "docs(examples): add example 16 to README"
```

---

## Task 5: Full Verification

- [ ] **Step 1: Run full test suite**

```bash
cd /Users/gditzler/git/SPECTRA && /Users/gditzler/.venvs/base/bin/pytest -q --tb=short 2>&1 | tail -5
```

- [ ] **Step 2: Confirm imports**

```bash
/Users/gditzler/.venvs/base/bin/python -c "
from spectra.datasets import WidebandDirectionFindingDataset, WidebandDFTarget
print('WidebandDF imports OK')
"
```
