# OTA Dataset Integration Implementation Plan

> **For agentic workers:** REQUIRED: Use superpowers:subagent-driven-development (if subagents available) or superpowers:executing-plans to implement this plan. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Add `LiveCapture` (SoapySDR-backed with ring buffer and file modes), `OTADataset`, and `HybridDataLoader` for mixing live/file OTA captures with synthetic data.

**Architecture:** A `CaptureBackend` ABC enables pluggable hardware. `SoapyBackend` is the reference implementation (optional dep). `LiveCapture` orchestrates capture into memory or SigMF files. `OTADataset` wraps captures as a PyTorch Dataset. `HybridDataLoader` interleaves synthetic + OTA batches at a configurable ratio. All core logic is testable with a `MockBackend` — no hardware needed for CI.

**Tech Stack:** Python 3.10+, NumPy, PyTorch, pytest. Optional: SoapySDR (`spectra[sdr]`).

**Spec:** `docs/superpowers/specs/2026-03-17-ota-integration-design.md`

---

## File Structure

| File | Responsibility |
|------|---------------|
| `python/spectra/capture/__init__.py` | Package init — exports |
| `python/spectra/capture/backend.py` | `CaptureBackend` ABC |
| `python/spectra/capture/metadata.py` | `CaptureMetadata` dataclass + RF measurement helpers |
| `python/spectra/capture/soapy.py` | `SoapyBackend` (optional SoapySDR wrapper) |
| `python/spectra/capture/live.py` | `LiveCapture` orchestrator (ring buffer + file modes) |
| `python/spectra/datasets/ota.py` | `OTADataset` |
| `python/spectra/datasets/__init__.py` | Modify: add `OTADataset` export |
| `python/spectra/streaming.py` | Modify: add `HybridDataLoader` |
| `pyproject.toml` | Modify: add `sdr` optional dependency |
| `tests/test_capture.py` | Tests for backend, metadata, LiveCapture |
| `tests/test_ota_dataset.py` | Tests for OTADataset |
| `tests/test_hybrid_loader.py` | Tests for HybridDataLoader |

---

## Task 1: CaptureBackend ABC + CaptureMetadata (`spectra/capture/`)

**Files:**
- Create: `python/spectra/capture/__init__.py`
- Create: `python/spectra/capture/backend.py`
- Create: `python/spectra/capture/metadata.py`
- Create: `tests/test_capture.py`

- [ ] **Step 1: Write the failing tests**

```python
# tests/test_capture.py
"""Tests for CaptureBackend, CaptureMetadata, and LiveCapture."""
import numpy as np
import pytest
import tempfile
import os


# ── MockBackend for testing without hardware ────────────────────────────────

class MockBackend:
    """Test backend that generates deterministic synthetic IQ."""

    def __init__(self, seed=42):
        self._rng = np.random.default_rng(seed)
        self._active = False
        self._configured = False

    def configure(self, center_frequency, sample_rate, gain, **kwargs):
        self._center_frequency = center_frequency
        self._sample_rate = sample_rate
        self._gain = gain
        self._configured = True

    def read(self, num_samples):
        return (self._rng.standard_normal(num_samples)
                + 1j * self._rng.standard_normal(num_samples)).astype(np.complex64)

    def start(self):
        self._active = True

    def stop(self):
        self._active = False

    @property
    def is_active(self):
        return self._active

    def __enter__(self):
        self.start()
        return self

    def __exit__(self, *args):
        self.stop()


# ── CaptureBackend ABC tests ───────────────────────────────────────────────

def test_backend_abc_not_instantiable():
    from spectra.capture.backend import CaptureBackend
    with pytest.raises(TypeError):
        CaptureBackend()


def test_mock_backend_satisfies_protocol():
    """MockBackend should work as a CaptureBackend."""
    backend = MockBackend()
    backend.configure(center_frequency=915e6, sample_rate=2.4e6, gain=30.0)
    backend.start()
    assert backend.is_active
    iq = backend.read(1024)
    assert iq.shape == (1024,)
    assert iq.dtype == np.complex64
    backend.stop()
    assert not backend.is_active


def test_backend_context_manager():
    backend = MockBackend()
    with backend:
        assert backend.is_active
    assert not backend.is_active


# ── CaptureMetadata tests ─────────────────────────────────────────────────

def test_metadata_fields():
    from spectra.capture.metadata import CaptureMetadata
    m = CaptureMetadata(
        center_frequency=915e6, sample_rate=2.4e6, gain=30.0,
        timestamp=1700000000.0, duration=0.01,
    )
    assert m.center_frequency == 915e6
    assert m.antenna is None
    assert m.noise_floor_dbm is None


def test_compute_rf_measurements():
    from spectra.capture.metadata import compute_rf_measurements
    rng = np.random.default_rng(42)
    iq = (rng.standard_normal(4096) + 1j * rng.standard_normal(4096)).astype(np.complex64)
    noise_floor, signal_power, occupancy = compute_rf_measurements(iq)
    assert isinstance(noise_floor, float)
    assert isinstance(signal_power, float)
    assert 0.0 <= occupancy <= 1.0
```

- [ ] **Step 2: Run tests to verify they fail**

```bash
/Users/gditzler/.venvs/base/bin/pytest tests/test_capture.py -v
```
Expected: `ModuleNotFoundError: No module named 'spectra.capture'`

- [ ] **Step 3: Write `backend.py`**

```python
# python/spectra/capture/backend.py
"""CaptureBackend abstract base class for SDR hardware abstraction."""

from __future__ import annotations

from abc import ABC, abstractmethod

import numpy as np


class CaptureBackend(ABC):
    """Abstract base class for SDR capture backends.

    Subclasses wrap specific hardware APIs (SoapySDR, GNU Radio, etc.).
    Context manager support enables automatic cleanup.
    """

    @abstractmethod
    def configure(
        self, center_frequency: float, sample_rate: float, gain: float, **kwargs
    ) -> None:
        """Configure the radio parameters."""

    @abstractmethod
    def read(self, num_samples: int) -> np.ndarray:
        """Read IQ samples from the device.

        Returns:
            Complex64 array of shape ``(num_samples,)``.
        """

    @abstractmethod
    def start(self) -> None:
        """Start the receive stream."""

    @abstractmethod
    def stop(self) -> None:
        """Stop the receive stream."""

    @property
    @abstractmethod
    def is_active(self) -> bool:
        """Whether the stream is currently active."""

    def __enter__(self):
        self.start()
        return self

    def __exit__(self, *args):
        self.stop()
```

- [ ] **Step 4: Write `metadata.py`**

```python
# python/spectra/capture/metadata.py
"""CaptureMetadata dataclass and RF measurement helpers."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Optional, Tuple

import numpy as np


@dataclass
class CaptureMetadata:
    """Metadata for a captured IQ segment.

    Attributes:
        center_frequency: Center frequency in Hz.
        sample_rate: Sample rate in Hz.
        gain: Receiver gain in dB.
        timestamp: Unix epoch timestamp of capture start.
        duration: Capture duration in seconds.
        antenna: Antenna port identifier.
        location: ``(latitude, longitude)`` tuple.
        capture_id: Unique capture identifier.
        notes: Free-form notes.
        noise_floor_dbm: Estimated noise floor in dBm.
        signal_power_dbm: Mean signal power in dBm.
        occupancy_fraction: Fraction of samples above noise floor + 6 dB.
    """

    center_frequency: float
    sample_rate: float
    gain: float
    timestamp: float
    duration: float
    antenna: Optional[str] = None
    location: Optional[Tuple[float, float]] = None
    capture_id: Optional[str] = None
    notes: Optional[str] = None
    noise_floor_dbm: Optional[float] = None
    signal_power_dbm: Optional[float] = None
    occupancy_fraction: Optional[float] = None


def compute_rf_measurements(iq: np.ndarray) -> Tuple[float, float, float]:
    """Compute RF measurements from an IQ segment.

    Returns:
        ``(noise_floor_dbm, signal_power_dbm, occupancy_fraction)``.
    """
    power = np.abs(iq) ** 2
    power_db = 10.0 * np.log10(power + 1e-30)

    noise_floor_dbm = float(np.percentile(power_db, 10))
    signal_power_dbm = float(10.0 * np.log10(np.mean(power) + 1e-30))
    occupancy_fraction = float(np.mean(power_db > noise_floor_dbm + 6.0))

    return noise_floor_dbm, signal_power_dbm, occupancy_fraction
```

- [ ] **Step 5: Write `__init__.py`**

```python
# python/spectra/capture/__init__.py
from spectra.capture.backend import CaptureBackend
from spectra.capture.metadata import CaptureMetadata, compute_rf_measurements

__all__ = [
    "CaptureBackend",
    "CaptureMetadata",
    "compute_rf_measurements",
]
```

- [ ] **Step 6: Run tests to verify they pass**

```bash
/Users/gditzler/.venvs/base/bin/pytest tests/test_capture.py -v
```
Expected: 5/5 PASS

- [ ] **Step 7: Commit**

```bash
git add python/spectra/capture/__init__.py python/spectra/capture/backend.py python/spectra/capture/metadata.py tests/test_capture.py
git commit -m "feat(capture): add CaptureBackend ABC and CaptureMetadata dataclass"
```

---

## Task 2: SoapyBackend + Optional Dependency

**Files:**
- Create: `python/spectra/capture/soapy.py`
- Modify: `python/spectra/capture/__init__.py`
- Modify: `pyproject.toml`

- [ ] **Step 1: Write `soapy.py`**

```python
# python/spectra/capture/soapy.py
"""SoapySDR capture backend.

Requires the ``spectra[sdr]`` optional dependency::

    pip install spectra[sdr]
"""

from __future__ import annotations

from typing import Dict, Optional

import numpy as np

from spectra.capture.backend import CaptureBackend

try:
    import SoapySDR
    from SoapySDR import SOAPY_SDR_CF32, SOAPY_SDR_RX

    _HAS_SOAPY = True
except ImportError:
    _HAS_SOAPY = False


class SoapyBackend(CaptureBackend):
    """SoapySDR capture backend.

    Wraps ``SoapySDR.Device`` for receive-mode IQ capture.
    Supports 50+ SDR devices (RTL-SDR, USRP, HackRF, LimeSDR, etc.).

    Args:
        driver: SoapySDR driver name (e.g., ``"rtlsdr"``, ``"uhd"``).
        device_args: Additional device arguments.
        channel: Receive channel index.
    """

    def __init__(
        self,
        driver: str = "rtlsdr",
        device_args: Optional[Dict[str, str]] = None,
        channel: int = 0,
    ) -> None:
        if not _HAS_SOAPY:
            raise ImportError(
                "SoapySDR is required for SoapyBackend. "
                "Install with: pip install spectra[sdr]"
            )
        args = {"driver": driver}
        if device_args:
            args.update(device_args)
        self._device = SoapySDR.Device(args)
        self._channel = channel
        self._stream = None
        self._active = False

    def configure(
        self, center_frequency: float, sample_rate: float, gain: float, **kwargs
    ) -> None:
        self._device.setSampleRate(SOAPY_SDR_RX, self._channel, sample_rate)
        self._device.setFrequency(SOAPY_SDR_RX, self._channel, center_frequency)
        self._device.setGain(SOAPY_SDR_RX, self._channel, gain)
        for key, val in kwargs.items():
            self._device.writeSetting(key, str(val))

    def start(self) -> None:
        self._stream = self._device.setupStream(SOAPY_SDR_RX, SOAPY_SDR_CF32, [self._channel])
        self._device.activateStream(self._stream)
        self._active = True

    def stop(self) -> None:
        if self._stream is not None:
            self._device.deactivateStream(self._stream)
            self._device.closeStream(self._stream)
            self._stream = None
        self._active = False

    def read(self, num_samples: int) -> np.ndarray:
        buf = np.zeros(num_samples, dtype=np.complex64)
        sr = self._device.readStream(self._stream, [buf], num_samples)
        if sr.ret > 0:
            return buf[: sr.ret]
        return buf

    @property
    def is_active(self) -> bool:
        return self._active
```

- [ ] **Step 2: Update `capture/__init__.py`**

Add conditional import:

```python
try:
    from spectra.capture.soapy import SoapyBackend
    __all__.append("SoapyBackend")
except ImportError:
    pass
```

- [ ] **Step 3: Add `sdr` optional dependency to `pyproject.toml`**

Add to `[project.optional-dependencies]`:

```toml
sdr = ["SoapySDR>=0.8"]
```

And add `"SoapySDR>=0.8"` to the `all` list.

- [ ] **Step 4: Verify import guard works**

```bash
/Users/gditzler/.venvs/base/bin/python -c "
from spectra.capture import CaptureBackend, CaptureMetadata
print('Base imports OK (no SoapySDR needed)')
try:
    from spectra.capture import SoapyBackend
    print('SoapyBackend available')
except ImportError:
    print('SoapyBackend not available (expected without SoapySDR)')
"
```

- [ ] **Step 5: Commit**

```bash
git add python/spectra/capture/soapy.py python/spectra/capture/__init__.py pyproject.toml
git commit -m "feat(capture): add SoapyBackend with optional SoapySDR dependency"
```

---

## Task 3: LiveCapture Orchestrator

**Files:**
- Create: `python/spectra/capture/live.py`
- Modify: `python/spectra/capture/__init__.py`
- Modify: `tests/test_capture.py`

- [ ] **Step 1: Add tests to `tests/test_capture.py`**

Append these tests (they use the `MockBackend` defined in Task 1):

```python
# ── LiveCapture tests ──────────────────────────────────────────────────────

def test_live_capture_memory_mode():
    from spectra.capture.live import LiveCapture
    backend = MockBackend(seed=42)
    backend.configure(center_frequency=915e6, sample_rate=2.4e6, gain=30.0)

    lc = LiveCapture(
        backend=backend, center_frequency=915e6, sample_rate=2.4e6, gain=30.0,
        segment_length=1024, mode="memory", buffer_size=10,
    )
    lc.capture(num_segments=3)
    assert len(lc) == 3

    iq, meta = lc.get_segment(0)
    assert iq.shape == (1024,)
    assert iq.dtype == np.complex64
    assert meta.center_frequency == 915e6
    assert meta.noise_floor_dbm is not None


def test_live_capture_ring_buffer_wraps():
    from spectra.capture.live import LiveCapture
    backend = MockBackend(seed=0)
    backend.configure(center_frequency=100e6, sample_rate=1e6, gain=20.0)

    lc = LiveCapture(
        backend=backend, center_frequency=100e6, sample_rate=1e6, gain=20.0,
        segment_length=256, mode="memory", buffer_size=5,
    )
    lc.capture(num_segments=8)
    assert len(lc) == 5  # buffer wraps at 5


def test_live_capture_file_mode():
    from spectra.capture.live import LiveCapture
    backend = MockBackend(seed=42)
    backend.configure(center_frequency=915e6, sample_rate=2.4e6, gain=30.0)

    with tempfile.TemporaryDirectory() as tmpdir:
        lc = LiveCapture(
            backend=backend, center_frequency=915e6, sample_rate=2.4e6, gain=30.0,
            segment_length=512, mode="file", output_dir=tmpdir,
        )
        lc.capture(num_segments=2)
        assert len(lc) == 2

        iq, meta = lc.get_segment(0)
        assert iq.shape == (512,)
        assert meta.center_frequency == 915e6

        # Verify SigMF files were created
        sigmf_files = [f for f in os.listdir(tmpdir) if f.endswith(".sigmf-meta")]
        assert len(sigmf_files) == 2


def test_live_capture_continuous():
    from spectra.capture.live import LiveCapture
    backend = MockBackend(seed=42)
    backend.configure(center_frequency=915e6, sample_rate=1e6, gain=30.0)

    lc = LiveCapture(
        backend=backend, center_frequency=915e6, sample_rate=1e6, gain=30.0,
        segment_length=1000, mode="memory", buffer_size=100,
    )
    # 0.005 seconds at 1 MHz sample rate = 5000 samples = 5 segments of 1000
    lc.capture_continuous(duration=0.005)
    assert len(lc) == 5


def test_live_capture_metadata_environment():
    from spectra.capture.live import LiveCapture
    backend = MockBackend(seed=42)
    backend.configure(center_frequency=915e6, sample_rate=2.4e6, gain=30.0)

    lc = LiveCapture(
        backend=backend, center_frequency=915e6, sample_rate=2.4e6, gain=30.0,
        segment_length=256, mode="memory", buffer_size=10,
        antenna="RX1", location=(40.0, -74.0),
    )
    lc.capture(num_segments=1)
    _, meta = lc.get_segment(0)
    assert meta.antenna == "RX1"
    assert meta.location == (40.0, -74.0)
```

- [ ] **Step 2: Write `live.py`**

```python
# python/spectra/capture/live.py
"""LiveCapture orchestrator for real-time IQ capture."""

from __future__ import annotations

import time
import uuid
from typing import List, Optional, Tuple

import numpy as np

from spectra.capture.backend import CaptureBackend
from spectra.capture.metadata import CaptureMetadata, compute_rf_measurements


class LiveCapture:
    """Orchestrates IQ capture from an SDR backend.

    Supports two modes:

    - ``"memory"``: Ring buffer in RAM. Fast, but lost on exit.
    - ``"file"``: Writes SigMF files to disk. Persistent.

    Args:
        backend: Configured :class:`CaptureBackend` instance.
        center_frequency: Capture center frequency in Hz.
        sample_rate: Capture sample rate in Hz.
        gain: Receiver gain in dB.
        segment_length: IQ samples per capture segment.
        mode: ``"memory"`` or ``"file"``.
        buffer_size: Max segments in ring buffer (memory mode only).
        output_dir: Directory for SigMF output (file mode only).
        antenna: Antenna port identifier.
        location: ``(lat, lon)`` tuple.
    """

    def __init__(
        self,
        backend: CaptureBackend,
        center_frequency: float,
        sample_rate: float,
        gain: float,
        segment_length: int = 4096,
        mode: str = "memory",
        buffer_size: int = 1000,
        output_dir: Optional[str] = None,
        antenna: Optional[str] = None,
        location: Optional[Tuple[float, float]] = None,
    ) -> None:
        if mode not in ("memory", "file"):
            raise ValueError(f"mode must be 'memory' or 'file', got {mode!r}")
        if mode == "file" and output_dir is None:
            raise ValueError("output_dir is required for file mode")

        self.backend = backend
        self.center_frequency = center_frequency
        self.sample_rate = sample_rate
        self.gain = gain
        self.segment_length = segment_length
        self.mode = mode
        self.buffer_size = buffer_size
        self.output_dir = output_dir
        self.antenna = antenna
        self.location = location

        # Memory mode state
        self._buffer = np.zeros((buffer_size, segment_length), dtype=np.complex64)
        self._metadata: List[Optional[CaptureMetadata]] = [None] * buffer_size
        self._write_idx = 0
        self._count = 0

        # File mode state
        self._file_paths: List[str] = []

    def capture(self, num_segments: int = 1) -> None:
        """Capture one or more segments from the backend."""
        for _ in range(num_segments):
            iq = self.backend.read(self.segment_length)
            if len(iq) < self.segment_length:
                padded = np.zeros(self.segment_length, dtype=np.complex64)
                padded[: len(iq)] = iq
                iq = padded

            ts = time.time()
            duration = self.segment_length / self.sample_rate
            nf, sp, occ = compute_rf_measurements(iq)
            cap_id = str(uuid.uuid4())[:8]

            meta = CaptureMetadata(
                center_frequency=self.center_frequency,
                sample_rate=self.sample_rate,
                gain=self.gain,
                timestamp=ts,
                duration=duration,
                antenna=self.antenna,
                location=self.location,
                capture_id=cap_id,
                noise_floor_dbm=nf,
                signal_power_dbm=sp,
                occupancy_fraction=occ,
            )

            if self.mode == "memory":
                idx = self._write_idx % self.buffer_size
                self._buffer[idx] = iq
                self._metadata[idx] = meta
                self._write_idx += 1
                self._count = min(self._count + 1, self.buffer_size)
            else:
                self._write_file(iq, meta, cap_id, ts)

    def capture_continuous(self, duration: float) -> None:
        """Capture continuously for the specified duration in seconds."""
        total_samples = int(duration * self.sample_rate)
        num_segments = max(1, total_samples // self.segment_length)
        self.capture(num_segments=num_segments)

    def get_segment(self, idx: int) -> Tuple[np.ndarray, CaptureMetadata]:
        """Retrieve a captured segment by index."""
        if self.mode == "memory":
            if idx >= self._count:
                raise IndexError(f"Segment {idx} not available (have {self._count})")
            start = (self._write_idx - self._count) % self.buffer_size
            actual_idx = (start + idx) % self.buffer_size
            return self._buffer[actual_idx].copy(), self._metadata[actual_idx]
        else:
            if idx >= len(self._file_paths):
                raise IndexError(f"Segment {idx} not available (have {len(self._file_paths)})")
            from spectra.utils.file_handlers import SigMFReader
            reader = SigMFReader()
            iq, sig_meta = reader.read(self._file_paths[idx])
            # Reconstruct CaptureMetadata from SigMF extra fields
            extra = sig_meta.extra
            meta = CaptureMetadata(
                center_frequency=sig_meta.center_frequency or self.center_frequency,
                sample_rate=sig_meta.sample_rate or self.sample_rate,
                gain=extra.get("gain", self.gain),
                timestamp=extra.get("timestamp", 0.0),
                duration=extra.get("duration", 0.0),
                antenna=extra.get("antenna"),
                location=extra.get("location"),
                capture_id=extra.get("capture_id"),
                noise_floor_dbm=extra.get("noise_floor_dbm"),
                signal_power_dbm=extra.get("signal_power_dbm"),
                occupancy_fraction=extra.get("occupancy_fraction"),
            )
            return iq, meta

    def __len__(self) -> int:
        if self.mode == "memory":
            return self._count
        return len(self._file_paths)

    def _write_file(self, iq: np.ndarray, meta: CaptureMetadata, cap_id: str, ts: float) -> None:
        import os
        from spectra.utils.file_handlers.sigmf_writer import SigMFWriter
        base = os.path.join(self.output_dir, f"capture_{cap_id}_{int(ts)}")
        writer = SigMFWriter(
            base_path=base,
            sample_rate=self.sample_rate,
            center_frequency=self.center_frequency,
            extra_global={
                "gain": self.gain,
                "timestamp": ts,
                "duration": meta.duration,
                "antenna": self.antenna,
                "location": self.location,
                "capture_id": cap_id,
                "noise_floor_dbm": meta.noise_floor_dbm,
                "signal_power_dbm": meta.signal_power_dbm,
                "occupancy_fraction": meta.occupancy_fraction,
            },
        )
        writer.write(iq)
        self._file_paths.append(f"{base}.sigmf-meta")
```

- [ ] **Step 3: Update `capture/__init__.py`**

Add `LiveCapture` import and export:

```python
from spectra.capture.live import LiveCapture
```

Add `"LiveCapture"` to `__all__`.

- [ ] **Step 4: Run tests**

```bash
/Users/gditzler/.venvs/base/bin/pytest tests/test_capture.py -v
```
Expected: 10/10 PASS

- [ ] **Step 5: Commit**

```bash
git add python/spectra/capture/live.py python/spectra/capture/__init__.py tests/test_capture.py
git commit -m "feat(capture): add LiveCapture with memory and file modes"
```

---

## Task 4: OTADataset

**Files:**
- Create: `python/spectra/datasets/ota.py`
- Modify: `python/spectra/datasets/__init__.py`
- Create: `tests/test_ota_dataset.py`

- [ ] **Step 1: Write the failing tests**

```python
# tests/test_ota_dataset.py
"""Tests for OTADataset."""
import numpy as np
import pytest
import tempfile
import torch


# Reuse MockBackend from test_capture
class MockBackend:
    def __init__(self, seed=42):
        self._rng = np.random.default_rng(seed)
        self._active = False
    def configure(self, center_frequency, sample_rate, gain, **kwargs):
        pass
    def read(self, num_samples):
        return (self._rng.standard_normal(num_samples)
                + 1j * self._rng.standard_normal(num_samples)).astype(np.complex64)
    def start(self):
        self._active = True
    def stop(self):
        self._active = False
    @property
    def is_active(self):
        return self._active


def _make_live_capture(num_segments=10, segment_length=512):
    from spectra.capture.live import LiveCapture
    backend = MockBackend(seed=42)
    backend.configure(center_frequency=915e6, sample_rate=1e6, gain=30.0)
    lc = LiveCapture(
        backend=backend, center_frequency=915e6, sample_rate=1e6, gain=30.0,
        segment_length=segment_length, mode="memory", buffer_size=100,
    )
    lc.capture(num_segments=num_segments)
    return lc


def test_ota_dataset_from_live_capture():
    from spectra.datasets.ota import OTADataset
    lc = _make_live_capture(num_segments=5, segment_length=256)
    ds = OTADataset(source=lc, num_iq_samples=256)
    assert len(ds) == 5
    data, label = ds[0]
    assert isinstance(data, torch.Tensor)
    assert data.shape == (2, 256)
    assert label == -1  # unlabelled


def test_ota_dataset_with_fixed_label():
    from spectra.datasets.ota import OTADataset
    lc = _make_live_capture(num_segments=3)
    ds = OTADataset(source=lc, num_iq_samples=512, label="QPSK")
    _, label = ds[0]
    assert label == 0  # first (only) class


def test_ota_dataset_from_directory():
    from spectra.datasets.ota import OTADataset
    from spectra.capture.live import LiveCapture

    backend = MockBackend(seed=42)
    backend.configure(center_frequency=915e6, sample_rate=1e6, gain=30.0)

    with tempfile.TemporaryDirectory() as tmpdir:
        lc = LiveCapture(
            backend=backend, center_frequency=915e6, sample_rate=1e6, gain=30.0,
            segment_length=256, mode="file", output_dir=tmpdir,
        )
        lc.capture(num_segments=3)

        ds = OTADataset(source=tmpdir, num_iq_samples=256, sample_rate=1e6)
        assert len(ds) >= 3
        data, label = ds[0]
        assert data.shape == (2, 256)


def test_ota_dataset_truncate_pad():
    from spectra.datasets.ota import OTADataset
    lc = _make_live_capture(num_segments=2, segment_length=1024)
    ds = OTADataset(source=lc, num_iq_samples=512)
    data, _ = ds[0]
    assert data.shape == (2, 512)
```

- [ ] **Step 2: Run tests to verify they fail**

```bash
/Users/gditzler/.venvs/base/bin/pytest tests/test_ota_dataset.py -v
```
Expected: `ModuleNotFoundError`

- [ ] **Step 3: Write `ota.py`**

```python
# python/spectra/datasets/ota.py
"""OTADataset: PyTorch Dataset wrapping live or file-based OTA captures."""

from __future__ import annotations

import os
from typing import Callable, Dict, List, Optional, Tuple, Union

import numpy as np
import torch
from torch.utils.data import Dataset

from spectra.datasets.iq_utils import iq_to_tensor, truncate_pad
from spectra.scene.signal_desc import SignalDescription
from spectra.utils.file_handlers import SigMFReader, get_reader, supported_extensions


class OTADataset(Dataset):
    """Dataset wrapping OTA captures from a LiveCapture or a directory of files.

    Args:
        source: A :class:`~spectra.capture.live.LiveCapture` instance (memory
            or file mode) or a path to a directory of captured IQ files.
        num_iq_samples: Samples per item (truncate or zero-pad).
        transform: Optional callable applied to the IQ tensor.
        impairments: Optional impairment pipeline applied to raw IQ.
        sample_rate: Sample rate for impairments (auto-detected from metadata
            when possible).
        label: Fixed label for all samples (scenario-based). If ``None`` and
            no manifest, samples are unlabelled (``label=-1``).
        manifest_path: CSV/JSON with per-file labels.
        reader_overrides: Custom file reader overrides.
    """

    def __init__(
        self,
        source: Union["LiveCapture", str],
        num_iq_samples: int = 4096,
        transform: Optional[Callable] = None,
        impairments: Optional[Callable] = None,
        sample_rate: Optional[float] = None,
        label: Optional[str] = None,
        manifest_path: Optional[str] = None,
        reader_overrides: Optional[Dict] = None,
    ) -> None:
        self.num_iq_samples = num_iq_samples
        self.transform = transform
        self.impairments = impairments
        self.sample_rate = sample_rate
        self.reader_overrides = reader_overrides

        self._is_live = not isinstance(source, str)

        if self._is_live:
            self._live_capture = source
            self._file_paths: List[str] = []
        else:
            self._live_capture = None
            exts = supported_extensions()
            self._file_paths = sorted([
                os.path.join(source, f)
                for f in os.listdir(source)
                if any(f.endswith(ext) for ext in exts)
            ])

        # Labelling
        self._label_map: Optional[Dict[str, int]] = None
        self._fixed_label: Optional[int] = None

        if manifest_path is not None:
            self._label_map = self._load_manifest(manifest_path)
        elif label is not None:
            self._fixed_label = 0
            self._class_names = [label]
        else:
            self._fixed_label = -1

    def __len__(self) -> int:
        if self._is_live:
            return len(self._live_capture)
        return len(self._file_paths)

    def __getitem__(self, idx: int) -> Tuple[torch.Tensor, int]:
        if self._is_live:
            iq, meta = self._live_capture.get_segment(idx)
            sr = meta.sample_rate if self.sample_rate is None else self.sample_rate
        else:
            filepath = self._file_paths[idx]
            # Handle compound extensions (.sigmf-meta) that os.path.splitext
            # cannot match — use SigMFReader directly for SigMF files.
            if filepath.endswith(".sigmf-meta"):
                from spectra.utils.file_handlers import SigMFReader
                reader = SigMFReader()
            else:
                reader = get_reader(filepath, self.reader_overrides)
            iq, sig_meta = reader.read(filepath)
            sr = sig_meta.sample_rate or self.sample_rate or 1.0

        iq = truncate_pad(iq, self.num_iq_samples)

        if self.impairments is not None and sr:
            desc = SignalDescription(
                t_start=0.0,
                t_stop=self.num_iq_samples / sr,
                f_low=-sr / 2,
                f_high=sr / 2,
                label="unknown",
                snr=0.0,
            )
            iq, desc = self.impairments(iq, desc, sample_rate=sr)

        if self.transform is not None:
            data = self.transform(iq)
        else:
            data = iq_to_tensor(iq)

        # Determine label
        if self._fixed_label is not None:
            lbl = self._fixed_label
        elif self._label_map is not None and not self._is_live:
            fname = os.path.basename(self._file_paths[idx])
            lbl = self._label_map.get(fname, -1)
        else:
            lbl = -1

        return data, lbl

    def _load_manifest(self, path: str) -> Dict[str, int]:
        import json
        if path.endswith(".json"):
            with open(path) as f:
                entries = json.load(f)
        else:
            import csv
            with open(path) as f:
                reader = csv.DictReader(f)
                entries = list(reader)

        labels = sorted(set(e["label"] for e in entries))
        label_to_idx = {l: i for i, l in enumerate(labels)}
        return {e["file"]: label_to_idx[e["label"]] for e in entries}
```

- [ ] **Step 4: Add export to `datasets/__init__.py`**

Add:
```python
from spectra.datasets.ota import OTADataset
```
Add `"OTADataset"` to `__all__`.

- [ ] **Step 5: Run tests**

```bash
/Users/gditzler/.venvs/base/bin/pytest tests/test_ota_dataset.py -v
```
Expected: 4/4 PASS

- [ ] **Step 6: Commit**

```bash
git add python/spectra/datasets/ota.py python/spectra/datasets/__init__.py tests/test_ota_dataset.py
git commit -m "feat(datasets): add OTADataset for live and file-based OTA captures"
```

---

## Task 5: HybridDataLoader

**Files:**
- Modify: `python/spectra/streaming.py`
- Create: `tests/test_hybrid_loader.py`

- [ ] **Step 1: Write the failing tests**

```python
# tests/test_hybrid_loader.py
"""Tests for HybridDataLoader."""
import numpy as np
import pytest
import torch
from torch.utils.data import Dataset, TensorDataset


def _make_synthetic_dataset(n=100):
    """Simple synthetic dataset: random tensors with label 0."""
    data = torch.randn(n, 2, 64)
    labels = torch.zeros(n, dtype=torch.long)
    return TensorDataset(data, labels)


def _make_ota_dataset(n=50):
    """Simple OTA dataset: random tensors with label -1."""
    data = torch.randn(n, 2, 64)
    labels = torch.full((n,), -1, dtype=torch.long)
    return TensorDataset(data, labels)


def test_hybrid_loader_batch_size():
    from spectra.streaming import HybridDataLoader
    syn = _make_synthetic_dataset(100)
    ota = _make_ota_dataset(50)
    loader = HybridDataLoader(
        synthetic_dataset=syn, ota_dataset=ota,
        synthetic_ratio=0.75, batch_size=8, seed=42,
    )
    dl = loader.loader()
    batch = next(iter(dl))
    data, labels, domain_flags = batch
    assert data.shape[0] == 8
    assert len(domain_flags) == 8


def test_hybrid_loader_domain_flags():
    from spectra.streaming import HybridDataLoader
    syn = _make_synthetic_dataset(100)
    ota = _make_ota_dataset(50)
    loader = HybridDataLoader(
        synthetic_dataset=syn, ota_dataset=ota,
        synthetic_ratio=0.5, batch_size=10, seed=42,
    )
    dl = loader.loader()
    batch = next(iter(dl))
    _, _, domain_flags = batch
    # Should have mix of True (synthetic) and False (OTA)
    assert domain_flags.sum().item() > 0  # some synthetic
    assert (~domain_flags).sum().item() > 0  # some OTA


def test_hybrid_loader_ratio():
    from spectra.streaming import HybridDataLoader
    syn = _make_synthetic_dataset(200)
    ota = _make_ota_dataset(200)
    loader = HybridDataLoader(
        synthetic_dataset=syn, ota_dataset=ota,
        synthetic_ratio=0.8, batch_size=20, seed=42,
    )
    dl = loader.loader()
    # Check first batch has ~80% synthetic
    _, _, domain_flags = next(iter(dl))
    n_syn = domain_flags.sum().item()
    assert n_syn == 16  # floor(20 * 0.8)


def test_hybrid_loader_full_epoch():
    from spectra.streaming import HybridDataLoader
    syn = _make_synthetic_dataset(40)
    ota = _make_ota_dataset(20)
    loader = HybridDataLoader(
        synthetic_dataset=syn, ota_dataset=ota,
        synthetic_ratio=0.6, batch_size=10, seed=42,
    )
    dl = loader.loader()
    total = 0
    for batch in dl:
        total += batch[0].shape[0]
    assert total > 0
```

- [ ] **Step 2: Run tests to verify they fail**

```bash
/Users/gditzler/.venvs/base/bin/pytest tests/test_hybrid_loader.py -v
```
Expected: `ImportError` (HybridDataLoader doesn't exist yet)

- [ ] **Step 3: Add `HybridDataLoader` to `streaming.py`**

Read the existing `streaming.py` first, then append:

```python
class _HybridDataset(Dataset):
    """Internal dataset that routes indices to synthetic or OTA sources."""

    def __init__(self, synthetic_dataset, ota_dataset, n_syn, n_ota):
        self.synthetic = synthetic_dataset
        self.ota = ota_dataset
        self.n_syn = n_syn
        self.n_ota = n_ota

    def __len__(self):
        return self.n_syn + self.n_ota

    def __getitem__(self, idx):
        if idx < self.n_syn:
            data, label = self.synthetic[idx % len(self.synthetic)]
            return data, label, True  # domain_flag = synthetic
        else:
            ota_idx = (idx - self.n_syn) % len(self.ota)
            data, label = self.ota[ota_idx]
            return data, label, False  # domain_flag = OTA


class _HybridBatchSampler:
    """Sampler that ensures each batch has the right synthetic/OTA mix."""

    def __init__(self, n_syn, n_ota, batch_size, syn_per_batch, shuffle, generator):
        self.n_syn = n_syn
        self.n_ota = n_ota
        self.batch_size = batch_size
        self.syn_per_batch = syn_per_batch
        self.ota_per_batch = batch_size - syn_per_batch
        self.shuffle = shuffle
        self.generator = generator

    def __iter__(self):
        if self.shuffle:
            syn_indices = torch.randperm(self.n_syn, generator=self.generator).tolist()
            ota_indices = (torch.randperm(self.n_ota, generator=self.generator) + self.n_syn).tolist()
        else:
            syn_indices = list(range(self.n_syn))
            ota_indices = list(range(self.n_syn, self.n_syn + self.n_ota))

        syn_pos = 0
        ota_pos = 0
        n_batches = min(
            self.n_syn // max(self.syn_per_batch, 1),
            self.n_ota // max(self.ota_per_batch, 1),
        ) if self.ota_per_batch > 0 else self.n_syn // max(self.syn_per_batch, 1)

        for _ in range(n_batches):
            batch = []
            for _ in range(self.syn_per_batch):
                batch.append(syn_indices[syn_pos % len(syn_indices)])
                syn_pos += 1
            for _ in range(self.ota_per_batch):
                batch.append(ota_indices[ota_pos % len(ota_indices)])
                ota_pos += 1
            yield batch

    def __len__(self):
        if self.ota_per_batch > 0:
            return min(
                self.n_syn // max(self.syn_per_batch, 1),
                self.n_ota // max(self.ota_per_batch, 1),
            )
        return self.n_syn // max(self.syn_per_batch, 1)


def _hybrid_collate(batch):
    """Collate function that separates domain flags."""
    data = torch.stack([b[0] for b in batch])
    labels = torch.tensor([b[1] for b in batch], dtype=torch.long)
    domain_flags = torch.tensor([b[2] for b in batch], dtype=torch.bool)
    return data, labels, domain_flags


class HybridDataLoader:
    """DataLoader mixing synthetic and OTA data within each batch.

    Args:
        synthetic_dataset: Synthetic data source (e.g., NarrowbandDataset).
        ota_dataset: OTA data source (e.g., OTADataset).
        synthetic_ratio: Fraction of each batch from synthetic (0 to 1).
        batch_size: Samples per batch.
        shuffle: Whether to shuffle indices.
        num_workers: DataLoader workers.
        seed: Random seed for shuffling.
    """

    def __init__(
        self,
        synthetic_dataset: Dataset,
        ota_dataset: Dataset,
        synthetic_ratio: float = 0.7,
        batch_size: int = 32,
        shuffle: bool = True,
        num_workers: int = 0,
        seed: int = 0,
    ) -> None:
        self.synthetic_ratio = synthetic_ratio
        self.batch_size = batch_size
        self.shuffle = shuffle
        self.num_workers = num_workers
        self.seed = seed

        self.syn_per_batch = int(batch_size * synthetic_ratio)
        self.ota_per_batch = batch_size - self.syn_per_batch

        n_syn = len(synthetic_dataset)
        n_ota = len(ota_dataset)

        self._dataset = _HybridDataset(synthetic_dataset, ota_dataset, n_syn, n_ota)
        self._generator = torch.Generator().manual_seed(seed)
        self._sampler = _HybridBatchSampler(
            n_syn, n_ota, batch_size, self.syn_per_batch, shuffle, self._generator
        )

    def loader(self) -> DataLoader:
        """Return a PyTorch DataLoader with hybrid batching."""
        return DataLoader(
            self._dataset,
            batch_sampler=self._sampler,
            collate_fn=_hybrid_collate,
            num_workers=self.num_workers,
        )
```

Also add import at top of file:
```python
from torch.utils.data import Dataset, DataLoader
```

- [ ] **Step 4: Run tests**

```bash
/Users/gditzler/.venvs/base/bin/pytest tests/test_hybrid_loader.py -v
```
Expected: 4/4 PASS

- [ ] **Step 5: Commit**

```bash
git add python/spectra/streaming.py tests/test_hybrid_loader.py
git commit -m "feat(streaming): add HybridDataLoader for synthetic/OTA batch mixing"
```

---

## Task 6: Full Verification

- [ ] **Step 1: Run all tests**

```bash
/Users/gditzler/.venvs/base/bin/pytest tests/ -q --tb=short
```
Expected: all tests pass, no regressions.

- [ ] **Step 2: Verify imports**

```bash
/Users/gditzler/.venvs/base/bin/python -c "
from spectra.capture import CaptureBackend, CaptureMetadata, LiveCapture, compute_rf_measurements
from spectra.datasets import OTADataset
from spectra.streaming import HybridDataLoader
print('All imports OK')
"
```

- [ ] **Step 3: Commit any remaining changes**

```bash
git status
```
