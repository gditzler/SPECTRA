# OTA Dataset Integration — Design Spec

**Goal:** Add a `LiveCapture` source (wrapping SoapySDR via an extensible backend ABC) and a `HybridDataLoader` that mixes live/file-based OTA captures with synthetic data at configurable ratios.

**Scope:** v1 implements hardware abstraction, ring-buffer + file capture modes, an unlabelled OTA dataset, and ratio-based batch interleaving. Curriculum-aware mixing ratios and classifier-assisted labelling are deferred to v2.

---

## Architecture Overview

```
SDR Hardware → CaptureBackend ABC → LiveCapture → OTADataset ──┐
               (SoapySDR impl)      (ring buf /               │
                                     SigMF file)              ├→ HybridDataLoader → Training
                                                               │
               Waveform.generate() → NarrowbandDataset ────────┘
                                     (synthetic)
```

The `HybridDataLoader` draws from both datasets per batch at a user-specified `synthetic_ratio`, providing `domain_flags` so training loops can implement domain-aware losses.

---

## Sub-project 1: CaptureBackend ABC + SoapySDR Implementation (`spectra/capture/`)

### Files

- `python/spectra/capture/__init__.py`
- `python/spectra/capture/backend.py`
- `python/spectra/capture/soapy.py`
- `python/spectra/capture/metadata.py`

### `backend.py` — `CaptureBackend` ABC

```python
class CaptureBackend(ABC):
    def configure(self, center_frequency: float, sample_rate: float,
                  gain: float, **kwargs) -> None: ...
    def read(self, num_samples: int) -> np.ndarray: ...  # returns complex64
    def start(self) -> None: ...
    def stop(self) -> None: ...
    @property
    def is_active(self) -> bool: ...
```

Stateful — `configure()` sets up the radio, `start()`/`stop()` control streaming, `read()` pulls a block of IQ samples. Context manager support (`__enter__`/`__exit__`) for automatic cleanup.

### `soapy.py` — `SoapyBackend(CaptureBackend)`

Wraps `SoapySDR.Device`. Constructor takes `driver` (e.g., `"rtlsdr"`, `"uhd"`, `"hackrf"`) and optional `device_args` dict. Maps `configure()`/`read()`/`start()`/`stop()` to SoapySDR's stream API. Optional dependency: `spectra[sdr]` adds `SoapySDR`.

### `metadata.py` — `CaptureMetadata` dataclass

```python
@dataclass
class CaptureMetadata:
    # Hardware
    center_frequency: float      # Hz
    sample_rate: float           # Hz
    gain: float                  # dB
    timestamp: float             # Unix epoch seconds
    duration: float              # seconds
    # Environment (optional)
    antenna: Optional[str] = None
    location: Optional[Tuple[float, float]] = None  # (lat, lon)
    capture_id: Optional[str] = None
    notes: Optional[str] = None
    # RF measurements (computed from IQ)
    noise_floor_dbm: Optional[float] = None
    signal_power_dbm: Optional[float] = None
    occupancy_fraction: Optional[float] = None
```

RF measurements are computed by `LiveCapture` after each segment is captured:
- `noise_floor_dbm`: 10th percentile of per-sample power in dBm
- `signal_power_dbm`: mean power in dBm
- `occupancy_fraction`: fraction of samples exceeding `noise_floor + 6 dB`

---

## Sub-project 2: LiveCapture Orchestrator (`spectra/capture/live.py`)

### Files

- `python/spectra/capture/live.py`

### `LiveCapture` class

```python
class LiveCapture:
    def __init__(
        self,
        backend: CaptureBackend,
        center_frequency: float,
        sample_rate: float,
        gain: float,
        segment_length: int = 4096,
        mode: str = "memory",              # "memory" or "file"
        buffer_size: int = 1000,           # max segments in ring buffer
        output_dir: Optional[str] = None,  # required for file mode
        antenna: Optional[str] = None,
        location: Optional[Tuple[float, float]] = None,
    ):
```

**Memory mode (`mode="memory"`):**
- Ring buffer: numpy array `(buffer_size, segment_length)` complex64
- Parallel `CaptureMetadata` list of same length
- `.capture(num_segments=1)` reads from backend, computes RF measurements, stores in buffer
- `.get_segment(idx)` returns `(iq, metadata)`
- Oldest segments overwritten when full

**File mode (`mode="file"`):**
- `.capture(num_segments=1)` writes each segment as SigMF file pair
- File naming: `capture_{capture_id}_{timestamp}.sigmf-meta`
- RF measurements stored in SigMF annotation fields
- `.get_segment(idx)` reads back via `SigMFReader`

**Common methods:**
- `.capture(num_segments=1)` — capture one or more segments
- `.capture_continuous(duration: float)` — capture for a duration
- `.get_segment(idx) -> Tuple[np.ndarray, CaptureMetadata]`
- `__len__()` — segments currently available
- Context manager support

---

## Sub-project 3: OTADataset (`spectra/datasets/ota.py`)

### Files

- `python/spectra/datasets/ota.py`
- `python/spectra/datasets/__init__.py` (modify: add export)

### `OTADataset` class

```python
class OTADataset(Dataset):
    def __init__(
        self,
        source: Union[LiveCapture, str],   # LiveCapture or capture directory
        num_iq_samples: int = 4096,
        transform: Optional[Callable] = None,
        impairments: Optional[Callable] = None,
        sample_rate: Optional[float] = None,
        label: Optional[str] = None,       # fixed label (scenario-based)
        manifest_path: Optional[str] = None,  # per-file labels
        reader_overrides: Optional[Dict] = None,
    ):
```

**Labelling:**
- `label` provided: all samples get that label (scenario-based)
- `manifest_path` provided: per-file labels from CSV/JSON
- Neither: `label=-1` (unlabelled)

**Output:** `(Tensor[2, num_iq_samples], int)` — same format as `NarrowbandDataset`.

---

## Sub-project 4: HybridDataLoader (`spectra/streaming.py`)

### Files

- `python/spectra/streaming.py` (modify: add `HybridDataLoader`)

### `HybridDataLoader` class

```python
class HybridDataLoader:
    def __init__(
        self,
        synthetic_dataset: Dataset,
        ota_dataset: Dataset,
        synthetic_ratio: float = 0.7,
        batch_size: int = 32,
        shuffle: bool = True,
        num_workers: int = 0,
        seed: int = 0,
    ):
```

**Mixing:** wraps both datasets into `_HybridDataset` with per-index routing. Custom `Sampler` ensures each batch has `floor(batch_size * synthetic_ratio)` synthetic samples and remainder from OTA.

**Output per batch:** `(Tensor[B, 2, N], labels, domain_flags)` where `domain_flags` is boolean (`True` = synthetic, `False` = OTA).

**Curriculum extension point:** `synthetic_ratio` can be accepted from `CurriculumSchedule` params dict. Not implemented v1, but API supports it.

---

## Build Order & Dependencies

```
Sub-project 1: capture/backend + metadata  (no dependencies)
Sub-project 2: capture/soapy              (depends on 1, optional SoapySDR)
Sub-project 3: capture/live               (depends on 1-2, existing SigMFWriter)
Sub-project 4: datasets/ota               (depends on 3, existing FileReader)
Sub-project 5: streaming HybridDataLoader  (depends on 4)
```

---

## Optional Dependency

`spectra[sdr]` adds `SoapySDR` to `pyproject.toml`. Only `SoapyBackend` requires the package. Import is guarded with try/except.

---

## Future Work (out of scope for v1)

- **Curriculum-aware mixing:** `synthetic_ratio` controlled by `CurriculumSchedule`
- **Classifier-assisted labelling:** run `CyclostationaryAMC` for pseudo-labels with confidence gating
- **Scenario-based labelling:** structured descriptors for known transmitter setups
- **GNU Radio backend:** `GNURadioBackend(CaptureBackend)` wrapping `gr-osmosdr`
- **Network streaming:** capture from remote SDRs via TCP/ZMQ
- **Multi-antenna capture:** extend `CaptureBackend` for MIMO receivers
