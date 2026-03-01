# SPECTRA Architecture Design

**Date:** 2026-03-01
**Status:** Approved

## Overview

SPECTRA is a library for generating realistic radar and communication waveforms with seamless PyTorch integration. It addresses key limitations in existing tools (notably TorchSig): limited waveform configurability, shallow protocol fidelity, simple channel models, poor performance at scale, and no wideband scene compositing with detection-ready labels.

## Design Decisions

| Decision | Choice | Rationale |
|---|---|---|
| Audience | ML researchers + RF engineers | Layered API: sensible defaults for ML, full parametric control for RF |
| Backend language | Rust (PyO3) | Memory safety, performance, modern toolchain |
| Rust/Python boundary | Rust engine + Python orchestration | Rust for compute-intensive primitives, Python for composition and dataset logic |
| GPU strategy | Rust -> CPU NumPy arrays | DataLoader `num_workers` parallelism is sufficient; avoids libtorch version coupling |
| Waveform priority | Comms first (modulation + basic framing) | Broader user base; radar slots in later with same engine |
| Impairments | Composable transform pipeline | Chainable like torchvision transforms; presets can layer on top later |
| Data pipeline | On-the-fly generation via `__getitem__()` | No disk I/O, infinite variability, deterministic from index |
| Signal scope | Narrowband + wideband compositing | Supports both AMC and signal detection use cases |
| Label format | Physical units + COCO-style pixel conversion | Ground truth in seconds/Hz; utility converts to pixel coords given STFT params |

## Package Structure

```
spectra/
├── pyproject.toml              # maturin-based build (Rust + Python)
├── Cargo.toml                  # Rust workspace
├── rust/
│   └── src/
│       ├── lib.rs              # PyO3 module entry
│       ├── oscillators.rs      # Tone/chirp/carrier generation
│       ├── modulators.rs       # PSK, QAM, FSK, OFDM symbol mapping + pulse shaping
│       ├── filters.rs          # FIR/IIR, polyphase resampler, pulse shaping filters
│       ├── noise.rs            # AWGN, colored noise
│       └── fft.rs              # FFT/IFFT (wrapping rustfft)
├── python/
│   └── spectra/
│       ├── __init__.py
│       ├── waveforms/
│       │   ├── base.py         # Abstract Waveform class
│       │   ├── psk.py          # BPSK, QPSK, 8PSK
│       │   ├── qam.py          # 16QAM, 64QAM, 256QAM
│       │   ├── fsk.py          # FSK, GFSK, MSK
│       │   ├── ofdm.py         # Generic OFDM with subcarrier config
│       │   ├── dsss.py         # Direct-sequence spread spectrum
│       │   └── fhss.py         # Frequency-hopping spread spectrum
│       ├── framing/
│       │   ├── base.py         # Abstract Frame class
│       │   └── generic.py      # Preamble + header + payload structure
│       ├── impairments/
│       │   ├── base.py         # Abstract Transform
│       │   ├── awgn.py
│       │   ├── phase_noise.py
│       │   ├── iq_imbalance.py
│       │   ├── frequency_offset.py
│       │   ├── multipath.py
│       │   ├── doppler.py
│       │   └── compose.py      # Chain transforms together
│       ├── scene/
│       │   ├── signal_desc.py  # SignalDescription: physical units metadata
│       │   ├── composer.py     # Wideband scene compositor
│       │   └── labels.py       # BBox conversion: physical -> pixel coords
│       ├── datasets/
│       │   ├── narrowband.py   # NarrowbandDataset(torch.Dataset)
│       │   └── wideband.py     # WidebandDataset(torch.Dataset)
│       └── transforms/
│           └── stft.py         # STFT transform + spectrogram generation
```

### Rust/Python Boundary

Rust exports stateless functions that operate on arrays:

- `spectra._rust.generate_qpsk_symbols(num_symbols) -> np.ndarray`
- `spectra._rust.apply_rrc_filter(symbols, rolloff, span, sps) -> np.ndarray`
- `spectra._rust.generate_chirp(duration, fs, f0, f1) -> np.ndarray`

Python waveform classes call these Rust primitives and handle orchestration (parameter validation, framing, composition).

## Wideband Scene Composition

### Data Flow

```
SceneConfig
  - capture_duration, capture_bandwidth, sample_rate
  - num_signals: int | (min, max)
  - signal_pool: List[Waveform]
  - snr_range: (min, max)
  - allow_overlap: bool
        │
        ▼
Composer.generate()
  For each signal i in num_signals:
    1. Sample waveform type from signal_pool
    2. Generate baseband IQ via Rust
    3. Apply per-signal impairments
    4. Frequency-shift to center_freq[i]
    5. Record SignalDescription:
       - t_start, t_stop (seconds)
       - f_center, bandwidth (Hz)
       - waveform_class, modulation_params, snr
  Sum all signals into composite IQ
  Apply scene-level impairments (AWGN, etc.)
        │
        ▼
Output per sample:
  - iq_data: np.ndarray (complex64)
  - signal_descriptions: List[SignalDescription]
    Each has: t_start, t_stop, f_low, f_high, label, snr
        │
        ▼
Label conversion (when needed):
  labels.to_coco(signal_descs, stft_params)
  -> boxes: Tensor[N, 4]  (pixel coords in spectrogram)
  -> labels: Tensor[N]    (class indices)
  -> metadata per box (snr, modulation, etc.)
```

### Multi-Signal Overlap

- Signals sum in the complex domain (physically correct)
- Center frequencies and start times can be random, grid-based, or user-specified
- Overlapping bounding boxes are preserved as-is; detection models learn to handle this
- Per-signal impairments apply before compositing; scene-level impairments apply after

## Labeling Architecture

Ground truth is stored in physical units, decoupled from any specific STFT resolution:

```python
@dataclass
class SignalDescription:
    t_start: float          # seconds
    t_stop: float           # seconds
    f_low: float            # Hz (f_center - bandwidth/2)
    f_high: float           # Hz (f_center + bandwidth/2)
    label: str              # e.g., "QPSK"
    snr: float              # dB
    modulation_params: dict # waveform-specific metadata
```

Conversion to pixel-space bounding boxes is a separate utility:

```python
def to_coco(signal_descs, stft_params) -> Dict:
    """Convert physical-unit descriptions to COCO-style detection targets.

    Args:
        signal_descs: List of SignalDescription
        stft_params: STFTParams(nfft, hop_length, sample_rate, ...)

    Returns:
        {"boxes": Tensor[N, 4], "labels": Tensor[N], "signal_descs": [...]}
    """
```

## PyTorch Dataset Integration

### NarrowbandDataset

For automatic modulation classification (AMC). One signal per sample.

```python
NarrowbandDataset(
    waveform_pool: List[Waveform],
    num_samples: int,
    num_iq_samples: int,
    sample_rate: float,
    impairments: Optional[Compose],
    transform: Optional[Callable],
    target_transform: Optional[Callable],
    seed: Optional[int],
)

__getitem__(idx) -> Tuple[Tensor, int]
```

### WidebandDataset

For signal detection. Multiple signals per scene.

```python
WidebandDataset(
    scene_config: SceneConfig,
    num_samples: int,
    impairments: Optional[Compose],
    transform: Optional[Callable],
    target_transform: Optional[Callable],
    seed: Optional[int],
)

__getitem__(idx) -> Tuple[Tensor, Dict]
# Dict contains: boxes, labels, signal_descs
```

### Key Properties

- **Deterministic:** `__getitem__(idx)` seeds RNG from `(seed, idx)` for reproducibility
- **Stateless:** No disk I/O, no shared state between workers
- **Standard interface:** Works with `DataLoader(num_workers=N, pin_memory=True)`
- **Custom collate:** `spectra.datasets.collate_fn` handles variable-length box lists

## Impairments Pipeline

Composable transforms following the torchvision pattern:

```python
class Transform(ABC):
    def __call__(self, iq, signal_desc) -> Tuple[np.ndarray, SignalDescription]:
        ...

class Compose:
    def __init__(self, transforms: List[Transform]): ...
    def __call__(self, iq, signal_desc) -> Tuple[np.ndarray, SignalDescription]:
        for t in self.transforms:
            iq, signal_desc = t(iq, signal_desc)
        return iq, signal_desc
```

Available impairments (v1):

| Impairment | Parameters | Rust-accelerated |
|---|---|---|
| AWGN | snr (dB) or snr_range | No (NumPy sufficient) |
| FrequencyOffset | max_offset (Hz) | No |
| PhaseNoise | noise_power_db | No |
| IQImbalance | amplitude_db, phase_deg | No |
| Multipath | num_paths, max_delay, power_profile | Yes (convolution) |
| Doppler | max_shift (Hz) | No |

Transforms take and return `(iq, signal_desc)` so the signal description tracks what was applied (e.g., effective center frequency after offset).

Parameters accept fixed values or ranges. Ranges sample uniformly per call for dataset variability.

## Waveform API

Layered for both audiences:

```python
class Waveform(ABC):
    @abstractmethod
    def generate(self, num_symbols, sample_rate, frame=None) -> np.ndarray: ...

    @property
    @abstractmethod
    def bandwidth(self) -> float: ...

    @property
    @abstractmethod
    def label(self) -> str: ...
```

- Every parameter has a sensible default (ML researcher path)
- All parameters are overridable (RF engineer path)
- Parameters can be ranges for randomization: `QPSK(samples_per_symbol=(4, 16))`
- Waveforms are stateless generators, safe across DataLoader workers
- Framing is optional and separate from modulation

## Build System

- **maturin** for building the Rust/Python hybrid package
- `pyproject.toml` with `[build-system] requires = ["maturin"]`
- Rust workspace in `rust/` with `Cargo.toml`
- PyO3 for Rust-Python FFI, numpy crate for zero-copy array returns
- CI builds wheels for Linux (manylinux), macOS (x86_64 + arm64)

## Future Work (Not in v1)

- Radar waveforms (LFM, phase-coded, frequency-agile, PRI patterns)
- Full protocol-level comms (WiFi 802.11, LTE/5G NR, Bluetooth)
- SigMF export and ingestion (real-data pipelines)
- Preset channel profiles (urban, rural, indoor)
- GPU-accelerated generation via tch-rs or CUDA kernels
- Waveform-level augmentation (time stretch, frequency warp)
