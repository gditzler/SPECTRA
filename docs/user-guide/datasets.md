# Datasets

## NarrowbandDataset

Single-signal IQ dataset for AMC classification tasks.

**Constructor:**

| Parameter | Type | Description |
|-----------|------|-------------|
| `waveform_pool` | `List[Waveform]` | Waveforms to draw from; pool index is the class label |
| `num_samples` | `int` | Total number of samples in the dataset |
| `num_iq_samples` | `int` | Number of complex samples per segment |
| `sample_rate` | `float` | Receiver sample rate in Hz |
| `impairments` | `Compose` (optional) | Channel impairment pipeline |
| `transform` | `callable` (optional) | Applied to IQ tensor; if `None`, returns `[2, num_iq_samples]` |
| `target_transform` | `callable` (optional) | Applied to integer class label |
| `seed` | `int` (optional) | Base seed for reproducible generation |
| `class_weights` | `List[float]` (optional) | Per-class sampling weights; `None` for uniform |
| `mimo_config` | `Dict` (optional) | MIMO config dict with `n_tx`, `n_rx`, `channel_type` keys |

**Output from `__getitem__`:** `(Tensor[2, num_iq_samples], int)` — channel 0 is I, channel 1 is Q. With `mimo_config`, output shape is `[n_rx*2, num_iq_samples]`.

```python
from spectra import NarrowbandDataset, QPSK, BPSK, AWGN, Compose

dataset = NarrowbandDataset(
    waveform_pool=[QPSK(), BPSK()],
    num_samples=10_000,
    num_iq_samples=1024,
    sample_rate=1e6,
    impairments=Compose([AWGN(snr_range=(-5.0, 20.0))]),
    seed=42,
)
iq, label = dataset[0]  # iq: Tensor[2, 1024], label: int
```

---

## WidebandDataset

Multi-signal composite dataset for detection and localization tasks.

**Constructor:**

| Parameter | Type | Description |
|-----------|------|-------------|
| `config` | `SceneConfig` | Scene configuration |
| `num_samples` | `int` | Number of scenes to generate |
| `seed` | `int` | Base seed |
| `impairments` | `Compose` (optional) | Per-signal impairment pipeline |
| `transform` | `callable` (optional) | Applied to spectrogram tensor |
| `stft_params` | `STFTParams` (optional) | STFT grid for COCO box conversion |

**Output from `__getitem__`:** `(Tensor[1, F, T], Dict)` — spectrogram and COCO targets dict.

Use `dataset.collate_fn` as `DataLoader(collate_fn=dataset.collate_fn)` to handle
variable numbers of signals per scene.

---

## CyclostationaryDataset

Multi-representation dataset for CSP-based AMC.

The `representations` dict maps string keys to transform objects. Each transform
is applied to the IQ signal and returned as a separate tensor in the output dict.

```python
from spectra import CyclostationaryDataset, QPSK, BPSK, AWGN, Compose, SCD, Cumulants

dataset = CyclostationaryDataset(
    waveform_pool=[QPSK(), BPSK()],
    num_samples=3000,
    num_iq_samples=4096,
    sample_rate=1e6,
    representations={"scd": SCD(), "cumulants": Cumulants()},
    impairments=Compose([AWGN(snr=10.0)]),
    seed=42,
)

data, label = dataset[0]
# data: {"scd": Tensor[...], "cumulants": Tensor[...]}
# label: int
```

---

## SNRSweepDataset

Generates signals at fixed SNR levels for evaluation curves. Used by
`evaluate_snr_sweep()` to produce accuracy-vs-SNR plots.

```python
from spectra.datasets.snr_sweep import SNRSweepDataset
from spectra.impairments import AWGN, Compose
from spectra.waveforms import BPSK, QPSK

def impairments_fn(snr_db: float) -> Compose:
    """Build the impairment chain for a given SNR."""
    return Compose([AWGN(snr=snr_db)])

dataset = SNRSweepDataset(
    waveform_pool=[QPSK(), BPSK()],
    snr_levels=[-10, -5, 0, 5, 10, 15, 20],
    samples_per_cell=500,
    num_iq_samples=1024,
    sample_rate=1e6,
    impairments_fn=impairments_fn,
    seed=42,
)

iq, class_idx, snr_db = dataset[0]   # __getitem__ returns (Tensor, int, float)
```

---

## SignalFolderDataset

Loads IQ recordings from a class-per-directory structure on disk (analogous to
`torchvision.datasets.ImageFolder`).

**Directory convention:**

```
root/
  QPSK/
    capture_001.npy
    capture_002.npy
  BPSK/
    capture_001.npy
```

Supported file extensions: `.npy`, `.bin`, `.sigmf-data`, `.h5`.

```python
from spectra.datasets.folder import SignalFolderDataset

dataset = SignalFolderDataset(
    root="path/to/recordings",
    num_iq_samples=1024,
)
```

---

## ManifestDataset

Loads IQ recordings described by a CSV or JSON manifest file.

**CSV format** (`file,label`):

```csv
data/qpsk_001.npy,QPSK
data/bpsk_001.npy,BPSK
```

**JSON format:**

```json
[
  {"file": "data/qpsk_001.npy", "label": "QPSK"},
  {"file": "data/bpsk_001.npy", "label": "BPSK"}
]
```

```python
from spectra.datasets.manifest import ManifestDataset

dataset = ManifestDataset(
    manifest_path="recordings.csv",
    num_iq_samples=1024,
)
```

---

## Class Balancing

Imbalanced datasets are common in AMC research. SPECTRA provides two mechanisms:

### Weighted Class Selection

Pass `class_weights` to `NarrowbandDataset` to control the probability of each waveform class being sampled:

```python
from spectra import NarrowbandDataset, BPSK, QPSK, FM

dataset = NarrowbandDataset(
    waveform_pool=[BPSK(), QPSK(), FM()],
    num_samples=10_000,
    num_iq_samples=1024,
    sample_rate=1e6,
    class_weights=[3.0, 1.0, 1.0],  # BPSK sampled 3x more often
    seed=42,
)
```

### Balanced Sampler

Use `balanced_sampler()` to create a `WeightedRandomSampler` for your DataLoader:

```python
from spectra import balanced_sampler
from torch.utils.data import DataLoader

sampler = balanced_sampler(dataset, num_classes=3)
loader = DataLoader(dataset, batch_size=64, sampler=sampler)
```

---

## MixUp and CutMix Augmentations

### Signal-Level Augmentations

`MixUp` and `CutMix` operate on raw IQ arrays as part of a transform pipeline:

```python
from spectra import MixUp, CutMix

mixup = MixUp(alpha=0.2)         # Beta(0.2, 0.2) mixing coefficient
cutmix = CutMix(alpha=1.0)       # Beta(1.0, 1.0) cut ratio
```

### Dataset-Level Wrappers

`MixUpDataset` and `CutMixDataset` wrap any classification dataset to blend two samples and return soft labels:

```python
from spectra import MixUpDataset, CutMixDataset

mixup_ds = MixUpDataset(dataset, alpha=0.2)
x, (y1, y2, lam) = mixup_ds[0]
# x = lam * sample1 + (1 - lam) * sample2
# Use soft labels: loss = lam * CE(pred, y1) + (1-lam) * CE(pred, y2)

cutmix_ds = CutMixDataset(dataset, alpha=1.0)
x, (y1, y2, lam) = cutmix_ds[0]
# x has a random segment replaced from sample2
```

---

## MIMO Dataset Integration

`NarrowbandDataset` supports multi-antenna generation via the `mimo_config` parameter:

```python
from spectra import NarrowbandDataset, QPSK, BPSK

dataset = NarrowbandDataset(
    waveform_pool=[QPSK(), BPSK()],
    num_samples=5000,
    num_iq_samples=1024,
    sample_rate=1e6,
    mimo_config={
        "n_tx": 2,
        "n_rx": 4,
        "channel_type": "flat",  # or "tdl"
    },
    seed=42,
)

x, label = dataset[0]
# x: Tensor[8, 1024]  — 4 RX antennas x 2 (I/Q) channels
```

Each TX antenna generates an independent waveform stream (with different sub-seeds). The streams pass through a `MIMOChannel` impairment and are returned as interleaved I/Q per receive antenna: shape `[n_rx * 2, num_iq_samples]`.

---

## Deterministic Generation

!!! note "Why `(base_seed, idx)` seeding is safe for multi-worker DataLoaders"
    All synthetic datasets seed NumPy via `np.random.default_rng(seed=(base_seed, idx))`.
    Because the seed encodes the sample index, every worker produces the exact same
    IQ signal for a given index regardless of which worker processes it or in what
    order. This eliminates the seed-collision bugs that occur when workers share
    or independently advance a global RNG.

---

## Radar Datasets

Two datasets cover the radar signal-processing pipeline at different
levels of abstraction.

### RadarDataset

On-the-fly **range-profile** dataset for target-detection training. Each
`__getitem__` returns `(Tensor[num_range_bins], RadarTarget)` where
`RadarTarget` carries `range_bins`, `snrs`, `num_targets`, and
`waveform_label`. Targets are point scatterers at random range bins with
matched-filter range profiles plus AWGN.

```python
from spectra.datasets.radar import RadarDataset
from spectra.waveforms import LFM, BarkerCodedPulse, PolyphaseCodedPulse

ds = RadarDataset(
    waveform_pool=[LFM(), BarkerCodedPulse(), PolyphaseCodedPulse(code_type="p4")],
    num_range_bins=512,
    sample_rate=1e6,
    snr_range=(5.0, 25.0),
    num_targets_range=(1, 3),
    num_samples=1000,
    seed=42,
)
profile, target = ds[0]
# profile: Tensor[512]
# target.num_targets, target.range_bins, target.snrs, target.waveform_label
```

### RadarPipelineDataset

End-to-end multi-CPI pipeline producing waveform → target injection →
clutter → matched filter → MTI → CFAR → Kalman tracker training data.
Returns `(Tensor[sequence_length, num_range_bins], RadarPipelineTarget)`
where the target carries the full pipeline state including detections
and tracker state.

!!! note "Range units"
    `ConstantVelocity` (and other trajectory models) express `initial_range`
    and `velocity` in **range-bin units**, not metres. Scale accordingly.

```python
from spectra.datasets.radar_pipeline import RadarPipelineDataset
from spectra.targets.trajectory import ConstantVelocity
from spectra.waveforms import LFM

ds = RadarPipelineDataset(
    waveform_pool=[LFM()],
    trajectory_pool=[ConstantVelocity(initial_range=128, velocity=5, dt=1e-3)],
    num_range_bins=256,
    sample_rate=1e6,
    pulses_per_cpi=16,
    sequence_length=4,
    apply_mti=True,
    cfar_type="ca",
    num_samples=500,
    seed=0,
)
iq, target = ds[0]
# iq: Tensor[4, 256]  — sequence_length CPIs × num_range_bins
# target.kf_states, target.detections, target.true_ranges, ...
```

---

## Direction-Finding Datasets

### DirectionFindingDataset

Snapshot-matrix dataset for narrowband DoA-estimation training. Returns
`(Tensor[num_elements, 2, num_snapshots], DirectionFindingTarget)` where
the last two dimensions are I and Q channels and `DirectionFindingTarget`
carries `azimuths`, `elevations`, and `num_sources`.

```python
from spectra.arrays import ula
from spectra.datasets.direction_finding import DirectionFindingDataset
from spectra.waveforms import QPSK

array = ula(num_elements=8, spacing=0.5, frequency=1e9)
ds = DirectionFindingDataset(
    array=array,
    signal_pool=[QPSK()],
    num_signals=2,
    num_snapshots=128,
    sample_rate=1e6,
    snr_range=(10.0, 20.0),
    num_samples=1000,
    seed=0,
)
snapshot, target = ds[0]
# snapshot: Tensor[8, 2, 128]  — elements × {I,Q} × snapshots
# target.num_sources, target.azimuths, target.elevations
```

**Note:** `default_collate` cannot batch `DirectionFindingTarget` directly —
pass a custom `collate_fn`:

```python
from torch.utils.data import DataLoader

def collate_fn(batch):
    return torch.stack([x for x, _ in batch]), [t for _, t in batch]

loader = DataLoader(ds, batch_size=8, collate_fn=collate_fn)
```

### WidebandDirectionFindingDataset

Joint wideband spectrum + DoA dataset where each source occupies a
distinct sub-band. Each source must be at least `min_freq_separation` Hz
apart in frequency; angular separation can also be enforced via
`min_angular_separation` (radians). Returns
`(Tensor[num_elements, 2, num_snapshots], WidebandDFTarget)`.

```python
from spectra.arrays import ula
from spectra.datasets.wideband_df import WidebandDirectionFindingDataset
from spectra.waveforms import BPSK, QPSK

array = ula(num_elements=8, spacing=0.5, frequency=1e9)
ds = WidebandDirectionFindingDataset(
    array=array,
    signal_pool=[BPSK(), QPSK()],
    num_signals=3,
    num_snapshots=256,
    sample_rate=10e6,
    capture_bandwidth=8e6,
    snr_range=(5.0, 20.0),
    num_samples=500,
    seed=0,
)
snapshot, target = ds[0]
# snapshot: Tensor[8, 2, 256]
```

See [Direction Finding](direction-finding.md) for the complete DoA
estimation workflow that consumes these datasets.
