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

**Output from `__getitem__`:** `(Tensor[2, num_iq_samples], int)` — channel 0 is I, channel 1 is Q.

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

dataset = SNRSweepDataset(
    waveform_pool=[QPSK(), BPSK()],
    num_samples_per_snr=500,
    snr_values=[-10, -5, 0, 5, 10, 15, 20],
    num_iq_samples=1024,
    sample_rate=1e6,
    seed=42,
)
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

## Deterministic Generation

!!! note "Why `(base_seed, idx)` seeding is safe for multi-worker DataLoaders"
    All synthetic datasets seed NumPy via `np.random.default_rng(seed=(base_seed, idx))`.
    Because the seed encodes the sample index, every worker produces the exact same
    IQ signal for a given index regardless of which worker processes it or in what
    order. This eliminates the seed-collision bugs that occur when workers share
    or independently advance a global RNG.
