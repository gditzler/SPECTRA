# Curriculum & Streaming

## CurriculumSchedule

`CurriculumSchedule` linearly interpolates a parameter (such as SNR range or
number of impairments) between a starting and ending value based on training
progress from 0.0 to 1.0.

**Constructor:**

| Parameter | Type | Description |
|-----------|------|-------------|
| `start` | `any` | Value at the beginning of training (`progress=0.0`) |
| `end` | `any` | Value at the end of training (`progress=1.0`) |

**`at(progress)`** — returns the interpolated value for a given progress fraction.

```python
from spectra.curriculum import CurriculumSchedule

# Start easy (high SNR), end hard (low SNR)
snr_schedule = CurriculumSchedule(start=20.0, end=-5.0)

print(snr_schedule.at(0.0))   # 20.0  (easy start)
print(snr_schedule.at(0.5))   # 7.5   (midpoint)
print(snr_schedule.at(1.0))   # -5.0  (hard end)
```

You can also schedule other parameters:

```python
# Increase number of impairments during training
num_impairments = CurriculumSchedule(start=1, end=4)

current = int(num_impairments.at(epoch / total_epochs))
active_pipeline = Compose(all_impairments[:current])
```

---

## StreamingDataLoader

`StreamingDataLoader` wraps a dataset factory with epoch-aware generation. On each
epoch, it passes the current training progress to the factory so the dataset can
be re-created with updated curriculum parameters.

**Constructor:**

| Parameter | Type | Description |
|-----------|------|-------------|
| `dataset_factory` | `callable` | Function `(params: dict) -> Dataset` |
| `num_epochs` | `int` | Total number of training epochs |
| `batch_size` | `int` | DataLoader batch size |
| `num_workers` | `int` | DataLoader workers |
| `**loader_kwargs` | — | Forwarded to `DataLoader` |

**`epoch(n)`** — returns a `DataLoader` for epoch `n`, with `progress = n / num_epochs`.

```python
from spectra.streaming import StreamingDataLoader
from spectra import NarrowbandDataset, QPSK, BPSK, AWGN, Compose
from spectra.curriculum import CurriculumSchedule

snr_schedule = CurriculumSchedule(start=20.0, end=-5.0)

def dataset_factory(params):
    snr_max = params["snr_max"]
    return NarrowbandDataset(
        waveform_pool=[QPSK(), BPSK()],
        num_samples=10_000,
        num_iq_samples=1024,
        sample_rate=1e6,
        impairments=Compose([AWGN(snr_range=(-5.0, snr_max))]),
        seed=params.get("epoch", 0),
    )

streaming = StreamingDataLoader(
    dataset_factory=dataset_factory,
    num_epochs=50,
    batch_size=64,
    num_workers=4,
)

for epoch_idx in range(50):
    loader = streaming.epoch(epoch_idx)
    for iq, labels in loader:
        # train on batch
        pass
```

---

## Factory Pattern

The `dataset_factory` receives a `params` dict with at minimum `"progress"` (float
0–1) and `"epoch"` (int). Add any curriculum-driven values by wrapping the factory:

```python
from spectra.curriculum import CurriculumSchedule

snr_schedule  = CurriculumSchedule(start=20.0, end=-5.0)
n_sig_schedule = CurriculumSchedule(start=1, end=5)

def dataset_factory(params):
    progress = params["progress"]
    return NarrowbandDataset(
        waveform_pool=[QPSK(), BPSK()],
        num_samples=10_000,
        num_iq_samples=1024,
        sample_rate=1e6,
        impairments=Compose([
            AWGN(snr_range=(-5.0, snr_schedule.at(progress))),
        ]),
        seed=params["epoch"],
    )
```
