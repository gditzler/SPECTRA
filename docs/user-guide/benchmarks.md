# Benchmarks

## Available Benchmarks

Built-in configs live under `spectra/benchmarks/configs/` and are resolved by
name (for example `load_benchmark("spectra-18")`).

### `load_benchmark()` — narrowband, wideband, direction finding

| Name | Task | Notes |
|------|------|--------|
| `spectra-18` | AMC (narrowband) | 18 classes; 50k / 10k / 10k samples; SNR −20 to 30 dB |
| `spectra-40` | AMC (narrowband) | 40 classes; 100k / 20k / 20k; SNR −10 to 30 dB |
| `spectra-18-wideband` | Signal detection | Same 18 classes in multi-signal scenes |
| `spectra-5g` | AMC (narrowband) | 5G NR–oriented waveform mix |
| `spectra-radar` | AMC (narrowband) | Radar-oriented waveform mix |
| `spectra-spread` | AMC (narrowband) | Spread-spectrum waveform mix |
| `spectra-protocol` | AMC (narrowband) | Protocol frame generators |
| `spectra-airport` | Wideband | Airport / ISM–style scenes |
| `spectra-maritime-vhf` | Wideband | Maritime VHF–style scenes |
| `spectra-congested-ism` | Wideband | Congested ISM band scenes |
| `spectra-df` | Direction finding | ULA snapshots → `DirectionFindingDataset` |

Configs with `task: narrowband_channel` or `task: narrowband_snr_sweep` are **not**
loaded by `load_benchmark()` — use the helpers below.

### `load_channel_benchmark()` — `spectra-channel`

| Name | Task | Description |
|------|------|-------------|
| `spectra-channel` | Channel robustness | Five fixed impairment stacks; pass `condition=` to select one |

### `load_snr_sweep()` — `spectra-snr`

| Name | Task | Description |
|------|------|-------------|
| `spectra-snr` | SNR grid | `SNRSweepDataset` over configured SNR levels |

---

## load_benchmark()

`load_benchmark()` loads a YAML benchmark and returns a PyTorch `Dataset` (or a
3-tuple when `split="all"`).

| Parameter | Type | Description |
|-----------|------|-------------|
| `name` | `str` | Built-in name (e.g. `"spectra-18"`) or path to a `.yaml` file |
| `split` | `str` | `"train"`, `"val"`, `"test"`, or `"all"` |

Returns a `NarrowbandDataset`, `WidebandDataset`, or `DirectionFindingDataset`
depending on `task` in the YAML (`narrowband`, `wideband`, or `direction_finding`).

```python
from spectra.benchmarks.loader import load_benchmark

train = load_benchmark("spectra-18", split="train")
val   = load_benchmark("spectra-18", split="val")
test  = load_benchmark("spectra-18", split="test")

print(f"Train size: {len(train)}")  # 50000
print(f"Classes: {train.waveform_pool[0].label}, ...")

from torch.utils.data import DataLoader
loader = DataLoader(train, batch_size=64, num_workers=4)
```

Direction-finding example (`spectra-df`):

```python
from spectra.benchmarks import load_benchmark

ds = load_benchmark("spectra-df", split="train")
iq, target = ds[0]  # iq: [n_elements, 2, num_snapshots], target: DirectionFindingTarget
```

---

## Evaluation Helpers

### evaluate_snr_sweep()

Evaluates a callable predictor over an `SNRSweepDataset` and returns
per-SNR accuracy.

```python
import torch
from spectra.benchmarks.evaluate import evaluate_snr_sweep
from spectra.benchmarks.loader import load_snr_sweep

def predict_fn(batch: torch.Tensor) -> torch.Tensor:
    """Accepts Tensor[B, 2, N], returns Tensor[B] of predicted class indices."""
    return model(batch).argmax(dim=-1)

dataset = load_snr_sweep("spectra-snr")
results = evaluate_snr_sweep(predict_fn, dataset, batch_size=64)

# results: {snr_db: {"accuracy": float, "per_class": Dict[int, float]}}
for snr_db in sorted(results):
    acc = results[snr_db]["accuracy"]
    print(f"SNR {snr_db:+5.1f} dB: {acc:.1%}")
```
