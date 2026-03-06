# Benchmarks

## Available Benchmarks

| Name | Task | Classes | Train / Val / Test | SNR range |
|------|------|---------|-------------------|-----------|
| `spectra-18` | AMC classification | 18 | 54k / 9k / 9k | -20 to 30 dB |
| `spectra-18-wideband` | Signal detection | 18 | 9k / 1.5k / 1.5k scenes | 0 to 20 dB |
| `spectra-channel` | Channel conditions | 5 | configurable | varies |

---

## load_benchmark()

`load_benchmark()` returns train, val, and test `Dataset` objects with fixed seeds
and configurations for reproducible evaluation.

| Parameter | Type | Description |
|-----------|------|-------------|
| `name` | `str` | Benchmark name (e.g., `"spectra-18"`) |
| `split` | `str` | `"train"`, `"val"`, or `"test"` |
| `transform` | `callable` (optional) | Applied to each IQ segment |
| `target_transform` | `callable` (optional) | Applied to each label |

```python
from spectra.benchmarks.loader import load_benchmark

train = load_benchmark("spectra-18", split="train")
val   = load_benchmark("spectra-18", split="val")
test  = load_benchmark("spectra-18", split="test")

print(f"Train size: {len(train)}")  # 54000
print(f"Classes: {train.waveform_pool[0].label}, ...")

from torch.utils.data import DataLoader
loader = DataLoader(train, batch_size=64, num_workers=4)
```

---

## Evaluation Helpers

### evaluate_snr_sweep()

Evaluates a trained model or classifier across a range of SNR values.

```python
from spectra.benchmarks.evaluate import evaluate_snr_sweep

results = evaluate_snr_sweep(
    model=my_model,           # callable: iq_batch -> label_batch
    benchmark_name="spectra-18",
    snr_values=[-10, -5, 0, 5, 10, 15, 20],
    num_samples_per_snr=500,
    device="cpu",
)

# results: {"snr_db": [...], "accuracy": [...]}
for snr, acc in zip(results["snr_db"], results["accuracy"]):
    print(f"SNR {snr:+3d} dB: {acc:.1%}")
```
