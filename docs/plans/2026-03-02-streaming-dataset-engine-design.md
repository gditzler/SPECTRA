# Streaming-First Dataset Engine — Design Document

**Date:** 2026-03-02
**Status:** Approved

## Problem

ML researchers training AMC (Automatic Modulation Classification) and RF signal detection models currently rely on TorchSig, which requires downloading multi-hundred-GB dataset artifacts to disk, offers no built-in difficulty progression, and generates data in pure Python. SPECTRA can offer a fundamentally better workflow.

## One-Line Pitch

*Stream infinite, curriculum-scheduled RF training data at Rust speed — no downloads, no disk, fully reproducible.*

## Architecture

Three new components layer on top of SPECTRA's existing `NarrowbandDataset` / `WidebandDataset` foundation. No changes to the Rust layer, Waveform ABC, or Transform ABC.

```
┌──────────────────────────────────────────────┐
│              User Training Loop               │
│  for epoch in range(N):                       │
│      for batch in loader.epoch(epoch): ...    │
└────────────────┬─────────────────────────────┘
                 │
┌────────────────▼─────────────────────────────┐
│          StreamingDataLoader                  │
│  - Manages epoch-aware seeding               │
│  - Calls CurriculumSchedule.at(progress)     │
│  - Builds fresh dataset per epoch via factory │
│  - Returns standard PyTorch DataLoader        │
└────────────────┬─────────────────────────────┘
                 │
┌────────────────▼─────────────────────────────┐
│          CurriculumSchedule                   │
│  - Maps progress [0.0, 1.0] to param ranges  │
│  - Linear interpolation between start/end    │
│  - Stateless: .at(progress) -> dict           │
└──────────────────────────────────────────────┘
                 │
┌────────────────▼─────────────────────────────┐
│     Existing NarrowbandDataset /              │
│     WidebandDataset                           │
│  - Per-sample deterministic seeding           │
│  - On-the-fly Rust-accelerated generation     │
└──────────────────────────────────────────────┘

┌──────────────────────────────────────────────┐
│          Benchmark Configs (YAML)             │
│  - Fully specify datasets declaratively       │
│  - load_benchmark("spectra-18") -> Dataset    │
│  - Zero-download reproducible benchmarks      │
└──────────────────────────────────────────────┘
```

## Component 1: CurriculumSchedule

A small, stateless object that maps training progress to parameter ranges via linear interpolation.

```python
class CurriculumSchedule:
    def __init__(self, snr_range=None, num_signals=None, impairments=None)
    def at(self, progress: float) -> dict
```

**Parameters:**
- `snr_range`: `{"start": (high_lo, high_hi), "end": (low_lo, low_hi)}` — easy-to-hard SNR
- `num_signals`: `{"start": (min, max), "end": (min, max)}` — sparse-to-dense scenes (wideband only)
- `impairments`: `{"impairment_name": {"start": severity_lo, "end": severity_hi}}` — ramps severity bounds

**Behavior:**
- `at(0.0)` returns start values, `at(1.0)` returns end values
- Intermediate progress values are linearly interpolated
- Returns a flat dict: `{"snr_range": (lo, hi), "num_signals": (lo, hi), "impairments": {...}}`
- `None` fields are omitted from output (caller uses defaults)

**Design constraints:**
- Linear interpolation only for v1 (no cosine/step curves — YAGNI)
- Does not interact with datasets directly — purely a parameter calculator
- Fully testable without any dataset or DataLoader dependencies

## Component 2: StreamingDataLoader

A wrapper that manages epoch-aware seeding and curriculum injection, returning standard PyTorch DataLoaders.

```python
class StreamingDataLoader:
    def __init__(
        self,
        dataset_factory: Callable[[dict], Dataset],
        base_seed: int,
        num_epochs: int,
        curriculum: Optional[CurriculumSchedule] = None,
        **dataloader_kwargs,
    )
    def epoch(self, epoch: int) -> DataLoader
```

**`dataset_factory` pattern:**
- Takes a callable `f(params) -> Dataset` instead of a dataset instance
- Each `.epoch(n)` call:
  1. Computes `progress = n / (num_epochs - 1)` (or 0.0 if single epoch)
  2. Gets curriculum params via `schedule.at(progress)` (if curriculum provided)
  3. Computes `epoch_seed = hash(base_seed, epoch)`
  4. Merges `{"seed": epoch_seed, **curriculum_params}` into a params dict
  5. Calls `dataset_factory(params)` to build a fresh dataset
  6. Wraps in `torch.utils.data.DataLoader` with stored kwargs
  7. Returns the DataLoader

**Epoch-aware seeding formula:**
```
epoch_seed = hash((base_seed, epoch))
per_sample_seed = (epoch_seed, sample_idx)   # handled by existing dataset
```

This ensures:
- Same `(base_seed, epoch, idx)` triple always produces identical data
- Different epochs produce different data
- Reproducible across machines given the same base_seed

**Without curriculum:**
- Factory receives `{"seed": epoch_seed}` only
- Still provides infinite unique data via epoch-varying seeds
- Drop-in upgrade path for existing users

**What this is NOT:**
- Not an IterableDataset (avoids worker-partitioning complexity)
- Not a custom sampler (datasets handle their own randomness)
- Not a Trainer (no optimizer, no loss, no training loop ownership)

## Component 3: Benchmark Configs

YAML files that fully specify reproducible datasets, shipped in-package.

**YAML schema:**
```yaml
name: "spectra-18"
version: "1.0"
description: "18-class narrowband AMC benchmark"
task: "narrowband"           # or "wideband"

sample_rate: 1_000_000
num_iq_samples: 1024
num_samples:
  train: 50_000
  val: 10_000
  test: 10_000

seed:
  train: 1000
  val: 2000
  test: 3000

waveform_pool:
  - {type: "BPSK"}
  - {type: "QPSK"}
  # ... full list of waveform types with optional params

snr_range: [-10, 30]

impairments:
  - {type: "FrequencyOffset", params: {max_offset: 50000}}
  - {type: "PhaseOffset", params: {max_offset: 3.14159}}
  - {type: "AWGN"}

# Wideband-only fields (ignored for narrowband):
scene:
  capture_bandwidth: 1_000_000
  capture_duration: 0.001
  num_signals: [1, 5]
  allow_overlap: true
```

**Python API:**
```python
from spectra.benchmarks import load_benchmark

train_ds, val_ds, test_ds = load_benchmark("spectra-18", split="all")
train_ds = load_benchmark("spectra-18", split="train")
```

**Resolution:** `load_benchmark(name)` checks:
1. Built-in configs via `importlib.resources` from `spectra/benchmarks/configs/`
2. File path if `name` ends in `.yaml` or `.yml`

**Design constraints:**
- YAML only (human-readable, citable in paper appendices)
- Train/val/test splits via separate seeds, same parameters
- Version field for forward compatibility
- No CLI for v1

## Flagship Benchmarks

### spectra-18 (Narrowband)
18-class AMC benchmark: BPSK, QPSK, 8PSK, QAM16, QAM64, QAM256, 2-FSK, 4-FSK, MSK, GMSK, OFDM, LFM, CostasCode, FrankCode, P1Code, AM, Barker, Noise.

50k train / 10k val / 10k test samples, 1024 IQ samples each, SNR range [-10, 30] dB.

### spectra-18-wideband (Wideband)
Same 18 signal classes composed into multi-signal scenes for detection tasks. 1-5 signals per scene, COCO-format bounding box targets.

## What Does NOT Ship (YAGNI)

- CLI tooling for benchmarks
- PyTorch Lightning / HuggingFace Trainer integration
- SigMF export
- Pretrained models
- Nonlinear curriculum curves (cosine, step)
- Waveform-specific curriculum scheduling

## Dependencies

No new dependencies. NumPy, PyTorch, and PyYAML (transitive via PyTorch) are sufficient.
Rust layer unchanged.

## Implementation Order

1. Feature expansion (impairments, waveforms, transforms) — already planned in `2026-03-01-spectra-feature-expansion.md`
2. `CurriculumSchedule` — small, standalone, fully testable
3. `StreamingDataLoader` — depends on curriculum + existing datasets
4. Benchmark config loader (`load_benchmark`) — depends on all waveforms/impairments existing
5. `spectra-18` YAML files — last step, authoring configs

## Sources

Competitive landscape research:
- [TorchSig GitHub](https://github.com/TorchDSP/torchsig)
- [TorchSig 2.0 GRCon 2025 presentation](https://events.gnuradio.org/event/26/contributions/752/)
- [SigMF specification](https://sigmf.org/)
- [Overview of Open RF Datasets](https://panoradio-sdr.de/overview-of-open-datasets-for-rf-signal-classification/)
