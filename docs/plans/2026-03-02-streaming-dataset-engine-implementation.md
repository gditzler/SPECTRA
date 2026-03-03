# Streaming Dataset Engine Implementation Plan

> **For Claude:** REQUIRED SUB-SKILL: Use superpowers:executing-plans to implement this plan task-by-task.

**Goal:** Add CurriculumSchedule, StreamingDataLoader, and reproducible benchmark configs (spectra-18) to make SPECTRA the first streaming-first RF/ML dataset engine.

**Architecture:** Three new Python modules layer on existing datasets. `CurriculumSchedule` is a stateless parameter interpolator. `StreamingDataLoader` wraps PyTorch DataLoader with epoch-aware seeding and a dataset factory pattern. Benchmark configs are YAML files parsed by `load_benchmark()` into configured dataset instances. No Rust changes.

**Tech Stack:** Python 3.10+, NumPy, PyTorch, PyYAML (new dependency)

**Prerequisites:** The feature expansion plan (`2026-03-01-spectra-feature-expansion.md`) must be completed first — the benchmark configs reference AM, Noise, BarkerCode, ZadoffChu waveforms and PhaseOffset, IQImbalance, DCOffset impairments.

---

### Task 1: Add PyYAML dependency

**Files:**
- Modify: `pyproject.toml:11-14`

**Context:** Benchmark configs are YAML files. PyYAML is needed to parse them. This is the only new dependency in the entire streaming engine.

**Step 1: Add pyyaml to dependencies**

In `pyproject.toml`, change the `dependencies` list from:
```toml
dependencies = [
    "numpy>=1.24",
    "torch>=2.0",
]
```
to:
```toml
dependencies = [
    "numpy>=1.24",
    "torch>=2.0",
    "pyyaml>=6.0",
]
```

**Step 2: Install the new dependency**

Run: `uv pip install pyyaml`
Expected: Successfully installed pyyaml

**Step 3: Verify import works**

Run: `python -c "import yaml; print(yaml.__version__)"`
Expected: Prints version number (e.g., `6.0.2`)

**Step 4: Commit**

```bash
git add pyproject.toml
git commit -m "build: add pyyaml dependency for benchmark configs"
```

---

### Task 2: CurriculumSchedule — core interpolation

**Files:**
- Create: `python/spectra/curriculum.py`
- Create: `tests/test_curriculum.py`

**Context:** `CurriculumSchedule` maps training progress (float 0.0→1.0) to parameter ranges via linear interpolation. It's a stateless calculator — given progress, it returns a dict of interpolated values. It has no knowledge of datasets, DataLoaders, or training loops.

Each schedulable parameter is a dict with `"start"` and `"end"` keys. For tuple-valued params (like `snr_range`), both start and end are tuples and each element is interpolated independently. For scalar params (like impairment severities), start and end are floats.

**Step 1: Write the failing tests**

```python
# tests/test_curriculum.py
import pytest


class TestCurriculumSchedule:
    def test_snr_at_start(self):
        from spectra.curriculum import CurriculumSchedule

        schedule = CurriculumSchedule(
            snr_range={"start": (20.0, 30.0), "end": (0.0, 10.0)},
        )
        params = schedule.at(0.0)
        assert params["snr_range"] == pytest.approx((20.0, 30.0))

    def test_snr_at_end(self):
        from spectra.curriculum import CurriculumSchedule

        schedule = CurriculumSchedule(
            snr_range={"start": (20.0, 30.0), "end": (0.0, 10.0)},
        )
        params = schedule.at(1.0)
        assert params["snr_range"] == pytest.approx((0.0, 10.0))

    def test_snr_at_midpoint(self):
        from spectra.curriculum import CurriculumSchedule

        schedule = CurriculumSchedule(
            snr_range={"start": (20.0, 30.0), "end": (0.0, 10.0)},
        )
        params = schedule.at(0.5)
        assert params["snr_range"] == pytest.approx((10.0, 20.0))

    def test_num_signals_interpolation(self):
        from spectra.curriculum import CurriculumSchedule

        schedule = CurriculumSchedule(
            num_signals={"start": (1, 2), "end": (4, 8)},
        )
        params = schedule.at(0.5)
        # Linear interp: (2.5, 5.0) -> rounded to ints: (2, 5)
        assert params["num_signals"] == (2, 5)

    def test_num_signals_at_boundaries(self):
        from spectra.curriculum import CurriculumSchedule

        schedule = CurriculumSchedule(
            num_signals={"start": (1, 2), "end": (4, 8)},
        )
        assert schedule.at(0.0)["num_signals"] == (1, 2)
        assert schedule.at(1.0)["num_signals"] == (4, 8)

    def test_impairment_scheduling(self):
        from spectra.curriculum import CurriculumSchedule

        schedule = CurriculumSchedule(
            impairments={
                "phase_offset": {"start": 0.0, "end": 0.5},
                "iq_imbalance": {"start": 0.0, "end": 0.2},
            },
        )
        params = schedule.at(0.5)
        assert params["impairments"]["phase_offset"] == pytest.approx(0.25)
        assert params["impairments"]["iq_imbalance"] == pytest.approx(0.1)

    def test_impairments_at_boundaries(self):
        from spectra.curriculum import CurriculumSchedule

        schedule = CurriculumSchedule(
            impairments={"phase_offset": {"start": 0.0, "end": 1.0}},
        )
        assert schedule.at(0.0)["impairments"]["phase_offset"] == pytest.approx(0.0)
        assert schedule.at(1.0)["impairments"]["phase_offset"] == pytest.approx(1.0)

    def test_combined_schedule(self):
        from spectra.curriculum import CurriculumSchedule

        schedule = CurriculumSchedule(
            snr_range={"start": (20.0, 30.0), "end": (0.0, 10.0)},
            num_signals={"start": (1, 2), "end": (3, 6)},
            impairments={"phase_offset": {"start": 0.0, "end": 0.5}},
        )
        params = schedule.at(0.25)
        assert params["snr_range"] == pytest.approx((15.0, 25.0))
        assert params["num_signals"] == (1, 3)
        assert params["impairments"]["phase_offset"] == pytest.approx(0.125)

    def test_none_fields_omitted(self):
        from spectra.curriculum import CurriculumSchedule

        schedule = CurriculumSchedule(snr_range={"start": (20.0, 30.0), "end": (0.0, 10.0)})
        params = schedule.at(0.5)
        assert "snr_range" in params
        assert "num_signals" not in params
        assert "impairments" not in params

    def test_progress_clamped(self):
        from spectra.curriculum import CurriculumSchedule

        schedule = CurriculumSchedule(
            snr_range={"start": (20.0, 30.0), "end": (0.0, 10.0)},
        )
        assert schedule.at(-0.5)["snr_range"] == pytest.approx((20.0, 30.0))
        assert schedule.at(1.5)["snr_range"] == pytest.approx((0.0, 10.0))

    def test_empty_schedule(self):
        from spectra.curriculum import CurriculumSchedule

        schedule = CurriculumSchedule()
        params = schedule.at(0.5)
        assert params == {}
```

**Step 2: Run tests to verify they fail**

Run: `pytest tests/test_curriculum.py -v`
Expected: FAIL — `ModuleNotFoundError: No module named 'spectra.curriculum'`

**Step 3: Write minimal implementation**

```python
# python/spectra/curriculum.py
from typing import Any, Dict, Optional, Tuple


def _lerp(a: float, b: float, t: float) -> float:
    """Linear interpolation between a and b at parameter t."""
    return a + (b - a) * t


class CurriculumSchedule:
    """Maps training progress [0.0, 1.0] to parameter ranges via linear interpolation.

    Parameters
    ----------
    snr_range : dict, optional
        {"start": (lo, hi), "end": (lo, hi)} — SNR range in dB.
    num_signals : dict, optional
        {"start": (min, max), "end": (min, max)} — signal count range (wideband).
    impairments : dict, optional
        {"name": {"start": severity, "end": severity}} — per-impairment severity.
    """

    def __init__(
        self,
        snr_range: Optional[Dict[str, Tuple[float, float]]] = None,
        num_signals: Optional[Dict[str, Tuple[int, int]]] = None,
        impairments: Optional[Dict[str, Dict[str, float]]] = None,
    ):
        self.snr_range = snr_range
        self.num_signals = num_signals
        self.impairments = impairments

    def at(self, progress: float) -> Dict[str, Any]:
        """Return interpolated parameters at the given progress.

        Parameters
        ----------
        progress : float
            Training progress in [0.0, 1.0]. Values outside this range are clamped.

        Returns
        -------
        dict
            Interpolated parameter values. Only includes fields that were configured.
        """
        t = max(0.0, min(1.0, progress))
        result: Dict[str, Any] = {}

        if self.snr_range is not None:
            s = self.snr_range["start"]
            e = self.snr_range["end"]
            result["snr_range"] = (_lerp(s[0], e[0], t), _lerp(s[1], e[1], t))

        if self.num_signals is not None:
            s = self.num_signals["start"]
            e = self.num_signals["end"]
            result["num_signals"] = (
                round(_lerp(s[0], e[0], t)),
                round(_lerp(s[1], e[1], t)),
            )

        if self.impairments is not None:
            imp_result = {}
            for name, cfg in self.impairments.items():
                imp_result[name] = _lerp(cfg["start"], cfg["end"], t)
            result["impairments"] = imp_result

        return result
```

**Step 4: Run tests to verify they pass**

Run: `pytest tests/test_curriculum.py -v`
Expected: All 11 tests PASS

**Step 5: Commit**

```bash
git add python/spectra/curriculum.py tests/test_curriculum.py
git commit -m "feat: add CurriculumSchedule for difficulty progression"
```

---

### Task 3: Register CurriculumSchedule in public API

**Files:**
- Modify: `python/spectra/__init__.py`

**Step 1: Add import and export**

Add to `python/spectra/__init__.py`:

Import line (after existing imports):
```python
from spectra.curriculum import CurriculumSchedule
```

Add `"CurriculumSchedule"` to `__all__`.

**Step 2: Run smoke test**

Run: `python -c "from spectra import CurriculumSchedule; print(CurriculumSchedule)"`
Expected: `<class 'spectra.curriculum.CurriculumSchedule'>`

**Step 3: Run full test suite**

Run: `pytest tests/ -v`
Expected: All tests PASS (including existing smoke test)

**Step 4: Commit**

```bash
git add python/spectra/__init__.py
git commit -m "feat: export CurriculumSchedule from spectra public API"
```

---

### Task 4: StreamingDataLoader — epoch-aware seeding

**Files:**
- Create: `python/spectra/streaming.py`
- Create: `tests/test_streaming.py`

**Context:** `StreamingDataLoader` wraps PyTorch DataLoader with epoch-aware seeding and optional curriculum injection. It takes a `dataset_factory` callable that builds a fresh dataset from a parameter dict. Each call to `.epoch(n)` computes a unique seed, queries the curriculum, and returns a standard `DataLoader`.

The epoch seed is computed as `hash((base_seed, epoch))` to ensure determinism. We use Python's built-in `hash()` modulo `2**31` to stay in numpy's seed range.

**Step 1: Write the failing tests**

```python
# tests/test_streaming.py
import numpy as np
import torch
import pytest


class TestStreamingDataLoader:
    def test_epoch_returns_dataloader(self):
        from spectra.streaming import StreamingDataLoader
        from spectra.datasets import NarrowbandDataset
        from spectra.waveforms import QPSK

        def factory(params):
            return NarrowbandDataset(
                waveform_pool=[QPSK(samples_per_symbol=4)],
                num_samples=32,
                num_iq_samples=256,
                sample_rate=1e6,
                seed=params["seed"],
            )

        loader = StreamingDataLoader(
            dataset_factory=factory,
            base_seed=42,
            num_epochs=10,
            batch_size=8,
        )
        dl = loader.epoch(0)
        assert isinstance(dl, torch.utils.data.DataLoader)

    def test_epoch_produces_batches(self):
        from spectra.streaming import StreamingDataLoader
        from spectra.datasets import NarrowbandDataset
        from spectra.waveforms import QPSK

        def factory(params):
            return NarrowbandDataset(
                waveform_pool=[QPSK(samples_per_symbol=4)],
                num_samples=16,
                num_iq_samples=256,
                sample_rate=1e6,
                seed=params["seed"],
            )

        loader = StreamingDataLoader(
            dataset_factory=factory,
            base_seed=42,
            num_epochs=5,
            batch_size=8,
        )
        batches = list(loader.epoch(0))
        assert len(batches) == 2  # 16 samples / 8 batch_size
        data, labels = batches[0]
        assert data.shape == (8, 2, 256)

    def test_different_epochs_produce_different_data(self):
        from spectra.streaming import StreamingDataLoader
        from spectra.datasets import NarrowbandDataset
        from spectra.waveforms import QPSK

        def factory(params):
            return NarrowbandDataset(
                waveform_pool=[QPSK(samples_per_symbol=4)],
                num_samples=8,
                num_iq_samples=256,
                sample_rate=1e6,
                seed=params["seed"],
            )

        loader = StreamingDataLoader(
            dataset_factory=factory,
            base_seed=42,
            num_epochs=10,
            batch_size=8,
        )
        data_e0, _ = next(iter(loader.epoch(0)))
        data_e1, _ = next(iter(loader.epoch(1)))
        assert not torch.equal(data_e0, data_e1)

    def test_same_epoch_is_deterministic(self):
        from spectra.streaming import StreamingDataLoader
        from spectra.datasets import NarrowbandDataset
        from spectra.waveforms import QPSK

        def factory(params):
            return NarrowbandDataset(
                waveform_pool=[QPSK(samples_per_symbol=4)],
                num_samples=8,
                num_iq_samples=256,
                sample_rate=1e6,
                seed=params["seed"],
            )

        loader = StreamingDataLoader(
            dataset_factory=factory,
            base_seed=42,
            num_epochs=10,
            batch_size=8,
        )
        data_a, _ = next(iter(loader.epoch(3)))
        data_b, _ = next(iter(loader.epoch(3)))
        torch.testing.assert_close(data_a, data_b)

    def test_same_base_seed_reproduces_across_instances(self):
        from spectra.streaming import StreamingDataLoader
        from spectra.datasets import NarrowbandDataset
        from spectra.waveforms import QPSK

        def factory(params):
            return NarrowbandDataset(
                waveform_pool=[QPSK(samples_per_symbol=4)],
                num_samples=8,
                num_iq_samples=256,
                sample_rate=1e6,
                seed=params["seed"],
            )

        loader1 = StreamingDataLoader(
            dataset_factory=factory, base_seed=42, num_epochs=10, batch_size=8
        )
        loader2 = StreamingDataLoader(
            dataset_factory=factory, base_seed=42, num_epochs=10, batch_size=8
        )
        d1, _ = next(iter(loader1.epoch(5)))
        d2, _ = next(iter(loader2.epoch(5)))
        torch.testing.assert_close(d1, d2)

    def test_different_base_seeds_differ(self):
        from spectra.streaming import StreamingDataLoader
        from spectra.datasets import NarrowbandDataset
        from spectra.waveforms import QPSK

        def factory(params):
            return NarrowbandDataset(
                waveform_pool=[QPSK(samples_per_symbol=4)],
                num_samples=8,
                num_iq_samples=256,
                sample_rate=1e6,
                seed=params["seed"],
            )

        loader1 = StreamingDataLoader(
            dataset_factory=factory, base_seed=42, num_epochs=10, batch_size=8
        )
        loader2 = StreamingDataLoader(
            dataset_factory=factory, base_seed=99, num_epochs=10, batch_size=8
        )
        d1, _ = next(iter(loader1.epoch(0)))
        d2, _ = next(iter(loader2.epoch(0)))
        assert not torch.equal(d1, d2)


class TestStreamingDataLoaderWithCurriculum:
    def test_curriculum_params_passed_to_factory(self):
        from spectra.streaming import StreamingDataLoader
        from spectra.curriculum import CurriculumSchedule

        captured_params = []

        def factory(params):
            captured_params.append(params.copy())
            from spectra.datasets import NarrowbandDataset
            from spectra.waveforms import QPSK

            return NarrowbandDataset(
                waveform_pool=[QPSK(samples_per_symbol=4)],
                num_samples=4,
                num_iq_samples=256,
                sample_rate=1e6,
                seed=params["seed"],
            )

        schedule = CurriculumSchedule(
            snr_range={"start": (20.0, 30.0), "end": (0.0, 10.0)},
        )
        loader = StreamingDataLoader(
            dataset_factory=factory,
            base_seed=42,
            num_epochs=10,
            curriculum=schedule,
            batch_size=4,
        )

        # Epoch 0 → progress=0.0 → snr_range=(20, 30)
        list(loader.epoch(0))
        assert captured_params[0]["snr_range"] == pytest.approx((20.0, 30.0))

        # Epoch 9 → progress=1.0 → snr_range=(0, 10)
        list(loader.epoch(9))
        assert captured_params[1]["snr_range"] == pytest.approx((0.0, 10.0))

    def test_no_curriculum_passes_seed_only(self):
        captured_params = []

        def factory(params):
            captured_params.append(params.copy())
            from spectra.datasets import NarrowbandDataset
            from spectra.waveforms import QPSK

            return NarrowbandDataset(
                waveform_pool=[QPSK(samples_per_symbol=4)],
                num_samples=4,
                num_iq_samples=256,
                sample_rate=1e6,
                seed=params["seed"],
            )

        from spectra.streaming import StreamingDataLoader

        loader = StreamingDataLoader(
            dataset_factory=factory,
            base_seed=42,
            num_epochs=5,
            batch_size=4,
        )
        list(loader.epoch(0))
        assert "seed" in captured_params[0]
        assert "snr_range" not in captured_params[0]

    def test_single_epoch_progress_is_zero(self):
        captured_params = []

        def factory(params):
            captured_params.append(params.copy())
            from spectra.datasets import NarrowbandDataset
            from spectra.waveforms import QPSK

            return NarrowbandDataset(
                waveform_pool=[QPSK(samples_per_symbol=4)],
                num_samples=4,
                num_iq_samples=256,
                sample_rate=1e6,
                seed=params["seed"],
            )

        from spectra.streaming import StreamingDataLoader
        from spectra.curriculum import CurriculumSchedule

        schedule = CurriculumSchedule(
            snr_range={"start": (20.0, 30.0), "end": (0.0, 10.0)},
        )
        loader = StreamingDataLoader(
            dataset_factory=factory,
            base_seed=42,
            num_epochs=1,
            curriculum=schedule,
            batch_size=4,
        )
        list(loader.epoch(0))
        # Single epoch: progress=0.0
        assert captured_params[0]["snr_range"] == pytest.approx((20.0, 30.0))
```

**Step 2: Run tests to verify they fail**

Run: `pytest tests/test_streaming.py -v`
Expected: FAIL — `ModuleNotFoundError: No module named 'spectra.streaming'`

**Step 3: Write minimal implementation**

```python
# python/spectra/streaming.py
import hashlib
from typing import Any, Callable, Dict, Optional

import torch
from torch.utils.data import DataLoader, Dataset

from spectra.curriculum import CurriculumSchedule


def _epoch_seed(base_seed: int, epoch: int) -> int:
    """Deterministic epoch seed via hash.

    Uses SHA-256 to avoid Python's hash() randomization (PYTHONHASHSEED).
    Returns a positive integer in numpy's valid seed range.
    """
    h = hashlib.sha256(f"{base_seed}:{epoch}".encode()).hexdigest()
    return int(h[:8], 16)  # 32-bit unsigned int from first 8 hex chars


class StreamingDataLoader:
    """Epoch-aware DataLoader wrapper with optional curriculum scheduling.

    Each call to `.epoch(n)` builds a fresh dataset via the factory,
    injecting a unique deterministic seed and curriculum parameters.

    Parameters
    ----------
    dataset_factory : callable
        ``f(params: dict) -> Dataset``. Called once per epoch. The params dict
        always contains ``"seed"`` (int). If a curriculum is provided, it also
        contains the curriculum's interpolated parameters for that epoch.
    base_seed : int
        Root seed for deterministic generation.
    num_epochs : int
        Total number of epochs (used to compute curriculum progress).
    curriculum : CurriculumSchedule, optional
        Difficulty schedule. If None, only seed varies per epoch.
    **dataloader_kwargs
        Forwarded to ``torch.utils.data.DataLoader`` (batch_size, num_workers, etc.).
    """

    def __init__(
        self,
        dataset_factory: Callable[[Dict[str, Any]], Dataset],
        base_seed: int,
        num_epochs: int,
        curriculum: Optional[CurriculumSchedule] = None,
        **dataloader_kwargs: Any,
    ):
        self.dataset_factory = dataset_factory
        self.base_seed = base_seed
        self.num_epochs = num_epochs
        self.curriculum = curriculum
        self.dataloader_kwargs = dataloader_kwargs

    def epoch(self, epoch: int) -> DataLoader:
        """Build and return a DataLoader for the given epoch.

        Parameters
        ----------
        epoch : int
            Current epoch index (0-based).

        Returns
        -------
        DataLoader
            A standard PyTorch DataLoader wrapping the factory-built dataset.
        """
        seed = _epoch_seed(self.base_seed, epoch)

        params: Dict[str, Any] = {"seed": seed}

        if self.curriculum is not None:
            if self.num_epochs <= 1:
                progress = 0.0
            else:
                progress = epoch / (self.num_epochs - 1)
            curriculum_params = self.curriculum.at(progress)
            params.update(curriculum_params)

        dataset = self.dataset_factory(params)
        return DataLoader(dataset, **self.dataloader_kwargs)
```

**Step 4: Run tests to verify they pass**

Run: `pytest tests/test_streaming.py -v`
Expected: All 9 tests PASS

**Step 5: Commit**

```bash
git add python/spectra/streaming.py tests/test_streaming.py
git commit -m "feat: add StreamingDataLoader with epoch-aware seeding and curriculum"
```

---

### Task 5: Register StreamingDataLoader in public API

**Files:**
- Modify: `python/spectra/__init__.py`

**Step 1: Add import and export**

Add to `python/spectra/__init__.py`:

Import line:
```python
from spectra.streaming import StreamingDataLoader
```

Add `"StreamingDataLoader"` to `__all__`.

**Step 2: Run smoke test**

Run: `python -c "from spectra import StreamingDataLoader; print(StreamingDataLoader)"`
Expected: `<class 'spectra.streaming.StreamingDataLoader'>`

**Step 3: Run full test suite**

Run: `pytest tests/ -v`
Expected: All tests PASS

**Step 4: Commit**

```bash
git add python/spectra/__init__.py
git commit -m "feat: export StreamingDataLoader from spectra public API"
```

---

### Task 6: Benchmark config loader — YAML parsing and dataset construction

**Files:**
- Create: `python/spectra/benchmarks/__init__.py`
- Create: `python/spectra/benchmarks/loader.py`
- Create: `tests/test_benchmarks.py`

**Context:** `load_benchmark(name, split)` parses a YAML config and returns configured dataset instances. It resolves waveform/impairment types by name from a registry, builds the waveform pool and impairment chain, and returns `NarrowbandDataset` or `WidebandDataset` depending on the `task` field.

The loader checks for built-in configs via `importlib.resources` under `spectra/benchmarks/configs/`, and also accepts raw file paths ending in `.yaml`/`.yml`.

**Step 1: Write the failing tests**

```python
# tests/test_benchmarks.py
import os
import tempfile

import pytest
import torch


class TestLoadBenchmark:
    def test_load_from_yaml_file_narrowband(self, tmp_path):
        from spectra.benchmarks import load_benchmark

        config_content = """
name: "test-nb"
version: "1.0"
description: "Test narrowband benchmark"
task: "narrowband"
sample_rate: 1000000
num_iq_samples: 256
num_samples:
  train: 16
  val: 8
  test: 8
seed:
  train: 100
  val: 200
  test: 300
waveform_pool:
  - {type: "QPSK"}
  - {type: "BPSK"}
snr_range: [0, 20]
impairments:
  - {type: "AWGN"}
"""
        config_file = tmp_path / "test.yaml"
        config_file.write_text(config_content)

        train_ds = load_benchmark(str(config_file), split="train")
        assert len(train_ds) == 16
        data, label = train_ds[0]
        assert isinstance(data, torch.Tensor)
        assert data.shape == (2, 256)
        assert isinstance(label, int)

    def test_load_all_splits(self, tmp_path):
        from spectra.benchmarks import load_benchmark

        config_content = """
name: "test-splits"
version: "1.0"
task: "narrowband"
sample_rate: 1000000
num_iq_samples: 256
num_samples:
  train: 16
  val: 8
  test: 8
seed:
  train: 100
  val: 200
  test: 300
waveform_pool:
  - {type: "QPSK"}
snr_range: [0, 20]
impairments: []
"""
        config_file = tmp_path / "test.yaml"
        config_file.write_text(config_content)

        train_ds, val_ds, test_ds = load_benchmark(str(config_file), split="all")
        assert len(train_ds) == 16
        assert len(val_ds) == 8
        assert len(test_ds) == 8

    def test_different_splits_have_different_seeds(self, tmp_path):
        from spectra.benchmarks import load_benchmark

        config_content = """
name: "test-seeds"
version: "1.0"
task: "narrowband"
sample_rate: 1000000
num_iq_samples: 256
num_samples:
  train: 8
  val: 8
  test: 8
seed:
  train: 100
  val: 200
  test: 300
waveform_pool:
  - {type: "QPSK"}
snr_range: [0, 20]
impairments: []
"""
        config_file = tmp_path / "test.yaml"
        config_file.write_text(config_content)

        train_ds, val_ds, _ = load_benchmark(str(config_file), split="all")
        d_train, _ = train_ds[0]
        d_val, _ = val_ds[0]
        assert not torch.equal(d_train, d_val)

    def test_waveform_with_params(self, tmp_path):
        from spectra.benchmarks import load_benchmark

        config_content = """
name: "test-params"
version: "1.0"
task: "narrowband"
sample_rate: 1000000
num_iq_samples: 256
num_samples:
  train: 8
  val: 4
  test: 4
seed:
  train: 100
  val: 200
  test: 300
waveform_pool:
  - {type: "FSK", params: {order: 4}}
  - {type: "QPSK", params: {samples_per_symbol: 4}}
snr_range: [5, 25]
impairments: []
"""
        config_file = tmp_path / "test.yaml"
        config_file.write_text(config_content)

        ds = load_benchmark(str(config_file), split="train")
        data, label = ds[0]
        assert data.shape == (2, 256)

    def test_wideband_task(self, tmp_path):
        from spectra.benchmarks import load_benchmark

        config_content = """
name: "test-wb"
version: "1.0"
task: "wideband"
sample_rate: 2000000
num_iq_samples: 1024
num_samples:
  train: 8
  val: 4
  test: 4
seed:
  train: 100
  val: 200
  test: 300
waveform_pool:
  - {type: "QPSK"}
  - {type: "BPSK"}
snr_range: [5, 25]
impairments: []
scene:
  capture_bandwidth: 1000000
  capture_duration: 0.001
  num_signals: [1, 3]
  allow_overlap: true
"""
        config_file = tmp_path / "test.yaml"
        config_file.write_text(config_content)

        ds = load_benchmark(str(config_file), split="train")
        data, targets = ds[0]
        assert isinstance(data, torch.Tensor)
        assert isinstance(targets, dict)
        assert "signal_descs" in targets

    def test_invalid_file_raises(self):
        from spectra.benchmarks import load_benchmark

        with pytest.raises(FileNotFoundError):
            load_benchmark("/nonexistent/path.yaml", split="train")

    def test_invalid_split_raises(self, tmp_path):
        from spectra.benchmarks import load_benchmark

        config_content = """
name: "test"
version: "1.0"
task: "narrowband"
sample_rate: 1000000
num_iq_samples: 256
num_samples: {train: 8, val: 4, test: 4}
seed: {train: 1, val: 2, test: 3}
waveform_pool: [{type: "QPSK"}]
snr_range: [0, 20]
impairments: []
"""
        config_file = tmp_path / "test.yaml"
        config_file.write_text(config_content)

        with pytest.raises(ValueError, match="split"):
            load_benchmark(str(config_file), split="invalid")
```

**Step 2: Run tests to verify they fail**

Run: `pytest tests/test_benchmarks.py -v`
Expected: FAIL — `ModuleNotFoundError: No module named 'spectra.benchmarks'`

**Step 3: Write the loader implementation**

```python
# python/spectra/benchmarks/__init__.py
from spectra.benchmarks.loader import load_benchmark

__all__ = ["load_benchmark"]
```

```python
# python/spectra/benchmarks/loader.py
import importlib.resources
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple, Union

import yaml

from spectra.datasets import NarrowbandDataset, WidebandDataset
from spectra.impairments import AWGN, Compose, FrequencyOffset
from spectra.scene.composer import SceneConfig
from spectra.waveforms.base import Waveform

# Registry: name -> class. Lazy-populated on first use.
_WAVEFORM_REGISTRY: Optional[Dict[str, type]] = None
_IMPAIRMENT_REGISTRY: Optional[Dict[str, type]] = None


def _get_waveform_registry() -> Dict[str, type]:
    global _WAVEFORM_REGISTRY
    if _WAVEFORM_REGISTRY is None:
        from spectra import waveforms as wmod

        _WAVEFORM_REGISTRY = {}
        # Import all waveform classes from __all__
        for name in wmod.__all__:
            cls = getattr(wmod, name)
            _WAVEFORM_REGISTRY[name] = cls
    return _WAVEFORM_REGISTRY


def _get_impairment_registry() -> Dict[str, type]:
    global _IMPAIRMENT_REGISTRY
    if _IMPAIRMENT_REGISTRY is None:
        from spectra import impairments as imod

        _IMPAIRMENT_REGISTRY = {}
        for name in imod.__all__:
            cls = getattr(imod, name)
            if name != "Compose":
                _IMPAIRMENT_REGISTRY[name] = cls
    return _IMPAIRMENT_REGISTRY


def _build_waveform_pool(pool_config: List[Dict[str, Any]]) -> List[Waveform]:
    registry = _get_waveform_registry()
    pool = []
    for entry in pool_config:
        wtype = entry["type"]
        if wtype not in registry:
            raise ValueError(
                f"Unknown waveform type '{wtype}'. "
                f"Available: {sorted(registry.keys())}"
            )
        params = entry.get("params", {})
        pool.append(registry[wtype](**params))
    return pool


def _build_impairments(imp_config: List[Dict[str, Any]]) -> Optional[Compose]:
    if not imp_config:
        return None
    registry = _get_impairment_registry()
    transforms = []
    for entry in imp_config:
        itype = entry["type"]
        if itype not in registry:
            raise ValueError(
                f"Unknown impairment type '{itype}'. "
                f"Available: {sorted(registry.keys())}"
            )
        params = entry.get("params", {})
        transforms.append(registry[itype](**params))
    return Compose(transforms)


def _resolve_config_path(name: str) -> Path:
    """Resolve a benchmark name or file path to a Path object."""
    # If it looks like a file path, use it directly
    if name.endswith((".yaml", ".yml")) or "/" in name or "\\" in name:
        p = Path(name)
        if not p.exists():
            raise FileNotFoundError(f"Benchmark config not found: {name}")
        return p

    # Otherwise, look up built-in configs
    configs_pkg = "spectra.benchmarks.configs"
    try:
        ref = importlib.resources.files(configs_pkg) / f"{name}.yaml"
        with importlib.resources.as_file(ref) as path:
            if path.exists():
                return path
    except (ModuleNotFoundError, FileNotFoundError):
        pass

    raise FileNotFoundError(
        f"Benchmark '{name}' not found. Provide a file path or use a built-in name."
    )


def _build_narrowband(
    config: Dict[str, Any], split: str
) -> NarrowbandDataset:
    pool = _build_waveform_pool(config["waveform_pool"])
    impairments = _build_impairments(config.get("impairments", []))

    # AWGN with snr_range if not already in impairments list
    snr_range = tuple(config["snr_range"])
    has_awgn = impairments is not None and any(
        isinstance(t, AWGN) for t in (impairments.transforms if impairments else [])
    )
    if has_awgn:
        # Replace AWGN with one using the config's snr_range
        new_transforms = []
        for t in impairments.transforms:
            if isinstance(t, AWGN):
                new_transforms.append(AWGN(snr_range=snr_range))
            else:
                new_transforms.append(t)
        impairments = Compose(new_transforms)
    else:
        # Add AWGN with the config's snr_range
        existing = impairments.transforms if impairments else []
        impairments = Compose([*existing, AWGN(snr_range=snr_range)])

    return NarrowbandDataset(
        waveform_pool=pool,
        num_samples=config["num_samples"][split],
        num_iq_samples=config["num_iq_samples"],
        sample_rate=config["sample_rate"],
        impairments=impairments,
        seed=config["seed"][split],
    )


def _build_wideband(
    config: Dict[str, Any], split: str
) -> WidebandDataset:
    pool = _build_waveform_pool(config["waveform_pool"])
    impairments = _build_impairments(config.get("impairments", []))

    scene_cfg = config.get("scene", {})
    num_signals = scene_cfg.get("num_signals", (1, 3))
    if isinstance(num_signals, list):
        num_signals = tuple(num_signals)

    sc = SceneConfig(
        capture_duration=scene_cfg.get(
            "capture_duration",
            config["num_iq_samples"] / config["sample_rate"],
        ),
        capture_bandwidth=scene_cfg.get(
            "capture_bandwidth", config["sample_rate"] / 2
        ),
        sample_rate=config["sample_rate"],
        num_signals=num_signals,
        signal_pool=pool,
        snr_range=tuple(config["snr_range"]),
        allow_overlap=scene_cfg.get("allow_overlap", True),
    )

    return WidebandDataset(
        scene_config=sc,
        num_samples=config["num_samples"][split],
        impairments=impairments,
        seed=config["seed"][split],
    )


def load_benchmark(
    name: str, split: str = "train"
) -> Union[
    NarrowbandDataset,
    WidebandDataset,
    Tuple[NarrowbandDataset, NarrowbandDataset, NarrowbandDataset],
    Tuple[WidebandDataset, WidebandDataset, WidebandDataset],
]:
    """Load a benchmark dataset from a YAML config.

    Parameters
    ----------
    name : str
        Built-in benchmark name (e.g., ``"spectra-18"``) or path to a
        ``.yaml`` file.
    split : str
        ``"train"``, ``"val"``, ``"test"``, or ``"all"`` (returns a 3-tuple).

    Returns
    -------
    Dataset or tuple of Datasets
        Configured dataset(s) ready for use with PyTorch DataLoader.
    """
    valid_splits = {"train", "val", "test", "all"}
    if split not in valid_splits:
        raise ValueError(f"split must be one of {valid_splits}, got '{split}'")

    path = _resolve_config_path(name)
    with open(path, "r") as f:
        config = yaml.safe_load(f)

    task = config.get("task", "narrowband")
    builder = _build_narrowband if task == "narrowband" else _build_wideband

    if split == "all":
        return builder(config, "train"), builder(config, "val"), builder(config, "test")
    return builder(config, split)
```

**Step 4: Run tests to verify they pass**

Run: `pytest tests/test_benchmarks.py -v`
Expected: All 7 tests PASS

**Step 5: Commit**

```bash
git add python/spectra/benchmarks/__init__.py python/spectra/benchmarks/loader.py tests/test_benchmarks.py
git commit -m "feat: add benchmark config loader (YAML -> Dataset)"
```

---

### Task 7: Register benchmarks in public API

**Files:**
- Modify: `python/spectra/__init__.py`

**Step 1: Add import and export**

Add to `python/spectra/__init__.py`:

Import line:
```python
from spectra.benchmarks import load_benchmark
```

Add `"load_benchmark"` to `__all__`.

**Step 2: Run smoke test**

Run: `python -c "from spectra import load_benchmark; print(load_benchmark)"`
Expected: `<function load_benchmark at 0x...>`

**Step 3: Run full test suite**

Run: `pytest tests/ -v`
Expected: All tests PASS

**Step 4: Commit**

```bash
git add python/spectra/__init__.py
git commit -m "feat: export load_benchmark from spectra public API"
```

---

### Task 8: Ship spectra-18 narrowband benchmark config

**Files:**
- Create: `python/spectra/benchmarks/configs/__init__.py` (empty, makes it a package for importlib.resources)
- Create: `python/spectra/benchmarks/configs/spectra-18.yaml`
- Create: `tests/test_benchmark_spectra18.py`

**Context:** This is the flagship benchmark — an 18-class narrowband AMC dataset. It requires the feature expansion waveforms (AM, Noise, BarkerCode) and impairments (PhaseOffset) from the earlier plan to be implemented first.

**Step 1: Create the empty `__init__.py`**

```python
# python/spectra/benchmarks/configs/__init__.py
```

(Empty file — just makes the directory a Python package so `importlib.resources.files()` can find the YAML files.)

**Step 2: Write the benchmark config**

```yaml
# python/spectra/benchmarks/configs/spectra-18.yaml
name: "spectra-18"
version: "1.0"
description: >
  18-class narrowband Automatic Modulation Classification benchmark.
  Covers PSK, QAM, FSK, radar, and analog modulations at SNR range [-10, 30] dB
  with frequency offset and phase offset impairments.
task: "narrowband"

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
  - {type: "PSK8"}
  - {type: "QAM16"}
  - {type: "QAM64"}
  - {type: "QAM256"}
  - {type: "FSK", params: {order: 2}}
  - {type: "FSK", params: {order: 4}}
  - {type: "MSK"}
  - {type: "GMSK"}
  - {type: "OFDM"}
  - {type: "LFM"}
  - {type: "CostasCode"}
  - {type: "FrankCode"}
  - {type: "P1Code"}
  - {type: "AM"}
  - {type: "BarkerCode", params: {length: 13}}
  - {type: "Noise"}

snr_range: [-10, 30]

impairments:
  - {type: "FrequencyOffset", params: {max_offset: 50_000}}
  - {type: "PhaseOffset", params: {max_offset: 3.14159}}
  - {type: "AWGN"}
```

**Step 3: Write tests for loading the built-in benchmark**

```python
# tests/test_benchmark_spectra18.py
import torch
import pytest


class TestSpectra18Benchmark:
    def test_load_by_name(self):
        from spectra.benchmarks import load_benchmark

        ds = load_benchmark("spectra-18", split="train")
        assert len(ds) == 50_000

    def test_load_val_split(self):
        from spectra.benchmarks import load_benchmark

        ds = load_benchmark("spectra-18", split="val")
        assert len(ds) == 10_000

    def test_load_test_split(self):
        from spectra.benchmarks import load_benchmark

        ds = load_benchmark("spectra-18", split="test")
        assert len(ds) == 10_000

    def test_load_all_splits(self):
        from spectra.benchmarks import load_benchmark

        train, val, test = load_benchmark("spectra-18", split="all")
        assert len(train) == 50_000
        assert len(val) == 10_000
        assert len(test) == 10_000

    def test_sample_shape(self):
        from spectra.benchmarks import load_benchmark

        ds = load_benchmark("spectra-18", split="train")
        data, label = ds[0]
        assert data.shape == (2, 1024)
        assert data.dtype == torch.float32
        assert isinstance(label, int)
        assert 0 <= label < 18

    def test_deterministic(self):
        from spectra.benchmarks import load_benchmark

        ds1 = load_benchmark("spectra-18", split="train")
        ds2 = load_benchmark("spectra-18", split="train")
        d1, l1 = ds1[42]
        d2, l2 = ds2[42]
        torch.testing.assert_close(d1, d2)
        assert l1 == l2

    def test_splits_differ(self):
        from spectra.benchmarks import load_benchmark

        train_ds = load_benchmark("spectra-18", split="train")
        val_ds = load_benchmark("spectra-18", split="val")
        d_train, _ = train_ds[0]
        d_val, _ = val_ds[0]
        assert not torch.equal(d_train, d_val)
```

**Step 4: Run tests to verify they pass**

Run: `pytest tests/test_benchmark_spectra18.py -v`
Expected: All 7 tests PASS

**Step 5: Commit**

```bash
git add python/spectra/benchmarks/configs/__init__.py python/spectra/benchmarks/configs/spectra-18.yaml tests/test_benchmark_spectra18.py
git commit -m "feat: ship spectra-18 narrowband benchmark config"
```

---

### Task 9: Ship spectra-18-wideband benchmark config

**Files:**
- Create: `python/spectra/benchmarks/configs/spectra-18-wideband.yaml`
- Create: `tests/test_benchmark_spectra18_wideband.py`

**Step 1: Write the wideband benchmark config**

```yaml
# python/spectra/benchmarks/configs/spectra-18-wideband.yaml
name: "spectra-18-wideband"
version: "1.0"
description: >
  18-class wideband signal detection benchmark.
  Same signal classes as spectra-18, composed into multi-signal scenes
  with 1-5 signals per capture for COCO-format object detection.
task: "wideband"

sample_rate: 2_000_000
num_iq_samples: 4096
num_samples:
  train: 20_000
  val: 4_000
  test: 4_000

seed:
  train: 4000
  val: 5000
  test: 6000

waveform_pool:
  - {type: "BPSK"}
  - {type: "QPSK"}
  - {type: "PSK8"}
  - {type: "QAM16"}
  - {type: "QAM64"}
  - {type: "QAM256"}
  - {type: "FSK", params: {order: 2}}
  - {type: "FSK", params: {order: 4}}
  - {type: "MSK"}
  - {type: "GMSK"}
  - {type: "OFDM"}
  - {type: "LFM"}
  - {type: "CostasCode"}
  - {type: "FrankCode"}
  - {type: "P1Code"}
  - {type: "AM"}
  - {type: "BarkerCode", params: {length: 13}}
  - {type: "Noise"}

snr_range: [0, 25]

impairments:
  - {type: "FrequencyOffset", params: {max_offset: 25_000}}
  - {type: "PhaseOffset", params: {max_offset: 1.5708}}

scene:
  capture_bandwidth: 1_000_000
  capture_duration: 0.002
  num_signals: [1, 5]
  allow_overlap: true
```

**Step 2: Write tests**

```python
# tests/test_benchmark_spectra18_wideband.py
import torch
import pytest


class TestSpectra18WidebandBenchmark:
    def test_load_by_name(self):
        from spectra.benchmarks import load_benchmark

        ds = load_benchmark("spectra-18-wideband", split="train")
        assert len(ds) == 20_000

    def test_load_all_splits(self):
        from spectra.benchmarks import load_benchmark

        train, val, test = load_benchmark("spectra-18-wideband", split="all")
        assert len(train) == 20_000
        assert len(val) == 4_000
        assert len(test) == 4_000

    def test_sample_returns_targets(self):
        from spectra.benchmarks import load_benchmark

        ds = load_benchmark("spectra-18-wideband", split="train")
        data, targets = ds[0]
        assert isinstance(data, torch.Tensor)
        assert isinstance(targets, dict)
        assert "boxes" in targets
        assert "labels" in targets
        assert "signal_descs" in targets

    def test_deterministic(self):
        from spectra.benchmarks import load_benchmark

        ds1 = load_benchmark("spectra-18-wideband", split="train")
        ds2 = load_benchmark("spectra-18-wideband", split="train")
        d1, t1 = ds1[5]
        d2, t2 = ds2[5]
        torch.testing.assert_close(d1, d2)
```

**Step 3: Run tests to verify they pass**

Run: `pytest tests/test_benchmark_spectra18_wideband.py -v`
Expected: All 4 tests PASS

**Step 4: Commit**

```bash
git add python/spectra/benchmarks/configs/spectra-18-wideband.yaml tests/test_benchmark_spectra18_wideband.py
git commit -m "feat: ship spectra-18-wideband detection benchmark config"
```

---

### Task 10: Full integration test — streaming + curriculum + benchmark

**Files:**
- Create: `tests/test_integration_streaming.py`

**Context:** This test verifies the full stack works end-to-end: load a benchmark config, wrap it in a StreamingDataLoader with curriculum scheduling, and verify that different epochs produce different data with expected difficulty progression.

**Step 1: Write the integration tests**

```python
# tests/test_integration_streaming.py
import torch
import pytest


class TestStreamingWithCurriculum:
    def test_full_pipeline_narrowband(self):
        """End-to-end: benchmark config -> streaming loader -> curriculum."""
        from spectra.benchmarks import load_benchmark
        from spectra.curriculum import CurriculumSchedule
        from spectra.streaming import StreamingDataLoader
        from spectra.datasets import NarrowbandDataset
        from spectra.waveforms import QPSK, BPSK
        from spectra.impairments import AWGN, Compose

        schedule = CurriculumSchedule(
            snr_range={"start": (20.0, 30.0), "end": (0.0, 10.0)},
        )

        def factory(params):
            return NarrowbandDataset(
                waveform_pool=[QPSK(samples_per_symbol=4), BPSK(samples_per_symbol=4)],
                num_samples=16,
                num_iq_samples=256,
                sample_rate=1e6,
                impairments=Compose([AWGN(snr_range=params.get("snr_range", (10, 20)))]),
                seed=params["seed"],
            )

        loader = StreamingDataLoader(
            dataset_factory=factory,
            base_seed=42,
            num_epochs=10,
            curriculum=schedule,
            batch_size=8,
        )

        # Epoch 0 and epoch 9 should produce different data
        data_e0, _ = next(iter(loader.epoch(0)))
        data_e9, _ = next(iter(loader.epoch(9)))
        assert data_e0.shape == (8, 2, 256)
        assert not torch.equal(data_e0, data_e9)

    def test_streaming_without_curriculum(self):
        """Streaming with no curriculum still varies per epoch."""
        from spectra.streaming import StreamingDataLoader
        from spectra.datasets import NarrowbandDataset
        from spectra.waveforms import QPSK

        def factory(params):
            return NarrowbandDataset(
                waveform_pool=[QPSK(samples_per_symbol=4)],
                num_samples=8,
                num_iq_samples=256,
                sample_rate=1e6,
                seed=params["seed"],
            )

        loader = StreamingDataLoader(
            dataset_factory=factory,
            base_seed=42,
            num_epochs=100,
            batch_size=8,
        )

        # Collect data from 5 different epochs
        epoch_data = []
        for e in range(5):
            data, _ = next(iter(loader.epoch(e)))
            epoch_data.append(data)

        # All epochs should differ
        for i in range(4):
            assert not torch.equal(epoch_data[i], epoch_data[i + 1])

    def test_training_loop_pattern(self):
        """Verify the intended usage pattern works."""
        from spectra.streaming import StreamingDataLoader
        from spectra.curriculum import CurriculumSchedule
        from spectra.datasets import NarrowbandDataset
        from spectra.waveforms import QPSK
        from spectra.impairments import AWGN, Compose

        schedule = CurriculumSchedule(
            snr_range={"start": (20.0, 30.0), "end": (0.0, 10.0)},
        )

        def factory(params):
            snr = params.get("snr_range", (10, 20))
            return NarrowbandDataset(
                waveform_pool=[QPSK(samples_per_symbol=4)],
                num_samples=8,
                num_iq_samples=128,
                sample_rate=1e6,
                impairments=Compose([AWGN(snr_range=snr)]),
                seed=params["seed"],
            )

        loader = StreamingDataLoader(
            dataset_factory=factory,
            base_seed=42,
            num_epochs=3,
            curriculum=schedule,
            batch_size=4,
        )

        total_batches = 0
        for epoch in range(3):
            for batch_data, batch_labels in loader.epoch(epoch):
                total_batches += 1
                assert batch_data.ndim == 3

        assert total_batches == 6  # 3 epochs * 2 batches (8 samples / 4 batch_size)
```

**Step 2: Run tests to verify they pass**

Run: `pytest tests/test_integration_streaming.py -v`
Expected: All 3 tests PASS

**Step 3: Run the full test suite**

Run: `pytest tests/ -v`
Expected: ALL tests PASS

**Step 4: Commit**

```bash
git add tests/test_integration_streaming.py
git commit -m "test: add end-to-end integration tests for streaming pipeline"
```

---

## Summary

| Task | Component | New Files | Key Class/Function |
|------|-----------|-----------|--------------------|
| 1 | PyYAML dependency | — | — |
| 2 | CurriculumSchedule | `curriculum.py`, `test_curriculum.py` | `CurriculumSchedule` |
| 3 | Export curriculum | — | — |
| 4 | StreamingDataLoader | `streaming.py`, `test_streaming.py` | `StreamingDataLoader` |
| 5 | Export streaming | — | — |
| 6 | Benchmark loader | `benchmarks/loader.py`, `test_benchmarks.py` | `load_benchmark()` |
| 7 | Export benchmarks | — | — |
| 8 | spectra-18 narrowband | `configs/spectra-18.yaml`, `test_benchmark_spectra18.py` | — |
| 9 | spectra-18 wideband | `configs/spectra-18-wideband.yaml`, `test_benchmark_spectra18_wideband.py` | — |
| 10 | Integration tests | `test_integration_streaming.py` | — |

**Prerequisite:** Feature expansion plan (`2026-03-01-spectra-feature-expansion.md`) must be completed first.

**After completion, SPECTRA's public API adds:**
- `CurriculumSchedule` — difficulty progression
- `StreamingDataLoader` — epoch-unique deterministic streaming
- `load_benchmark("spectra-18")` — zero-download reproducible benchmarks
