# ty Type-Error Cleanup Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Drive `uv tool run ty check` from 232 → 0 diagnostics by fixing real annotation gaps and refactoring patterns ty cannot infer.

**Architecture:** A `[tool.ty]` config in `pyproject.toml` already silences `unresolved-import` (Rust extension + optional deps). Remaining errors fall into seven categories with distinct fix patterns. We work easiest-first so each phase compounds confidence and reduces noise for the next phase.

**Tech Stack:** Python 3.10+, ty type checker (via uv tool), numpy, dataclasses, pytest.

**Conventions:**
- Run `uv tool run ty check 2>&1 | tail -3` after each task to see the new total — it must monotonically decrease.
- After each task, commit before moving on. Plan assumes `git status` is clean at start of each task.
- Never disable a rule globally to silence errors; use targeted `# type: ignore[CODE]` only when ty's inference is genuinely wrong.
- Where ty conflates a parameter and a re-binding (e.g. `y_true = np.asarray(y_true, ...)`), introduce a new local variable rather than fight ty.

---

## File Structure

No new modules. All edits are surgical changes to existing files:

| File | Phase | Type of change |
|---|---|---|
| `python/spectra/transforms/augmentations.py` | 1 | `Generator = None` → `Generator \| None = None` |
| `python/spectra/datasets/radar_pipeline.py` | 1 | Optional defaults |
| `python/spectra/utils/file_handlers/zarr_handler.py` | 1 | Optional defaults |
| `python/spectra/algorithms/radar.py` | 1 | One-off Optional default |
| `python/spectra/utils/dsp.py` | 1 | One-off Optional default |
| `python/spectra/utils/writer.py` | 1 | One-off Optional default |
| `python/spectra/metrics.py` | 2 | Rename re-bound parameters |
| `tests/test_csp_dataset.py` | 3 | Type fixture dicts as `dict[str, Any]` |
| `tests/test_radar_pipeline.py` | 3 | Type fixture dicts |
| `tests/test_direction_finding_dataset.py` | 3 | Type fixture dicts |
| `tests/test_wideband_df.py` | 3 | Type fixture dicts |
| `tests/test_environment_integration.py` | 3 | Type fixture dicts |
| `tests/test_radar_dataset.py` | 3 | Type fixture dicts |
| `tests/test_class_balancing.py` | 3 | Type fixture dicts |
| `python/spectra/impairments/clutter.py` | 4 | Type classmethod `defaults` dicts |
| `python/spectra/transforms/target_transforms.py` | 5 | Tighten `__call__` overrides via generics |
| `python/spectra/datasets/mixing.py` | 5 | `__getitem__` override |
| `python/spectra/datasets/_base.py` | 5 | Base `__getitem__` widening |
| `python/spectra/datasets/cyclo.py` | 5 | `__getitem__` override |
| `python/spectra/datasets/df_snr_sweep.py` | 5 | `__getitem__` override |
| `python/spectra/datasets/folder.py` | 5 | `__getitem__` override |
| `python/spectra/datasets/manifest.py` | 5 | `__getitem__` override |
| `python/spectra/datasets/radar_pipeline.py` | 5 | `__getitem__` override |
| `python/spectra/datasets/metadata.py` | 6 | Real bug — `build_dataset` calls wrong API |
| `tests/test_benchmark_spectra18_wideband.py` | 6 | ty mis-infers `load_benchmark` return type |
| `python/spectra/benchmarks/loader.py` | 6 | Tighten `load_benchmark` return annotation |
| `python/spectra/scene/labels.py` | 7 | `invalid-return-type` |
| `python/spectra/utils/file_handlers/raw_reader.py` | 7 | `iinfo` overload |
| `python/spectra/utils/file_handlers/raw_writer.py` | 7 | `iinfo` overload |
| `python/spectra/waveforms/multifunction/schedule.py` | 7 | Operator/callable issues |
| `python/spectra/datasets/wideband.py` | 7 | Misc unresolved attribute |
| `python/spectra/impairments/tdl_channel.py` | 7 | Misc invalid-assignment |
| `python/spectra/algorithms/radar.py` | 7 | Misc unsupported `<=` |
| `python/spectra/impairments/awgn.py` | 7 | Misc unsupported `<` |
| `python/spectra/datasets/direction_finding.py` | 7 | Misc unsupported `<` |
| `python/spectra/link/simulator.py` | 7 | Misc unresolved attribute |
| `tests/test_benchmark_spectra_df.py` | 7 | Stale attribute access |
| `tests/test_benchmark_spectra18.py` | 7 | Stale attribute access |
| `tests/test_benchmark_spectra40.py` | 7 | Stale attribute access |

---

## Phase 1 — `Optional[X] = None` parameter defaults (17 errors)

ty rejects `def f(x: T = None)` because `None` is not assignable to `T`. Fix by widening the annotation to `T | None`.

### Task 1.1: Fix `augmentations.py` — 10 sites of `rng: np.random.Generator = None`

**Files:**
- Modify: `python/spectra/transforms/augmentations.py:10,24,31,41,56,82,96,109,127,145`

- [ ] **Step 1: Read the file**

Run: `cat python/spectra/transforms/augmentations.py | head -160`

- [ ] **Step 2: Apply the same edit at every flagged line**

The pattern at each site is:
```python
def __call__(self, iq: np.ndarray, rng: np.random.Generator = None) -> np.ndarray:
```

Change `np.random.Generator = None` → `np.random.Generator | None = None` (use a single `replace_all` Edit since the substring is unique within the file).

**Edit:**
- old_string: `rng: np.random.Generator = None`
- new_string: `rng: np.random.Generator | None = None`
- replace_all: true

- [ ] **Step 3: Verify**

Run: `uv tool run ty check python/spectra/transforms/augmentations.py 2>&1 | tail -3`
Expected: `All checks passed!`

- [ ] **Step 4: Run pytest**

Run: `pytest tests/test_augmentations.py -q` (skip if file does not exist)
Expected: 0 regressions vs. baseline.

- [ ] **Step 5: Commit**

```bash
git add python/spectra/transforms/augmentations.py
git commit -m "fix(types): widen rng Generator defaults to Generator | None in augmentations"
```

---

### Task 1.2: Fix `radar_pipeline.py` — 2 sites

**Files:**
- Modify: `python/spectra/datasets/radar_pipeline.py` (2 sites flagged by ty)

- [ ] **Step 1: List the offending lines**

Run: `uv tool run ty check 2>&1 | grep -A 2 "radar_pipeline.py" | grep "invalid-parameter-default" -A 1`

- [ ] **Step 2: For each flagged line, widen the annotation**

For each parameter where ty reports `Default value of type 'None' is not assignable to annotated parameter type 'X'`:
- Open the file and the line
- Add `| None` to the annotation if missing, or change `Optional[X]` if already imported

If the existing annotation uses `Optional`, ensure `from typing import Optional` is imported. Prefer the `X | None` form (Python 3.10+).

- [ ] **Step 3: Verify and commit**

```bash
uv tool run ty check python/spectra/datasets/radar_pipeline.py 2>&1 | grep "invalid-parameter-default" || echo "clean"
git add python/spectra/datasets/radar_pipeline.py
git commit -m "fix(types): widen None-defaulted parameters in radar_pipeline"
```

---

### Task 1.3: Fix `zarr_handler.py` — 2 sites

**Files:**
- Modify: `python/spectra/utils/file_handlers/zarr_handler.py`

- [ ] **Step 1: Run ty against the file to find lines**

Run: `uv tool run ty check python/spectra/utils/file_handlers/zarr_handler.py 2>&1 | grep "invalid-parameter-default" -A 5`

- [ ] **Step 2: Apply the same `T | None = None` widening for each flagged param**

- [ ] **Step 3: Verify and commit**

```bash
git add python/spectra/utils/file_handlers/zarr_handler.py
git commit -m "fix(types): widen None-defaulted parameters in zarr_handler"
```

---

### Task 1.4: Fix the three one-off Optional defaults

**Files:**
- Modify: `python/spectra/algorithms/radar.py` (1 site)
- Modify: `python/spectra/utils/dsp.py` (1 site)
- Modify: `python/spectra/utils/writer.py` (1 site)

- [ ] **Step 1: Find the lines**

Run: `uv tool run ty check 2>&1 | grep -B 0 -A 2 "invalid-parameter-default" | grep -E "(algorithms/radar|utils/dsp|utils/writer)\.py"`

- [ ] **Step 2: Apply `T | None = None` widening at each site**

- [ ] **Step 3: Verify**

Run: `uv tool run ty check 2>&1 | grep "invalid-parameter-default" | wc -l`
Expected: `0`

- [ ] **Step 4: Commit**

```bash
git add python/spectra/algorithms/radar.py python/spectra/utils/dsp.py python/spectra/utils/writer.py
git commit -m "fix(types): widen None-defaulted parameters in radar/dsp/writer"
```

---

## Phase 2 — `metrics.py` ndarray re-binding (12 errors)

The functions in `metrics.py` declare `y_true: Sequence[int]` then immediately do `y_true = np.asarray(y_true, dtype=int)`, rebinding the parameter to `np.ndarray`. ty rejects this. Fix by introducing a new local name.

### Task 2.1: Rename rebound parameters in `metrics.py`

**Files:**
- Modify: `python/spectra/metrics.py`

- [ ] **Step 1: Read the file**

Run: `cat python/spectra/metrics.py`

- [ ] **Step 2: Replace each `np.asarray` rebind with a fresh local**

For each function with the pattern:
```python
def f(y_true: Sequence[int], y_pred: Sequence[int], ...) -> ...:
    y_true = np.asarray(y_true, dtype=int)
    y_pred = np.asarray(y_pred, dtype=int)
    ...
```

Rewrite as:
```python
def f(y_true: Sequence[int], y_pred: Sequence[int], ...) -> ...:
    y_true_arr = np.asarray(y_true, dtype=int)
    y_pred_arr = np.asarray(y_pred, dtype=int)
    ...
```

…and update every subsequent reference inside that function from `y_true` → `y_true_arr` and `y_pred` → `y_pred_arr`. Apply the same pattern to `snr_values` → `snr_values_arr` in `per_snr_accuracy` and `per_snr_rmse`, and `true_angles`/`estimated_angles` in `per_snr_rmse`.

Affected functions (by line, per ty output):
- `confusion_matrix` (lines 22-23)
- `accuracy` (lines 40-41)
- `classification_report` (lines 62-63)
- `per_snr_accuracy` (lines 109-111)
- `per_snr_rmse` (similar pattern below line 111)

The `bool.sum()` errors (`mask.sum()` on line 116 of `per_snr_accuracy`) resolve once `snr_values` is an ndarray throughout: `mask = snr_values_arr == snr` will be `np.ndarray[bool]`, which has `.sum()`.

- [ ] **Step 3: Run pytest**

Run: `pytest tests/test_metrics.py -q`
Expected: all pass.

- [ ] **Step 4: Verify ty**

Run: `uv tool run ty check python/spectra/metrics.py 2>&1 | tail -3`
Expected: `All checks passed!`

- [ ] **Step 5: Commit**

```bash
git add python/spectra/metrics.py
git commit -m "fix(types): introduce ndarray locals to avoid parameter rebinding in metrics"
```

---

## Phase 3 — Test fixture `**defaults` patterns (~93 errors)

Tests build a heterogeneous dict and unpack to a dataset constructor:
```python
defaults = {"array": arr, "signal_pool": [...], "num_snapshots": 64, ...}
return Dataset(**defaults)
```
ty infers `dict[str, Union[Array, list, int, ...]]` and reports a type error per kwarg. The fix is one-line: type the dict as `dict[str, Any]` so ty stops trying to validate per-key.

### Task 3.1: Type fixture dicts in `tests/test_csp_dataset.py` (26 errors)

**Files:**
- Modify: `tests/test_csp_dataset.py`

- [ ] **Step 1: Read the file**

Run: `cat tests/test_csp_dataset.py`

- [ ] **Step 2: Add `from typing import Any` if not already imported**

- [ ] **Step 3: Find every `defaults = {...}` (or similarly named dict) that is unpacked into a dataset constructor**

For each, change:
```python
defaults = {
    ...
}
```
to:
```python
defaults: dict[str, Any] = {
    ...
}
```

- [ ] **Step 4: Verify**

Run: `uv tool run ty check tests/test_csp_dataset.py 2>&1 | tail -3`
Expected: `All checks passed!`

- [ ] **Step 5: Run the tests**

Run: `pytest tests/test_csp_dataset.py -q`
Expected: same pass/fail set as before this task.

- [ ] **Step 6: Commit**

```bash
git add tests/test_csp_dataset.py
git commit -m "test(types): annotate fixture dicts as dict[str, Any] in test_csp_dataset"
```

---

### Task 3.2: Type fixture dicts in `tests/test_radar_pipeline.py` (17 errors)

**Files:**
- Modify: `tests/test_radar_pipeline.py`

- [ ] **Step 1: Apply the same pattern as Task 3.1**

Add `from typing import Any` if needed; annotate `defaults` (or whatever the fixture dict is named) as `dict[str, Any]`.

- [ ] **Step 2: Verify and run tests**

```bash
uv tool run ty check tests/test_radar_pipeline.py 2>&1 | tail -3
pytest tests/test_radar_pipeline.py -q
```

- [ ] **Step 3: Commit**

```bash
git add tests/test_radar_pipeline.py
git commit -m "test(types): annotate fixture dicts as dict[str, Any] in test_radar_pipeline"
```

---

### Task 3.3: Type fixture dicts in `tests/test_direction_finding_dataset.py` (14 errors)

**Files:**
- Modify: `tests/test_direction_finding_dataset.py`

- [ ] **Step 1: Apply Task 3.1 pattern**
- [ ] **Step 2: Verify and run tests**

```bash
uv tool run ty check tests/test_direction_finding_dataset.py 2>&1 | tail -3
pytest tests/test_direction_finding_dataset.py -q
```

- [ ] **Step 3: Commit**

```bash
git add tests/test_direction_finding_dataset.py
git commit -m "test(types): annotate fixture dicts in test_direction_finding_dataset"
```

---

### Task 3.4: Type fixture dicts in `tests/test_wideband_df.py` (14 errors)

**Files:**
- Modify: `tests/test_wideband_df.py`

- [ ] **Step 1: Apply Task 3.1 pattern**
- [ ] **Step 2: Verify and commit**

```bash
uv tool run ty check tests/test_wideband_df.py 2>&1 | tail -3
pytest tests/test_wideband_df.py -q
git add tests/test_wideband_df.py
git commit -m "test(types): annotate fixture dicts in test_wideband_df"
```

---

### Task 3.5: Type fixture dicts in `tests/test_environment_integration.py` (9 errors)

**Files:**
- Modify: `tests/test_environment_integration.py`

- [ ] **Step 1: Apply Task 3.1 pattern**
- [ ] **Step 2: Verify and commit**

```bash
uv tool run ty check tests/test_environment_integration.py 2>&1 | tail -3
pytest tests/test_environment_integration.py -q
git add tests/test_environment_integration.py
git commit -m "test(types): annotate fixture dicts in test_environment_integration"
```

---

### Task 3.6: Type fixture dicts in `tests/test_radar_dataset.py` (7 errors)

**Files:**
- Modify: `tests/test_radar_dataset.py`

- [ ] **Step 1: Apply Task 3.1 pattern**
- [ ] **Step 2: Verify and commit**

```bash
uv tool run ty check tests/test_radar_dataset.py 2>&1 | tail -3
pytest tests/test_radar_dataset.py -q
git add tests/test_radar_dataset.py
git commit -m "test(types): annotate fixture dicts in test_radar_dataset"
```

---

### Task 3.7: Type fixture dicts in `tests/test_class_balancing.py` (6 errors)

**Files:**
- Modify: `tests/test_class_balancing.py`

- [ ] **Step 1: Apply Task 3.1 pattern**
- [ ] **Step 2: Verify and commit**

```bash
uv tool run ty check tests/test_class_balancing.py 2>&1 | tail -3
pytest tests/test_class_balancing.py -q
git add tests/test_class_balancing.py
git commit -m "test(types): annotate fixture dicts in test_class_balancing"
```

---

## Phase 4 — `clutter.py` classmethod factories (18 errors)

`RadarClutter.ground()`, `.sea()`, `.weather()` all build a `defaults = dict(cnr=..., spectral_shape="gaussian", ...)` whose inferred type is `dict[str, int | float | str]` and unpack into `__init__`. Same fix as Phase 3.

### Task 4.1: Type classmethod `defaults` dicts in `clutter.py`

**Files:**
- Modify: `python/spectra/impairments/clutter.py:131-139` and the analogous `sea()`/`weather()` methods further down

- [ ] **Step 1: Read the file**

Run: `cat python/spectra/impairments/clutter.py`

- [ ] **Step 2: Add `from typing import Any` if missing**

- [ ] **Step 3: For each classmethod (`ground`, `sea`, `weather`), annotate the `defaults` local**

For `ground` (line 131):
```python
defaults: dict[str, Any] = dict(
    cnr=cnr,
    doppler_spread=spread,
    sample_rate=sample_rate,
    doppler_center=0.0,
    spectral_shape="gaussian",
)
```
Apply the same pattern to `sea()` and `weather()` (and any other `defaults = dict(...)` followed by `cls(**defaults)`).

- [ ] **Step 4: Verify**

Run: `uv tool run ty check python/spectra/impairments/clutter.py 2>&1 | tail -3`
Expected: `All checks passed!`

- [ ] **Step 5: Run tests**

Run: `pytest tests/test_clutter.py -q` (skip if no test file)

- [ ] **Step 6: Commit**

```bash
git add python/spectra/impairments/clutter.py
git commit -m "fix(types): annotate classmethod defaults dicts in RadarClutter factories"
```

---

## Phase 5 — Override signature mismatches (14 errors)

### Task 5.1: Tighten `TargetTransform.__call__` signature in `target_transforms.py`

**Files:**
- Modify: `python/spectra/transforms/target_transforms.py:66-68` (base class) and subclasses

The base declares:
```python
class TargetTransform(ABC):
    @abstractmethod
    def __call__(self, target: Any) -> Any: ...
```
…but subclasses narrow the parameter (`label: str`, `targets: List[Dict]`). Liskov says you can widen params, not narrow them — ty correctly rejects this.

- [ ] **Step 1: Read the file**

Run: `cat python/spectra/transforms/target_transforms.py`

- [ ] **Step 2: Make `TargetTransform` generic**

Replace lines 66-68 with:
```python
from typing import Generic, TypeVar

T_in = TypeVar("T_in")
T_out = TypeVar("T_out")


class TargetTransform(ABC, Generic[T_in, T_out]):
    @abstractmethod
    def __call__(self, target: T_in) -> T_out: ...
```

- [ ] **Step 3: Parameterize each subclass**

Update each subclass to bind the type parameters. Example:
```python
class ClassIndex(TargetTransform[str, int]):
    ...

class FamilyName(TargetTransform[str, str]):
    ...

class FamilyIndex(TargetTransform[str, int]):
    ...

class YOLOLabel(TargetTransform[List[Dict], List[List[float]]]):
    ...

class BoxesNormalize(TargetTransform[<read existing signature>, <existing return>]):
    ...
```
Read each subclass's `__call__` and copy its existing param/return types into the brackets.

- [ ] **Step 4: Verify**

Run: `uv tool run ty check python/spectra/transforms/target_transforms.py 2>&1 | tail -3`
Expected: no `invalid-method-override` errors remain in this file.

- [ ] **Step 5: Run tests**

Run: `pytest tests/test_target_transforms.py -q` (skip if no test file)
Expected: all pass.

- [ ] **Step 6: Commit**

```bash
git add python/spectra/transforms/target_transforms.py
git commit -m "fix(types): make TargetTransform generic over input/output types"
```

---

### Task 5.2: Widen `__getitem__` return type on dataset base class

**Files:**
- Read: `python/spectra/datasets/_base.py` first
- Modify: `python/spectra/datasets/_base.py` (1 error)
- Modify: `python/spectra/datasets/cyclo.py` (1 error)
- Modify: `python/spectra/datasets/df_snr_sweep.py` (1 error)
- Modify: `python/spectra/datasets/folder.py` (1 error)
- Modify: `python/spectra/datasets/manifest.py` (1 error)
- Modify: `python/spectra/datasets/mixing.py` (2 errors)
- Modify: `python/spectra/datasets/radar_pipeline.py` (1 error)

- [ ] **Step 1: Read the base**

Run: `cat python/spectra/datasets/_base.py`

- [ ] **Step 2: Decide the strategy**

Each dataset subclass returns a different tuple shape (`(Tensor, int)`, `(Tensor, dict)`, `(Tensor, DirectionFindingTarget)`, etc.). Two options:

**Option A (preferred):** Make `_BaseDataset` (or whatever the base is called) generic over the target type:
```python
T_target = TypeVar("T_target")

class _BaseDataset(torch.utils.data.Dataset, Generic[T_target]):
    @abstractmethod
    def __getitem__(self, idx: int) -> Tuple[torch.Tensor, T_target]: ...
```
Then each subclass binds: `class NarrowbandDataset(_BaseDataset[int])`, `class WidebandDataset(_BaseDataset[Dict])`, etc.

**Option B:** Widen the base return type to `Tuple[torch.Tensor, Any]`.

Use Option A unless the base class definition makes generics awkward (e.g. if there is no base class and each subclass directly inherits `torch.utils.data.Dataset`). In that case, fall back to Option B by explicitly annotating each subclass's `__getitem__`.

- [ ] **Step 3: Apply the chosen pattern**

For each file in the modify list, update the `__getitem__` signature so the override is compatible with the base.

- [ ] **Step 4: Verify**

Run: `uv tool run ty check 2>&1 | grep "invalid-method-override" | wc -l`
Expected: `0`

- [ ] **Step 5: Run dataset tests**

```bash
pytest tests/test_narrowband_dataset.py tests/test_wideband_dataset.py tests/test_csp_dataset.py tests/test_direction_finding_dataset.py -q
```
Expected: same pass/fail as baseline.

- [ ] **Step 6: Commit**

```bash
git add python/spectra/datasets/
git commit -m "fix(types): make dataset base class generic over target type"
```

---

## Phase 6 — Real bugs (4 errors, but worth investigation time)

### Task 6.1: Fix `metadata.py` `build_dataset()` calls with wrong API

**Files:**
- Modify: `python/spectra/datasets/metadata.py:53-61` (NarrowbandMetadata.build_dataset)
- Modify: `python/spectra/datasets/metadata.py:80-86` (WidebandMetadata.build_dataset)

ty reports:
- `NarrowbandDataset` does not accept `waveform_labels` and is missing required `waveform_pool`/`sample_rate`
- `WidebandDataset` is missing required `scene_config`

These are genuine bugs — `build_dataset()` was likely written before the dataset signatures were finalized.

- [ ] **Step 1: Inspect the current signatures**

```bash
grep -A 20 "class NarrowbandDataset" python/spectra/datasets/narrowband.py | head -25
grep -A 20 "class WidebandDataset" python/spectra/datasets/wideband.py | head -25
```

- [ ] **Step 2: Decide whether `build_dataset` is used anywhere**

Run: `grep -rn "\.build_dataset()" python/ tests/ examples/ benchmarks/`

If unused: delete both `build_dataset` methods.

If used: rewrite each method to construct the dataset correctly. For `NarrowbandMetadata`, that likely means:
```python
def build_dataset(self) -> "NarrowbandDataset":
    from spectra.datasets import NarrowbandDataset
    from spectra.waveforms import waveform_from_label

    waveform_pool = [waveform_from_label(name) for name in self.waveform_labels]
    return NarrowbandDataset(
        waveform_pool=waveform_pool,
        sample_rate=getattr(self, "sample_rate", 1e6),
        num_iq_samples=self.num_iq_samples,
        num_samples=self.num_samples,
        snr_range=self.snr_range,
        seed=self.seed,
    )
```
For `WidebandMetadata`, build a `SceneConfig` from the metadata fields and pass as `scene_config`.

If `waveform_from_label` does not exist, the cleanest fix is to delete `build_dataset` and have callers construct datasets directly.

- [ ] **Step 3: Decide and document the choice**

Add a one-line commit note explaining whether `build_dataset` was deleted or rewritten.

- [ ] **Step 4: Run metadata tests**

```bash
pytest tests/test_metadata.py -q
```
Expected: pass after fix. If a test calls `build_dataset()` and you deleted it, also delete or rewrite the test.

- [ ] **Step 5: Verify ty**

Run: `uv tool run ty check python/spectra/datasets/metadata.py 2>&1 | tail -3`
Expected: `All checks passed!`

- [ ] **Step 6: Commit**

```bash
git add python/spectra/datasets/metadata.py tests/test_metadata.py
git commit -m "fix(metadata): repair build_dataset() to match current dataset signatures"
```

---

### Task 6.2: Fix `load_benchmark` return type so `ds[5]` is not flagged as out-of-bounds

**Files:**
- Read first: `python/spectra/benchmarks/loader.py`
- Modify: `python/spectra/benchmarks/loader.py`

ty thinks `load_benchmark("spectra-18-wideband", split="train")` returns `tuple[Dataset, Dataset, Dataset]` (a 3-tuple of train/val/test), so `ds[5]` is index-out-of-bounds. But when `split` is passed, the function returns a single dataset.

- [ ] **Step 1: Read the function**

Run: `cat python/spectra/benchmarks/loader.py | head -120`

- [ ] **Step 2: Add `@overload` declarations**

Replace the bare `load_benchmark` with explicit overloads:
```python
from typing import Literal, Tuple, Union, overload

@overload
def load_benchmark(name: str, *, split: None = None) -> Tuple[Dataset, Dataset, Dataset]: ...

@overload
def load_benchmark(name: str, *, split: Literal["train", "val", "test"]) -> Dataset: ...

def load_benchmark(name: str, *, split: str | None = None) -> Union[Dataset, Tuple[Dataset, Dataset, Dataset]]:
    ...  # existing body
```

Replace `Dataset` with the actual return type in this codebase (could be `NarrowbandDataset`, `WidebandDataset`, etc., probably `torch.utils.data.Dataset` as the common base).

- [ ] **Step 3: Verify the test now passes ty**

Run: `uv tool run ty check tests/test_benchmark_spectra18_wideband.py 2>&1 | tail -3`
Expected: index-out-of-bounds and not-subscriptable errors gone.

- [ ] **Step 4: Run benchmark tests**

```bash
pytest tests/test_benchmark_spectra18_wideband.py tests/test_benchmark_spectra18.py tests/test_benchmark_spectra40.py -q
```
Expected: pass.

- [ ] **Step 5: Commit**

```bash
git add python/spectra/benchmarks/loader.py
git commit -m "fix(types): add @overload declarations to load_benchmark for split-aware return type"
```

---

## Phase 7 — Long-tail miscellaneous (~15 errors)

These are one-offs. Work through them sequentially; each is a quick fix.

### Task 7.1: Fix `scene/labels.py` invalid-return-type (2 errors)

**Files:**
- Modify: `python/spectra/scene/labels.py`

- [ ] **Step 1: Find the lines**

Run: `uv tool run ty check python/spectra/scene/labels.py 2>&1 | grep -A 5 "invalid-return-type"`

- [ ] **Step 2: Inspect each function and either narrow the return value or widen the annotation**

If the function declares `-> Foo` but a branch returns `None`, change to `-> Optional[Foo]`. If it returns multiple types, widen to a union.

- [ ] **Step 3: Verify and commit**

```bash
uv tool run ty check python/spectra/scene/labels.py 2>&1 | tail -3
git add python/spectra/scene/labels.py
git commit -m "fix(types): correct return annotations in scene/labels"
```

---

### Task 7.2: Fix `raw_reader.py` and `raw_writer.py` `np.iinfo` overload mismatch (2 errors)

**Files:**
- Modify: `python/spectra/utils/file_handlers/raw_reader.py`
- Modify: `python/spectra/utils/file_handlers/raw_writer.py`

`np.iinfo` accepts an integer dtype; ty rejects passing a non-integer (likely a `np.dtype` of unknown integer-ness, or a string).

- [ ] **Step 1: Find the call sites**

```bash
grep -n "np.iinfo" python/spectra/utils/file_handlers/raw_reader.py python/spectra/utils/file_handlers/raw_writer.py
```

- [ ] **Step 2: For each, narrow the argument**

If the call is `np.iinfo(some_dtype)` where `some_dtype` is a `np.dtype` object, change to `np.iinfo(np.dtype(some_dtype).type)` so the integer type class is passed. If `some_dtype` is a string like `"int8"`, leave it — strings are accepted; the issue may be a typing stub gap, in which case add `# type: ignore[no-matching-overload]` on the call.

- [ ] **Step 3: Run file_handler tests**

```bash
pytest tests/test_file_handlers.py -q  # adjust path if different
```

- [ ] **Step 4: Verify and commit**

```bash
uv tool run ty check python/spectra/utils/file_handlers/raw_reader.py python/spectra/utils/file_handlers/raw_writer.py 2>&1 | tail -3
git add python/spectra/utils/file_handlers/raw_reader.py python/spectra/utils/file_handlers/raw_writer.py
git commit -m "fix(types): correct iinfo argument typing in raw IQ readers/writers"
```

---

### Task 7.3: Fix `multifunction/schedule.py` operator and call-top-callable (4 errors)

**Files:**
- Modify: `python/spectra/waveforms/multifunction/schedule.py`

ty reports 2 unsupported `<` operations and 2 `call-top-callable` errors. These typically mean a comparison or call against a value of unknown type.

- [ ] **Step 1: Find the lines**

Run: `uv tool run ty check python/spectra/waveforms/multifunction/schedule.py 2>&1 | grep -A 5 -E "unsupported-operator|call-top-callable"`

- [ ] **Step 2: Tighten annotations**

For each `unsupported-operator: '<'` site: the lhs is likely a `dict.get(...)` or untyped attribute. Add a `cast(float, ...)` or assign to a typed local:
```python
weight = float(self.weights.get(key, 0.0))
if weight < threshold:
    ...
```

For each `call-top-callable`: a callable was stored without a callable type annotation. Add `Callable[..., T]` to the field. Example:
```python
self._factory: Callable[..., Waveform] = waveform_factory
```

- [ ] **Step 3: Verify and commit**

```bash
uv tool run ty check python/spectra/waveforms/multifunction/schedule.py 2>&1 | tail -3
pytest tests/test_multifunction.py -q  # adjust if different
git add python/spectra/waveforms/multifunction/schedule.py
git commit -m "fix(types): tighten annotations in multifunction schedule"
```

---

### Task 7.4: Fix `tdl_channel.py` invalid-assignment (1 error) and remaining unsupported-operator one-offs (3 errors)

**Files:**
- Modify: `python/spectra/impairments/tdl_channel.py`
- Modify: `python/spectra/algorithms/radar.py`
- Modify: `python/spectra/impairments/awgn.py`
- Modify: `python/spectra/datasets/direction_finding.py`

- [ ] **Step 1: List the offending lines**

```bash
uv tool run ty check 2>&1 | grep -B 0 -A 4 -E "(tdl_channel|algorithms/radar|impairments/awgn|datasets/direction_finding)\.py" | grep -E "(unsupported-operator|invalid-assignment)" -A 3
```

- [ ] **Step 2: For each, fix at the source**

Common patterns:
- Operator on `Optional[X]` → add a `None` check first or use `assert x is not None`.
- Operator on `int | float | str` (untyped dict value) → cast at the assignment site.
- `invalid-assignment` to a dict key with the wrong value type → annotate the dict more loosely or convert the value first.

- [ ] **Step 3: Verify and commit**

```bash
uv tool run ty check 2>&1 | tail -3
git add python/spectra/impairments/tdl_channel.py python/spectra/algorithms/radar.py python/spectra/impairments/awgn.py python/spectra/datasets/direction_finding.py
git commit -m "fix(types): resolve operator/assignment typing in impairments and dsp modules"
```

---

### Task 7.5: Fix unresolved-attribute one-offs (~7 errors)

**Files:**
- Modify: `python/spectra/datasets/wideband.py` (2 errors)
- Modify: `python/spectra/utils/file_handlers/zarr_handler.py` (2 errors — separate from Task 1.3 which was about defaults)
- Modify: `python/spectra/benchmarks/loader.py` (1 error — separate from Task 6.2)
- Modify: `python/spectra/link/simulator.py` (1 error)
- Modify: `tests/test_benchmark_spectra_df.py` (3 errors)
- Modify: `tests/test_benchmark_spectra18.py` (2 errors)
- Modify: `tests/test_benchmark_spectra40.py` (2 errors)

Most of these are `Object of type 'X' has no attribute 'Y'` — either a stale attribute reference after a refactor, or `X` is wider than expected (e.g. `None | T`).

- [ ] **Step 1: For each file, list its remaining errors**

```bash
for f in python/spectra/datasets/wideband.py python/spectra/utils/file_handlers/zarr_handler.py python/spectra/benchmarks/loader.py python/spectra/link/simulator.py tests/test_benchmark_spectra_df.py tests/test_benchmark_spectra18.py tests/test_benchmark_spectra40.py; do
  echo "=== $f ==="
  uv tool run ty check "$f" 2>&1 | grep -E "(error|warning)" | head -10
done
```

- [ ] **Step 2: For each error, decide:**
  - **Stale reference (test was not updated after a refactor):** update the test to use the current attribute name.
  - **Optional widening:** add a `None` check before access, or `assert x is not None`.
  - **Wrong inferred type:** narrow with `cast()` or `isinstance()`.

- [ ] **Step 3: Run affected tests**

```bash
pytest tests/test_benchmark_spectra_df.py tests/test_benchmark_spectra18.py tests/test_benchmark_spectra40.py -q
```

- [ ] **Step 4: Verify and commit**

```bash
uv tool run ty check 2>&1 | tail -3
git add python/spectra/datasets/wideband.py python/spectra/utils/file_handlers/zarr_handler.py python/spectra/benchmarks/loader.py python/spectra/link/simulator.py tests/test_benchmark_spectra_df.py tests/test_benchmark_spectra18.py tests/test_benchmark_spectra40.py
git commit -m "fix(types): resolve stale attribute references in datasets/tests"
```

---

## Final verification

### Task 8: Confirm ty is clean and run the full test suite

- [ ] **Step 1: Run ty**

Run: `uv tool run ty check 2>&1 | tail -3`
Expected: `All checks passed!`

- [ ] **Step 2: Run full pytest**

Run: `pytest tests/ -q`
Expected: same pass/fail count as before this plan started (regressions = bug; investigate before merging).

- [ ] **Step 3: Run ruff check too**

Run: `ruff check`
Expected: `All checks passed!`

- [ ] **Step 4: Final commit / push**

```bash
git log --oneline -20  # sanity-check commit history
git push origin <branch>
```

---

## Notes for the executor

- **Always run `uv tool run ty check 2>&1 | tail -3` after each task.** The total count must monotonically decrease. If it goes up, the most recent edit introduced a new error — investigate before committing.
- **Do not disable ty rules globally to make errors disappear.** If you cannot fix a specific site cleanly, add a narrowly scoped `# ty: ignore[<rule>]` comment on the offending line with a one-word reason (e.g. `# ty: ignore[no-matching-overload] # numpy stub gap`).
- **Skip a task gracefully if its target file no longer matches the description** — e.g. if Task 6.1 finds `build_dataset` was already deleted upstream, mark the task done and proceed.
- **Pytest regressions are blockers.** If a Phase-3 fixture-typing change makes tests fail, the change broke runtime behavior — revert and investigate. The annotation `dict[str, Any]` should not affect runtime.
