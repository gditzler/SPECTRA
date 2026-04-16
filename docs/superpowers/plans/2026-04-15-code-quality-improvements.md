# Code Quality Improvements: Readability, Reusability, Efficiency

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Eliminate duplicated code, improve readability, and fix efficiency issues across the Python and Rust layers of SPECTRA.

**Architecture:** This plan is organized into independent refactoring tasks, each producing a self-contained commit. We extract shared utilities, deduplicate hot paths, and clean up dead code. No public API changes.

**Tech Stack:** Python (NumPy, PyTorch), Rust (PyO3, num-complex, rustfft)

---

## File Structure

### New files to create:
- `python/spectra/impairments/_param_utils.py` — shared parameter validation and resolution for impairments
- `python/spectra/datasets/_base.py` — base dataset class with shared initialization, RNG, and `__len__`
- `tests/helpers.py` — shared test utilities (`make_signal_description`, `assert_varies`)

### Files to modify:
- `python/spectra/impairments/frequency_offset.py` — use shared param resolution
- `python/spectra/impairments/phase_offset.py` — use shared param resolution
- `python/spectra/impairments/dc_offset.py` — use shared param resolution
- `python/spectra/impairments/sample_rate_offset.py` — use shared param resolution
- `python/spectra/impairments/doppler.py` — use shared param resolution
- `python/spectra/datasets/narrowband.py` — inherit from base dataset
- `python/spectra/datasets/wideband.py` — inherit from base dataset
- `python/spectra/datasets/direction_finding.py` — inherit from base dataset
- `python/spectra/datasets/snr_sweep.py` — inherit from base dataset
- `python/spectra/datasets/radar.py` — inherit from base dataset
- `python/spectra/utils/dsp.py` — remove dead code, add named constant
- `python/spectra/__init__.py` — add missing exports
- `rust/src/modulators.rs` — extract shared QAM constellation builder and normalization helper
- `tests/conftest.py` — add `signal_description` fixture
- Various test files — use shared fixtures instead of local `_make_desc()`

---

### Task 1: Extract shared parameter validation for impairments

**Files:**
- Create: `python/spectra/impairments/_param_utils.py`
- Modify: `python/spectra/impairments/frequency_offset.py`
- Modify: `python/spectra/impairments/phase_offset.py`
- Modify: `python/spectra/impairments/dc_offset.py`
- Modify: `python/spectra/impairments/sample_rate_offset.py`
- Test: `tests/test_impairments_param_utils.py`

Five impairment classes duplicate the same init pattern: validate that one of `param`/`max_param` is provided, then resolve which value to use in `__call__`. This task extracts that into two utility functions.

- [ ] **Step 1: Write the failing test for `validate_fixed_or_random` and `resolve_param`**

```python
# tests/test_impairments_param_utils.py
import pytest
import numpy as np
from spectra.impairments._param_utils import validate_fixed_or_random, resolve_param


class TestValidateFixedOrRandom:
    def test_both_none_raises(self):
        with pytest.raises(ValueError, match="Must provide either"):
            validate_fixed_or_random(None, None, "offset")

    def test_fixed_only_passes(self):
        validate_fixed_or_random(1.0, None, "offset")  # no exception

    def test_random_only_passes(self):
        validate_fixed_or_random(None, 1.0, "offset")  # no exception

    def test_both_provided_passes(self):
        # Both provided is valid — fixed is ignored when max is set
        validate_fixed_or_random(1.0, 2.0, "offset")


class TestResolveParam:
    def test_fixed_returns_value(self):
        assert resolve_param(5.0, None) == 5.0

    def test_random_returns_within_range(self):
        results = [resolve_param(None, 10.0) for _ in range(100)]
        assert all(-10.0 <= r <= 10.0 for r in results)

    def test_random_varies(self):
        results = [resolve_param(None, 10.0) for _ in range(20)]
        assert len(set(results)) > 1, "Random values should vary"
```

- [ ] **Step 2: Run test to verify it fails**

Run: `pytest tests/test_impairments_param_utils.py -v`
Expected: FAIL with "ModuleNotFoundError: No module named 'spectra.impairments._param_utils'"

- [ ] **Step 3: Write the implementation**

```python
# python/spectra/impairments/_param_utils.py
"""Shared parameter validation and resolution for impairment transforms."""

from typing import Optional

import numpy as np


def validate_fixed_or_random(
    fixed: Optional[float],
    max_val: Optional[float],
    name: str,
) -> None:
    """Validate that at least one of fixed or max_val is provided.

    Raises:
        ValueError: If both are None.
    """
    if fixed is None and max_val is None:
        raise ValueError(f"Must provide either {name} or max_{name}")


def resolve_param(fixed: Optional[float], max_val: Optional[float]) -> float:
    """Return the fixed value or sample uniformly from [-max_val, max_val]."""
    if max_val is not None:
        return float(np.random.uniform(-max_val, max_val))
    return float(fixed)
```

- [ ] **Step 4: Run test to verify it passes**

Run: `pytest tests/test_impairments_param_utils.py -v`
Expected: PASS (all 5 tests)

- [ ] **Step 5: Refactor `FrequencyOffset` to use shared utilities**

Replace `python/spectra/impairments/frequency_offset.py`:

```python
from typing import Optional, Tuple

import numpy as np

from spectra.impairments._param_utils import resolve_param, validate_fixed_or_random
from spectra.impairments.base import Transform
from spectra.scene.signal_desc import SignalDescription


class FrequencyOffset(Transform):
    def __init__(
        self,
        offset: Optional[float] = None,
        max_offset: Optional[float] = None,
    ):
        validate_fixed_or_random(offset, max_offset, "offset")
        self.offset = offset
        self.max_offset = max_offset

    def __call__(
        self, iq: np.ndarray, desc: SignalDescription, **kwargs
    ) -> Tuple[np.ndarray, SignalDescription]:
        sample_rate = kwargs.get("sample_rate")
        if sample_rate is None:
            raise ValueError("FrequencyOffset requires sample_rate kwarg")

        fo = resolve_param(self.offset, self.max_offset)

        t = np.arange(len(iq)) / sample_rate
        shift = np.exp(1j * 2.0 * np.pi * fo * t).astype(np.complex64)
        shifted_iq = iq * shift

        from dataclasses import replace

        new_desc = replace(desc, f_low=desc.f_low + fo, f_high=desc.f_high + fo)
        return shifted_iq, new_desc
```

- [ ] **Step 6: Refactor `PhaseOffset`, `DCOffset`, and `SampleRateOffset` similarly**

`python/spectra/impairments/phase_offset.py` — replace lines 15-16 with `validate_fixed_or_random(offset, max_offset, "offset")` and lines 23-26 with `theta = resolve_param(self.offset, self.max_offset)`.

`python/spectra/impairments/dc_offset.py` — replace lines 15-16 with `validate_fixed_or_random(offset, max_offset, "offset")`. Lines 23-28: `DCOffset` uses complex random values, so keep its resolve logic inline but use `validate_fixed_or_random` for init.

`python/spectra/impairments/sample_rate_offset.py` — replace lines 15-16 with `validate_fixed_or_random(ppm, max_ppm, "ppm")` and lines 23-26 with `ppm_val = resolve_param(self.ppm, self.max_ppm)`.

- [ ] **Step 7: Run existing impairment tests to verify no regressions**

Run: `pytest tests/test_impairments_*.py -v`
Expected: All PASS

- [ ] **Step 8: Commit**

```bash
git add python/spectra/impairments/_param_utils.py tests/test_impairments_param_utils.py python/spectra/impairments/frequency_offset.py python/spectra/impairments/phase_offset.py python/spectra/impairments/dc_offset.py python/spectra/impairments/sample_rate_offset.py
git commit -m "refactor(impairments): extract shared param validation and resolution"
```

---

### Task 2: Extract base dataset class to eliminate shared boilerplate

**Files:**
- Create: `python/spectra/datasets/_base.py`
- Modify: `python/spectra/datasets/narrowband.py`
- Modify: `python/spectra/datasets/wideband.py`
- Modify: `python/spectra/datasets/direction_finding.py`
- Modify: `python/spectra/datasets/snr_sweep.py`
- Modify: `python/spectra/datasets/radar.py`
- Test: `tests/test_datasets_base.py`

Five dataset classes repeat: `self.seed = seed if seed is not None else 0`, `__len__` returning `self.num_samples`, and `np.random.default_rng(seed=(self.seed, idx))` in `__getitem__`. Extract to a base class.

- [ ] **Step 1: Write the failing test for `BaseIQDataset`**

```python
# tests/test_datasets_base.py
import numpy as np
import pytest
from spectra.datasets._base import BaseIQDataset


class ConcreteDataset(BaseIQDataset):
    """Minimal concrete subclass for testing."""

    def __getitem__(self, idx):
        rng = self._make_rng(idx)
        return rng.random()


class TestBaseIQDataset:
    def test_len(self):
        ds = ConcreteDataset(num_samples=100, seed=42)
        assert len(ds) == 100

    def test_seed_none_defaults_to_zero(self):
        ds = ConcreteDataset(num_samples=10, seed=None)
        assert ds.seed == 0

    def test_deterministic_rng(self):
        ds = ConcreteDataset(num_samples=10, seed=123)
        assert ds[5] == ds[5]  # same index -> same value

    def test_different_indices_differ(self):
        ds = ConcreteDataset(num_samples=10, seed=123)
        assert ds[0] != ds[1]

    def test_different_seeds_differ(self):
        ds1 = ConcreteDataset(num_samples=10, seed=1)
        ds2 = ConcreteDataset(num_samples=10, seed=2)
        assert ds1[0] != ds2[0]
```

- [ ] **Step 2: Run test to verify it fails**

Run: `pytest tests/test_datasets_base.py -v`
Expected: FAIL with "ModuleNotFoundError"

- [ ] **Step 3: Write the implementation**

```python
# python/spectra/datasets/_base.py
"""Base class for on-the-fly IQ datasets."""

from abc import abstractmethod
from typing import Optional

import numpy as np
from torch.utils.data import Dataset


class BaseIQDataset(Dataset):
    """Base class providing shared seed management and deterministic RNG creation.

    Subclasses must implement ``__getitem__`` and should call ``self._make_rng(idx)``
    to get a deterministic ``np.random.Generator`` for each sample index.
    """

    def __init__(self, num_samples: int, seed: Optional[int] = None):
        self.num_samples = num_samples
        self.seed = seed if seed is not None else 0

    def __len__(self) -> int:
        return self.num_samples

    def _make_rng(self, idx: int) -> np.random.Generator:
        """Create a deterministic RNG from (base_seed, idx)."""
        return np.random.default_rng(seed=(self.seed, idx))

    @abstractmethod
    def __getitem__(self, idx: int):
        ...
```

- [ ] **Step 4: Run test to verify it passes**

Run: `pytest tests/test_datasets_base.py -v`
Expected: PASS (all 5 tests)

- [ ] **Step 5: Refactor `NarrowbandDataset` to use `BaseIQDataset`**

In `python/spectra/datasets/narrowband.py`, change:
- Replace `from torch.utils.data import Dataset` with `from spectra.datasets._base import BaseIQDataset`
- Change `class NarrowbandDataset(Dataset):` to `class NarrowbandDataset(BaseIQDataset):`
- In `__init__`, replace `self.num_samples = num_samples` and `self.seed = seed if seed is not None else 0` with `super().__init__(num_samples=num_samples, seed=seed)`
- Remove the `__len__` method (inherited from base)
- In `__getitem__`, replace `rng = np.random.default_rng(seed=(self.seed, idx))` with `rng = self._make_rng(idx)`

- [ ] **Step 6: Refactor `WidebandDataset`, `DirectionFindingDataset`, `SNRSweepDataset`, `RadarDataset` similarly**

Apply the same pattern to each:
- Inherit from `BaseIQDataset` instead of `Dataset`
- Call `super().__init__(num_samples=..., seed=...)` in `__init__`
- Remove `__len__` method
- Replace `np.random.default_rng(seed=(self.seed, idx))` with `self._make_rng(idx)`

- [ ] **Step 7: Run all dataset tests to verify no regressions**

Run: `pytest tests/test_datasets_*.py -v`
Expected: All PASS

- [ ] **Step 8: Commit**

```bash
git add python/spectra/datasets/_base.py tests/test_datasets_base.py python/spectra/datasets/narrowband.py python/spectra/datasets/wideband.py python/spectra/datasets/direction_finding.py python/spectra/datasets/snr_sweep.py python/spectra/datasets/radar.py
git commit -m "refactor(datasets): extract BaseIQDataset with shared seed/RNG logic"
```

---

### Task 3: Deduplicate QAM constellation builder in Rust

**Files:**
- Modify: `rust/src/modulators.rs`
- Test: existing `cargo test`

The QAM constellation-building code (grid + normalization) is copy-pasted three times: `generate_qam_symbols` (lines 117-135), `generate_qam_symbols_with_indices` (lines 303-320), and `get_qam_constellation` (lines 421-438). Extract to a shared helper.

- [ ] **Step 1: Run existing Rust tests to establish baseline**

Run: `cargo test --manifest-path rust/Cargo.toml`
Expected: All PASS

- [ ] **Step 2: Extract `build_qam_constellation` helper function**

Add this function near the top of `rust/src/modulators.rs` (after the `Xorshift64` impl):

```rust
/// Build a normalized QAM constellation for the given order.
/// Order must be a perfect square (16, 64, 256, ...).
fn build_qam_constellation(order: usize) -> Result<Vec<Complex32>, String> {
    let side = (order as f64).sqrt() as usize;
    if side * side != order {
        return Err("QAM order must be a perfect square (16, 64, 256, ...)".to_string());
    }
    let mut constellation = Vec::with_capacity(order);
    for i in 0..side {
        for j in 0..side {
            let re = 2.0 * i as f64 - (side - 1) as f64;
            let im = 2.0 * j as f64 - (side - 1) as f64;
            constellation.push(Complex32::new(re as f32, im as f32));
        }
    }
    normalize_constellation(&mut constellation);
    Ok(constellation)
}
```

- [ ] **Step 3: Extract `normalize_constellation` helper**

Add this helper just above `build_qam_constellation`:

```rust
/// Normalize a constellation to unit average power in-place.
fn normalize_constellation(constellation: &mut [Complex32]) {
    let n = constellation.len();
    if n == 0 {
        return;
    }
    let avg_power: f64 = constellation
        .iter()
        .map(|c| (c.re * c.re + c.im * c.im) as f64)
        .sum::<f64>()
        / n as f64;
    if avg_power > 0.0 {
        let scale = 1.0 / avg_power.sqrt() as f32;
        for c in constellation.iter_mut() {
            c.re *= scale;
            c.im *= scale;
        }
    }
}
```

- [ ] **Step 4: Refactor `generate_qam_symbols` to use the helper**

Replace lines 110-135 of `generate_qam_symbols` with:

```rust
    let constellation = build_qam_constellation(order)
        .map_err(pyo3::exceptions::PyValueError::new_err)?;
    let mut rng = Xorshift64::new(seed);
    let symbols = Array1::from_shape_fn(num_symbols, |_| {
        let idx = (rng.next() as usize) % order;
        constellation[idx]
    });
    Ok(symbols.into_pyarray(py))
```

- [ ] **Step 5: Refactor `generate_qam_symbols_with_indices` to use the helper**

Replace lines 297-320 of `generate_qam_symbols_with_indices` with:

```rust
    let constellation = build_qam_constellation(order)
        .map_err(pyo3::exceptions::PyValueError::new_err)?;
    let mut rng = Xorshift64::new(seed);
    // ... rest stays the same (symbol + index generation loop)
```

- [ ] **Step 6: Refactor `get_qam_constellation` to use the helper**

Replace lines 415-438 of `get_qam_constellation` with:

```rust
    let constellation = build_qam_constellation(order)
        .map_err(pyo3::exceptions::PyValueError::new_err)?;
    Ok(Array1::from_vec(constellation).into_pyarray(py))
```

- [ ] **Step 7: Also refactor ASK normalization to use `normalize_constellation`**

In `generate_ask_symbols_with_indices` (lines 350-356) and `get_ask_constellation` (lines 453-458), replace the manual normalization with:
```rust
    let mut constellation: Vec<Complex32> = levels
        .iter()
        .map(|l| Complex32::new(*l as f32, 0.0))
        .collect();
    normalize_constellation(&mut constellation);
```

- [ ] **Step 8: Run Rust tests and clippy**

Run: `cargo test --manifest-path rust/Cargo.toml && cargo clippy --manifest-path rust/Cargo.toml --all-targets -- -D warnings`
Expected: All PASS, no warnings

- [ ] **Step 9: Rebuild and run Python tests**

Run: `maturin develop --release && pytest tests/test_rust_modulators.py -v`
Expected: All PASS

- [ ] **Step 10: Commit**

```bash
git add rust/src/modulators.rs
git commit -m "refactor(rust): extract shared QAM constellation builder and normalization"
```

---

### Task 4: Remove dead code and add named constant in `dsp.py`

**Files:**
- Modify: `python/spectra/utils/dsp.py`
- Test: existing tests + manual verification

- [ ] **Step 1: Remove the dead `for part_fn in [np.real, np.imag]: pass` loop**

In `python/spectra/utils/dsp.py`, delete lines 144-145:
```python
        for part_fn in [np.real, np.imag]:
            pass
```

And remove the comment on line 146 (`# Simpler approach: shape in frequency domain`) — the remaining code is the only approach, not an alternative.

- [ ] **Step 2: Add named constant for the filter tap multiplier**

At the top of `dsp.py` (after imports), add:
```python
_RESAMPLE_TAPS_MULTIPLIER = 64  # taps-per-rate for polyphase anti-aliasing filter
```

Then in `multistage_resampler` (line 104), change:
```python
    taps = low_pass(64 * max(up, down) + 1, cutoff)
```
to:
```python
    taps = low_pass(_RESAMPLE_TAPS_MULTIPLIER * max(up, down) + 1, cutoff)
```

- [ ] **Step 3: Run tests**

Run: `pytest tests/ -v -k "dsp or resample or noise"` (if any exist), otherwise `pytest tests/ -v`
Expected: All PASS

- [ ] **Step 4: Commit**

```bash
git add python/spectra/utils/dsp.py
git commit -m "fix(dsp): remove dead code loop, add named constant for filter tap count"
```

---

### Task 5: Add missing exports to `__init__.py`

**Files:**
- Modify: `python/spectra/__init__.py`

`DopplerShift` and `RadarClutter` are exported from `spectra.impairments` but not from the top-level `spectra` package.

- [ ] **Step 1: Read current `__init__.py` imports to find insertion point**

Run: Read `python/spectra/__init__.py` and find the impairment import block.

- [ ] **Step 2: Add missing imports**

Add `DopplerShift` and `RadarClutter` to the impairments import block and to `__all__`.

- [ ] **Step 3: Write a quick smoke test**

```python
# In a temporary test or added to an existing test file
def test_top_level_exports():
    from spectra import DopplerShift, RadarClutter
    assert DopplerShift is not None
    assert RadarClutter is not None
```

- [ ] **Step 4: Run tests**

Run: `pytest tests/ -v -k "export or import" --co` to check, then `python -c "from spectra import DopplerShift, RadarClutter; print('OK')"`
Expected: "OK"

- [ ] **Step 5: Commit**

```bash
git add python/spectra/__init__.py
git commit -m "fix(exports): add DopplerShift and RadarClutter to top-level __init__"
```

---

### Task 6: Consolidate test fixtures — shared `SignalDescription` factory

**Files:**
- Create: `tests/helpers.py`
- Modify: `tests/conftest.py`
- Modify: `tests/test_impairments_phase_offset.py`
- Modify: `tests/test_impairments_dc_offset.py`
- Modify: `tests/test_impairments_fading.py`
- Modify: `tests/test_impairments_iq_imbalance.py`
- Modify: `tests/test_impairments_doppler.py`

Six+ test files each define their own `_make_desc()` that creates `SignalDescription(0.0, 0.001, -5e3, 5e3, "QPSK", 20.0)`.

- [ ] **Step 1: Create shared `make_signal_description` in `tests/helpers.py`**

```python
# tests/helpers.py
"""Shared test utilities for SPECTRA tests."""

from spectra.scene.signal_desc import SignalDescription


def make_signal_description(
    t_start: float = 0.0,
    t_stop: float = 0.001,
    f_low: float = -5e3,
    f_high: float = 5e3,
    label: str = "QPSK",
    snr: float = 20.0,
    **kwargs,
) -> SignalDescription:
    """Create a SignalDescription with test defaults."""
    return SignalDescription(
        t_start=t_start,
        t_stop=t_stop,
        f_low=f_low,
        f_high=f_high,
        label=label,
        snr=snr,
        **kwargs,
    )
```

- [ ] **Step 2: Add `signal_description` fixture to `conftest.py`**

Append to `tests/conftest.py`:

```python
from tests.helpers import make_signal_description

@pytest.fixture
def signal_description():
    """Default SignalDescription for impairment tests."""
    return make_signal_description()
```

- [ ] **Step 3: Update test files to use shared factory**

In each of the affected test files, replace:
```python
def _make_desc():
    return SignalDescription(0.0, 0.001, -5e3, 5e3, "QPSK", 20.0)
```
with:
```python
from tests.helpers import make_signal_description
```
And replace `_make_desc()` calls with `make_signal_description()`.

For `test_impairments_doppler.py` which needs a custom `f_center`, use:
```python
make_signal_description(f_low=f_center - 5e3, f_high=f_center + 5e3)
```

- [ ] **Step 4: Run all impairment tests**

Run: `pytest tests/test_impairments_*.py -v`
Expected: All PASS

- [ ] **Step 5: Commit**

```bash
git add tests/helpers.py tests/conftest.py tests/test_impairments_*.py
git commit -m "refactor(tests): extract shared make_signal_description factory"
```

---

### Task 7: Extract shared window+FFT helper in Rust

**Files:**
- Modify: `rust/src/cyclo_spectral.rs`
- Modify: `rust/src/reassigned_gabor.rs`
- Modify: `rust/src/cwd.rs`

The pattern of applying a window to a slice of Complex32 samples then running FFT is repeated across three modules.

- [ ] **Step 1: Run existing Rust tests to establish baseline**

Run: `cargo test --manifest-path rust/Cargo.toml`
Expected: All PASS

- [ ] **Step 2: Create `apply_window` helper in `cyclo_spectral.rs` (as `pub(crate)`)**

Add near the top of `cyclo_spectral.rs`:

```rust
/// Apply a real-valued window to complex samples element-wise.
pub(crate) fn apply_window(samples: &[Complex32], window: &[f32]) -> Vec<Complex32> {
    samples
        .iter()
        .zip(window.iter())
        .map(|(&s, &w)| Complex32::new(s.re * w, s.im * w))
        .collect()
}
```

- [ ] **Step 3: Replace inline windowing in `cyclo_spectral.rs`**

Find all occurrences of the `.iter().zip(window.iter()).map(|(&s, &w)| Complex32::new(s.re * w, s.im * w)).collect()` pattern in `cyclo_spectral.rs` and replace with `apply_window(&samples[start..end], &window)`.

- [ ] **Step 4: Use `apply_window` in `reassigned_gabor.rs` and `cwd.rs`**

Add `use crate::cyclo_spectral::apply_window;` to both files and replace inline windowing.

- [ ] **Step 5: Run Rust tests and clippy**

Run: `cargo test --manifest-path rust/Cargo.toml && cargo clippy --manifest-path rust/Cargo.toml --all-targets -- -D warnings`
Expected: All PASS

- [ ] **Step 6: Rebuild and run Python CSP tests**

Run: `maturin develop --release && pytest tests/ -m csp -v`
Expected: All PASS

- [ ] **Step 7: Commit**

```bash
git add rust/src/cyclo_spectral.rs rust/src/reassigned_gabor.rs rust/src/cwd.rs
git commit -m "refactor(rust): extract shared apply_window helper for windowed FFT"
```

---

### Task 8: Eliminate redundant `to_vec()` copies in Rust CSP functions

**Files:**
- Modify: `rust/src/cyclo_spectral.rs`

Several functions copy the input array to a Vec when they only need read access.

- [ ] **Step 1: Audit `to_vec()` usage in `cyclo_spectral.rs`**

Read the file and identify every `iq.as_array().to_vec()` call. For each, determine whether `as_slice()` would work instead (i.e., the data is only read, not mutated).

- [ ] **Step 2: Replace `to_vec()` with `as_slice()` where possible**

For read-only access patterns, change:
```rust
let samples: Vec<Complex32> = iq.as_array().to_vec();
```
to:
```rust
let samples = iq.as_slice()?;
```

Note: If the function needs `&[Complex32]` and the data is contiguous (which PyO3 arrays are), `as_slice()` avoids the allocation.

- [ ] **Step 3: Run Rust tests**

Run: `cargo test --manifest-path rust/Cargo.toml`
Expected: All PASS

- [ ] **Step 4: Rebuild and run Python CSP tests**

Run: `maturin develop --release && pytest tests/ -m csp -v`
Expected: All PASS

- [ ] **Step 5: Commit**

```bash
git add rust/src/cyclo_spectral.rs
git commit -m "perf(rust): avoid unnecessary to_vec() copies in CSP functions"
```

---

### Task 9: Reduce dtype promotion overhead in impairments

**Files:**
- Modify: `python/spectra/impairments/frequency_offset.py`
- Modify: `python/spectra/impairments/doppler.py`
- Modify: `python/spectra/impairments/phase_offset.py`

When `iq` is `complex64`, multiplying by `np.exp(1j * phase)` (which is `complex128`) promotes to `complex128`, then `.astype(np.complex64)` converts back. We can avoid the promotion by computing the modulator in `complex64`.

- [ ] **Step 1: Run existing tests to establish baseline**

Run: `pytest tests/test_impairments_frequency_offset.py tests/test_impairments_doppler.py tests/test_impairments_phase_offset.py -v`
Expected: All PASS

- [ ] **Step 2: Fix `FrequencyOffset`**

In `frequency_offset.py`, change line 33:
```python
        shift = np.exp(1j * 2.0 * np.pi * fo * t).astype(np.complex64)
        shifted_iq = iq * shift
```
to:
```python
        phase = (2.0 * np.pi * fo * t).astype(np.float32)
        shifted_iq = (iq * np.exp(1j * phase)).astype(np.complex64)
```

This computes the phase in float32, so `np.exp(1j * phase)` produces complex64 directly.

- [ ] **Step 3: Fix `DopplerShift`**

In `doppler.py`, line 101:
```python
        out = (iq * np.exp(1j * phase).astype(np.complex64)).astype(np.complex64)
```
Change to:
```python
        phase = phase.astype(np.float32)
        out = (iq * np.exp(1j * phase)).astype(np.complex64)
```

- [ ] **Step 4: Fix `PhaseOffset`**

In `phase_offset.py`, line 28:
```python
        rotated = (iq * np.exp(1j * theta)).astype(np.complex64)
```
Change to:
```python
        rotated = (iq * np.complex64(np.exp(1j * theta))).astype(np.complex64)
```

- [ ] **Step 5: Run tests**

Run: `pytest tests/test_impairments_frequency_offset.py tests/test_impairments_doppler.py tests/test_impairments_phase_offset.py -v`
Expected: All PASS

- [ ] **Step 6: Commit**

```bash
git add python/spectra/impairments/frequency_offset.py python/spectra/impairments/doppler.py python/spectra/impairments/phase_offset.py
git commit -m "perf(impairments): reduce complex128 promotion in phase shift operations"
```

---

## Self-Review Checklist

1. **Spec coverage:** All three areas covered:
   - Readability: Tasks 4 (dead code), 5 (missing exports), 6 (test fixtures)
   - Reusability: Tasks 1 (impairment params), 2 (dataset base), 3 (QAM builder), 6 (test factories), 7 (window+FFT)
   - Efficiency: Tasks 8 (to_vec elimination), 9 (dtype promotion)

2. **Placeholder scan:** No TBD/TODO/placeholders found. All code blocks are complete.

3. **Type consistency:** `validate_fixed_or_random` and `resolve_param` names are used consistently across Tasks 1 and 9. `BaseIQDataset` and `_make_rng` are used consistently in Task 2. `build_qam_constellation` and `normalize_constellation` are consistent in Task 3.
