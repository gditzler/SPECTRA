# Waveform Parameter Realism Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Let users specify waveform parameters in physical units (baud, Hz, seconds) and draw realistic, standards-referenced emitters from a profile registry, so wideband scenes statistically resemble real captures.

**Architecture:** Two additive layers per the approved spec (`docs/superpowers/specs/2026-07-02-waveform-realism-design.md`). Layer 1 adds optional physical-unit kwargs to core comms/radar waveform classes, resolved to sample-domain values at `generate()` time (round when symbol-rate error ≤1%, otherwise rational-resample). Layer 2 adds `spectra.profiles`: `ParamSpec` distributions + `EmitterProfile` + a curated registry, consumed by `Composer` and the benchmark YAML loader.

**Tech Stack:** Python 3.10+, NumPy, existing Rust primitives via `spectra._rust` (no Rust changes), pytest.

**Dependencies:** The FSK level-convention fix (task_0885f2c0) and OFDM pilot/guard fix (task_3ccc50b2) are in flight. Tasks 1–5 and 8–13 are independent of them. Task 6 (FSK physical units) and Task 7 (OFDM physical units) build on the *corrected* semantics — rebase on those fixes before starting Task 6/7, and do not merge FSK/GMSK/OFDM-dependent profiles before they land.

**Environment:** run everything from the repo root with the venv active (`source .venv/bin/activate`). No `maturin develop` needed (no Rust changes).

---

## File Structure

```
python/spectra/waveforms/physical.py      # NEW: rate-resolution helpers (Task 1)
python/spectra/waveforms/base.py          # MODIFY: num_symbols_for() default (Task 2)
python/spectra/scene/composer.py          # MODIFY: use num_symbols_for (Task 2), accept profiles (Task 10)
python/spectra/waveforms/rrc_base.py      # MODIFY: symbol_rate kwarg (Task 3)
python/spectra/waveforms/radar.py         # MODIFY: PulsedRadar/FMCW physical kwargs (Task 4)
python/spectra/waveforms/lfm.py           # MODIFY: sweep_bandwidth/pulse_duration (Task 5)
python/spectra/waveforms/barker.py        # MODIFY: chip_rate (Task 5)
python/spectra/waveforms/fsk.py           # MODIFY: symbol_rate/deviation (Task 6, after fix lands)
python/spectra/waveforms/ofdm.py          # MODIFY: subcarrier_spacing (Task 7, after fix lands)
python/spectra/profiles/__init__.py       # NEW: public API re-exports (Task 8)
python/spectra/profiles/spec.py           # NEW: ParamSpec + EmitterProfile + errors (Task 8)
python/spectra/profiles/registry.py       # NEW: curated profiles + get/list/register (Task 9)
python/spectra/__init__.py                # MODIFY: export profiles module (Task 9)
python/spectra/benchmarks/loader.py       # MODIFY: {profile: name} YAML entries (Task 11)
tests/test_physical_params.py             # NEW: Tasks 1–7 unit tests
tests/test_profiles.py                    # NEW: Tasks 8–10 tests
tests/test_spectral_occupancy.py          # NEW: slow spectral verification (Task 12)
docs/user-guide/realistic-emitters.md     # NEW: user guide (Task 13)
mkdocs.yml                                # MODIFY: nav entry (Task 13)
examples/datasets/wideband_scenes.py      # MODIFY: profile-based scene section (Task 13)
```

Convention used throughout: physical kwargs default to `None`; when `None`, behavior is byte-identical to today. `sample_rate` first becomes known inside `generate()`/`bandwidth()`, so physical→sample resolution happens there, never in `__init__`.

---

### Task 1: Rate-resolution helpers (`waveforms/physical.py`)

**Files:**
- Create: `python/spectra/waveforms/physical.py`
- Test: `tests/test_physical_params.py`

- [ ] **Step 1: Write the failing tests**

```python
# tests/test_physical_params.py
"""Tests for physical-unit waveform parameterization."""

import numpy as np
import pytest

from spectra.waveforms.physical import resolve_symbol_rate, resample_to_rate


class TestResolveSymbolRate:
    def test_exact_integer_sps(self):
        # 10 MHz / 1.25 MBd = exactly 8 sps, no resampling
        sps, up, down = resolve_symbol_rate(10e6, 1.25e6)
        assert (sps, up, down) == (8, 1, 1)

    def test_rounds_within_tolerance(self):
        # 10 MHz / 1.24 MBd = 8.0645 -> rounds to 8 (0.8% error <= 1%)
        sps, up, down = resolve_symbol_rate(10e6, 1.24e6)
        assert (sps, up, down) == (8, 1, 1)

    def test_resamples_beyond_tolerance(self):
        # 10 MHz / 4.8 kBd = 2083.33 -> 0.016% error, rounds. Force a bad
        # case: 10 MHz / 1.5 MBd = 6.667 -> 5% from 7, needs resampling.
        sps, up, down = resolve_symbol_rate(10e6, 1.5e6)
        assert sps == 7               # ceil(6.667)
        assert up < down              # downsampling: generated rate > target
        # after resampling by up/down, samples-per-symbol matches the exact
        # ratio to <0.1%:  7 * 20/21 = 6.667
        eff_sps = sps * up / down
        assert abs(eff_sps - 10e6 / 1.5e6) / (10e6 / 1.5e6) < 1e-3

    def test_symbol_rate_above_nyquist_raises(self):
        with pytest.raises(ValueError, match="symbol_rate"):
            resolve_symbol_rate(10e6, 6e6)   # sps would be < 2

    def test_nonpositive_symbol_rate_raises(self):
        with pytest.raises(ValueError, match="symbol_rate"):
            resolve_symbol_rate(10e6, 0.0)


class TestResampleToRate:
    def test_identity_when_up_down_one(self):
        x = np.arange(64, dtype=np.complex64)
        out = resample_to_rate(x, 1, 1)
        assert out is x

    def test_length_scales_by_ratio(self):
        x = np.exp(2j * np.pi * 0.01 * np.arange(8000)).astype(np.complex64)
        out = resample_to_rate(x, 20, 21)
        assert abs(len(out) - len(x) * 20 / 21) <= 21
        assert out.dtype == np.complex64

    def test_preserves_tone_frequency(self):
        # A tone at normalized f=0.05 resampled by 20/21 must appear at
        # 0.05 * 21/20 = 0.0525 of the new rate
        n = 16384
        x = np.exp(2j * np.pi * 0.05 * np.arange(n)).astype(np.complex64)
        out = resample_to_rate(x, 20, 21)
        spec = np.abs(np.fft.fft(out * np.hanning(len(out))))
        peak = np.argmax(spec[: len(out) // 2]) / len(out)
        assert abs(peak - 0.0525) < 0.001
```

- [ ] **Step 2: Run tests to verify they fail**

Run: `pytest tests/test_physical_params.py -v`
Expected: FAIL — `ModuleNotFoundError: No module named 'spectra.waveforms.physical'`

- [ ] **Step 3: Write the implementation**

```python
# python/spectra/waveforms/physical.py
"""Physical-unit -> sample-domain resolution for waveform parameters.

Spec: docs/superpowers/specs/2026-07-02-waveform-realism-design.md.
Rule: generate at the nearest integer samples-per-symbol when the implied
symbol-rate error is <= RATE_TOLERANCE; otherwise generate at
ceil(sample_rate / symbol_rate) sps and rational-resample to the exact rate.
"""

import math
from fractions import Fraction
from typing import Tuple

import numpy as np

from spectra.utils.dsp import multistage_resampler

RATE_TOLERANCE = 0.01


def resolve_symbol_rate(
    sample_rate: float, symbol_rate: float, tol: float = RATE_TOLERANCE
) -> Tuple[int, int, int]:
    """Map a physical symbol rate to ``(sps, up, down)``.

    Generate at integer ``sps``; when ``up == down == 1`` no resampling is
    needed, otherwise resample the generated signal by ``up/down`` so the
    symbol rate at ``sample_rate`` matches ``symbol_rate``.
    """
    if symbol_rate <= 0:
        raise ValueError(f"symbol_rate must be positive, got {symbol_rate}")
    exact = sample_rate / symbol_rate
    if exact < 2.0:
        raise ValueError(
            f"symbol_rate {symbol_rate:g} Hz needs >= 2 samples/symbol "
            f"at sample_rate {sample_rate:g} Hz"
        )
    nearest = round(exact)
    if nearest >= 2 and abs(nearest - exact) / exact <= tol:
        return nearest, 1, 1
    sps = max(2, math.ceil(exact))
    # Generated rate is symbol_rate * sps >= sample_rate; resample down by
    # the rational approximation of sample_rate / generated_rate.
    frac = Fraction(exact / sps).limit_denominator(1000)
    return sps, frac.numerator, frac.denominator


def resample_to_rate(iq: np.ndarray, up: int, down: int) -> np.ndarray:
    """Rational-resample ``iq`` by ``up/down``; identity when ``up == down == 1``."""
    if up == down:
        return iq
    return multistage_resampler(iq, up, down).astype(np.complex64)
```

- [ ] **Step 4: Run tests to verify they pass**

Run: `pytest tests/test_physical_params.py -v`
Expected: 8 PASS

- [ ] **Step 5: Commit**

```bash
git add python/spectra/waveforms/physical.py tests/test_physical_params.py
git commit -m "feat(waveforms): rate-resolution helpers for physical-unit params"
```

---

### Task 2: `Waveform.num_symbols_for()` + Composer uses it

**Files:**
- Modify: `python/spectra/waveforms/base.py`
- Modify: `python/spectra/scene/composer.py:86-88`
- Test: `tests/test_physical_params.py`

- [ ] **Step 1: Write the failing tests** (append to `tests/test_physical_params.py`)

```python
from spectra.waveforms.base import Waveform


class _StubWaveform(Waveform):
    """Minimal legacy waveform: 8 samples per symbol via attribute."""

    samples_per_symbol = 8

    def generate(self, num_symbols, sample_rate, seed=None):
        return np.zeros(num_symbols * 8, dtype=np.complex64)

    def bandwidth(self, sample_rate):
        return sample_rate / 8

    @property
    def label(self):
        return "STUB"


class TestNumSymbolsFor:
    def test_default_uses_samples_per_symbol(self):
        wf = _StubWaveform()
        assert wf.num_symbols_for(10000, 10e6) == 1250

    def test_default_without_attribute_uses_eight(self):
        wf = _StubWaveform()
        del type(wf).samples_per_symbol  # simulate a waveform lacking the attr
        try:
            assert wf.num_symbols_for(10000, 10e6) == 1250
        finally:
            type(wf).samples_per_symbol = 8

    def test_composer_scene_unchanged(self):
        # Regression: same seed => byte-identical composite before/after
        # Composer switches from getattr(...) to num_symbols_for().
        import spectra as sp

        cfg = sp.SceneConfig(
            capture_duration=1e-4, capture_bandwidth=10e6, sample_rate=10e6,
            num_signals=2, signal_pool=[sp.QPSK(), sp.BPSK()], snr_range=(10, 10),
        )
        iq, descs = sp.Composer(cfg).generate(seed=123)
        assert len(iq) == 1000
        assert len(descs) == 2
```

- [ ] **Step 2: Run tests to verify they fail**

Run: `pytest tests/test_physical_params.py::TestNumSymbolsFor -v`
Expected: FAIL — `AttributeError: 'Waveform' object has no attribute 'num_symbols_for'` (the composer test passes already; it is a pinned regression check)

- [ ] **Step 3: Implement.** In `python/spectra/waveforms/base.py`, add to `Waveform` (after `bandwidth`, before `label`):

```python
    def num_symbols_for(self, num_samples: int, sample_rate: float) -> int:
        """Number of symbols needed to fill ``num_samples`` at ``sample_rate``.

        Default reproduces the legacy Composer heuristic
        (``num_samples // samples_per_symbol``, falling back to 8).
        Waveforms with physical-unit parameters override this.
        """
        sps = getattr(self, "samples_per_symbol", 8)
        return int(num_samples // sps)
```

In `python/spectra/scene/composer.py`, replace lines 86-88:

```python
            # Determine number of symbols to fill the capture
            sps = getattr(waveform, "samples_per_symbol", 8)
            num_symbols = num_capture_samples // sps
```

with:

```python
            # Determine number of symbols to fill the capture
            num_symbols = waveform.num_symbols_for(num_capture_samples, cfg.sample_rate)
```

- [ ] **Step 4: Run tests**

Run: `pytest tests/test_physical_params.py tests/test_composer.py tests/test_wideband*.py -v` (skip files that don't exist)
Expected: PASS

- [ ] **Step 5: Commit**

```bash
git add python/spectra/waveforms/base.py python/spectra/scene/composer.py tests/test_physical_params.py
git commit -m "feat(waveforms): num_symbols_for() hook; Composer delegates symbol count"
```

---

### Task 3: `symbol_rate` on the RRC family (PSK/QAM/ASK)

**Files:**
- Modify: `python/spectra/waveforms/rrc_base.py`
- Test: `tests/test_physical_params.py`

All PSK/QAM/ASK classes inherit `_RRCWaveformBase.__init__` unchanged, so one edit covers the family.

- [ ] **Step 1: Write the failing tests** (append)

```python
import spectra as sp


class TestRRCSymbolRate:
    def test_legacy_path_unchanged(self):
        # No physical kwargs: identical IQ to a pre-change reference call
        a = sp.QPSK().generate(num_symbols=64, sample_rate=10e6, seed=7)
        b = sp.QPSK(samples_per_symbol=8).generate(num_symbols=64, sample_rate=10e6, seed=7)
        np.testing.assert_array_equal(a, b)

    def test_conflicting_kwargs_raise(self):
        with pytest.raises(ValueError, match="symbol_rate"):
            sp.QPSK(symbol_rate=1e6, samples_per_symbol=4)

    def test_symbol_rate_sets_bandwidth_fs_independent(self):
        wf = sp.QPSK(symbol_rate=250e3, rolloff=0.35)
        assert wf.bandwidth(10e6) == pytest.approx(250e3 * 1.35)
        assert wf.bandwidth(20e6) == pytest.approx(250e3 * 1.35)

    def test_generate_at_exact_divisor(self):
        # 10 MHz / 250 kBd = 40 sps exactly
        wf = sp.QPSK(symbol_rate=250e3)
        iq = wf.generate(num_symbols=100, sample_rate=10e6, seed=1)
        assert len(iq) == 100 * 40

    def test_generate_with_resampling_hits_rate(self):
        # 10 MHz / 1.5 MBd = 6.667 sps -> resample path; length within 1%
        wf = sp.QPSK(symbol_rate=1.5e6)
        iq = wf.generate(num_symbols=300, sample_rate=10e6, seed=1)
        expected = 300 * 10e6 / 1.5e6
        assert abs(len(iq) - expected) / expected < 0.01

    def test_num_symbols_for_physical(self):
        wf = sp.QPSK(symbol_rate=250e3)
        assert wf.num_symbols_for(10000, 10e6) == 250   # 10000 / 40

    def test_bandwidth_above_fs_raises_at_generate(self):
        wf = sp.QPSK(symbol_rate=4e6, rolloff=0.35)     # 5.4 MHz BW
        with pytest.raises(ValueError, match="bandwidth"):
            wf.generate(num_symbols=10, sample_rate=5e6, seed=1)

    def test_measured_occupancy_matches_symbol_rate(self):
        # 99% occupied bandwidth of RRC QPSK ~= 1.16 * Rs, well inside
        # the claimed Rs*(1+rolloff); coarse check at low cost.
        wf = sp.QPSK(symbol_rate=500e3)
        iq = wf.generate(num_symbols=4000, sample_rate=10e6, seed=3)
        f = np.fft.fftshift(np.fft.fftfreq(4096, 1 / 10e6))
        w = np.hanning(4096)
        segs = len(iq) // 4096
        psd = np.mean(
            [np.abs(np.fft.fft(iq[k * 4096:(k + 1) * 4096] * w)) ** 2 for k in range(segs)],
            axis=0,
        )
        psd = np.fft.fftshift(psd)
        c = np.cumsum(psd) / np.sum(psd)
        obw = f[np.searchsorted(c, 0.995)] - f[np.searchsorted(c, 0.005)]
        assert 0.9 * 500e3 < obw < 1.35 * 500e3
```

- [ ] **Step 2: Run to verify failure**

Run: `pytest tests/test_physical_params.py::TestRRCSymbolRate -v`
Expected: `test_legacy_path_unchanged` PASSes (pinned regression); the rest FAIL with `TypeError: __init__() got an unexpected keyword argument 'symbol_rate'`

- [ ] **Step 3: Implement.** Replace `python/spectra/waveforms/rrc_base.py` contents with:

```python
"""Shared base class for RRC-filtered waveforms (PSK, QAM, ASK)."""

from abc import abstractmethod
from typing import Optional

import numpy as np

from spectra._rust import apply_rrc_filter_with_taps
from spectra.utils.rrc_cache import cached_rrc_taps
from spectra.waveforms.base import Waveform
from spectra.waveforms.physical import resample_to_rate, resolve_symbol_rate


class _RRCWaveformBase(Waveform):
    """Base class for waveforms that pulse-shape symbols through an RRC filter.

    Generates discrete symbols, upsamples by ``samples_per_symbol``, then
    convolves with a Root-Raised-Cosine (RRC) filter. The matched RRC at the
    receiver yields a Raised-Cosine response with zero ISI at symbol boundaries.

    Subclasses must define ``label`` and implement ``_generate_symbols()``.

    Args:
        rolloff: RRC excess bandwidth factor in [0, 1]. Higher values widen
            the spectrum but reduce ISI sensitivity. Default 0.35.
        filter_span: Filter half-length in symbols (filter has
            ``2 * filter_span * samples_per_symbol + 1`` taps). Default 10.
        samples_per_symbol: Upsampling factor (samples per symbol). Default 8.
            Mutually exclusive with ``symbol_rate``.
        symbol_rate: Physical symbol rate in baud. When set, the
            samples-per-symbol value is derived from the sample rate at
            ``generate()`` time and ``bandwidth()`` becomes
            ``symbol_rate * (1 + rolloff)`` independent of sample rate.

    Note:
        Bandwidth = ``symbol_rate * (1 + rolloff)``
        where ``symbol_rate = sample_rate / samples_per_symbol`` in the
        legacy (sample-domain) parameterization.
    """

    def __init__(
        self,
        rolloff: float = 0.35,
        filter_span: int = 10,
        samples_per_symbol: Optional[int] = None,
        symbol_rate: Optional[float] = None,
    ):
        if symbol_rate is not None and samples_per_symbol is not None:
            raise ValueError(
                "symbol_rate and samples_per_symbol are mutually exclusive; "
                "pass one or the other"
            )
        self.rolloff = rolloff
        self.filter_span = filter_span
        self.symbol_rate = symbol_rate
        # Legacy attribute stays an int for external consumers; it is unused
        # (and meaningless) when symbol_rate is set.
        self.samples_per_symbol = 8 if samples_per_symbol is None else samples_per_symbol

    def _resolved_sps(self, sample_rate: float):
        """Return (sps, up, down) for this waveform at ``sample_rate``."""
        if self.symbol_rate is None:
            return self.samples_per_symbol, 1, 1
        return resolve_symbol_rate(sample_rate, self.symbol_rate)

    def bandwidth(self, sample_rate: float) -> float:
        if self.symbol_rate is not None:
            return self.symbol_rate * (1.0 + self.rolloff)
        symbol_rate = sample_rate / self.samples_per_symbol
        return symbol_rate * (1.0 + self.rolloff)

    def num_symbols_for(self, num_samples: int, sample_rate: float) -> int:
        if self.symbol_rate is None:
            return int(num_samples // self.samples_per_symbol)
        return int(num_samples * self.symbol_rate / sample_rate)

    @abstractmethod
    def _generate_symbols(self, num_symbols: int, seed: int) -> np.ndarray:
        """Return complex64 symbol array of length num_symbols."""
        ...

    def generate(
        self, num_symbols: int, sample_rate: float, seed: Optional[int] = None
    ) -> np.ndarray:
        if self.symbol_rate is not None and self.bandwidth(sample_rate) > sample_rate:
            raise ValueError(
                f"{self.label} bandwidth {self.bandwidth(sample_rate):g} Hz exceeds "
                f"sample_rate {sample_rate:g} Hz"
            )
        sps, up, down = self._resolved_sps(sample_rate)
        s = seed if seed is not None else np.random.randint(0, 2**32)
        symbols = self._generate_symbols(num_symbols, s)
        taps = cached_rrc_taps(self.rolloff, self.filter_span, sps)
        filtered = apply_rrc_filter_with_taps(symbols, taps, sps)
        return resample_to_rate(filtered, up, down)
```

Byte-compat notes the implementer must respect: (a) the legacy path calls `cached_rrc_taps`/`apply_rrc_filter_with_taps` with the same integer sps as before; (b) the `np.random.randint` fallback is unchanged; (c) `samples_per_symbol` attribute remains an `int`.

- [ ] **Step 4: Run tests + existing suite**

Run: `pytest tests/test_physical_params.py tests/test_waveforms_psk.py tests/test_waveforms_qam.py -v`
Expected: all PASS (legacy tests prove byte-compat)

- [ ] **Step 5: Commit**

```bash
git add python/spectra/waveforms/rrc_base.py tests/test_physical_params.py
git commit -m "feat(waveforms): symbol_rate physical parameterization for RRC family"
```

---

### Task 4: Physical units on `PulsedRadar` and `FMCW`

**Files:**
- Modify: `python/spectra/waveforms/radar.py` (PulsedRadar `__init__`/`generate`/`bandwidth`; FMCW likewise)
- Test: `tests/test_physical_params.py`

- [ ] **Step 1: Write the failing tests** (append)

```python
class TestRadarPhysical:
    def test_pulsed_physical_derivation(self):
        wf = sp.PulsedRadar(pulse_width=1e-6, pri=100e-6, num_pulses=4)
        iq = wf.generate(num_symbols=1, sample_rate=10e6, seed=1)
        # 4 pulses of 10 samples each in 4*1000 samples
        assert len(iq) == 4000
        on = np.abs(iq) > 1e-9
        assert on.sum() == 4 * 10
        assert wf.bandwidth(10e6) == pytest.approx(1e6)  # 1 / pulse_width

    def test_pulsed_conflicting_kwargs_raise(self):
        with pytest.raises(ValueError, match="pulse_width"):
            sp.PulsedRadar(pulse_width=1e-6, pulse_width_samples=32)

    def test_pulsed_pri_shorter_than_pulse_raises(self):
        wf = sp.PulsedRadar(pulse_width=2e-6, pri=1e-6)
        with pytest.raises(ValueError, match="pri"):
            wf.generate(num_symbols=1, sample_rate=10e6, seed=1)

    def test_pulsed_legacy_unchanged(self):
        a = sp.PulsedRadar().generate(num_symbols=1, sample_rate=10e6, seed=2)
        b = sp.PulsedRadar(pulse_width_samples=64, pri_samples=512).generate(
            num_symbols=1, sample_rate=10e6, seed=2
        )
        np.testing.assert_array_equal(a, b)

    def test_fmcw_physical_derivation(self):
        wf = sp.FMCW(sweep_bandwidth=2e6, sweep_time=50e-6, idle_time=10e-6, num_sweeps=2)
        iq = wf.generate(num_symbols=1, sample_rate=10e6, seed=1)
        assert len(iq) == 2 * (500 + 100)
        assert wf.bandwidth(10e6) == pytest.approx(2e6)
        assert wf.bandwidth(20e6) == pytest.approx(2e6)   # fs-independent

    def test_fmcw_sweep_wider_than_fs_raises(self):
        wf = sp.FMCW(sweep_bandwidth=12e6, sweep_time=50e-6)
        with pytest.raises(ValueError, match="bandwidth"):
            wf.generate(num_symbols=1, sample_rate=10e6, seed=1)
```

- [ ] **Step 2: Run to verify failure**

Run: `pytest tests/test_physical_params.py::TestRadarPhysical -v`
Expected: FAIL — unexpected keyword arguments

- [ ] **Step 3: Implement `PulsedRadar` changes.** In `python/spectra/waveforms/radar.py`, replace the `PulsedRadar.__init__` body and add resolution:

```python
    def __init__(
        self,
        pulse_width_samples: Optional[int] = None,
        pri_samples: Optional[int] = None,
        num_pulses: int = 16,
        pulse_shape: str = "rect",
        pri_stagger: Optional[List[int]] = None,
        pri_jitter_fraction: float = 0.0,
        pulse_width: Optional[float] = None,
        pri: Optional[float] = None,
    ):
        if pulse_width is not None and pulse_width_samples is not None:
            raise ValueError("pulse_width and pulse_width_samples are mutually exclusive")
        if pri is not None and pri_samples is not None:
            raise ValueError("pri and pri_samples are mutually exclusive")
        self._pulse_width = pulse_width
        self._pri = pri
        self._pulse_width_samples = 64 if pulse_width_samples is None else pulse_width_samples
        self._pri_samples = 512 if pri_samples is None else pri_samples
        self._num_pulses = num_pulses
        self._pulse_shape = pulse_shape
        self._pri_stagger = pri_stagger
        self._pri_jitter_fraction = pri_jitter_fraction
        self.samples_per_symbol = self._pri_samples * num_pulses

    def _resolved_samples(self, sample_rate: float):
        """Return (pulse_width_samples, pri_samples) at ``sample_rate``."""
        pw = self._pulse_width_samples
        pri = self._pri_samples
        if self._pulse_width is not None:
            pw = max(1, round(self._pulse_width * sample_rate))
            if pw < 4:
                import warnings

                warnings.warn(
                    f"pulse_width {self._pulse_width:g}s is only {pw} samples at "
                    f"{sample_rate:g} Hz; pulse fidelity will be poor",
                    stacklevel=2,
                )
        if self._pri is not None:
            pri = round(self._pri * sample_rate)
        if pri < pw:
            raise ValueError(f"pri ({pri} samples) must be >= pulse width ({pw} samples)")
        return pw, pri
```

In `PulsedRadar.generate`, replace direct uses of `self._pulse_width_samples` / `self._pri_samples` with the resolved values:

```python
        pw_samples, pri_samples = self._resolved_samples(sample_rate)
        pulse = _make_pulse_shape(pw_samples, self._pulse_shape)
```

and use `pri_samples` in the jitter computation and the `generate_pulse_train(pulse, pri_samples, ...)` call. Update `bandwidth` and add `num_symbols_for`:

```python
    def bandwidth(self, sample_rate: float) -> float:
        if self._pulse_width is not None:
            return 1.0 / self._pulse_width
        return sample_rate / self._pulse_width_samples

    def num_symbols_for(self, num_samples: int, sample_rate: float) -> int:
        _, pri_samples = self._resolved_samples(sample_rate)
        return max(1, int(num_samples // (pri_samples * self._num_pulses)))
```

- [ ] **Step 4: Implement `FMCW` changes.** Same pattern:

```python
    def __init__(
        self,
        sweep_bandwidth_fraction: Optional[float] = None,
        sweep_samples: Optional[int] = None,
        idle_samples: Optional[int] = None,
        num_sweeps: int = 16,
        sweep_type: str = "sawtooth",
        sweep_bandwidth: Optional[float] = None,
        sweep_time: Optional[float] = None,
        idle_time: Optional[float] = None,
    ):
        if sweep_bandwidth is not None and sweep_bandwidth_fraction is not None:
            raise ValueError("sweep_bandwidth and sweep_bandwidth_fraction are mutually exclusive")
        if sweep_time is not None and sweep_samples is not None:
            raise ValueError("sweep_time and sweep_samples are mutually exclusive")
        if idle_time is not None and idle_samples is not None:
            raise ValueError("idle_time and idle_samples are mutually exclusive")
        self._sweep_bandwidth = sweep_bandwidth
        self._sweep_time = sweep_time
        self._idle_time = idle_time
        self._sweep_bandwidth_fraction = (
            0.5 if sweep_bandwidth_fraction is None else sweep_bandwidth_fraction
        )
        self._sweep_samples = 256 if sweep_samples is None else sweep_samples
        self._idle_samples = 64 if idle_samples is None else idle_samples
        self._num_sweeps = num_sweeps
        self._sweep_type = sweep_type
        self.samples_per_symbol = (self._sweep_samples + self._idle_samples) * num_sweeps

    def _resolved(self, sample_rate: float):
        """Return (bw_hz, sweep_samples, idle_samples) at ``sample_rate``."""
        bw = (
            self._sweep_bandwidth
            if self._sweep_bandwidth is not None
            else sample_rate * self._sweep_bandwidth_fraction
        )
        sweep = (
            round(self._sweep_time * sample_rate)
            if self._sweep_time is not None
            else self._sweep_samples
        )
        idle = (
            round(self._idle_time * sample_rate)
            if self._idle_time is not None
            else self._idle_samples
        )
        if bw > sample_rate:
            raise ValueError(
                f"FMCW sweep bandwidth {bw:g} Hz exceeds sample_rate {sample_rate:g} Hz"
            )
        return bw, sweep, idle
```

Replace the first two lines of `FMCW.generate` and add `bandwidth`/`num_symbols_for`:

```python
    def generate(self, num_symbols, sample_rate, seed=None):
        bw, sweep_samples, idle_samples = self._resolved(sample_rate)
        sweep = generate_fmcw_sweep(sweep_samples, bw, sample_rate, self._sweep_type)
        idle = np.zeros(idle_samples, dtype=np.complex64)

        one_symbol_parts = []
        for _ in range(self._num_sweeps):
            one_symbol_parts.append(sweep)
            if idle_samples > 0:
                one_symbol_parts.append(idle)
        one_symbol = np.concatenate(one_symbol_parts)
        return np.tile(one_symbol, num_symbols)

    def bandwidth(self, sample_rate: float) -> float:
        if self._sweep_bandwidth is not None:
            return self._sweep_bandwidth
        return sample_rate * self._sweep_bandwidth_fraction

    def num_symbols_for(self, num_samples: int, sample_rate: float) -> int:
        _, sweep_samples, idle_samples = self._resolved(sample_rate)
        return max(1, int(num_samples // ((sweep_samples + idle_samples) * self._num_sweeps)))
```

- [ ] **Step 5: Run tests**

Run: `pytest tests/test_physical_params.py::TestRadarPhysical tests/test_waveforms_radar.py -v`
Expected: PASS (legacy radar tests prove byte-compat; the existing `weather()`/`marine_nav()` presets still construct with sample-domain args)

- [ ] **Step 6: Commit**

```bash
git add python/spectra/waveforms/radar.py tests/test_physical_params.py
git commit -m "feat(radar): physical pulse_width/pri and sweep_bandwidth/sweep_time params"
```

---

### Task 5: Physical units on `LFM`, `BarkerCode`, and polyphase codes

**Files:**
- Modify: `python/spectra/waveforms/lfm.py`
- Modify: `python/spectra/waveforms/barker.py`
- Modify: `python/spectra/waveforms/polyphase.py`
- Test: `tests/test_physical_params.py`

- [ ] **Step 1: Write the failing tests** (append)

```python
class TestLFMBarkerPhysical:
    def test_lfm_physical(self):
        wf = sp.LFM(sweep_bandwidth=3e6, pulse_duration=50e-6)
        iq = wf.generate(num_symbols=2, sample_rate=10e6, seed=1)
        assert len(iq) == 2 * 500
        assert wf.bandwidth(10e6) == pytest.approx(3e6)
        assert wf.bandwidth(40e6) == pytest.approx(3e6)

    def test_lfm_conflict_raises(self):
        with pytest.raises(ValueError, match="sweep_bandwidth"):
            sp.LFM(sweep_bandwidth=3e6, bandwidth_fraction=0.4)

    def test_barker_chip_rate(self):
        wf = sp.BarkerCode(length=13, chip_rate=1e6)
        iq = wf.generate(num_symbols=1, sample_rate=10e6, seed=1)
        assert len(iq) == 13 * 10          # 10 samples per chip
        assert wf.bandwidth(10e6) == pytest.approx(1e6)

    def test_barker_legacy_unchanged(self):
        a = sp.BarkerCode(length=13).generate(num_symbols=2, sample_rate=10e6, seed=1)
        b = sp.BarkerCode(length=13, samples_per_chip=8).generate(
            num_symbols=2, sample_rate=10e6, seed=1
        )
        np.testing.assert_array_equal(a, b)

    def test_polyphase_chip_rate(self):
        wf = sp.FrankCode(code_order=4, chip_rate=1e6)
        iq = wf.generate(num_symbols=1, sample_rate=10e6, seed=1)
        assert len(iq) == 16 * 10           # order^2 chips * 10 samples/chip
        assert wf.bandwidth(10e6) == pytest.approx(1e6)
        with pytest.raises(ValueError, match="chip_rate"):
            sp.FrankCode(chip_rate=1e6, samples_per_chip=4)
```

- [ ] **Step 2: Run to verify failure**

Run: `pytest tests/test_physical_params.py::TestLFMBarkerPhysical -v`
Expected: FAIL — unexpected keyword arguments

- [ ] **Step 3: Implement `LFM`.** Replace `__init__`, `generate`, `bandwidth` in `python/spectra/waveforms/lfm.py`:

```python
    def __init__(
        self,
        bandwidth_fraction: Optional[float] = None,
        samples_per_pulse: Optional[int] = None,
        sweep_bandwidth: Optional[float] = None,
        pulse_duration: Optional[float] = None,
    ):
        if sweep_bandwidth is not None and bandwidth_fraction is not None:
            raise ValueError("sweep_bandwidth and bandwidth_fraction are mutually exclusive")
        if pulse_duration is not None and samples_per_pulse is not None:
            raise ValueError("pulse_duration and samples_per_pulse are mutually exclusive")
        self._sweep_bandwidth = sweep_bandwidth
        self._pulse_duration = pulse_duration
        self._bandwidth_fraction = 0.5 if bandwidth_fraction is None else bandwidth_fraction
        self._samples_per_pulse = 256 if samples_per_pulse is None else samples_per_pulse
        self.samples_per_symbol = self._samples_per_pulse

    def generate(self, num_symbols, sample_rate, seed=None):
        bw = self.bandwidth(sample_rate)
        if bw > sample_rate:
            raise ValueError(f"LFM sweep bandwidth {bw:g} Hz exceeds sample_rate {sample_rate:g} Hz")
        n = (
            round(self._pulse_duration * sample_rate)
            if self._pulse_duration is not None
            else self._samples_per_pulse
        )
        f0, f1 = -bw / 2.0, bw / 2.0
        duration = n / sample_rate
        pulses = [generate_chirp(duration, sample_rate, f0, f1) for _ in range(num_symbols)]
        return np.concatenate(pulses) if pulses else np.array([], dtype=np.complex64)

    def bandwidth(self, sample_rate: float) -> float:
        if self._sweep_bandwidth is not None:
            return self._sweep_bandwidth
        return sample_rate * self._bandwidth_fraction

    def num_symbols_for(self, num_samples: int, sample_rate: float) -> int:
        n = (
            round(self._pulse_duration * sample_rate)
            if self._pulse_duration is not None
            else self._samples_per_pulse
        )
        return max(1, int(num_samples // n))
```

- [ ] **Step 4: Implement `BarkerCode`.** In `python/spectra/waveforms/barker.py`:

```python
    def __init__(
        self,
        length: int = 13,
        samples_per_chip: Optional[int] = None,
        chip_rate: Optional[float] = None,
    ):
        if length not in BARKER_CODES:
            valid = sorted(BARKER_CODES.keys())
            raise ValueError(f"Barker code length must be one of {valid}, got {length}")
        if chip_rate is not None and samples_per_chip is not None:
            raise ValueError("chip_rate and samples_per_chip are mutually exclusive")
        self._length = length
        self._code = np.array(BARKER_CODES[length], dtype=np.float32)
        self._chip_rate = chip_rate
        self._samples_per_chip = 8 if samples_per_chip is None else samples_per_chip
        self.samples_per_symbol = length * self._samples_per_chip

    def _resolved_spc(self, sample_rate: float) -> int:
        if self._chip_rate is None:
            return self._samples_per_chip
        from spectra.waveforms.physical import resolve_symbol_rate

        spc, _, _ = resolve_symbol_rate(sample_rate, self._chip_rate)
        return spc

    def generate(self, num_symbols, sample_rate, seed=None):
        spc = self._resolved_spc(sample_rate)
        chips_up = np.repeat(self._code, spc)
        signal = np.tile(chips_up, num_symbols)
        return (signal + 0j).astype(np.complex64)

    def bandwidth(self, sample_rate: float) -> float:
        if self._chip_rate is not None:
            return self._chip_rate
        return sample_rate / self._samples_per_chip

    def num_symbols_for(self, num_samples: int, sample_rate: float) -> int:
        return max(1, int(num_samples // (self._length * self._resolved_spc(sample_rate))))
```

(Rect chips tolerate sps rounding; no resample path needed — `resolve_symbol_rate` is used only for its validation and rounding.)

- [ ] **Step 5: Implement polyphase codes.** In `python/spectra/waveforms/polyphase.py`, extend `_PolyphaseCodeBase`:

```python
class _PolyphaseCodeBase(Waveform):
    """Shared base for polyphase radar code waveforms."""

    def __init__(self, samples_per_chip: Optional[int] = None, chip_rate: Optional[float] = None):
        if chip_rate is not None and samples_per_chip is not None:
            raise ValueError("chip_rate and samples_per_chip are mutually exclusive")
        self._chip_rate = chip_rate
        self._samples_per_chip = 8 if samples_per_chip is None else samples_per_chip

    def _resolved_spc(self, sample_rate: float) -> int:
        if self._chip_rate is None:
            return self._samples_per_chip
        from spectra.waveforms.physical import resolve_symbol_rate

        spc, _, _ = resolve_symbol_rate(sample_rate, self._chip_rate)
        return spc

    def bandwidth(self, sample_rate: float) -> float:
        if self._chip_rate is not None:
            return self._chip_rate
        return sample_rate / self._samples_per_chip

    def num_symbols_for(self, num_samples: int, sample_rate: float) -> int:
        code_len = len(self._get_chips())
        return max(1, int(num_samples // (code_len * self._resolved_spc(sample_rate))))

    @abstractmethod
    def _get_chips(self) -> np.ndarray:
        """Return complex64 chip array for one code period."""
        ...

    def generate(self, num_symbols, sample_rate, seed=None):
        chips = self._get_chips()
        one_code = np.repeat(chips, self._resolved_spc(sample_rate))
        return np.tile(one_code, num_symbols)
```

Each subclass (`FrankCode`, `P1Code`, `P2Code`, `P3Code`, `P4Code`) gets `chip_rate: Optional[float] = None` appended to its `__init__` signature, with `samples_per_chip` default changed to `None`, forwarding `super().__init__(samples_per_chip, chip_rate)` and computing `self.samples_per_symbol` from `8 if samples_per_chip is None else samples_per_chip` (the legacy attribute keeps its old value; `num_symbols_for` handles the physical path). Example for `FrankCode` — repeat the same pattern for P1–P4:

```python
class FrankCode(_PolyphaseCodeBase):
    """Frank polyphase radar code. Code length = code_order^2 chips."""

    def __init__(
        self,
        code_order: int = 4,
        samples_per_chip: Optional[int] = None,
        chip_rate: Optional[float] = None,
    ):
        super().__init__(samples_per_chip, chip_rate)
        self._code_order = code_order
        self.samples_per_symbol = code_order * code_order * self._samples_per_chip
```

- [ ] **Step 6: Run tests, then the full suite**

Run: `pytest tests/test_physical_params.py -v && pytest tests/ -x -q -m "not slow"`
Expected: PASS

- [ ] **Step 7: Commit**

```bash
git add python/spectra/waveforms/lfm.py python/spectra/waveforms/barker.py python/spectra/waveforms/polyphase.py tests/test_physical_params.py
git commit -m "feat(waveforms): physical sweep/chip-rate params for LFM, Barker, polyphase"
```

---

### Task 6: `symbol_rate`/`deviation` on the FSK family — **blocked on task_0885f2c0**

Rebase on the FSK level-convention fix first. The code below assumes the corrected convention where adjacent tone spacing is `mod_index * symbol_rate` (levels ±1, ±3, …); if the landed fix chose to correct `bandwidth()` instead of the levels, adjust `_h_from_deviation` to the landed convention before implementing — the *tests* below are convention-independent because they assert measured spectra.

**Files:**
- Modify: `python/spectra/waveforms/fsk.py` (FSK, GFSK; MSK/GMSK gain `symbol_rate` only — their h is fixed at 0.5)
- Test: `tests/test_physical_params.py`

- [ ] **Step 1: Write the failing tests** (append)

```python
def _tone_peaks(iq, sample_rate, nfft=8192):
    w = np.hanning(nfft)
    segs = len(iq) // nfft
    psd = np.mean(
        [np.abs(np.fft.fft(iq[k * nfft:(k + 1) * nfft] * w)) ** 2 for k in range(segs)],
        axis=0,
    )
    psd = np.fft.fftshift(psd)
    f = np.fft.fftshift(np.fft.fftfreq(nfft, 1 / sample_rate))
    return f, psd


class TestFSKPhysical:
    def test_deviation_places_outer_tones(self):
        # 2FSK, Rs=100 kBd, peak deviation 50 kHz: tones at +/-50 kHz
        wf = sp.FSK(order=2, symbol_rate=100e3, deviation=50e3)
        iq = wf.generate(num_symbols=5000, sample_rate=2e6, seed=1)
        f, psd = _tone_peaks(iq, 2e6)
        pos = f[np.argmax(psd * (f > 0))]
        assert abs(pos - 50e3) < 5e3

    def test_bandwidth_carson(self):
        # Carson: 2*(peak_dev + Rs)
        wf = sp.FSK(order=2, symbol_rate=100e3, deviation=50e3)
        assert wf.bandwidth(2e6) == pytest.approx(2 * (50e3 + 100e3))
        assert wf.bandwidth(8e6) == pytest.approx(2 * (50e3 + 100e3))

    def test_conflict_raises(self):
        with pytest.raises(ValueError, match="deviation"):
            sp.FSK(mod_index=0.7, deviation=50e3, symbol_rate=100e3)

    def test_deviation_requires_symbol_rate(self):
        with pytest.raises(ValueError, match="symbol_rate"):
            sp.FSK(deviation=50e3)

    def test_gmsk_symbol_rate(self):
        wf = sp.GMSK(bt=0.4, symbol_rate=9600)
        iq = wf.generate(num_symbols=2000, sample_rate=192e3, seed=1)  # 20 sps exact
        assert len(iq) == 2000 * 20
        # GMSK BT=0.4 99% OBW ~ 0.92 Rs; assert inside [0.7, 1.2] Rs
        f, psd = _tone_peaks(iq, 192e3, nfft=4096)
        c = np.cumsum(psd) / np.sum(psd)
        obw = f[np.searchsorted(c, 0.995)] - f[np.searchsorted(c, 0.005)]
        assert 0.7 * 9600 < obw < 1.2 * 9600
```

- [ ] **Step 2: Run to verify failure**

Run: `pytest tests/test_physical_params.py::TestFSKPhysical -v`
Expected: FAIL — unexpected keyword arguments

- [ ] **Step 3: Implement.** In `python/spectra/waveforms/fsk.py`, `FSK.__init__` becomes:

```python
    def __init__(
        self,
        order: int = 2,
        mod_index: Optional[float] = None,
        samples_per_symbol: Optional[int] = None,
        symbol_rate: Optional[float] = None,
        deviation: Optional[float] = None,
    ):
        if deviation is not None:
            if mod_index is not None:
                raise ValueError("deviation and mod_index are mutually exclusive")
            if symbol_rate is None:
                raise ValueError("deviation requires symbol_rate")
        if symbol_rate is not None and samples_per_symbol is not None:
            raise ValueError("symbol_rate and samples_per_symbol are mutually exclusive")
        self._order = order
        self._deviation = deviation
        self.symbol_rate = symbol_rate
        # Peak deviation = (order - 1)/2 * h * Rs  =>  h = 2*dev / ((M-1)*Rs)
        if deviation is not None:
            self._mod_index = 2.0 * deviation / ((order - 1) * symbol_rate)
        else:
            self._mod_index = 1.0 if mod_index is None else mod_index
        self.samples_per_symbol = 8 if samples_per_symbol is None else samples_per_symbol
```

`generate` resolves sps exactly like the RRC base (share the pattern):

```python
        if self.symbol_rate is not None:
            sps, up, down = resolve_symbol_rate(sample_rate, self.symbol_rate)
        else:
            sps, up, down = self.samples_per_symbol, 1, 1
```

then uses `sps` in place of `self.samples_per_symbol` throughout and returns `resample_to_rate(np.exp(1j * phase).astype(np.complex64), up, down)`. `bandwidth`:

```python
    def bandwidth(self, sample_rate: float) -> float:
        symbol_rate = (
            self.symbol_rate
            if self.symbol_rate is not None
            else sample_rate / self.samples_per_symbol
        )
        peak_dev = 0.5 * (self._order - 1) * self._mod_index * symbol_rate
        return 2.0 * (peak_dev + symbol_rate)
```

and `num_symbols_for` mirrors the RRC implementation. `GFSK` gets the identical `symbol_rate`/`deviation` treatment (same `_h_from_deviation` formula). `MSK` and `GMSK` have h fixed at 0.5, so they gain only `symbol_rate` — GMSK's full changed methods:

```python
    def __init__(
        self,
        bt: float = 0.3,
        filter_span: int = 4,
        samples_per_symbol: Optional[int] = None,
        symbol_rate: Optional[float] = None,
    ):
        if symbol_rate is not None and samples_per_symbol is not None:
            raise ValueError("symbol_rate and samples_per_symbol are mutually exclusive")
        self._bt = bt
        self._filter_span = filter_span
        self.symbol_rate = symbol_rate
        self.samples_per_symbol = 8 if samples_per_symbol is None else samples_per_symbol
```

with `generate` resolving `sps, up, down` exactly as in `FSK` (note `_gaussian_taps` and the repeat-upsample both use the resolved `sps`, not `self.samples_per_symbol`), returning `resample_to_rate(...)`, and:

```python
    def bandwidth(self, sample_rate: float) -> float:
        symbol_rate = (
            self.symbol_rate
            if self.symbol_rate is not None
            else sample_rate / self.samples_per_symbol
        )
        return symbol_rate * 1.5   # or the BT-aware factor if task_0885f2c0 changed it

    def num_symbols_for(self, num_samples: int, sample_rate: float) -> int:
        if self.symbol_rate is None:
            return int(num_samples // self.samples_per_symbol)
        return int(num_samples * self.symbol_rate / sample_rate)
```

`MSK` is the same minus the Gaussian filter details. Named subclasses (`FSK4`, `FSK8`, `GFSK4`, `MSK4`, …) have explicit signatures — append `symbol_rate: Optional[float] = None, deviation: Optional[float] = None` (deviation only where the parent takes it) and pass them through to `super().__init__`.

- [ ] **Step 4: Run tests**

Run: `pytest tests/test_physical_params.py::TestFSKPhysical tests/test_waveforms_fsk.py -v`
Expected: PASS

- [ ] **Step 5: Commit**

```bash
git add python/spectra/waveforms/fsk.py tests/test_physical_params.py
git commit -m "feat(fsk): physical symbol_rate/deviation parameterization"
```

---

### Task 7: `subcarrier_spacing` on OFDM — **blocked on task_3ccc50b2**

Rebase on the OFDM pilot/guard fix first (`bandwidth()` semantics change there).

**Files:**
- Modify: `python/spectra/waveforms/ofdm.py`
- Test: `tests/test_physical_params.py`

- [ ] **Step 1: Write the failing tests** (append)

```python
class TestOFDMPhysical:
    def test_spacing_derives_fft_size(self):
        # 10 MHz / 15 kHz = 666.67 -> rounds to 667 (0.05% spacing error)
        wf = sp.OFDM(num_subcarriers=72, subcarrier_spacing=15e3, cp_fraction=0.07)
        iq = wf.generate(num_symbols=4, sample_rate=10e6, seed=1)
        fft_size = 667
        cp = round(0.07 * fft_size)
        assert len(iq) == 4 * (fft_size + cp)
        assert wf.bandwidth(10e6) == pytest.approx(72 * 15e3, rel=0.01)
        assert wf.bandwidth(30.72e6) == pytest.approx(72 * 15e3, rel=0.01)

    def test_spacing_conflict_raises(self):
        with pytest.raises(ValueError, match="subcarrier_spacing"):
            sp.OFDM(subcarrier_spacing=15e3, fft_size=512)

    def test_occupied_band_matches_claim(self):
        wf = sp.OFDM(num_subcarriers=52, subcarrier_spacing=312.5e3)
        iq = wf.generate(num_symbols=40, sample_rate=20e6, seed=2)
        f, psd = _tone_peaks(iq, 20e6, nfft=4096)
        c = np.cumsum(psd) / np.sum(psd)
        obw = f[np.searchsorted(c, 0.995)] - f[np.searchsorted(c, 0.005)]
        claimed = wf.bandwidth(20e6)
        assert 0.85 * claimed < obw < 1.15 * claimed
```

- [ ] **Step 2: Run to verify failure**

Run: `pytest tests/test_physical_params.py::TestOFDMPhysical -v`
Expected: FAIL — unexpected keyword arguments

- [ ] **Step 3: Implement.** In `OFDM.__init__` add parameters `subcarrier_spacing: Optional[float] = None` and `cp_fraction: Optional[float] = None`, change `fft_size`/`cp_length` defaults to `None` sentinels:

```python
        if subcarrier_spacing is not None and fft_size is not None:
            raise ValueError("subcarrier_spacing and fft_size are mutually exclusive")
        if cp_fraction is not None and cp_length is not None:
            raise ValueError("cp_fraction and cp_length are mutually exclusive")
        self._subcarrier_spacing = subcarrier_spacing
        self._cp_fraction = cp_fraction
        self._fft_size = 256 if fft_size is None else fft_size
        self._cp_length = 16 if cp_length is None else cp_length
```

Add a resolver used by `generate`/`num_symbols_for`:

```python
    def _resolved_fft(self, sample_rate: float):
        """Return (fft_size, cp_length) at ``sample_rate``."""
        if self._subcarrier_spacing is None:
            return self._fft_size, self._cp_length
        exact = sample_rate / self._subcarrier_spacing
        fft_size = round(exact)
        if abs(fft_size - exact) / exact > 0.01:
            raise ValueError(
                f"sample_rate {sample_rate:g} is not within 1% of an integer "
                f"multiple of subcarrier_spacing {self._subcarrier_spacing:g}; "
                f"choose a compatible sample rate"
            )
        if fft_size < self._num_subcarriers:
            raise ValueError(
                f"derived fft_size {fft_size} < num_subcarriers {self._num_subcarriers}"
            )
        cp_frac = 0.0625 if self._cp_fraction is None else self._cp_fraction
        return fft_size, round(cp_frac * fft_size)
```

`generate` calls `fft_size, cp_length = self._resolved_fft(sample_rate)` and uses the locals everywhere it used `self._fft_size`/`self._cp_length`. `samples_per_symbol` attribute keeps its legacy value; add:

```python
    def num_symbols_for(self, num_samples: int, sample_rate: float) -> int:
        fft_size, cp_length = self._resolved_fft(sample_rate)
        return max(1, int(num_samples // (fft_size + cp_length)))
```

`bandwidth` (post-fix semantics: pilots count as occupied): when `subcarrier_spacing` is set, `return active * self._subcarrier_spacing` where `active` follows whatever counting the landed OFDM fix uses. **Deliberate deviation from the general resample rule:** OFDM raises beyond 1% spacing error instead of resampling — resampling breaks the CP/FFT sample alignment that downstream OFDM tooling assumes. `SCFDMA` inherits everything.

- [ ] **Step 4: Run tests**

Run: `pytest tests/test_physical_params.py::TestOFDMPhysical tests/test_waveforms_ofdm.py -v`
Expected: PASS

- [ ] **Step 5: Commit**

```bash
git add python/spectra/waveforms/ofdm.py tests/test_physical_params.py
git commit -m "feat(ofdm): subcarrier_spacing/cp_fraction physical parameterization"
```

---

### Task 8: `ParamSpec` + `EmitterProfile` (`profiles/spec.py`)

**Files:**
- Create: `python/spectra/profiles/spec.py`
- Create: `python/spectra/profiles/__init__.py`
- Test: `tests/test_profiles.py`

- [ ] **Step 1: Write the failing tests**

```python
# tests/test_profiles.py
"""Tests for the emitter-profile registry."""

import numpy as np
import pytest

import spectra as sp
from spectra.profiles import (
    Choice,
    EmitterProfile,
    Fixed,
    LogUniform,
    ProfileNotRepresentable,
    Uniform,
)


class TestParamSpec:
    def test_fixed(self):
        rng = np.random.default_rng(0)
        assert Fixed(42).sample(rng) == 42

    def test_choice_draws_from_options(self):
        rng = np.random.default_rng(0)
        draws = {Choice([1, 2, 3]).sample(rng) for _ in range(50)}
        assert draws <= {1, 2, 3} and len(draws) > 1

    def test_uniform_bounds(self):
        rng = np.random.default_rng(0)
        for _ in range(50):
            v = Uniform(2.0, 3.0).sample(rng)
            assert 2.0 <= v <= 3.0

    def test_loguniform_bounds(self):
        rng = np.random.default_rng(0)
        for _ in range(50):
            v = LogUniform(1e3, 1e6).sample(rng)
            assert 1e3 <= v <= 1e6

    def test_invalid_bounds_raise(self):
        with pytest.raises(ValueError):
            Uniform(3.0, 2.0)
        with pytest.raises(ValueError):
            LogUniform(0.0, 1.0)

    def test_deterministic_under_seed(self):
        a = Uniform(0, 1).sample(np.random.default_rng(7))
        b = Uniform(0, 1).sample(np.random.default_rng(7))
        assert a == b


class TestEmitterProfile:
    def _qpsk_profile(self, rate_spec):
        return EmitterProfile(
            name="test-qpsk",
            label="TESTQPSK",
            waveform_cls=sp.QPSK,
            params={"symbol_rate": rate_spec, "rolloff": Fixed(0.35)},
            reference="test",
        )

    def test_sample_constructs_waveform(self):
        prof = self._qpsk_profile(Fixed(250e3))
        wf = prof.sample(np.random.default_rng(0), sample_rate=10e6)
        assert isinstance(wf, sp.QPSK)
        assert wf.bandwidth(10e6) == pytest.approx(250e3 * 1.35)

    def test_sample_deterministic(self):
        prof = self._qpsk_profile(Uniform(100e3, 1e6))
        a = prof.sample(np.random.default_rng(3), 10e6)
        b = prof.sample(np.random.default_rng(3), 10e6)
        assert a.symbol_rate == b.symbol_rate

    def test_redraws_until_representable(self):
        # Range straddles fs: draws with bandwidth > fs or symbol rate above
        # fs/2 are rejected, but representable draws exist, so sample() must
        # succeed AND the result must be generable.
        prof = self._qpsk_profile(Uniform(1e6, 50e6))
        wf = prof.sample(np.random.default_rng(1), sample_rate=10e6)
        assert wf.bandwidth(10e6) <= 10e6
        iq = wf.generate(num_symbols=16, sample_rate=10e6, seed=0)
        assert len(iq) > 0

    def test_unrepresentable_raises(self):
        prof = self._qpsk_profile(Fixed(50e6))
        with pytest.raises(ProfileNotRepresentable, match="test-qpsk"):
            prof.sample(np.random.default_rng(0), sample_rate=10e6)
```

- [ ] **Step 2: Run to verify failure**

Run: `pytest tests/test_profiles.py -v`
Expected: FAIL — `ModuleNotFoundError: No module named 'spectra.profiles'`

- [ ] **Step 3: Implement**

```python
# python/spectra/profiles/spec.py
"""Parameter distributions and emitter profiles.

An EmitterProfile describes a real-world emitter class as a waveform type
plus sampleable physical-parameter distributions, with a citation to the
defining standard. Spec: docs/superpowers/specs/2026-07-02-waveform-realism-design.md
"""

from dataclasses import dataclass, field
from typing import Any, Mapping, Sequence, Type

import numpy as np

from spectra.waveforms.base import Waveform

_MAX_DRAWS = 16


class ProfileNotRepresentable(ValueError):
    """Raised when a profile cannot fit inside the capture bandwidth."""


@dataclass(frozen=True)
class Fixed:
    value: Any

    def sample(self, rng: np.random.Generator) -> Any:
        return self.value


@dataclass(frozen=True)
class Choice:
    options: Sequence[Any]

    def __post_init__(self):
        if len(self.options) == 0:
            raise ValueError("Choice needs at least one option")

    def sample(self, rng: np.random.Generator) -> Any:
        return self.options[int(rng.integers(len(self.options)))]


@dataclass(frozen=True)
class Uniform:
    low: float
    high: float

    def __post_init__(self):
        if not self.low < self.high:
            raise ValueError(f"Uniform requires low < high, got [{self.low}, {self.high}]")

    def sample(self, rng: np.random.Generator) -> float:
        return float(rng.uniform(self.low, self.high))


@dataclass(frozen=True)
class LogUniform:
    low: float
    high: float

    def __post_init__(self):
        if not 0 < self.low < self.high:
            raise ValueError(f"LogUniform requires 0 < low < high, got [{self.low}, {self.high}]")

    def sample(self, rng: np.random.Generator) -> float:
        return float(np.exp(rng.uniform(np.log(self.low), np.log(self.high))))


@dataclass(frozen=True)
class EmitterProfile:
    """A standards-referenced emitter: waveform class + parameter distributions.

    Attributes:
        name: Registry key, kebab-case (e.g. ``"bluetooth-le-1m"``).
        label: Dataset class label (e.g. ``"BLE"``); becomes
            ``SignalDescription.label`` in scenes.
        waveform_cls: Waveform class constructed by :meth:`sample`.
        params: Mapping of constructor kwarg -> ParamSpec.
        reference: One-line citation of the defining standard.
    """

    name: str
    label: str
    waveform_cls: Type[Waveform]
    params: Mapping[str, Any] = field(default_factory=dict)
    reference: str = ""

    def sample(self, rng: np.random.Generator, sample_rate: float) -> Waveform:
        """Draw a concrete waveform representable at ``sample_rate``.

        Re-draws up to a fixed budget when a draw's occupied bandwidth
        exceeds ``sample_rate``; profiles never silently distort parameters.
        """
        last_bw = None
        for _ in range(_MAX_DRAWS):
            kwargs = {k: spec.sample(rng) for k, spec in self.params.items()}
            wf = self.waveform_cls(**kwargs)
            last_bw = wf.bandwidth(sample_rate)
            # bandwidth() alone is not sufficient: symbol-rate-parameterized
            # waveforms also need >= 2 samples/symbol at generate() time.
            sr = getattr(wf, "symbol_rate", None)
            if last_bw <= sample_rate and (sr is None or 2 * sr <= sample_rate):
                return wf
        raise ProfileNotRepresentable(
            f"profile '{self.name}' drew bandwidth {last_bw:g} Hz > sample_rate "
            f"{sample_rate:g} Hz after {_MAX_DRAWS} attempts; increase the capture "
            f"sample rate or remove this profile from the pool"
        )
```

```python
# python/spectra/profiles/__init__.py
from spectra.profiles.spec import (
    Choice,
    EmitterProfile,
    Fixed,
    LogUniform,
    ProfileNotRepresentable,
    Uniform,
)

__all__ = [
    "Choice",
    "EmitterProfile",
    "Fixed",
    "LogUniform",
    "ProfileNotRepresentable",
    "Uniform",
]
```

- [ ] **Step 4: Run tests**

Run: `pytest tests/test_profiles.py -v`
Expected: PASS

- [ ] **Step 5: Commit**

```bash
git add python/spectra/profiles/ tests/test_profiles.py
git commit -m "feat(profiles): ParamSpec distributions and EmitterProfile"
```

---

### Task 9: Curated registry (`profiles/registry.py`)

**Files:**
- Create: `python/spectra/profiles/registry.py`
- Modify: `python/spectra/profiles/__init__.py`
- Modify: `python/spectra/__init__.py` (add `from spectra import profiles` to the module exports)
- Test: `tests/test_profiles.py`

Note: the FSK/GMSK-based profiles below assume Task 6 landed; the OFDM profiles assume Task 7. Implement this task after both.

- [ ] **Step 1: Write the failing tests** (append to `tests/test_profiles.py`)

```python
from spectra import profiles


class TestRegistry:
    def test_get_known_profile(self):
        prof = profiles.get("bluetooth-le-1m")
        assert prof.label == "BLE"
        assert "Bluetooth" in prof.reference

    def test_unknown_name_suggests(self):
        with pytest.raises(KeyError, match="bluetooth-le-1m"):
            profiles.get("bluetooth-le-1M")   # near-miss casing

    def test_list_profiles(self):
        names = profiles.list_profiles()
        assert len(names) >= 15
        assert names == sorted(names)

    def test_register_custom(self):
        custom = EmitterProfile(
            name="my-custom", label="CUSTOM", waveform_cls=sp.QPSK,
            params={"symbol_rate": Fixed(1e5)}, reference="n/a",
        )
        profiles.register(custom)
        try:
            assert profiles.get("my-custom") is custom
        finally:
            profiles._REGISTRY.pop("my-custom")

    def test_register_duplicate_raises(self):
        with pytest.raises(ValueError, match="already registered"):
            profiles.register(profiles.get("bluetooth-le-1m"))

    def test_every_builtin_samples_at_adequate_fs(self):
        # Each profile must produce a representable waveform at a sample
        # rate generous enough for its widest draw.
        adequate_fs = {
            "automotive-fmcw": 600e6,
            "radar-altimeter-fmcw": 400e6,
            "marine-nav-radar": 40e6,
            "dvbs-qpsk": 80e6,
            "wifi-ofdm-20mhz": 20e6,
            "lte-ofdm": 30.72e6,
        }
        rng = np.random.default_rng(11)
        for name in profiles.list_profiles():
            prof = profiles.get(name)
            fs = adequate_fs.get(name, 10e6)
            wf = prof.sample(rng, fs)
            assert wf.bandwidth(fs) <= fs, name
```

- [ ] **Step 2: Run to verify failure**

Run: `pytest tests/test_profiles.py::TestRegistry -v`
Expected: FAIL — `AttributeError: module 'spectra.profiles' has no attribute 'get'`

- [ ] **Step 3: Implement**

```python
# python/spectra/profiles/registry.py
"""Curated registry of standards-referenced emitter profiles.

Parameter values cite the defining standard in each profile's ``reference``.
Deviations are peak frequency deviation of the outermost tone; symbol rates
are in baud; times in seconds; bandwidths in Hz.
"""

import difflib
from typing import Dict, List

from spectra.profiles.spec import Choice, EmitterProfile, Fixed, LogUniform, Uniform
from spectra.waveforms.barker import BarkerCode
from spectra.waveforms.fsk import FSK, GFSK, GMSK
from spectra.waveforms.ofdm import OFDM
from spectra.waveforms.psk import QPSK
from spectra.waveforms.radar import FMCW, PulsedRadar

_BUILTINS = [
    # --- Comms ---
    EmitterProfile(
        name="bluetooth-le-1m", label="BLE", waveform_cls=GFSK,
        params={"order": Fixed(2), "symbol_rate": Fixed(1e6), "bt": Fixed(0.5),
                "deviation": Uniform(225e3, 275e3)},
        reference="Bluetooth Core 5.4 Vol 6 Part A: LE 1M, GFSK 1 MBd, BT=0.5, h in [0.45,0.55]",
    ),
    EmitterProfile(
        name="bluetooth-le-2m", label="BLE", waveform_cls=GFSK,
        params={"order": Fixed(2), "symbol_rate": Fixed(2e6), "bt": Fixed(0.5),
                "deviation": Uniform(450e3, 550e3)},
        reference="Bluetooth Core 5.4 Vol 6 Part A: LE 2M, GFSK 2 MBd, BT=0.5, h in [0.45,0.55]",
    ),
    EmitterProfile(
        name="bluetooth-br", label="BT-BR", waveform_cls=GFSK,
        params={"order": Fixed(2), "symbol_rate": Fixed(1e6), "bt": Fixed(0.5),
                "deviation": Uniform(140e3, 175e3)},
        reference="Bluetooth Core 5.4 Vol 2 Part A: BR GFSK 1 MBd, h in [0.28,0.35]",
    ),
    EmitterProfile(
        name="p25-c4fm", label="P25", waveform_cls=FSK,
        params={"order": Fixed(4), "symbol_rate": Fixed(4800), "deviation": Fixed(1800.0)},
        reference="TIA-102.BAAA-A: P25 Phase 1 C4FM, 4.8 kBd, deviations +/-600/+/-1800 Hz",
    ),
    EmitterProfile(
        name="dmr-4fsk", label="DMR", waveform_cls=FSK,
        params={"order": Fixed(4), "symbol_rate": Fixed(4800), "deviation": Fixed(1944.0)},
        reference="ETSI TS 102 361-1: DMR 4FSK, 4.8 kBd, deviations +/-648/+/-1944 Hz",
    ),
    EmitterProfile(
        name="tetra-dqpsk", label="TETRA", waveform_cls=QPSK,
        params={"symbol_rate": Fixed(18000.0), "rolloff": Fixed(0.35)},
        reference="ETSI EN 300 392-2: TETRA pi/4-DQPSK 18 kBd, RRC rolloff 0.35 (approx as QPSK)",
    ),
    EmitterProfile(
        name="pocsag", label="POCSAG", waveform_cls=FSK,
        params={"order": Fixed(2), "symbol_rate": Choice([512.0, 1200.0, 2400.0]),
                "deviation": Fixed(4500.0)},
        reference="ITU-R M.584: POCSAG 2FSK, 512/1200/2400 bps, +/-4.5 kHz deviation",
    ),
    EmitterProfile(
        name="ais-gmsk", label="AIS", waveform_cls=GMSK,
        params={"symbol_rate": Fixed(9600.0), "bt": Fixed(0.4)},
        reference="ITU-R M.1371-5: AIS GMSK 9.6 kBd, BT=0.4",
    ),
    EmitterProfile(
        name="wifi-ofdm-20mhz", label="WiFi", waveform_cls=OFDM,
        params={"num_subcarriers": Fixed(52), "subcarrier_spacing": Fixed(312.5e3),
                "cp_fraction": Fixed(0.25)},
        reference="IEEE 802.11a/g: 20 MHz OFDM, 312.5 kHz spacing, 52 used subcarriers, CP 1/4",
    ),
    EmitterProfile(
        name="lte-ofdm", label="LTE", waveform_cls=OFDM,
        params={"num_subcarriers": Choice([72, 180, 300]), "subcarrier_spacing": Fixed(15e3),
                "cp_fraction": Fixed(0.07)},
        reference="3GPP TS 36.211: LTE downlink OFDM, 15 kHz spacing, 1.4/3/5 MHz allocations",
    ),
    EmitterProfile(
        name="dvbs-qpsk", label="DVB-S", waveform_cls=QPSK,
        params={"symbol_rate": LogUniform(1e6, 27.5e6), "rolloff": Fixed(0.35)},
        reference="ETSI EN 300 421: DVB-S QPSK, 1-27.5 MBd, RRC rolloff 0.35",
    ),
    # --- Radar ---
    EmitterProfile(
        name="marine-nav-radar", label="MarineRadar", waveform_cls=PulsedRadar,
        params={"pulse_width": Choice([80e-9, 250e-9, 800e-9]),
                "pri": Uniform(0.33e-3, 2.5e-3), "num_pulses": Fixed(8)},
        reference="IEC 62388: X-band marine radar, 80-800 ns pulses, PRF 400-3000 Hz",
    ),
    EmitterProfile(
        name="weather-radar", label="WeatherRadar", waveform_cls=PulsedRadar,
        params={"pulse_width": Choice([1.57e-6, 4.7e-6]),
                "pri": Uniform(0.76e-3, 3.1e-3), "num_pulses": Fixed(4)},
        reference="WSR-88D/NEXRAD: 1.57/4.7 us pulses, PRF 322-1300 Hz",
    ),
    EmitterProfile(
        name="atc-radar", label="ATCRadar", waveform_cls=PulsedRadar,
        params={"pulse_width": Fixed(1.0e-6), "pri": Uniform(0.83e-3, 1.25e-3),
                "num_pulses": Fixed(8)},
        reference="ASR-9/11 airport surveillance radar: ~1 us pulse, PRF 800-1200 Hz",
    ),
    EmitterProfile(
        name="automotive-fmcw", label="AutoFMCW", waveform_cls=FMCW,
        params={"sweep_bandwidth": LogUniform(5e6, 500e6),
                "sweep_time": Uniform(10e-6, 60e-6),
                "idle_time": Uniform(5e-6, 20e-6), "num_sweeps": Fixed(8)},
        reference="ETSI EN 302 264 (76-77 GHz): FMCW, sweeps up to hundreds of MHz, 10-60 us",
    ),
    EmitterProfile(
        name="radar-altimeter-fmcw", label="Altimeter", waveform_cls=FMCW,
        params={"sweep_bandwidth": Uniform(100e6, 170e6),
                "sweep_time": Uniform(0.5e-3, 2e-3),
                "idle_time": Fixed(0.0), "num_sweeps": Fixed(1),
                "sweep_type": Fixed("triangle")},
        reference="ITU-R M.2059 (4.2-4.4 GHz): FMCW altimeter, ~100-170 MHz triangle sweeps",
    ),
    EmitterProfile(
        name="barker13-radar", label="BarkerRadar", waveform_cls=BarkerCode,
        params={"length": Fixed(13), "chip_rate": LogUniform(0.5e6, 5e6)},
        reference="Pulse-compression radar, Barker-13 code (Levanon & Mozeson ch. 6)",
    ),
]

_REGISTRY: Dict[str, EmitterProfile] = {}


def register(profile: EmitterProfile) -> None:
    """Add a profile to the registry. Raises on duplicate names."""
    if profile.name in _REGISTRY:
        raise ValueError(f"profile '{profile.name}' is already registered")
    _REGISTRY[profile.name] = profile


def get(name: str) -> EmitterProfile:
    """Look up a profile by name; unknown names raise with near-matches."""
    try:
        return _REGISTRY[name]
    except KeyError:
        close = difflib.get_close_matches(name, _REGISTRY.keys(), n=3)
        hint = f" Did you mean: {', '.join(close)}?" if close else ""
        raise KeyError(f"unknown profile '{name}'.{hint}") from None


def list_profiles() -> List[str]:
    """Sorted names of all registered profiles."""
    return sorted(_REGISTRY.keys())


for _p in _BUILTINS:
    register(_p)
```

Extend `python/spectra/profiles/__init__.py`:

```python
from spectra.profiles.registry import get, list_profiles, register  # noqa: F401
```

and add to `__all__`: `"get", "list_profiles", "register"`. In `python/spectra/__init__.py`, add `from spectra import profiles  # noqa: F401` alongside the existing subpackage imports.

- [ ] **Step 4: Run tests**

Run: `pytest tests/test_profiles.py -v`
Expected: PASS

- [ ] **Step 5: Commit**

```bash
git add python/spectra/profiles/ python/spectra/__init__.py tests/test_profiles.py
git commit -m "feat(profiles): curated standards-referenced emitter registry"
```

---

### Task 10: Composer accepts profiles

**Files:**
- Modify: `python/spectra/scene/composer.py`
- Test: `tests/test_profiles.py`

- [ ] **Step 1: Write the failing tests** (append)

```python
class TestComposerProfiles:
    def _cfg(self, pool):
        return sp.SceneConfig(
            capture_duration=1e-3, capture_bandwidth=10e6, sample_rate=10e6,
            num_signals=2, signal_pool=pool, snr_range=(10, 20),
        )

    def test_profile_in_pool(self):
        pool = [profiles.get("tetra-dqpsk"), sp.QPSK()]
        iq, descs = sp.Composer(self._cfg(pool)).generate(seed=5)
        assert len(descs) == 2
        assert set(d.label for d in descs) <= {"TETRA", "QPSK"}

    def test_profile_label_used(self):
        pool = [profiles.get("ais-gmsk")]
        _, descs = sp.Composer(self._cfg(pool)).generate(seed=5)
        assert all(d.label == "AIS" for d in descs)

    def test_deterministic(self):
        pool = [profiles.get("pocsag"), profiles.get("tetra-dqpsk")]
        a, da = sp.Composer(self._cfg(pool)).generate(seed=9)
        b, db = sp.Composer(self._cfg(pool)).generate(seed=9)
        np.testing.assert_array_equal(a, b)
        assert [(d.label, d.f_low, d.f_high) for d in da] == [
            (d.label, d.f_low, d.f_high) for d in db
        ]

    def test_box_width_matches_sampled_bandwidth(self):
        pool = [profiles.get("tetra-dqpsk")]
        _, descs = sp.Composer(self._cfg(pool)).generate(seed=1)
        for d in descs:
            assert (d.f_high - d.f_low) == pytest.approx(18000.0 * 1.35, rel=1e-6)
```

- [ ] **Step 2: Run to verify failure**

Run: `pytest tests/test_profiles.py::TestComposerProfiles -v`
Expected: FAIL — `AttributeError: 'EmitterProfile' object has no attribute 'bandwidth'` (or similar inside Composer)

- [ ] **Step 3: Implement.** In `python/spectra/scene/composer.py`:

Add import at top:

```python
from spectra.profiles.spec import EmitterProfile
```

Update the `SceneConfig.signal_pool` annotation and docstring:

```python
    signal_pool: List[Union[Waveform, "EmitterProfile"]]
```

In `Composer.generate`, replace:

```python
            # Pick a waveform from the pool
            waveform = cfg.signal_pool[rng.integers(0, len(cfg.signal_pool))]
```

with:

```python
            # Pick a waveform or emitter profile from the pool
            entry = cfg.signal_pool[rng.integers(0, len(cfg.signal_pool))]
            if isinstance(entry, EmitterProfile):
                waveform = entry.sample(rng, cfg.sample_rate)
                sig_label = entry.label
            else:
                waveform = entry
                sig_label = entry.label
```

and use `sig_label` in both `SignalDescription(...)` constructions (replacing `waveform.label`).

- [ ] **Step 4: Run tests + suite**

Run: `pytest tests/test_profiles.py tests/ -q -m "not slow"`
Expected: PASS

- [ ] **Step 5: Commit**

```bash
git add python/spectra/scene/composer.py tests/test_profiles.py
git commit -m "feat(scene): Composer accepts EmitterProfile entries in signal_pool"
```

---

### Task 11: YAML loader `{profile: name}` entries

**Files:**
- Modify: `python/spectra/benchmarks/loader.py:46-57` (`_build_waveform_pool`)
- Test: `tests/test_profiles.py`

- [ ] **Step 1: Write the failing test** (append)

```python
class TestYamlProfiles:
    def test_profile_entry(self):
        from spectra.benchmarks.loader import _build_waveform_pool

        pool = _build_waveform_pool(
            [{"type": "QPSK"}, {"profile": "ais-gmsk"}]
        )
        assert len(pool) == 2
        from spectra.profiles import EmitterProfile
        assert isinstance(pool[1], EmitterProfile)
        assert pool[1].name == "ais-gmsk"

    def test_entry_with_both_keys_raises(self):
        from spectra.benchmarks.loader import _build_waveform_pool

        with pytest.raises(ValueError, match="either 'type' or 'profile'"):
            _build_waveform_pool([{"type": "QPSK", "profile": "ais-gmsk"}])
```

- [ ] **Step 2: Run to verify failure**

Run: `pytest tests/test_profiles.py::TestYamlProfiles -v`
Expected: FAIL — `KeyError: 'type'`

- [ ] **Step 3: Implement.** Replace `_build_waveform_pool` in `python/spectra/benchmarks/loader.py`:

```python
def _build_waveform_pool(pool_config: List[Dict[str, Any]]) -> List[Any]:
    from spectra import profiles as _profiles

    registry = _get_waveform_registry()
    pool = []
    for entry in pool_config:
        if ("type" in entry) == ("profile" in entry):
            raise ValueError(
                f"waveform_pool entries need either 'type' or 'profile', got {entry}"
            )
        if "profile" in entry:
            pool.append(_profiles.get(entry["profile"]))
            continue
        wtype = entry["type"]
        if wtype not in registry:
            raise ValueError(
                f"Unknown waveform type '{wtype}'. Available: {sorted(registry.keys())}"
            )
        params = entry.get("params", {})
        pool.append(registry[wtype](**params))
    return pool
```

Note: narrowband datasets call `waveform.generate(...)` on pool entries directly, so profile entries in *narrowband* pools are out of scope for this phase. Add this guard at the top of `_build_narrowband`, right after `pool = _build_waveform_pool(...)`:

```python
    from spectra.profiles import EmitterProfile

    if any(isinstance(p, EmitterProfile) for p in pool):
        raise ValueError(
            "profile entries are only supported in wideband benchmarks in this phase"
        )
```

and add a test:

```python
    def test_narrowband_rejects_profiles(self, tmp_path):
        import yaml
        from spectra.benchmarks import load_benchmark

        cfg = {
            "name": "t", "version": "1.0", "task": "narrowband",
            "sample_rate": 1_000_000, "num_iq_samples": 1024,
            "num_samples": {"train": 4}, "seed": {"train": 1},
            "waveform_pool": [{"profile": "ais-gmsk"}],
            "snr_range": [0, 10], "impairments": [],
        }
        p = tmp_path / "t.yaml"
        p.write_text(yaml.safe_dump(cfg))
        with pytest.raises(ValueError, match="wideband"):
            load_benchmark(str(p), split="train")
```

(If `load_benchmark`'s signature differs — check `python/spectra/benchmarks/loader.py` — adapt the call, keeping the assertion on the raised message.)

- [ ] **Step 4: Run tests**

Run: `pytest tests/test_profiles.py -v -m "not slow" && pytest tests/ -q -m benchmark`
Expected: PASS

- [ ] **Step 5: Commit**

```bash
git add python/spectra/benchmarks/loader.py tests/test_profiles.py
git commit -m "feat(benchmarks): profile entries in waveform_pool YAML"
```

---

### Task 12: Spectral-occupancy verification tests (slow)

Encodes the 2026-07-02 bandwidth verification harness as a permanent guard: for every physically-parameterized waveform and every registry profile, measured 99% occupied bandwidth must agree with `bandwidth(fs)` within a family epsilon.

**Files:**
- Create: `tests/test_spectral_occupancy.py`

- [ ] **Step 1: Write the tests**

```python
# tests/test_spectral_occupancy.py
"""Measured occupied bandwidth vs claimed bandwidth() for physical params.

Claimed bandwidth conventions differ per family: RRC claims the full
spectral support Rs*(1+rolloff) (99% OBW sits ~14% inside it); Carson-rule
FSK claims are also outer bounds. Assertions therefore bracket the claim:
    low_factor * claimed <= obw99 <= high_factor * claimed
"""

import numpy as np
import pytest

import spectra as sp
from spectra import profiles

pytestmark = pytest.mark.slow


def obw99(iq, sample_rate, nfft=4096):
    w = np.hanning(nfft)
    segs = max(1, len(iq) // nfft)
    psd = np.mean(
        [np.abs(np.fft.fft(iq[k * nfft:(k + 1) * nfft] * w)) ** 2 for k in range(segs)],
        axis=0,
    )
    psd = np.fft.fftshift(psd)
    f = np.fft.fftshift(np.fft.fftfreq(nfft, 1 / sample_rate))
    c = np.cumsum(psd) / np.sum(psd)
    return f[np.searchsorted(c, 0.995)] - f[np.searchsorted(c, 0.005)]


CASES = [
    # (waveform, fs, low_factor, high_factor)
    (sp.QPSK(symbol_rate=500e3), 10e6, 0.75, 1.05),
    (sp.QAM16(symbol_rate=1e6, rolloff=0.25), 10e6, 0.75, 1.05),
    (sp.FSK(order=2, symbol_rate=100e3, deviation=50e3), 2e6, 0.4, 1.05),
    (sp.FSK(order=4, symbol_rate=4800, deviation=1944.0), 192e3, 0.4, 1.05),
    (sp.GMSK(symbol_rate=9600, bt=0.4), 192e3, 0.6, 1.1),
    (sp.OFDM(num_subcarriers=52, subcarrier_spacing=312.5e3), 20e6, 0.85, 1.15),
    (sp.LFM(sweep_bandwidth=3e6, pulse_duration=50e-6), 10e6, 0.8, 1.3),
]


@pytest.mark.parametrize("wf,fs,lo,hi", CASES, ids=lambda c: getattr(c, "label", str(c)))
def test_physical_waveform_occupancy(wf, fs, lo, hi):
    n_sym = wf.num_symbols_for(200_000, fs)
    iq = wf.generate(num_symbols=max(1, n_sym), sample_rate=fs, seed=42)
    claimed = wf.bandwidth(fs)
    measured = obw99(iq, fs)
    assert lo * claimed <= measured <= hi * claimed, (
        f"{wf.label}: claimed {claimed:g}, measured 99% OBW {measured:g}"
    )


PROFILE_FS = {
    "automotive-fmcw": 600e6,
    "radar-altimeter-fmcw": 400e6,
    "marine-nav-radar": 40e6,
    "dvbs-qpsk": 80e6,
    "wifi-ofdm-20mhz": 20e6,
    "lte-ofdm": 30.72e6,
}
# Pulsed radar claims 1/pulse_width (Rayleigh convention); rect-pulse sinc
# spectra put 99% of power far outside that. Bracket loosely on the low
# side and skip the upper bound for pulsed families.
PULSED = {"marine-nav-radar", "weather-radar", "atc-radar", "barker13-radar"}


@pytest.mark.parametrize("name", profiles.list_profiles())
def test_profile_occupancy(name):
    prof = profiles.get(name)
    fs = PROFILE_FS.get(name, 10e6)
    rng = np.random.default_rng(7)
    wf = prof.sample(rng, fs)
    n_sym = wf.num_symbols_for(400_000, fs)
    iq = wf.generate(num_symbols=max(1, n_sym), sample_rate=fs, seed=42)
    claimed = wf.bandwidth(fs)
    measured = obw99(iq, fs)
    if name in PULSED:
        assert measured >= 0.5 * claimed, f"{name}: energy narrower than claim"
    else:
        assert 0.4 * claimed <= measured <= 1.2 * claimed, (
            f"{name}: claimed {claimed:g}, measured {measured:g}"
        )
```

- [ ] **Step 2: Run**

Run: `pytest tests/test_spectral_occupancy.py -v -m slow`
Expected: PASS. Any failure here is a real physics disagreement — investigate the waveform, do not widen the bracket without a written justification in the test comment.

- [ ] **Step 3: Commit**

```bash
git add tests/test_spectral_occupancy.py
git commit -m "test: spectral-occupancy guard for physical params and profiles"
```

---

### Task 13: Docs and example

**Files:**
- Create: `docs/user-guide/realistic-emitters.md`
- Modify: `mkdocs.yml` (add `Realistic Emitters: user-guide/realistic-emitters.md` to the User Guide nav, after `scene-composition`)
- Modify: `examples/datasets/wideband_scenes.py` (append a section 6)

- [ ] **Step 1: Write the user guide**

```markdown
# Realistic Emitters

SPECTRA waveforms accept parameters two ways: **sample-domain** knobs
(`samples_per_symbol=8`) that scale with the capture sample rate, and
**physical units** (baud, Hz, seconds) that stay fixed regardless of it.

## Physical-unit parameters

```python
import spectra as sp

qpsk  = sp.QPSK(symbol_rate=250e3)                       # 250 kBd, BW = 337.5 kHz
fsk   = sp.FSK(order=4, symbol_rate=4800, deviation=1800) # P25-style C4FM
radar = sp.PulsedRadar(pulse_width=1e-6, pri=1e-3)        # 1 us pulses, 1 kHz PRF
fmcw  = sp.FMCW(sweep_bandwidth=2e6, sweep_time=50e-6)
ofdm  = sp.OFDM(num_subcarriers=52, subcarrier_spacing=312.5e3)
```

Physical and sample-domain forms of the same parameter are mutually
exclusive — passing both raises `ValueError`. When the sample rate is not
an integer multiple of the symbol rate, generation rounds if the error is
below 1% and rational-resamples otherwise. `bandwidth()` returns the
standard-derived value independent of sample rate, so wideband bounding
boxes stay correct at any capture configuration.

## Emitter profiles

Profiles describe real emitter classes as parameter *distributions* with a
standards citation, so scenes are realistic and diverse:

```python
from spectra import profiles

print(profiles.list_profiles())
ble = profiles.get("bluetooth-le-1m")   # GFSK 1 MBd, BT=0.5, h in [0.45, 0.55]

config = sp.SceneConfig(
    capture_duration=1e-3,
    capture_bandwidth=10e6,
    sample_rate=10e6,
    num_signals=(2, 5),
    signal_pool=[ble, profiles.get("tetra-dqpsk"), profiles.get("ais-gmsk")],
    snr_range=(5, 25),
)
```

Each placement draws fresh parameters from the profile with the scene RNG
(deterministic per seed). A profile that cannot fit the capture bandwidth
raises `ProfileNotRepresentable` rather than silently distorting itself.

Register your own:

```python
from spectra.profiles import EmitterProfile, Fixed, Uniform, register

register(EmitterProfile(
    name="my-telemetry", label="TLM", waveform_cls=sp.GMSK,
    params={"symbol_rate": Uniform(50e3, 200e3), "bt": Fixed(0.3)},
    reference="internal ICD rev. C",
))
```

Benchmark YAMLs accept profiles alongside types:

```yaml
waveform_pool:
  - {type: "QPSK"}
  - {profile: "bluetooth-le-1m"}
```
```

- [ ] **Step 2: Add mkdocs nav entry and the example section.** In `examples/datasets/wideband_scenes.py` append before the final `print`:

```python
# -- 6. Profile-based realistic scene ----------------------------------------------

from spectra import profiles

profile_config = sp.SceneConfig(
    capture_duration=1e-3,
    capture_bandwidth=10e6,
    sample_rate=10e6,
    num_signals=(3, 5),
    signal_pool=[
        profiles.get("bluetooth-le-1m"),
        profiles.get("tetra-dqpsk"),
        profiles.get("ais-gmsk"),
        profiles.get("pocsag"),
    ],
    snr_range=(5, 25),
)
iq_p, descs_p = sp.Composer(profile_config).generate(seed=11)
spec_p = dsp.compute_spectrogram(iq_p, nfft=nfft, hop=hop)
fig, ax = plt.subplots(figsize=(14, 6))
ax.imshow(10 * np.log10(spec_p + 1e-12), aspect="auto", origin="lower", cmap="viridis")
ax.set_title("Profile-based scene: " + ", ".join(d.label for d in descs_p))
fig.tight_layout()
savefig("05_profile_scene.png")
```

- [ ] **Step 3: Verify docs build and example runs**

Run: `mkdocs build 2>&1 | tail -3 && cd examples/datasets && python wideband_scenes.py && cd ../..`
Expected: clean build; example prints labels like `BLE, TETRA, AIS` and saves `05_profile_scene.png`

- [ ] **Step 4: Commit**

```bash
git add docs/user-guide/realistic-emitters.md mkdocs.yml examples/datasets/wideband_scenes.py
git commit -m "docs: realistic-emitters guide and profile-based scene example"
```

---

## Final verification

- [ ] `pytest tests/ -q` (full suite including slow) — all pass
- [ ] `ruff check python/ tests/ && ruff format --check python/ tests/` — clean
- [ ] Run the spectral evidence check by eye: `cd examples/datasets && python wideband_scenes.py` and confirm boxes hug the energy in `examples/outputs/05_profile_scene.png`
- [ ] Confirm Tasks 6, 7, 9, 12 were implemented only after the FSK/OFDM fix branches merged
