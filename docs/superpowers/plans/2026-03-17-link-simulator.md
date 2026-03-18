# Link-Level Simulator Implementation Plan

> **For agentic workers:** REQUIRED: Use superpowers:subagent-driven-development (if subagents available) or superpowers:executing-plans to implement this plan. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Add a `LinkSimulator` that generates BER/SER/PER vs. Eb/N0 curves by running transmitter → channel → coherent receiver loops.

**Architecture:** Four sub-projects: (1) Rust modulator extensions returning symbol indices + constellation access functions, (2) BER/SER/PER metric functions, (3) coherent receiver with FEC decoder stubs, (4) LinkSimulator orchestration class. Direct analytical noise injection bypasses the AWGN transform for correct Eb/N0 handling.

**Tech Stack:** Rust (PyO3), Python 3.10+, NumPy, pytest. No new dependencies.

**Spec:** `docs/superpowers/specs/2026-03-17-link-simulator-design.md`

---

## File Structure

| File | Responsibility |
|------|---------------|
| `rust/src/modulators.rs` | Modify: add `_with_indices` generators + `get_*_constellation` functions |
| `rust/src/lib.rs` | Modify: register new Rust functions |
| `python/spectra/metrics.py` | Modify: add `bit_error_rate`, `symbol_error_rate`, `packet_error_rate` |
| `python/spectra/receivers/__init__.py` | Package init — exports |
| `python/spectra/receivers/base.py` | Receiver ABC, Decoder ABC, PassthroughDecoder, ViterbiDecoder/LDPCDecoder stubs |
| `python/spectra/receivers/coherent.py` | CoherentReceiver: matched filter + slicer + demapper |
| `python/spectra/link/__init__.py` | Package init — exports |
| `python/spectra/link/simulator.py` | LinkSimulator class |
| `python/spectra/link/results.py` | LinkResults dataclass |
| `tests/test_link_metrics.py` | Tests for BER/SER/PER metric functions |
| `tests/test_receivers.py` | Tests for CoherentReceiver + decoder stubs |
| `tests/test_link_simulator.py` | Tests for LinkSimulator end-to-end |

---

## Task 1: Rust Modulator Extensions

**Files:**
- Modify: `rust/src/modulators.rs`
- Modify: `rust/src/lib.rs`

- [ ] **Step 1: Add `_with_indices` generator functions to `modulators.rs`**

Add these functions after the existing generators. Each reuses the same `Xorshift64` PRNG and constellation logic, but returns `(symbols, indices)`:

```rust
// ── Symbol generators with indices ─────────────────────────────────────

#[pyfunction]
pub fn generate_bpsk_symbols_with_indices<'py>(
    py: Python<'py>,
    num_symbols: usize,
    seed: u64,
) -> (Bound<'py, PyArray1<Complex32>>, Bound<'py, PyArray1<u32>>) {
    let mut rng = Xorshift64::new(seed);
    let mut symbols = Vec::with_capacity(num_symbols);
    let mut indices = Vec::with_capacity(num_symbols);
    for _ in 0..num_symbols {
        let idx = (rng.next() % 2) as u32;
        indices.push(idx);
        if idx == 0 {
            symbols.push(Complex32::new(1.0, 0.0));
        } else {
            symbols.push(Complex32::new(-1.0, 0.0));
        }
    }
    (
        Array1::from_vec(symbols).into_pyarray(py),
        Array1::from_vec(indices).into_pyarray(py),
    )
}

#[pyfunction]
pub fn generate_qpsk_symbols_with_indices<'py>(
    py: Python<'py>,
    num_symbols: usize,
    seed: u64,
) -> (Bound<'py, PyArray1<Complex32>>, Bound<'py, PyArray1<u32>>) {
    let mut rng = Xorshift64::new(seed);
    let mut symbols = Vec::with_capacity(num_symbols);
    let mut indices = Vec::with_capacity(num_symbols);
    for _ in 0..num_symbols {
        let idx = (rng.next() % 4) as u32;
        indices.push(idx);
        symbols.push(QPSK_CONSTELLATION[idx as usize]);
    }
    (
        Array1::from_vec(symbols).into_pyarray(py),
        Array1::from_vec(indices).into_pyarray(py),
    )
}

#[pyfunction]
pub fn generate_psk_symbols_with_indices<'py>(
    py: Python<'py>,
    num_symbols: usize,
    order: usize,
    seed: u64,
) -> PyResult<(Bound<'py, PyArray1<Complex32>>, Bound<'py, PyArray1<u32>>)> {
    if order < 2 {
        return Err(pyo3::exceptions::PyValueError::new_err(
            "PSK order must be at least 2",
        ));
    }
    let constellation: Vec<Complex32> = (0..order)
        .map(|k| {
            let angle = 2.0 * std::f64::consts::PI * k as f64 / order as f64;
            Complex32::new(angle.cos() as f32, angle.sin() as f32)
        })
        .collect();
    let mut rng = Xorshift64::new(seed);
    let mut symbols = Vec::with_capacity(num_symbols);
    let mut indices = Vec::with_capacity(num_symbols);
    for _ in 0..num_symbols {
        let idx = (rng.next() as usize) % order;
        indices.push(idx as u32);
        symbols.push(constellation[idx]);
    }
    Ok((
        Array1::from_vec(symbols).into_pyarray(py),
        Array1::from_vec(indices).into_pyarray(py),
    ))
}

#[pyfunction]
pub fn generate_qam_symbols_with_indices<'py>(
    py: Python<'py>,
    num_symbols: usize,
    order: usize,
    seed: u64,
) -> PyResult<(Bound<'py, PyArray1<Complex32>>, Bound<'py, PyArray1<u32>>)> {
    let side = (order as f64).sqrt() as usize;
    if side * side != order {
        return Err(pyo3::exceptions::PyValueError::new_err(
            "QAM order must be a perfect square (16, 64, 256, ...)",
        ));
    }
    let mut constellation = Vec::with_capacity(order);
    for i in 0..side {
        for j in 0..side {
            let re = 2.0 * i as f64 - (side - 1) as f64;
            let im = 2.0 * j as f64 - (side - 1) as f64;
            constellation.push(Complex32::new(re as f32, im as f32));
        }
    }
    let avg_power: f64 = constellation
        .iter()
        .map(|c| (c.re * c.re + c.im * c.im) as f64)
        .sum::<f64>()
        / order as f64;
    let scale = 1.0 / avg_power.sqrt() as f32;
    for c in &mut constellation {
        c.re *= scale;
        c.im *= scale;
    }
    let mut rng = Xorshift64::new(seed);
    let mut symbols = Vec::with_capacity(num_symbols);
    let mut indices = Vec::with_capacity(num_symbols);
    for _ in 0..num_symbols {
        let idx = (rng.next() as usize) % order;
        indices.push(idx as u32);
        symbols.push(constellation[idx]);
    }
    Ok((
        Array1::from_vec(symbols).into_pyarray(py),
        Array1::from_vec(indices).into_pyarray(py),
    ))
}

#[pyfunction]
pub fn generate_ask_symbols_with_indices<'py>(
    py: Python<'py>,
    num_symbols: usize,
    order: usize,
    seed: u64,
) -> PyResult<(Bound<'py, PyArray1<Complex32>>, Bound<'py, PyArray1<u32>>)> {
    if order < 2 {
        return Err(pyo3::exceptions::PyValueError::new_err(
            "ASK order must be at least 2",
        ));
    }
    let levels: Vec<f64> = (0..order).map(|k| k as f64).collect();
    let avg_power: f64 = levels.iter().map(|l| l * l).sum::<f64>() / order as f64;
    let scale = if avg_power > 0.0 {
        1.0 / avg_power.sqrt()
    } else {
        1.0
    };
    let constellation: Vec<Complex32> = levels
        .iter()
        .map(|l| Complex32::new((l * scale) as f32, 0.0))
        .collect();
    let mut rng = Xorshift64::new(seed);
    let mut symbols = Vec::with_capacity(num_symbols);
    let mut indices = Vec::with_capacity(num_symbols);
    for _ in 0..num_symbols {
        let idx = (rng.next() as usize) % order;
        indices.push(idx as u32);
        symbols.push(constellation[idx]);
    }
    Ok((
        Array1::from_vec(symbols).into_pyarray(py),
        Array1::from_vec(indices).into_pyarray(py),
    ))
}
```

- [ ] **Step 2: Add constellation access functions to `modulators.rs`**

```rust
// ── Constellation access ───────────────────────────────────────────────

#[pyfunction]
pub fn get_bpsk_constellation<'py>(
    py: Python<'py>,
) -> Bound<'py, PyArray1<Complex32>> {
    Array1::from_vec(vec![
        Complex32::new(1.0, 0.0),
        Complex32::new(-1.0, 0.0),
    ]).into_pyarray(py)
}

#[pyfunction]
pub fn get_qpsk_constellation<'py>(
    py: Python<'py>,
) -> Bound<'py, PyArray1<Complex32>> {
    Array1::from_vec(QPSK_CONSTELLATION.to_vec()).into_pyarray(py)
}

#[pyfunction]
pub fn get_psk_constellation<'py>(
    py: Python<'py>,
    order: usize,
) -> PyResult<Bound<'py, PyArray1<Complex32>>> {
    if order < 2 {
        return Err(pyo3::exceptions::PyValueError::new_err(
            "PSK order must be at least 2",
        ));
    }
    let constellation: Vec<Complex32> = (0..order)
        .map(|k| {
            let angle = 2.0 * std::f64::consts::PI * k as f64 / order as f64;
            Complex32::new(angle.cos() as f32, angle.sin() as f32)
        })
        .collect();
    Ok(Array1::from_vec(constellation).into_pyarray(py))
}

#[pyfunction]
pub fn get_qam_constellation<'py>(
    py: Python<'py>,
    order: usize,
) -> PyResult<Bound<'py, PyArray1<Complex32>>> {
    let side = (order as f64).sqrt() as usize;
    if side * side != order {
        return Err(pyo3::exceptions::PyValueError::new_err(
            "QAM order must be a perfect square",
        ));
    }
    let mut constellation = Vec::with_capacity(order);
    for i in 0..side {
        for j in 0..side {
            let re = 2.0 * i as f64 - (side - 1) as f64;
            let im = 2.0 * j as f64 - (side - 1) as f64;
            constellation.push(Complex32::new(re as f32, im as f32));
        }
    }
    let avg_power: f64 = constellation
        .iter()
        .map(|c| (c.re * c.re + c.im * c.im) as f64)
        .sum::<f64>()
        / order as f64;
    let scale = 1.0 / avg_power.sqrt() as f32;
    for c in &mut constellation {
        c.re *= scale;
        c.im *= scale;
    }
    Ok(Array1::from_vec(constellation).into_pyarray(py))
}

#[pyfunction]
pub fn get_ask_constellation<'py>(
    py: Python<'py>,
    order: usize,
) -> PyResult<Bound<'py, PyArray1<Complex32>>> {
    if order < 2 {
        return Err(pyo3::exceptions::PyValueError::new_err(
            "ASK order must be at least 2",
        ));
    }
    let levels: Vec<f64> = (0..order).map(|k| k as f64).collect();
    let avg_power: f64 = levels.iter().map(|l| l * l).sum::<f64>() / order as f64;
    let scale = if avg_power > 0.0 {
        1.0 / avg_power.sqrt()
    } else {
        1.0
    };
    let constellation: Vec<Complex32> = levels
        .iter()
        .map(|l| Complex32::new((l * scale) as f32, 0.0))
        .collect();
    Ok(Array1::from_vec(constellation).into_pyarray(py))
}
```

- [ ] **Step 3: Register all new functions in `lib.rs`**

Add to the `#[pymodule]` block alongside the existing modulator registrations:

```rust
// Symbol generators with indices
m.add_function(wrap_pyfunction!(modulators::generate_bpsk_symbols_with_indices, m)?)?;
m.add_function(wrap_pyfunction!(modulators::generate_qpsk_symbols_with_indices, m)?)?;
m.add_function(wrap_pyfunction!(modulators::generate_psk_symbols_with_indices, m)?)?;
m.add_function(wrap_pyfunction!(modulators::generate_qam_symbols_with_indices, m)?)?;
m.add_function(wrap_pyfunction!(modulators::generate_ask_symbols_with_indices, m)?)?;
// Constellation access
m.add_function(wrap_pyfunction!(modulators::get_bpsk_constellation, m)?)?;
m.add_function(wrap_pyfunction!(modulators::get_qpsk_constellation, m)?)?;
m.add_function(wrap_pyfunction!(modulators::get_psk_constellation, m)?)?;
m.add_function(wrap_pyfunction!(modulators::get_qam_constellation, m)?)?;
m.add_function(wrap_pyfunction!(modulators::get_ask_constellation, m)?)?;
```

- [ ] **Step 4: Build and verify Rust compiles**

```bash
maturin develop --release --manifest-path rust/Cargo.toml
```
Expected: builds without errors.

- [ ] **Step 5: Quick smoke test from Python**

```bash
/Users/gditzler/.venvs/base/bin/python -c "
from spectra._rust import (
    generate_bpsk_symbols_with_indices,
    generate_qpsk_symbols_with_indices,
    get_bpsk_constellation,
    get_qpsk_constellation,
    get_qam_constellation,
)
import numpy as np

syms, idx = generate_bpsk_symbols_with_indices(100, 42)
print(f'BPSK: {len(syms)} symbols, {len(idx)} indices, idx range [{idx.min()}, {idx.max()}]')

syms, idx = generate_qpsk_symbols_with_indices(100, 42)
print(f'QPSK: {len(syms)} symbols, idx range [{idx.min()}, {idx.max()}]')

c = get_bpsk_constellation()
print(f'BPSK constellation: {c}')

c = get_qpsk_constellation()
print(f'QPSK constellation: {len(c)} points')

c = get_qam_constellation(16)
print(f'QAM16 constellation: {len(c)} points')

print('All Rust extensions OK')
"
```
Expected: prints confirmation lines, no errors.

- [ ] **Step 6: Run existing Rust tests (no regressions)**

```bash
cargo test --manifest-path rust/Cargo.toml
```
Expected: all existing tests pass.

- [ ] **Step 7: Commit**

```bash
git add rust/src/modulators.rs rust/src/lib.rs
git commit -m "feat(rust): add _with_indices symbol generators and get_*_constellation functions"
```

---

## Task 2: BER/SER/PER Metrics

**Files:**
- Modify: `python/spectra/metrics.py`
- Create: `tests/test_link_metrics.py`

- [ ] **Step 1: Write the failing tests**

```python
# tests/test_link_metrics.py
"""Tests for BER, SER, and PER metric functions."""
import numpy as np
import pytest


def test_ber_perfect():
    from spectra.metrics import bit_error_rate
    tx = np.array([0, 1, 0, 1, 1, 0])
    rx = np.array([0, 1, 0, 1, 1, 0])
    assert bit_error_rate(tx, rx) == 0.0


def test_ber_all_wrong():
    from spectra.metrics import bit_error_rate
    tx = np.array([0, 0, 0, 0])
    rx = np.array([1, 1, 1, 1])
    assert bit_error_rate(tx, rx) == 1.0


def test_ber_half():
    from spectra.metrics import bit_error_rate
    tx = np.array([0, 0, 1, 1])
    rx = np.array([0, 1, 1, 0])
    assert bit_error_rate(tx, rx) == pytest.approx(0.5)


def test_ser_perfect():
    from spectra.metrics import symbol_error_rate
    tx = np.array([0, 1, 2, 3])
    rx = np.array([0, 1, 2, 3])
    assert symbol_error_rate(tx, rx) == 0.0


def test_ser_one_error():
    from spectra.metrics import symbol_error_rate
    tx = np.array([0, 1, 2, 3])
    rx = np.array([0, 1, 2, 0])  # last symbol wrong
    assert symbol_error_rate(tx, rx) == pytest.approx(0.25)


def test_per_no_errors():
    from spectra.metrics import packet_error_rate
    tx = np.zeros(100, dtype=int)
    rx = np.zeros(100, dtype=int)
    assert packet_error_rate(tx, rx, packet_length=10) == 0.0


def test_per_one_packet_error():
    from spectra.metrics import packet_error_rate
    tx = np.zeros(100, dtype=int)
    rx = np.zeros(100, dtype=int)
    rx[5] = 1  # one bit error in first packet
    assert packet_error_rate(tx, rx, packet_length=10) == pytest.approx(0.1)


def test_per_all_packets_error():
    from spectra.metrics import packet_error_rate
    tx = np.zeros(40, dtype=int)
    rx = np.ones(40, dtype=int)
    assert packet_error_rate(tx, rx, packet_length=10) == 1.0


def test_per_truncates():
    from spectra.metrics import packet_error_rate
    # 15 bits with packet_length=10 -> only 1 full packet evaluated
    tx = np.zeros(15, dtype=int)
    rx = np.zeros(15, dtype=int)
    rx[3] = 1
    assert packet_error_rate(tx, rx, packet_length=10) == 1.0
```

- [ ] **Step 2: Run tests to verify they fail**

```bash
/Users/gditzler/.venvs/base/bin/pytest tests/test_link_metrics.py -v
```
Expected: `ImportError` (functions don't exist yet)

- [ ] **Step 3: Write the implementation**

Add to the end of `python/spectra/metrics.py`:

```python
def bit_error_rate(tx_bits: np.ndarray, rx_bits: np.ndarray) -> float:
    """Fraction of bits that differ between transmitted and received sequences."""
    tx_bits = np.asarray(tx_bits)
    rx_bits = np.asarray(rx_bits)
    return float(np.mean(tx_bits != rx_bits))


def symbol_error_rate(tx_indices: np.ndarray, rx_indices: np.ndarray) -> float:
    """Fraction of symbol indices that differ."""
    tx_indices = np.asarray(tx_indices)
    rx_indices = np.asarray(rx_indices)
    return float(np.mean(tx_indices != rx_indices))


def packet_error_rate(
    tx_bits: np.ndarray, rx_bits: np.ndarray, packet_length: int
) -> float:
    """Fraction of fixed-length packets containing at least one bit error.

    Truncates to the largest multiple of ``packet_length``.
    """
    tx_bits = np.asarray(tx_bits)
    rx_bits = np.asarray(rx_bits)
    n_packets = len(tx_bits) // packet_length
    if n_packets == 0:
        return 0.0
    usable = n_packets * packet_length
    tx_blocks = tx_bits[:usable].reshape(n_packets, packet_length)
    rx_blocks = rx_bits[:usable].reshape(n_packets, packet_length)
    packet_errors = np.any(tx_blocks != rx_blocks, axis=1)
    return float(np.mean(packet_errors))
```

- [ ] **Step 4: Run tests to verify they pass**

```bash
/Users/gditzler/.venvs/base/bin/pytest tests/test_link_metrics.py -v
```
Expected: 9/9 PASS

- [ ] **Step 5: Commit**

```bash
git add python/spectra/metrics.py tests/test_link_metrics.py
git commit -m "feat(metrics): add bit_error_rate, symbol_error_rate, packet_error_rate"
```

---

## Task 3: Receivers (`spectra/receivers/`)

**Files:**
- Create: `python/spectra/receivers/__init__.py`
- Create: `python/spectra/receivers/base.py`
- Create: `python/spectra/receivers/coherent.py`
- Create: `tests/test_receivers.py`

- [ ] **Step 1: Write the failing tests**

```python
# tests/test_receivers.py
"""Tests for CoherentReceiver and Decoder stubs."""
import numpy as np
import pytest


def test_passthrough_decoder():
    from spectra.receivers.base import PassthroughDecoder
    dec = PassthroughDecoder()
    bits = np.array([0, 1, 1, 0, 1])
    out = dec.decode(bits)
    assert np.array_equal(out, bits)


def test_viterbi_stub_raises():
    from spectra.receivers.base import ViterbiDecoder
    dec = ViterbiDecoder(constraint_length=7, generators=[0o171, 0o133])
    with pytest.raises(NotImplementedError):
        dec.decode(np.array([0, 1, 0]))


def test_ldpc_stub_raises():
    from spectra.receivers.base import LDPCDecoder
    dec = LDPCDecoder(parity_check_matrix=np.eye(4), max_iterations=50)
    with pytest.raises(NotImplementedError):
        dec.decode(np.array([0, 1, 0]))


def test_coherent_receiver_bpsk_noiseless():
    """Noiseless BPSK should demodulate perfectly."""
    from spectra.receivers.coherent import CoherentReceiver
    from spectra.waveforms import BPSK
    from spectra._rust import generate_bpsk_symbols_with_indices, apply_rrc_filter_with_taps
    from spectra.utils.rrc_cache import cached_rrc_taps

    wf = BPSK(samples_per_symbol=8)
    num_sym = 200
    seed = 42

    # Transmit
    symbols, tx_indices = generate_bpsk_symbols_with_indices(num_sym, seed)
    taps = cached_rrc_taps(wf.rolloff, wf.filter_span, wf.samples_per_symbol)
    tx_iq = np.array(apply_rrc_filter_with_taps(symbols, taps, wf.samples_per_symbol))

    # Receive (noiseless)
    rx = CoherentReceiver(wf)
    rx_indices, rx_bits = rx.demodulate(tx_iq)

    # Should match perfectly (allow edge trimming from filter delay)
    trim = wf.filter_span  # RRC filter group delay in symbols
    assert np.array_equal(rx_indices[trim:-trim], np.array(tx_indices)[trim:-trim])


def test_coherent_receiver_qpsk_noiseless():
    from spectra.receivers.coherent import CoherentReceiver
    from spectra.waveforms import QPSK
    from spectra._rust import generate_qpsk_symbols_with_indices, apply_rrc_filter_with_taps
    from spectra.utils.rrc_cache import cached_rrc_taps

    wf = QPSK(samples_per_symbol=8)
    num_sym = 200
    seed = 99

    symbols, tx_indices = generate_qpsk_symbols_with_indices(num_sym, seed)
    taps = cached_rrc_taps(wf.rolloff, wf.filter_span, wf.samples_per_symbol)
    tx_iq = np.array(apply_rrc_filter_with_taps(symbols, taps, wf.samples_per_symbol))

    rx = CoherentReceiver(wf)
    rx_indices, rx_bits = rx.demodulate(tx_iq)

    trim = wf.filter_span
    assert np.array_equal(rx_indices[trim:-trim], np.array(tx_indices)[trim:-trim])


def test_coherent_receiver_output_shapes():
    from spectra.receivers.coherent import CoherentReceiver
    from spectra.waveforms import QPSK
    from spectra._rust import generate_qpsk_symbols_with_indices, apply_rrc_filter_with_taps
    from spectra.utils.rrc_cache import cached_rrc_taps

    wf = QPSK(samples_per_symbol=8)
    symbols, tx_indices = generate_qpsk_symbols_with_indices(100, 42)
    taps = cached_rrc_taps(wf.rolloff, wf.filter_span, wf.samples_per_symbol)
    tx_iq = np.array(apply_rrc_filter_with_taps(symbols, taps, wf.samples_per_symbol))

    rx = CoherentReceiver(wf)
    rx_indices, rx_bits = rx.demodulate(tx_iq)

    assert rx_indices.ndim == 1
    assert rx_bits.ndim == 1
    # QPSK: 2 bits per symbol
    assert len(rx_bits) == len(rx_indices) * 2


def test_constellation_to_bits():
    from spectra.receivers.coherent import constellation_to_bits
    # BPSK: 1 bit per symbol
    bits = constellation_to_bits(np.array([0, 1, 0, 1], dtype=np.uint32), 2)
    assert len(bits) == 4
    assert bits[0] == 0
    assert bits[1] == 1


def test_receiver_abc():
    from spectra.receivers.base import Receiver
    # Receiver is abstract — can't instantiate directly
    with pytest.raises(TypeError):
        Receiver()
```

- [ ] **Step 2: Run tests to verify they fail**

```bash
/Users/gditzler/.venvs/base/bin/pytest tests/test_receivers.py -v
```
Expected: `ModuleNotFoundError: No module named 'spectra.receivers'`

- [ ] **Step 3: Write `base.py`**

```python
# python/spectra/receivers/base.py
"""Receiver and Decoder abstract base classes with FEC stubs."""

from __future__ import annotations

from abc import ABC, abstractmethod
from typing import Tuple

import numpy as np


class Receiver(ABC):
    """Abstract base class for receivers.

    Receivers map IQ samples to symbol indices and bits. Unlike
    :class:`~spectra.impairments.base.Transform` (which maps IQ → IQ),
    receivers output a fundamentally different domain.
    """

    @abstractmethod
    def demodulate(self, received_iq: np.ndarray) -> Tuple[np.ndarray, np.ndarray]:
        """Demodulate received IQ samples.

        Returns:
            Tuple of ``(symbol_indices, bits)`` as uint32/uint8 arrays.
        """


class Decoder(ABC):
    """Abstract base class for FEC decoders."""

    @abstractmethod
    def decode(self, bits: np.ndarray) -> np.ndarray:
        """Decode a block of (possibly coded) bits.

        Returns:
            Information bits after decoding.
        """


class PassthroughDecoder(Decoder):
    """No-op decoder. Returns input unchanged."""

    def decode(self, bits: np.ndarray) -> np.ndarray:
        return bits


class ViterbiDecoder(Decoder):
    """Convolutional code decoder stub.

    Args:
        constraint_length: Constraint length of the convolutional code.
        generators: Generator polynomials as integers.
    """

    def __init__(self, constraint_length: int, generators: list) -> None:
        self.constraint_length = constraint_length
        self.generators = generators

    def decode(self, bits: np.ndarray) -> np.ndarray:
        raise NotImplementedError(
            "ViterbiDecoder is a stub. Implement decode() for your code."
        )


class LDPCDecoder(Decoder):
    """LDPC decoder stub.

    Args:
        parity_check_matrix: Parity check matrix H.
        max_iterations: Maximum belief propagation iterations.
    """

    def __init__(self, parity_check_matrix: np.ndarray, max_iterations: int = 50) -> None:
        self.parity_check_matrix = parity_check_matrix
        self.max_iterations = max_iterations

    def decode(self, bits: np.ndarray) -> np.ndarray:
        raise NotImplementedError(
            "LDPCDecoder is a stub. Implement decode() for your code."
        )
```

- [ ] **Step 4: Write `coherent.py`**

```python
# python/spectra/receivers/coherent.py
"""Coherent receiver with matched filter and nearest-neighbor slicer."""

from __future__ import annotations

from typing import Tuple

import numpy as np

from spectra._rust import (
    apply_rrc_filter_with_taps,
    get_bpsk_constellation,
    get_qpsk_constellation,
    get_psk_constellation,
    get_qam_constellation,
    get_ask_constellation,
)
from spectra.receivers.base import Receiver
from spectra.waveforms.base import Waveform
from spectra.utils.rrc_cache import cached_rrc_taps


def constellation_to_bits(indices: np.ndarray, constellation_size: int) -> np.ndarray:
    """Convert symbol indices to bits using natural binary mapping.

    Args:
        indices: Symbol indices, shape ``(N,)``, dtype uint32.
        constellation_size: Number of constellation points M.

    Returns:
        Flat bit array, shape ``(N * bits_per_symbol,)``, dtype uint8.
    """
    bits_per_symbol = int(np.log2(constellation_size))
    n = len(indices)
    bits = np.zeros(n * bits_per_symbol, dtype=np.uint8)
    for b in range(bits_per_symbol):
        bits[b::bits_per_symbol] = (indices >> (bits_per_symbol - 1 - b)) & 1
    return bits


def _get_constellation(waveform: Waveform) -> np.ndarray:
    """Get the reference constellation for a waveform from Rust."""
    label = waveform.label.upper()
    order = getattr(waveform, "_order", None)

    if label == "BPSK":
        return np.array(get_bpsk_constellation(), dtype=np.complex64)
    elif label == "QPSK":
        return np.array(get_qpsk_constellation(), dtype=np.complex64)
    elif label == "OOK" or "ASK" in label:
        m = order if order else 2
        return np.array(get_ask_constellation(m), dtype=np.complex64)
    elif "QAM" in label:
        m = order if order else 16
        return np.array(get_qam_constellation(m), dtype=np.complex64)
    elif "PSK" in label:
        m = order if order else 8
        return np.array(get_psk_constellation(m), dtype=np.complex64)
    else:
        raise ValueError(f"Unsupported waveform for CoherentReceiver: {waveform.label}")


class CoherentReceiver(Receiver):
    """Perfect-synchronization coherent receiver.

    Pipeline: RRC matched filter → downsample → nearest-neighbor slicer → bit demap.

    Args:
        waveform: Transmit waveform (PSK, square QAM, or ASK).
    """

    def __init__(self, waveform: Waveform) -> None:
        self.waveform = waveform
        self.samples_per_symbol = waveform.samples_per_symbol
        self.rolloff = getattr(waveform, "rolloff", 0.35)
        self.filter_span = getattr(waveform, "filter_span", 10)
        self.constellation = _get_constellation(waveform)
        self.constellation_size = len(self.constellation)

    def demodulate(self, received_iq: np.ndarray) -> Tuple[np.ndarray, np.ndarray]:
        """Demodulate received IQ through matched filter + slicer.

        Returns:
            ``(symbol_indices, bits)`` — uint32 and uint8 arrays.
        """
        # 1. RRC matched filter
        taps = cached_rrc_taps(self.rolloff, self.filter_span, self.samples_per_symbol)
        filtered = np.array(
            apply_rrc_filter_with_taps(
                np.asarray(received_iq, dtype=np.complex64), taps, 1
            )
        )

        # 2. Downsample to symbol rate
        # Account for filter group delay
        delay = self.filter_span * self.samples_per_symbol
        symbols = filtered[delay::self.samples_per_symbol]

        # 3. Nearest-neighbor constellation slicer
        const = self.constellation
        # Vectorised distance: (N_symbols, 1) - (1, M) -> (N_symbols, M)
        diffs = symbols[:, np.newaxis] - const[np.newaxis, :]
        distances = np.abs(diffs) ** 2
        rx_indices = np.argmin(distances, axis=1).astype(np.uint32)

        # 4. Bit demapper
        rx_bits = constellation_to_bits(rx_indices, self.constellation_size)

        return rx_indices, rx_bits
```

- [ ] **Step 5: Write `__init__.py`**

```python
# python/spectra/receivers/__init__.py
from spectra.receivers.base import (
    Decoder,
    LDPCDecoder,
    PassthroughDecoder,
    Receiver,
    ViterbiDecoder,
)
from spectra.receivers.coherent import CoherentReceiver

__all__ = [
    "CoherentReceiver",
    "Decoder",
    "LDPCDecoder",
    "PassthroughDecoder",
    "Receiver",
    "ViterbiDecoder",
]
```

- [ ] **Step 6: Run tests to verify they pass**

```bash
/Users/gditzler/.venvs/base/bin/pytest tests/test_receivers.py -v
```
Expected: 8/8 PASS

- [ ] **Step 7: Commit**

```bash
git add python/spectra/receivers/__init__.py python/spectra/receivers/base.py python/spectra/receivers/coherent.py tests/test_receivers.py
git commit -m "feat(receivers): add CoherentReceiver, Decoder ABC, and FEC stubs"
```

---

## Task 4: Link Simulator (`spectra/link/`)

**Files:**
- Create: `python/spectra/link/__init__.py`
- Create: `python/spectra/link/results.py`
- Create: `python/spectra/link/simulator.py`
- Create: `tests/test_link_simulator.py`

- [ ] **Step 1: Write the failing tests**

```python
# tests/test_link_simulator.py
"""Tests for LinkSimulator."""
import numpy as np
import pytest


def test_link_results_fields():
    from spectra.link.results import LinkResults
    r = LinkResults(
        eb_n0_db=np.array([0, 5, 10]),
        ber=np.array([0.1, 0.01, 0.001]),
        ser=np.array([0.2, 0.02, 0.002]),
        per=np.array([0.5, 0.1, 0.01]),
        num_bits=10000,
        num_symbols=5000,
        packet_length=1000,
        waveform_label="BPSK",
    )
    assert len(r.eb_n0_db) == 3
    assert r.waveform_label == "BPSK"


def test_theoretical_ber_bpsk():
    from spectra.link.results import LinkResults
    r = LinkResults(
        eb_n0_db=np.array([0, 5, 10]),
        ber=np.zeros(3), ser=np.zeros(3), per=np.zeros(3),
        num_bits=1000, num_symbols=1000, packet_length=100,
        waveform_label="BPSK",
    )
    theory = r.theoretical_ber()
    assert theory is not None
    assert len(theory) == 3
    # At high Eb/N0, theoretical BER should be very small
    assert theory[2] < 1e-4


def test_theoretical_ber_qpsk_returns_none():
    from spectra.link.results import LinkResults
    r = LinkResults(
        eb_n0_db=np.array([0, 5, 10]),
        ber=np.zeros(3), ser=np.zeros(3), per=np.zeros(3),
        num_bits=1000, num_symbols=500, packet_length=100,
        waveform_label="QPSK",
    )
    assert r.theoretical_ber() is None


def test_link_simulator_bpsk_low_snr():
    """At low Eb/N0, BER should be high."""
    from spectra.link.simulator import LinkSimulator
    from spectra.waveforms import BPSK
    sim = LinkSimulator(waveform=BPSK(samples_per_symbol=8), num_symbols=2000, seed=42)
    results = sim.run(np.array([0.0]))  # 0 dB Eb/N0
    assert results.ber[0] > 0.01


def test_link_simulator_bpsk_high_snr():
    """At high Eb/N0, BER should be very low or zero."""
    from spectra.link.simulator import LinkSimulator
    from spectra.waveforms import BPSK
    sim = LinkSimulator(waveform=BPSK(samples_per_symbol=8), num_symbols=2000, seed=42)
    results = sim.run(np.array([15.0]))  # 15 dB Eb/N0
    assert results.ber[0] < 0.01


def test_link_simulator_sweep():
    from spectra.link.simulator import LinkSimulator
    from spectra.waveforms import BPSK
    sim = LinkSimulator(waveform=BPSK(samples_per_symbol=8), num_symbols=1000, seed=42)
    eb_n0 = np.array([0.0, 5.0, 10.0])
    results = sim.run(eb_n0)
    assert len(results.ber) == 3
    assert len(results.ser) == 3
    assert len(results.per) == 3
    # BER should decrease with increasing Eb/N0
    assert results.ber[0] >= results.ber[1]
    assert results.ber[1] >= results.ber[2]


def test_link_simulator_deterministic():
    from spectra.link.simulator import LinkSimulator
    from spectra.waveforms import BPSK
    sim1 = LinkSimulator(waveform=BPSK(samples_per_symbol=8), num_symbols=500, seed=42)
    sim2 = LinkSimulator(waveform=BPSK(samples_per_symbol=8), num_symbols=500, seed=42)
    r1 = sim1.run(np.array([5.0]))
    r2 = sim2.run(np.array([5.0]))
    assert np.allclose(r1.ber, r2.ber)


def test_link_simulator_qam16():
    from spectra.link.simulator import LinkSimulator
    from spectra.waveforms import QAM16
    sim = LinkSimulator(waveform=QAM16(samples_per_symbol=8), num_symbols=1000, seed=42)
    results = sim.run(np.array([15.0]))
    assert isinstance(results.ber[0], float)
    assert results.waveform_label == "16QAM"
```

- [ ] **Step 2: Run tests to verify they fail**

```bash
/Users/gditzler/.venvs/base/bin/pytest tests/test_link_simulator.py -v
```
Expected: `ModuleNotFoundError`

- [ ] **Step 3: Write `results.py`**

```python
# python/spectra/link/results.py
"""LinkResults dataclass for BER/SER/PER simulation results."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Optional

import numpy as np


@dataclass
class LinkResults:
    """Results from a link-level simulation sweep.

    Attributes:
        eb_n0_db: Eb/N0 values in dB, shape ``(num_points,)``.
        ber: Bit error rate per point.
        ser: Symbol error rate per point.
        per: Packet error rate per point.
        num_bits: Total bits simulated per point.
        num_symbols: Total symbols simulated per point.
        packet_length: Bits per packet for PER.
        waveform_label: Modulation label (e.g., "BPSK").
    """

    eb_n0_db: np.ndarray
    ber: np.ndarray
    ser: np.ndarray
    per: np.ndarray
    num_bits: int
    num_symbols: int
    packet_length: int
    waveform_label: str

    def theoretical_ber(self) -> Optional[np.ndarray]:
        """Return closed-form AWGN BER for BPSK, else None.

        BPSK BER = Q(sqrt(2 * Eb/N0)) = 0.5 * erfc(sqrt(Eb/N0)).
        Only BPSK is supported (inherently Gray-coded at M=2).
        """
        if self.waveform_label.upper() != "BPSK":
            return None
        import math

        eb_n0_lin = 10.0 ** (self.eb_n0_db / 10.0)
        return np.array([0.5 * math.erfc(math.sqrt(x)) for x in eb_n0_lin])
```

- [ ] **Step 4: Write `simulator.py`**

```python
# python/spectra/link/simulator.py
"""LinkSimulator: BER/SER/PER vs. Eb/N0 simulation."""

from __future__ import annotations

from typing import List, Optional

import numpy as np

from spectra._rust import (
    apply_rrc_filter_with_taps,
    generate_bpsk_symbols_with_indices,
    generate_qpsk_symbols_with_indices,
    generate_psk_symbols_with_indices,
    generate_qam_symbols_with_indices,
    generate_ask_symbols_with_indices,
)
from spectra.link.results import LinkResults
from spectra.metrics import bit_error_rate, packet_error_rate, symbol_error_rate
from spectra.receivers.base import Decoder, PassthroughDecoder
from spectra.receivers.coherent import CoherentReceiver, constellation_to_bits
from spectra.scene.signal_desc import SignalDescription
from spectra.waveforms.base import Waveform
from spectra.utils.rrc_cache import cached_rrc_taps


def _generate_with_indices(waveform: Waveform, num_symbols: int, seed: int):
    """Call the appropriate Rust _with_indices generator for a waveform."""
    label = waveform.label.upper()
    order = getattr(waveform, "_order", None)

    if label == "BPSK":
        return generate_bpsk_symbols_with_indices(num_symbols, seed)
    elif label == "QPSK":
        return generate_qpsk_symbols_with_indices(num_symbols, seed)
    elif label == "OOK" or "ASK" in label:
        m = order if order else 2
        return generate_ask_symbols_with_indices(num_symbols, m, seed)
    elif "QAM" in label:
        m = order if order else 16
        return generate_qam_symbols_with_indices(num_symbols, m, seed)
    elif "PSK" in label:
        m = order if order else 8
        return generate_psk_symbols_with_indices(num_symbols, m, seed)
    else:
        raise ValueError(f"Unsupported waveform for LinkSimulator: {waveform.label}")


class LinkSimulator:
    """Link-level simulator for BER/SER/PER vs. Eb/N0 curves.

    Args:
        waveform: Modulation waveform (PSK, square QAM, or ASK).
        channel: Optional impairments applied AFTER noise injection.
            AWGN is injected directly (not via the AWGN transform).
        decoder: FEC decoder. Default: PassthroughDecoder.
        num_symbols: Symbols per Eb/N0 point.
        packet_length: Bits per packet for PER.
        seed: Base seed for reproducibility.
    """

    def __init__(
        self,
        waveform: Waveform,
        channel: Optional[List] = None,
        decoder: Optional[Decoder] = None,
        num_symbols: int = 10000,
        packet_length: int = 1000,
        seed: int = 0,
    ) -> None:
        self.waveform = waveform
        self.channel = channel or []
        self.decoder = decoder or PassthroughDecoder()
        self.num_symbols = num_symbols
        self.packet_length = packet_length
        self.seed = seed
        self.receiver = CoherentReceiver(waveform)

    def run(self, eb_n0_points: np.ndarray) -> LinkResults:
        """Sweep Eb/N0 values and return BER/SER/PER curves."""
        eb_n0_points = np.asarray(eb_n0_points, dtype=float)
        n_points = len(eb_n0_points)

        ber_arr = np.zeros(n_points)
        ser_arr = np.zeros(n_points)
        per_arr = np.zeros(n_points)

        sps = self.waveform.samples_per_symbol
        rolloff = getattr(self.waveform, "rolloff", 0.35)
        filter_span = getattr(self.waveform, "filter_span", 10)
        constellation_size = self.receiver.constellation_size
        bits_per_symbol = int(np.log2(constellation_size))

        # Transmit (same for all Eb/N0 points)
        symbols, tx_indices = _generate_with_indices(
            self.waveform, self.num_symbols, self.seed
        )
        symbols = np.asarray(symbols, dtype=np.complex64)
        tx_indices = np.asarray(tx_indices, dtype=np.uint32)
        tx_bits = constellation_to_bits(tx_indices, constellation_size)

        # RRC pulse shaping
        taps = cached_rrc_taps(rolloff, filter_span, sps)
        tx_iq = np.array(apply_rrc_filter_with_taps(symbols, taps, sps))

        # Trim symbols affected by filter transients
        trim = filter_span
        valid_tx_indices = tx_indices[trim:-trim] if trim > 0 else tx_indices
        valid_tx_bits = constellation_to_bits(valid_tx_indices, constellation_size)

        for i, eb_n0_db in enumerate(eb_n0_points):
            # Noise injection (analytical, per-point seeded RNG)
            rng = np.random.default_rng((self.seed, i))
            signal_power = np.mean(np.abs(tx_iq) ** 2)
            eb = signal_power * sps / bits_per_symbol
            eb_n0_lin = 10.0 ** (eb_n0_db / 10.0)
            n0 = eb / eb_n0_lin
            noise_std = np.sqrt(n0 / 2.0)
            noise = noise_std * (
                rng.standard_normal(tx_iq.shape)
                + 1j * rng.standard_normal(tx_iq.shape)
            ).astype(np.complex64)
            rx_iq = tx_iq + noise

            # Optional channel impairments
            if self.channel:
                bw = self.waveform.bandwidth(1.0)
                desc = SignalDescription(
                    t_start=0.0,
                    t_stop=len(rx_iq) / sps,
                    f_low=-bw / 2, f_high=bw / 2,
                    label=self.waveform.label,
                    snr=float(eb_n0_db),
                )
                for transform in self.channel:
                    rx_iq, desc = transform(rx_iq, desc)

            # Demodulate
            rx_indices, rx_bits = self.receiver.demodulate(rx_iq)

            # Trim to valid region
            valid_rx_indices = rx_indices[:len(valid_tx_indices)]
            valid_rx_bits = constellation_to_bits(valid_rx_indices, constellation_size)

            # Decode
            decoded_bits = self.decoder.decode(valid_rx_bits)

            # Score
            min_len_bits = min(len(valid_tx_bits), len(decoded_bits))
            min_len_sym = min(len(valid_tx_indices), len(valid_rx_indices))

            ber_arr[i] = bit_error_rate(
                valid_tx_bits[:min_len_bits], decoded_bits[:min_len_bits]
            )
            ser_arr[i] = symbol_error_rate(
                valid_tx_indices[:min_len_sym], valid_rx_indices[:min_len_sym]
            )
            per_arr[i] = packet_error_rate(
                valid_tx_bits[:min_len_bits], decoded_bits[:min_len_bits],
                self.packet_length,
            )

        return LinkResults(
            eb_n0_db=eb_n0_points,
            ber=ber_arr,
            ser=ser_arr,
            per=per_arr,
            num_bits=len(valid_tx_bits),
            num_symbols=len(valid_tx_indices),
            packet_length=self.packet_length,
            waveform_label=self.waveform.label,
        )
```

- [ ] **Step 5: Write `__init__.py`**

```python
# python/spectra/link/__init__.py
from spectra.link.results import LinkResults
from spectra.link.simulator import LinkSimulator

__all__ = ["LinkResults", "LinkSimulator"]
```

- [ ] **Step 6: Run tests to verify they pass**

```bash
/Users/gditzler/.venvs/base/bin/pytest tests/test_link_simulator.py -v
```
Expected: 8/8 PASS

- [ ] **Step 7: Commit**

```bash
git add python/spectra/link/__init__.py python/spectra/link/results.py python/spectra/link/simulator.py tests/test_link_simulator.py
git commit -m "feat(link): add LinkSimulator for BER/SER/PER vs Eb/N0 curves"
```

---

## Task 5: Full Verification

- [ ] **Step 1: Run all tests**

```bash
/Users/gditzler/.venvs/base/bin/pytest tests/ -q --tb=short
```
Expected: all tests pass, no regressions.

- [ ] **Step 2: Verify imports**

```bash
/Users/gditzler/.venvs/base/bin/python -c "
from spectra.receivers import CoherentReceiver, Receiver, Decoder, PassthroughDecoder, ViterbiDecoder, LDPCDecoder
from spectra.link import LinkSimulator, LinkResults
from spectra.metrics import bit_error_rate, symbol_error_rate, packet_error_rate
print('All imports OK')
"
```

- [ ] **Step 3: Quick BER curve smoke test**

```bash
/Users/gditzler/.venvs/base/bin/python -c "
import numpy as np
from spectra.link import LinkSimulator
from spectra.waveforms import BPSK
sim = LinkSimulator(waveform=BPSK(samples_per_symbol=8), num_symbols=5000, seed=42)
results = sim.run(np.arange(0, 12, 2))
print('Eb/N0 (dB) | BER       | SER       | PER')
for i in range(len(results.eb_n0_db)):
    print(f'  {results.eb_n0_db[i]:5.1f}     | {results.ber[i]:.4e} | {results.ser[i]:.4e} | {results.per[i]:.4e}')
theory = results.theoretical_ber()
if theory is not None:
    print(f'Theoretical BPSK BER: {theory}')
print('Smoke test OK')
"
```
Expected: BER decreases with increasing Eb/N0, theoretical values printed.
