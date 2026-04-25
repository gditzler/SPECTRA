# Link Simulation

`LinkSimulator` produces BER/SER/PER vs. Eb/N0 curves for a chosen
modulation scheme, with optional channel impairments applied after AWGN
injection. Internally it uses Rust-backed `*_with_indices` symbol generators
so the ground-truth bit and symbol streams stay aligned with received samples
through RRC pulse shaping and any channel effects.

The simulator follows a standard digital-link model:

```
Symbols (Tx)
  → RRC pulse shaping (Rust)
  → AWGN injection (analytical, per-point seeded)
  → optional channel impairments (list of Transforms)
  → matched RRC filter + slicer (CoherentReceiver)
  → BER / SER / PER scoring
```

## Quickstart

```python
import numpy as np
from spectra.link.simulator import LinkSimulator
from spectra.waveforms import QPSK

sim = LinkSimulator(waveform=QPSK(), num_symbols=10000, seed=0)
results = sim.run(eb_n0_points=np.arange(0, 11, 2))  # 0, 2, 4, ..., 10 dB

print(results.eb_n0_db)   # array([0., 2., 4., 6., 8., 10.])
print(results.ber)        # bit error rate per point
print(results.ser)        # symbol error rate per point
print(results.per)        # packet error rate per point
```

## Constructor parameters

| Parameter | Type | Default | Purpose |
|-----------|------|---------|---------|
| `waveform` | `Waveform` | required | Modulation waveform — PSK, QAM (square), or ASK |
| `channel` | `list[Transform]` | `None` | Impairments applied **after** AWGN; each must follow the `(iq, desc) -> (iq, desc)` protocol |
| `decoder` | `Decoder` | `PassthroughDecoder()` | FEC decoder applied before error scoring |
| `num_symbols` | `int` | `10000` | Symbols simulated per Eb/N0 point |
| `packet_length` | `int` | `1000` | Bits per packet used for PER calculation |
| `seed` | `int` | `0` | Base seed; each Eb/N0 point derives `(seed, i)` for reproducibility |

## `run()` — sweeping Eb/N0

```python
results = sim.run(eb_n0_points: np.ndarray) -> LinkResults
```

`eb_n0_points` is a 1-D NumPy array of Eb/N0 values in **dB**. Each point
is simulated independently (AWGN variance computed analytically, seeded
from `(seed, i)`) so individual points are reproducible even if the sweep
range changes.

## LinkResults fields

`LinkResults` is a plain dataclass. All array fields share the same length
as `eb_n0_points`.

| Field | Type | Description |
|-------|------|-------------|
| `eb_n0_db` | `np.ndarray` | Eb/N0 sweep values (dB) — echo of the input |
| `ber` | `np.ndarray` | Bit Error Rate per point |
| `ser` | `np.ndarray` | Symbol Error Rate per point |
| `per` | `np.ndarray` | Packet Error Rate per point |
| `num_bits` | `int` | Valid bits evaluated (after trimming filter transients) |
| `num_symbols` | `int` | Valid symbols evaluated (after trimming filter transients) |
| `packet_length` | `int` | Bits per packet used for PER (mirrors constructor arg) |
| `waveform_label` | `str` | Label string of the waveform, e.g. `"QPSK"` |

### `theoretical_ber()`

`LinkResults` also exposes a helper method for BPSK:

```python
theory = results.theoretical_ber()  # returns None for non-BPSK waveforms
```

For BPSK this computes the closed-form AWGN BER:

$$\text{BER}_{\text{BPSK}} = \frac{1}{2} \,\text{erfc}\!\left(\sqrt{E_b/N_0}\right)$$

For all other waveforms it returns `None`.

## Comparing simulated and theoretical BER

```python
import numpy as np
import matplotlib.pyplot as plt
from spectra.link.simulator import LinkSimulator
from spectra.waveforms import BPSK

sim = LinkSimulator(waveform=BPSK(), num_symbols=10000, seed=42)
results = sim.run(np.arange(0, 12, 1.0))

theory = results.theoretical_ber()

fig, ax = plt.subplots()
ax.semilogy(results.eb_n0_db, results.ber, "o-", label="Simulated")
if theory is not None:
    ax.semilogy(results.eb_n0_db, theory, "k--", label="Theoretical")
ax.set_xlabel("Eb/N0 (dB)")
ax.set_ylabel("Bit Error Rate")
ax.legend()
ax.grid(True, which="both", alpha=0.3)
```

## Multi-modulation comparison

```python
import numpy as np
from spectra.link.simulator import LinkSimulator
from spectra.waveforms import BPSK, QPSK, QAM16

eb_n0 = np.arange(0, 14, 1.0)

for wf_cls, label in [(BPSK, "BPSK"), (QPSK, "QPSK"), (QAM16, "16QAM")]:
    sim = LinkSimulator(waveform=wf_cls(), num_symbols=10000, seed=0)
    results = sim.run(eb_n0)
    print(f"{label}: BER at 6 dB = {results.ber[6]:.4e}")
```

## Adding channel impairments

Channel impairments are passed as a list of SPECTRA transforms. Each
impairment is applied **after** AWGN injection at every Eb/N0 point, so the
total degradation is additive on top of thermal noise.

```python
import numpy as np
from spectra.link.simulator import LinkSimulator
from spectra.impairments import RayleighFading
from spectra.waveforms import QPSK

# RayleighFading parameters: num_taps (int), doppler_spread (float, normalized)
sim = LinkSimulator(
    waveform=QPSK(),
    channel=[RayleighFading(doppler_spread=0.01)],
    num_symbols=10000,
    seed=0,
)
results = sim.run(np.arange(0, 16, 2.0))
print(results.ber)
```

!!! note "Channel impairments vs. AWGN"
    The `channel` list is applied **after** the per-point AWGN injection, so
    Eb/N0 still controls thermal noise. Channel effects (fading, phase noise,
    IQ imbalance, etc.) add on top of that. Any SPECTRA transform that follows
    the `(iq, desc) -> (iq, desc)` protocol can be used here.

## Supported waveforms

`LinkSimulator` supports modulations that have Rust-backed
`*_with_indices` generators and coherent constellation demodulators:

| Family | Examples |
|--------|---------|
| BPSK | `BPSK` |
| QPSK | `QPSK` |
| M-PSK | `PSK8`, `PSK16`, ... |
| Square M-QAM | `QAM16`, `QAM64`, `QAM256`, ... |
| M-ASK / OOK | `OOK`, `ASK4`, ... |

Other waveform families (OFDM, FSK, radar, spread-spectrum, etc.) raise
`ValueError` at `run()` time because they do not have matching
`*_with_indices` Rust generators or coherent receivers.

## Packet length and PER

PER counts packets that contain **at least one bit error**. A packet is
defined as a contiguous block of `packet_length` bits. Choose `packet_length`
to match your protocol's frame size:

```python
# 802.11 OFDM payload = 1080 bytes → 8640 bits
sim = LinkSimulator(
    waveform=QPSK(),
    num_symbols=50000,
    packet_length=8640,
    seed=0,
)
results = sim.run(np.arange(0, 12, 1.0))
print(results.per)
```

## Reproducibility

The `seed` parameter controls all randomness:

- Tx symbols use `seed` directly.
- Each Eb/N0 point `i` uses `np.random.default_rng((seed, i))` for its AWGN
  noise draw.

This means:
- Two runs with the same `seed` and `eb_n0_points` produce identical results.
- Changing `eb_n0_points` (adding or removing points) does **not** affect the
  results at unchanged index positions, because each point is seeded by its
  index, not its value.

## See also

- API reference: `spectra.link.simulator.LinkSimulator`, `spectra.link.results.LinkResults`
- Receiver internals: `spectra.receivers.CoherentReceiver`
- Impairments: [Impairments](impairments.md)
- Example: `examples/communications/link_simulator.py`
