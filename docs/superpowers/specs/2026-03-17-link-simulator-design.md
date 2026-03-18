# Link-Level Simulator — Design Spec

**Goal:** Add a `LinkSimulator` that runs transmitter → channel → receiver loops to generate BER/SER/PER vs. Eb/N0 curves, validating that SPECTRA's generated waveforms have correct physical behaviour.

**Scope:** v1 implements a perfect-sync coherent receiver (no timing/frequency/phase recovery) with nearest-neighbor constellation slicing for PSK, QAM, and ASK families. FEC decoder stubs (Viterbi, LDPC) define the interface for future implementations.

---

## Architecture

```
Transmitter           Channel              Receiver              Metrics
(Rust symbols+idx)    (AWGN + impairments) (MF + slicer + demap) (BER/SER/PER)
     |                     |                    |                    |
     v                     v                    v                    v
generate_*_with_indices → RRC shape → AWGN(Eb/N0→SNR) → RRC MF → downsample →
  → [optional chain]  → nearest-neighbor slicer → Gray demap → [FEC decode] → compare bits
```

The `LinkSimulator` orchestrates this loop over an array of Eb/N0 points and returns a `LinkResults` dataclass.

---

## Sub-project 1: Rust Modulator Extensions

### Files

- `rust/src/modulators.rs` (modify)
- `rust/src/lib.rs` (modify — register new functions)

### New Functions

Add `_with_indices` variants for each modulation family. These use the same xorshift64 PRNG and constellation mappings as the existing generators, but additionally return the symbol index before mapping to a complex point.

**`generate_bpsk_symbols_with_indices(num_symbols, seed) -> (PyArray1<Complex32>, PyArray1<u32>)`**

**`generate_qpsk_symbols_with_indices(num_symbols, seed) -> (PyArray1<Complex32>, PyArray1<u32>)`**

**`generate_psk_symbols_with_indices(num_symbols, m_order, seed) -> (PyArray1<Complex32>, PyArray1<u32>)`**

**`generate_qam_symbols_with_indices(num_symbols, m_order, seed) -> (PyArray1<Complex32>, PyArray1<u32>)`**

Each returns a tuple of `(symbols, indices)` where:
- `symbols`: complex64 array of constellation points (same as existing functions)
- `indices`: uint32 array of constellation point indices (0 to M-1)

The existing `generate_*` functions remain unchanged — the `_with_indices` variants are only used by the link simulator.

### Gray Code Mapping

On the Python side, a `gray_map(m_order) -> np.ndarray` utility generates the standard Gray code bit mapping table for a given modulation order. Given an index `k` in `0..M-1`, `gray_map(M)[k]` returns the corresponding bit pattern as an integer. A `bits_from_indices(indices, m_order) -> np.ndarray` function converts index arrays to flat bit arrays.

---

## Sub-project 2: Metrics Extensions

### Files

- `python/spectra/metrics.py` (modify)

### New Functions

```python
def bit_error_rate(tx_bits: np.ndarray, rx_bits: np.ndarray) -> float:
    """Fraction of bits that differ between transmitted and received sequences."""

def symbol_error_rate(tx_indices: np.ndarray, rx_indices: np.ndarray) -> float:
    """Fraction of symbol indices that differ."""

def packet_error_rate(
    tx_bits: np.ndarray, rx_bits: np.ndarray, packet_length: int
) -> float:
    """Fraction of fixed-length packets containing at least one bit error.

    Truncates to the largest multiple of packet_length.
    """
```

These are standalone utility functions. `LinkSimulator` uses them internally, but they're also available for direct use.

---

## Sub-project 3: Receivers (`spectra/receivers/`)

### Files

- `python/spectra/receivers/__init__.py`
- `python/spectra/receivers/coherent.py`
- `python/spectra/receivers/base.py`

### `coherent.py` — `CoherentReceiver`

Assumes perfect synchronization. Processing pipeline:

1. **RRC matched filter** — convolve received IQ with the transmit RRC filter taps (reuses Rust `apply_rrc_filter_with_taps`), providing optimal SNR at symbol sampling points.
2. **Downsample** — take every `samples_per_symbol`-th sample to get symbol-rate samples.
3. **Constellation slicer** — nearest-neighbor decision: for each received sample, find the closest constellation point. Returns estimated symbol indices.
4. **Bit demapper** — convert symbol indices to bits via Gray code lookup.

```python
class CoherentReceiver:
    def __init__(self, waveform: Waveform):
        """Extract constellation, samples_per_symbol, RRC params from waveform."""

    def demodulate(self, received_iq: np.ndarray) -> Tuple[np.ndarray, np.ndarray]:
        """Returns (estimated_symbol_indices, estimated_bits)."""
```

The receiver extracts everything it needs from the `Waveform` instance — constellation map, `samples_per_symbol`, RRC `rolloff` and `filter_span`. No separate configuration needed.

**Supported modulation families:** PSK (BPSK, QPSK, 8PSK, 16PSK, 32PSK, 64PSK), QAM (16 through 1024), ASK (OOK, ASK4 through ASK64). All use the same nearest-neighbor slicer — the only difference is the constellation map.

**Constellation access:** The receiver needs the reference constellation points. These will be obtained by calling the `_with_indices` Rust function with a known sequence to extract all M constellation points, or by defining the constellations as Python-side lookup tables that match the Rust definitions.

### `base.py` — `Decoder` ABC and Stubs

```python
class Decoder(ABC):
    @abstractmethod
    def decode(self, bits: np.ndarray) -> np.ndarray:
        """Decode a block of (possibly coded) bits. Returns information bits."""

class PassthroughDecoder(Decoder):
    """No-op decoder. Returns input unchanged. Default when no FEC is used."""

class ViterbiDecoder(Decoder):
    """Convolutional code decoder stub.

    Args:
        constraint_length: Constraint length of the convolutional code.
        generators: Generator polynomials as integers (e.g., [0o171, 0o133]).

    Raises NotImplementedError on decode().
    """

class LDPCDecoder(Decoder):
    """LDPC decoder stub.

    Args:
        parity_check_matrix: Sparse parity check matrix H.
        max_iterations: Maximum belief propagation iterations.

    Raises NotImplementedError on decode().
    """
```

The stubs accept configuration parameters and store them, but raise `NotImplementedError` when `decode()` is called. This defines the interface so users know exactly what to implement.

---

## Sub-project 4: Link Simulator (`spectra/link/`)

### Files

- `python/spectra/link/__init__.py`
- `python/spectra/link/simulator.py`
- `python/spectra/link/results.py`

### `simulator.py` — `LinkSimulator`

```python
class LinkSimulator:
    def __init__(
        self,
        waveform: Waveform,
        channel: Optional[List[Transform]] = None,
        decoder: Optional[Decoder] = None,
        num_symbols: int = 10000,
        packet_length: int = 1000,
        seed: int = 0,
    ):
        """
        Args:
            waveform: Modulation waveform (must be PSK, QAM, or ASK).
            channel: Optional list of impairments to apply AFTER AWGN.
                AWGN is always applied internally at the specified Eb/N0.
            decoder: FEC decoder. Default: PassthroughDecoder (no coding).
            num_symbols: Symbols to simulate per Eb/N0 point.
            packet_length: Bits per packet for PER computation.
            seed: Base seed for reproducible simulation.
        """

    def run(self, eb_n0_points: np.ndarray) -> LinkResults:
        """Sweep Eb/N0 values and return BER/SER/PER curves."""
```

**Per Eb/N0 point, the `.run()` loop:**

1. **Transmit** — call `generate_*_with_indices(num_symbols, seed)` to get `(tx_symbols, tx_indices)`. Convert `tx_indices` to `tx_bits` via Gray demapping. Apply RRC pulse shaping to `tx_symbols` to get `tx_iq`.

2. **Eb/N0 → SNR conversion** — `SNR_dB = Eb/N0_dB + 10*log10(bits_per_symbol) - 10*log10(samples_per_symbol)`. This accounts for oversampling and modulation order.

3. **Channel** — apply AWGN at the computed SNR (using the existing `AWGN` transform). Then apply any additional impairments from `channel` list.

4. **Receive** — `CoherentReceiver.demodulate(rx_iq)` returns `(rx_indices, rx_bits)`.

5. **Decode** — `decoder.decode(rx_bits)` (passthrough by default).

6. **Score** — `bit_error_rate(tx_bits, rx_bits)` for BER, `symbol_error_rate(tx_indices, rx_indices)` for SER, `packet_error_rate(tx_bits, rx_bits, packet_length)` for PER.

**Seeding:** The transmitter seed is fixed per `LinkSimulator` instance. All Eb/N0 points use the same transmitted sequence (different noise realisations come from AWGN's internal randomness).

### `results.py` — `LinkResults`

```python
@dataclass
class LinkResults:
    eb_n0_db: np.ndarray       # (num_points,) Eb/N0 values in dB
    ber: np.ndarray            # (num_points,) bit error rate per point
    ser: np.ndarray            # (num_points,) symbol error rate per point
    per: np.ndarray            # (num_points,) packet error rate per point
    num_bits: int              # total bits simulated per point
    num_symbols: int           # total symbols simulated per point
    packet_length: int         # bits per packet
    waveform_label: str        # e.g., "QPSK"

    def theoretical_ber(self) -> Optional[np.ndarray]:
        """Return closed-form AWGN BER curve if known for this waveform.

        Supported:
        - BPSK: Q(sqrt(2 * Eb/N0))
        - QPSK: Q(sqrt(2 * Eb/N0))  (same as BPSK for Gray-coded)

        Returns None for other modulations.
        """
```

---

## Build Order & Dependencies

```
Sub-project 1: Rust modulators    (no Python dependencies)
Sub-project 2: Metrics            (standalone functions)
Sub-project 3: Receivers          (depends on waveform API, Rust _with_indices)
Sub-project 4: Link simulator     (depends on 1, 2, 3, impairments)
```

Sub-projects 1 and 2 are independent. Sub-project 3 depends on 1 (for constellation access). Sub-project 4 integrates everything.

---

## Future Work (out of scope for v1)

- **Synchronization:** symbol timing recovery (Mueller-Muller, Gardner), carrier frequency offset estimation, phase tracking
- **FEC implementations:** actual Viterbi and LDPC decoders (the stubs define the interface)
- **Soft-decision decoding:** LLR computation from received samples instead of hard bits
- **FSK/OFDM receivers:** different demodulation structures beyond matched-filter + slicer
- **Curriculum-by-BER:** use measured BER as feedback to `CurriculumSchedule` for adaptive training difficulty
- **Parallel simulation:** vectorise the Eb/N0 sweep for throughput
