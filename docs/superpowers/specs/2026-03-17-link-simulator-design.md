# Link-Level Simulator — Design Spec

**Goal:** Add a `LinkSimulator` that runs transmitter → channel → receiver loops to generate BER/SER/PER vs. Eb/N0 curves, validating that SPECTRA's generated waveforms have correct physical behaviour.

**Scope:** v1 implements a perfect-sync coherent receiver (no timing/frequency/phase recovery) with nearest-neighbor constellation slicing for PSK, square QAM, and ASK families. Cross-QAM (32, 128, 512) is excluded from v1 — see Future Work. FEC decoder stubs (Viterbi, LDPC) define the interface for future implementations.

---

## Architecture

```
Transmitter           Channel              Receiver              Metrics
(Rust symbols+idx)    (direct AWGN)        (MF + slicer + demap) (BER/SER/PER)
     |                     |                    |                    |
     v                     v                    v                    v
generate_*_with_indices → RRC shape → add_awgn(Eb/N0) → RRC MF → downsample →
  → [optional impairments] → nearest-neighbor slicer → bit demap → [FEC decode] → compare
```

The `LinkSimulator` orchestrates this loop over an array of Eb/N0 points and returns a `LinkResults` dataclass.

---

## Sub-project 1: Rust Modulator Extensions

### Files

- `rust/src/modulators.rs` (modify)
- `rust/src/lib.rs` (modify — register new functions)

### New Symbol Generator Functions

Add `_with_indices` variants for each modulation family. These use the same xorshift64 PRNG and constellation mappings as the existing generators, but additionally return the symbol index before mapping to a complex point.

**`generate_bpsk_symbols_with_indices(num_symbols, seed) -> (PyArray1<Complex32>, PyArray1<u32>)`**

**`generate_qpsk_symbols_with_indices(num_symbols, seed) -> (PyArray1<Complex32>, PyArray1<u32>)`**

**`generate_psk_symbols_with_indices(num_symbols, m_order, seed) -> (PyArray1<Complex32>, PyArray1<u32>)`**

**`generate_qam_symbols_with_indices(num_symbols, m_order, seed) -> (PyArray1<Complex32>, PyArray1<u32>)`**

Each returns a tuple of `(symbols, indices)` where:
- `symbols`: complex64 array of constellation points (same as existing functions)
- `indices`: uint32 array of constellation point indices (0 to M-1)

The existing `generate_*` functions remain unchanged — the `_with_indices` variants are only used by the link simulator.

### New Constellation Access Functions

Add functions to retrieve the reference constellation for each family:

**`get_bpsk_constellation() -> PyArray1<Complex32>`** — returns `[+1, -1]`

**`get_qpsk_constellation() -> PyArray1<Complex32>`** — returns the 4 QPSK points

**`get_psk_constellation(m_order) -> PyArray1<Complex32>`** — returns M-PSK points

**`get_qam_constellation(m_order) -> PyArray1<Complex32>`** — returns square M-QAM points

These guarantee the receiver uses the exact same constellation as the transmitter.

### Bit Mapping

The Rust constellation orderings use natural indexing (not Gray code). On the Python side:

- `constellation_to_bits(indices, constellation_size) -> np.ndarray` maps indices to bits using the **natural binary** mapping matching the Rust constellation ordering.
- `bits_to_indices(bits, constellation_size) -> np.ndarray` does the inverse.

Since the Rust constellations do not use Gray coding, `theoretical_ber()` formulas in `LinkResults` will only be provided for BPSK (which is inherently Gray-coded with M=2). For QPSK and higher orders, `theoretical_ber()` returns `None` — users can compare against simulation results instead.

**Note:** Adjacent constellation points in the Rust orderings may differ by more than 1 bit. This means BER will be slightly higher than Gray-coded theoretical curves, but this accurately reflects the actual SPECTRA waveform behaviour. A future enhancement could add Gray-coded constellation variants.

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

### Design Note: Receivers vs Transforms

Receivers are intentionally **not** `Transform` subclasses. The `Transform` interface maps `(iq, desc) -> (iq, desc)` — IQ in, IQ out. Receivers map IQ to a fundamentally different domain (symbol indices and bits). A `Receiver` ABC is defined in `base.py` alongside the `Decoder` ABC to establish the interface for future receiver types (e.g., FSK discriminator, OFDM FFT-based).

### `base.py` — `Receiver` ABC, `Decoder` ABC, and Stubs

```python
class Receiver(ABC):
    @abstractmethod
    def demodulate(self, received_iq: np.ndarray) -> Tuple[np.ndarray, np.ndarray]:
        """Demodulate received IQ to (symbol_indices, bits)."""

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

### `coherent.py` — `CoherentReceiver`

Assumes perfect synchronization. Processing pipeline:

1. **RRC matched filter** — convolve received IQ with the transmit RRC filter taps (reuses Rust `apply_rrc_filter_with_taps`), providing optimal SNR at symbol sampling points.
2. **Downsample** — take every `samples_per_symbol`-th sample to get symbol-rate samples.
3. **Constellation slicer** — nearest-neighbor decision: for each received sample, find the closest constellation point from the reference constellation (obtained via `get_*_constellation()` Rust functions). Returns estimated symbol indices.
4. **Bit demapper** — convert symbol indices to bits via `constellation_to_bits()` (natural binary mapping).

```python
class CoherentReceiver(Receiver):
    def __init__(self, waveform: Waveform):
        """Extract constellation, samples_per_symbol, RRC params from waveform.

        Obtains the reference constellation via the Rust get_*_constellation()
        functions to guarantee exact match with the transmitter.
        """

    def demodulate(self, received_iq: np.ndarray) -> Tuple[np.ndarray, np.ndarray]:
        """Returns (estimated_symbol_indices, estimated_bits)."""
```

The receiver extracts everything it needs from the `Waveform` instance — `samples_per_symbol`, RRC `rolloff` and `filter_span` — and loads the constellation from Rust.

**Supported modulation families (v1):** PSK (BPSK, QPSK, 8PSK, 16PSK, 32PSK, 64PSK), square QAM (QAM16, QAM64, QAM256, QAM1024), ASK (OOK, ASK4, ASK8, ASK16, ASK32, ASK64). Cross-QAM (QAM32, QAM128, QAM512) is excluded from v1 because those waveforms use Python-side constellation generation, not the Rust `generate_qam_symbols` path.

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
            waveform: Modulation waveform (must be PSK, square QAM, or ASK).
            channel: Optional list of impairments to apply AFTER noise injection.
                These use the Transform interface and receive a minimal
                SignalDescription stub. AWGN is NOT in this list — it is
                handled internally via direct noise injection.
            decoder: FEC decoder. Default: PassthroughDecoder (no coding).
            num_symbols: Symbols to simulate per Eb/N0 point.
            packet_length: Bits per packet for PER computation.
            seed: Base seed for reproducible simulation.
        """

    def run(self, eb_n0_points: np.ndarray) -> LinkResults:
        """Sweep Eb/N0 values and return BER/SER/PER curves."""
```

**Per Eb/N0 point, the `.run()` loop:**

1. **Transmit** — call `generate_*_with_indices(num_symbols, seed)` to get `(tx_symbols, tx_indices)`. Convert `tx_indices` to `tx_bits` via `constellation_to_bits()`. Apply RRC pulse shaping to `tx_symbols` to get `tx_iq`.

2. **Noise injection (direct, not via AWGN transform)** — compute noise variance analytically from Eb/N0:
   ```
   bits_per_symbol = log2(M)
   Eb = mean(|tx_iq|^2) * samples_per_symbol / bits_per_symbol
   N0 = Eb / (10^(eb_n0_db/10))
   noise_variance = N0 / 2  (per real/imag component)
   ```
   Generate circular complex Gaussian noise using a per-point seeded RNG: `rng = np.random.default_rng((self.seed, point_index))`. Add noise to `tx_iq`. This bypasses the `AWGN` transform entirely, avoiding the empirical power measurement issue and ensuring reproducible, independent noise per Eb/N0 point.

3. **Additional impairments** — if `channel` is provided, apply each `Transform` in sequence to the noisy IQ. A minimal `SignalDescription` is constructed with the waveform's label, bandwidth, and the current SNR.

4. **Receive** — `CoherentReceiver.demodulate(rx_iq)` returns `(rx_indices, rx_bits)`.

5. **Decode** — `decoder.decode(rx_bits)` (passthrough by default).

6. **Score** — `bit_error_rate(tx_bits, rx_bits)` for BER, `symbol_error_rate(tx_indices, rx_indices)` for SER, `packet_error_rate(tx_bits, rx_bits, packet_length)` for PER.

**Seeding:** The transmitter seed is fixed per `LinkSimulator` instance (same transmitted sequence for all Eb/N0 points). Each Eb/N0 point gets an independent noise realisation via `np.random.default_rng((self.seed, point_index))`, ensuring reproducibility and independence.

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

        Returns None for all other modulations (Rust constellation orderings
        are not Gray-coded, so standard formulas do not apply directly).
        """
```

---

## Build Order & Dependencies

```
Sub-project 1: Rust modulators    (no Python dependencies)
Sub-project 2: Metrics            (standalone functions)
Sub-project 3: Receivers          (depends on waveform API, Rust constellation access)
Sub-project 4: Link simulator     (depends on 1, 2, 3, impairments)
```

Sub-projects 1 and 2 are independent. Sub-project 3 depends on 1 (for constellation access). Sub-project 4 integrates everything.

---

## Scope Exclusions (v1)

- **Cross-QAM (QAM32, QAM128, QAM512):** These use Python-side constellation generation (`_CrossQAMBase`), not the Rust `generate_qam_symbols` path. Supporting them requires adding Python-side `generate_with_indices` methods, which is deferred to v2.
- **FSK/OFDM receivers:** Different demodulation structures beyond matched-filter + slicer.
- **Synchronization:** No timing, frequency, or phase recovery.

---

## Future Work (out of scope for v1)

- **Cross-QAM support:** Add Python-side `generate_with_indices` to `_CrossQAMBase`
- **Gray-coded constellations:** Add optional Gray-coded Rust constellation variants for theoretical BER validation
- **Synchronization:** symbol timing recovery (Mueller-Muller, Gardner), carrier frequency offset estimation, phase tracking
- **FEC implementations:** actual Viterbi and LDPC decoders (the stubs define the interface)
- **Soft-decision decoding:** LLR computation from received samples instead of hard bits
- **FSK/OFDM receivers:** different demodulation structures
- **Curriculum-by-BER:** use measured BER as feedback to `CurriculumSchedule` for adaptive training difficulty
- **Parallel simulation:** vectorise the Eb/N0 sweep for throughput
