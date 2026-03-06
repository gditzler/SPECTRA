# Waveforms

## Waveform ABC

All waveform generators inherit from `Waveform` and implement three members:

```python
from spectra.waveforms.base import Waveform

class Waveform(ABC):
    def generate(self, num_symbols, sample_rate, seed=None) -> np.ndarray: ...
    def bandwidth(self, sample_rate) -> float: ...
    @property
    def label(self) -> str: ...
```

- `generate()` returns a `complex64` NumPy array. The actual number of samples
  depends on `samples_per_symbol` for pulse-shaped waveforms.
- `bandwidth()` returns the occupied bandwidth in Hz, always <= `sample_rate`.
- `label` is the string class name used as the dataset target (e.g. `"QPSK"`).

**Minimal custom waveform:**

```python
import numpy as np
from spectra.waveforms.base import Waveform

class MySineWave(Waveform):
    def __init__(self, freq_hz: float = 1000.0):
        self.freq_hz = freq_hz

    def generate(self, num_symbols, sample_rate, seed=None):
        t = np.arange(num_symbols) / sample_rate
        return np.exp(1j * 2 * np.pi * self.freq_hz * t).astype(np.complex64)

    def bandwidth(self, sample_rate):
        return sample_rate / 100.0  # narrow tone

    @property
    def label(self):
        return "SineWave"
```

---

## RRC-Filtered Waveforms

Most digital modulations in SPECTRA use Root-Raised-Cosine (RRC) pulse shaping
via `_RRCWaveformBase`. Symbols are generated in Rust, upsampled by
`samples_per_symbol`, then convolved with an RRC filter.

**Constructor parameters (shared by all RRC waveforms):**

| Parameter | Default | Description |
|-----------|---------|-------------|
| `samples_per_symbol` | 8 | Upsampling factor |
| `rolloff` | 0.35 | Excess bandwidth factor in [0, 1] |
| `filter_span` | 10 | Filter half-length in symbols |

**Bandwidth formula:**

```
symbol_rate = sample_rate / samples_per_symbol
bandwidth   = symbol_rate * (1 + rolloff)
```

Higher `rolloff` widens the spectrum; lower values require more precise
symbol timing at the receiver.

---

## PSK Family

Gray-coded phase-shift keying with RRC pulse shaping.

| Class | Label | Constellation order |
|-------|-------|-------------------|
| `BPSK` | `"BPSK"` | 2 (±1 on real axis) |
| `QPSK` | `"QPSK"` | 4 (pi/4, 3pi/4, -3pi/4, -pi/4) |
| `PSK8` | `"8PSK"` | 8 |
| `PSK16` | `"16PSK"` | 16 |
| `PSK32` | `"32PSK"` | 32 |
| `PSK64` | `"64PSK"` | 64 |

```python
from spectra import QPSK, BPSK

iq = QPSK(samples_per_symbol=8, rolloff=0.35).generate(1000, sample_rate=1e6, seed=0)
```

---

## QAM Family

Quadrature amplitude modulation with square Gray-coded constellations and RRC shaping.

| Class | Label |
|-------|-------|
| `QAM16` | `"16QAM"` |
| `QAM32` | `"32QAM"` |
| `QAM64` | `"64QAM"` |
| `QAM128` | `"128QAM"` |
| `QAM256` | `"256QAM"` |
| `QAM512` | `"512QAM"` |
| `QAM1024` | `"1024QAM"` |

All inherit from `_QAMBase` and accept the same RRC constructor arguments.

---

## ASK / OOK Family

Amplitude-shift keying with RRC pulse shaping.

| Class | Label | Description |
|-------|-------|-------------|
| `OOK` | `"OOK"` | On-off keying (binary ASK, levels 0 and 1) |
| `ASK4` | `"4ASK"` | 4-level ASK |
| `ASK8` | `"8ASK"` | 8-level ASK |
| `ASK16` | `"16ASK"` | 16-level ASK |
| `ASK32` | `"32ASK"` | 32-level ASK |
| `ASK64` | `"64ASK"` | 64-level ASK |

---

## FSK Family

Frequency-shift keying variants. All accept `samples_per_symbol` and `modulation_index`.

| Class | Label | Description |
|-------|-------|-------------|
| `FSK` | `"FSK"` | Binary FSK |
| `FSK4` | `"4FSK"` | 4-tone FSK |
| `FSK8` | `"8FSK"` | 8-tone FSK |
| `FSK16` | `"16FSK"` | 16-tone FSK |
| `GFSK` | `"GFSK"` | Gaussian FSK (Gaussian pre-filter before FM modulation) |
| `GFSK4` | `"4GFSK"` | 4-tone GFSK |
| `GFSK8` | `"8GFSK"` | 8-tone GFSK |
| `GFSK16` | `"16GFSK"` | 16-tone GFSK |
| `GMSK` | `"GMSK"` | Gaussian MSK (continuous-phase GFSK with modulation index 0.5) |
| `GMSK4` | `"4GMSK"` | 4-tone GMSK |
| `GMSK8` | `"8GMSK"` | 8-tone GMSK |
| `MSK` | `"MSK"` | Minimum-shift keying (continuous-phase FSK, modulation index 0.5) |
| `MSK4` | `"4MSK"` | 4-tone MSK |
| `MSK8` | `"8MSK"` | 8-tone MSK |

**Key distinctions:**

- **FSK** — discontinuous phase; tones are switched abruptly between symbols.
- **GFSK** — Gaussian pre-filter smooths the frequency transitions.
- **MSK** — FSK with modulation index exactly 0.5 for continuous phase.
- **GMSK** — Gaussian-filtered MSK; standard in GSM/Bluetooth.

---

## AM / FM Family

Analog modulations using baseband message signals.

| Class | Label | Description |
|-------|-------|-------------|
| `AMDSB` | `"AM-DSB"` | Double-sideband AM with carrier |
| `AMDSB_SC` | `"AM-DSB-SC"` | Double-sideband suppressed-carrier |
| `AMLSB` | `"AM-LSB"` | Lower-sideband SSB |
| `AMUSB` | `"AM-USB"` | Upper-sideband SSB |
| `FM` | `"FM"` | Narrowband frequency modulation |

---

## OFDM Family

Orthogonal frequency-division multiplexing. `OFDM` accepts:

| Parameter | Default | Description |
|-----------|---------|-------------|
| `num_subcarriers` | 64 | Number of OFDM subcarriers |
| `cp_len` | 16 | Cyclic prefix length in samples |
| `pilot_spacing` | 8 | Pilot subcarrier interval |

Preconfigured variants:

| Class | Label | Subcarriers |
|-------|-------|-------------|
| `OFDM72` | `"OFDM-72"` | 72 |
| `OFDM128` | `"OFDM-128"` | 128 |
| `OFDM180` | `"OFDM-180"` | 180 |
| `OFDM300` | `"OFDM-300"` | 300 |
| `OFDM512` | `"OFDM-512"` | 512 |
| `OFDM600` | `"OFDM-600"` | 600 |
| `OFDM1024` | `"OFDM-1024"` | 1024 |
| `OFDM2048` | `"OFDM-2048"` | 2048 |

---

## Radar / Spread-Spectrum

| Class | Label | Key parameters | Description |
|-------|-------|----------------|-------------|
| `LFM` | `"LFM"` | `sweep_bandwidth`, `pulse_duration` | Linear frequency-modulated chirp pulse |
| `ChirpSS` | `"ChirpSS"` | `spreading_factor` | Chirp spread-spectrum (LoRa-style) |
| `DSSS_BPSK` | `"DSSS-BPSK"` | `chip_rate`, `spreading_factor` | Direct-sequence spread spectrum with BPSK |
| `ZadoffChu` | `"ZadoffChu"` | `root`, `length` | Constant-envelope Zadoff-Chu sequence |
| `BarkerCode` | `"Barker"` | `length` | Barker-coded pulse (lengths 2, 3, 4, 5, 7, 11, 13) |
| `FrankCode` | `"Frank"` | `M` | Frank polyphase code (M² chips) |
| `CostasCode` | `"Costas"` | `length` | Costas frequency-hopping array |
| `P1Code` | `"P1"` | `M` | P1 polyphase code |
| `P2Code` | `"P2"` | `M` | P2 polyphase code |
| `P3Code` | `"P3"` | `N` | P3 polyphase code |
| `P4Code` | `"P4"` | `N` | P4 polyphase code |

---

## Utility Waveforms

| Class | Label | Description |
|-------|-------|-------------|
| `Tone` | `"Tone"` | Complex sinusoid at a fixed frequency. Constructor: `freq_hz`. |
| `Noise` | `"Noise"` | Gaussian noise (baseline / no-signal class). |
