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
| `OFDM256` | `"OFDM-256"` | 256 |
| `OFDM300` | `"OFDM-300"` | 300 |
| `OFDM512` | `"OFDM-512"` | 512 |
| `OFDM600` | `"OFDM-600"` | 600 |
| `OFDM900` | `"OFDM-900"` | 900 |
| `OFDM1200` | `"OFDM-1200"` | 1200 |
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

---

## Radar — FMCW, Pulsed, and Coded

These classes model full radar emission modes. Each generates a time-domain IQ
sequence that includes pulse trains, sweep ramps, or coded bursts ready for
downstream matched filtering and Doppler processing.

| Class | Label | Key kwargs | Description |
|-------|-------|------------|-------------|
| `PulsedRadar` | `"PulsedRadar"` | `pulse_width_samples`, `pri_samples`, `num_pulses`, `pulse_shape`, `pri_stagger`, `pri_jitter_fraction` | Simple pulsed radar waveform with configurable pulse shape and PRI. |
| `PulseDoppler` | `"PulseDoppler"` | `prf_mode`, `num_pulses_per_cpi`, `pulse_width_samples`, `num_cpis` | Pulse-Doppler radar waveform. |
| `FMCW` | `"FMCW"` | `sweep_bandwidth_fraction`, `sweep_samples`, `idle_samples`, `num_sweeps`, `sweep_type` | Frequency-Modulated Continuous Wave radar waveform. |
| `SteppedFrequency` | `"SteppedFrequency"` | `num_steps`, `samples_per_step`, `freq_step_fraction`, `num_bursts` | Stepped-frequency radar waveform. |
| `NonlinearFM` | `"NonlinearFM"` | `sweep_type`, `bandwidth_fraction`, `num_samples` | Nonlinear frequency modulation radar waveform. |
| `BarkerCodedPulse` | `"BarkerCodedPulse"` | `barker_length`, `samples_per_chip`, `pri_samples`, `num_pulses` | Barker-coded pulsed radar waveform. |
| `PolyphaseCodedPulse` | `"PolyphaseCodedPulse"` | `code_type`, `code_order`, `samples_per_chip`, `pri_samples`, `num_pulses` | Polyphase-coded pulsed radar waveform. |

```python
from spectra.waveforms.radar import FMCW, PulsedRadar, PolyphaseCodedPulse

# FMCW with default sawtooth sweep
wf = FMCW(sweep_bandwidth_fraction=0.5, num_sweeps=16)
iq = wf.generate(num_symbols=1, sample_rate=50e6, seed=0)
print(wf.label, iq.shape)  # FMCW (5120,)

# Pulsed radar with Gaussian pulse shape and two-PRI stagger
pr = PulsedRadar(pulse_shape="gaussian", pri_stagger=[512, 576])
iq = pr.generate(num_symbols=1, sample_rate=10e6, seed=0)
print(pr.label, iq.shape)  # PulsedRadar (varies)

# Frank polyphase coded pulse
pcp = PolyphaseCodedPulse(code_type="frank", code_order=4)
iq = pcp.generate(num_symbols=1, sample_rate=10e6, seed=0)
print(pcp.label, iq.shape)  # PolyphaseCodedPulse (16384,)
```

**See also:** [Radar Datasets](datasets.md#radar-datasets) for dataset
wrappers that combine these waveforms with target injection, clutter, and
matched-filter post-processing.

---

## Spread Spectrum — DSSS, FHSS, THSS, CDMA

These classes extend the basic `DSSS_BPSK` / `LFM` / `ChirpSS` entries in the
table above with richer modulation models, multi-user CDMA, and time-hopping.

| Class | Label | Key kwargs | Description |
|-------|-------|------------|-------------|
| `DSSS_BPSK` | `"DSSS-BPSK"` | `code_type`, `code_order`, `samples_per_chip`, `code_index` | Direct-Sequence Spread Spectrum with BPSK modulation. |
| `DSSS_QPSK` | `"DSSS-QPSK"` | `code_type`, `code_order`, `samples_per_chip`, `code_index` | Direct-Sequence Spread Spectrum with QPSK modulation. |
| `FHSS` | `"FHSS"` | `num_channels`, `hop_pattern`, `dwell_samples`, `modulation` | Frequency Hopping Spread Spectrum. |
| `THSS` | `"THSS"` | `num_frames`, `slots_per_frame`, `pulse_samples`, `pulse_shape` | Time Hopping Spread Spectrum. |
| `CDMA_Forward` | `"CDMA-Forward"` | `num_users`, `spreading_factor`, `user_powers` | CDMA Forward Link (downlink) waveform. |
| `CDMA_Reverse` | `"CDMA-Reverse"` | `num_users`, `spreading_factor`, `user_powers` | CDMA Reverse Link (uplink) waveform. |
| `ChirpSS` | `"ChirpSS"` | `spreading_factor` | LoRa-like Chirp Spread Spectrum waveform. |

```python
from spectra.waveforms.spread_spectrum import DSSS_QPSK, FHSS, CDMA_Forward

# DSSS with Gold codes
wf = DSSS_QPSK(code_type="gold", code_order=5, samples_per_chip=4)
iq = wf.generate(num_symbols=128, sample_rate=10e6, seed=0)
print(wf.label, iq.shape)  # DSSS-QPSK (varies)

# Frequency hopping across 8 channels with BPSK data
fhss = FHSS(num_channels=8, hop_pattern="random", dwell_samples=64)
iq = fhss.generate(num_symbols=64, sample_rate=10e6, seed=0)
print(fhss.label, iq.shape)  # FHSS (4096,)

# 4-user CDMA forward link
cdma = CDMA_Forward(num_users=4, spreading_factor=64)
iq = cdma.generate(num_symbols=128, sample_rate=10e6, seed=0)
print(cdma.label, iq.shape)  # CDMA-Forward (8192,)
```
