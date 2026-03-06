# Impairments

## Transform ABC

All impairments implement `Transform.__call__`:

```python
def __call__(
    self,
    iq: np.ndarray,          # complex64 input signal
    desc: SignalDescription,  # ground-truth metadata
    **kwargs,                 # typically sample_rate=float
) -> Tuple[np.ndarray, SignalDescription]:
    ...
```

A transform may:
- Modify `iq` in-place or return a new array
- Update `desc` fields (e.g., `FrequencyOffset` shifts `f_low` and `f_high`)
- Leave `desc` unchanged (e.g., `AWGN` adds noise but doesn't change frequency bounds)

---

## Compose Pipeline

`Compose` chains transforms sequentially, threading `(iq, desc)` through each one.
All `**kwargs` are forwarded to every transform — always pass `sample_rate`.

```python
from spectra import AWGN, FrequencyOffset, PhaseNoise, Compose

pipeline = Compose([
    AWGN(snr_range=(-5.0, 20.0)),
    FrequencyOffset(max_offset=5000.0),
    PhaseNoise(level=-80.0),
])

iq_out, desc_out = pipeline(iq, desc, sample_rate=1e6)
```

---

## Impairment Reference

| Class | Key parameters | Description |
|-------|---------------|-------------|
| `AWGN` | `snr` or `snr_range` | Additive white Gaussian noise |
| `ColoredNoise` | `color`, `snr_range` | Colored noise (pink, brown, etc.) |
| `FrequencyOffset` | `max_offset` | Constant carrier frequency offset in Hz |
| `FrequencyDrift` | `max_drift`, `drift_rate` | Time-varying frequency drift |
| `DopplerShift` | `velocity`, `profile` | Doppler-shifted frequency via constant or linear velocity profile |
| `PhaseNoise` | `level` | Oscillator phase noise from PSD model |
| `PhaseOffset` | `max_offset` | Constant phase rotation |
| `SpectralInversion` | — | Conjugates signal (mirrors spectrum) |
| `IQImbalance` | `amplitude_imbalance`, `phase_imbalance` | Receiver I/Q gain and phase mismatch |
| `DCOffset` | `dc_offset` | Adds complex DC bias to signal |
| `Quantization` | `num_bits` | ADC quantization clipping and rounding |
| `SampleRateOffset` | `max_offset_ppm` | Clock frequency offset in parts-per-million |
| `PassbandRipple` | `ripple_db`, `num_taps` | Filter passband ripple |
| `RayleighFading` | `num_taps`, `max_doppler` | Rayleigh flat/selective fading (Jakes model) |
| `RicianFading` | `k_factor`, `num_taps`, `max_doppler` | Rician fading with LOS component |
| `AdjacentChannelInterference` | `interference_power`, `offset_hz` | ACI from an adjacent channel |
| `IntermodulationProducts` | `ip3_dbm` | Third-order intermodulation distortion |

---

## AWGN

`AWGN` computes noise power from the measured signal power:

```
signal_power = mean(|iq|²)
noise_power  = signal_power / 10^(SNR_dB / 10)
noise        = sqrt(noise_power / 2) * (randn + j*randn)
```

Use `snr` for a fixed SNR, or `snr_range=(min_db, max_db)` for per-sample
uniform random SNR — the latter creates diversity across the dataset.

```python
# Fixed SNR
awgn = AWGN(snr=10.0)

# Random SNR per call
awgn = AWGN(snr_range=(-5.0, 30.0))
```

---

## Fading Channels

### RayleighFading

Models a multipath channel with no dominant line-of-sight component using the
Jakes sum-of-sinusoids model.

```python
from spectra.impairments.fading import RayleighFading

fading = RayleighFading(num_taps=8, max_doppler=50.0)  # 50 Hz max Doppler
iq_faded, desc = fading(iq, desc, sample_rate=1e6)
```

Key parameters:
- `num_taps` — number of multipath delay taps
- `max_doppler` — maximum Doppler frequency in Hz (controls fade rate)

### RicianFading

Extends Rayleigh with a dominant LOS component controlled by the K-factor.
K = 0 collapses to Rayleigh; K → ∞ collapses to AWGN.

```python
from spectra.impairments.fading import RicianFading

fading = RicianFading(k_factor=3.0, num_taps=4, max_doppler=20.0)
```

---

## IQ Imbalance

Models receiver hardware mismatch between I and Q branches.

```python
from spectra.impairments.iq_imbalance import IQImbalance

iq_imb = IQImbalance(amplitude_imbalance=0.5, phase_imbalance=5.0)
# amplitude_imbalance: dB gain difference
# phase_imbalance: degrees of quadrature error
```

---

## Phase Noise

Models oscillator instability using a power spectral density (PSD) shaped noise
process. The `level` parameter sets the overall noise floor in dBc/Hz.

```python
from spectra.impairments.phase_noise import PhaseNoise

pn = PhaseNoise(level=-80.0)  # -80 dBc/Hz
```

---

## SNR Conventions

!!! note "SNR is per-signal, before scene mixing"
    The SNR set by `AWGN` or `SceneConfig.snr_range` is the per-signal SNR
    **before** multiple signals are summed in a wideband scene. After mixing,
    the effective SNR per signal depends on the number and power of other signals.
