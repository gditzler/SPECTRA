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
| `TDLChannel` | `profile`, `doppler_hz` | 3GPP TDL multipath fading (TDL-A through E, Pedestrian/Vehicular) |
| `RappPA` | `smoothness`, `saturation` | Rapp solid-state power amplifier nonlinearity |
| `SalehPA` | `alpha_a`, `beta_a`, `alpha_p`, `beta_p` | Saleh TWT power amplifier AM/AM and AM/PM |
| `FractionalDelay` | `max_delay` | Sub-sample timing offset via sinc interpolation |
| `SamplingJitter` | `jitter_std` | Random per-sample timing jitter |
| `MIMOChannel` | `n_tx`, `n_rx`, `channel_type` | MIMO flat/TDL channel with optional spatial correlation |

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

## MIMO Channel

`MIMOChannel` maps N_tx transmit streams to N_rx receive streams through a spatial channel matrix. It supports flat (i.i.d. Rayleigh) and TDL (per-element 3GPP) modes.

```python
from spectra.impairments.mimo_channel import MIMOChannel

# Flat Rayleigh MIMO
mimo = MIMOChannel(n_tx=2, n_rx=4, channel_type="flat")
rx, desc = mimo(tx_streams, desc, sample_rate=1e6)
# tx_streams: (2, N) -> rx: (4, N)
```

**Input handling:**

- 2D input `(n_tx, N)` — used directly
- 1D input `(N,)` with `n_tx=1` — auto-expanded to `(1, N)`
- 1D input `(N,)` with `n_tx > 1` — replicated to all TX antennas

### Spatial Correlation

Apply Kronecker-model spatial correlation:

```python
from spectra import exponential_correlation, MIMOChannel

R_rx = exponential_correlation(4, rho=0.8)
R_tx = exponential_correlation(2, rho=0.5)

mimo = MIMOChannel(
    n_tx=2, n_rx=4,
    channel_type="flat",
    spatial_correlation_rx=R_rx,
    spatial_correlation_tx=R_tx,
)
```

### Antenna Utilities

| Function | Description |
|----------|-------------|
| `steering_vector(n, angle, d_lambda)` | ULA steering vector |
| `exponential_correlation(n, rho)` | Exponential correlation matrix R[i,j] = rho^&#124;i-j&#124; |
| `kronecker_correlation(R_tx, R_rx)` | Full spatial correlation via Kronecker product |

```python
from spectra import steering_vector, exponential_correlation

# 8-element ULA at 30 degrees
a = steering_vector(8, angle_rad=np.pi / 6)

# Exponential correlation for 4 antennas
R = exponential_correlation(4, rho=0.9)
```

---

## SNR Conventions

!!! note "SNR is per-signal, before scene mixing"
    The SNR set by `AWGN` or `SceneConfig.snr_range` is the per-signal SNR
    **before** multiple signals are summed in a wideband scene. After mixing,
    the effective SNR per signal depends on the number and power of other signals.

## Auto-impairment Chain (from Propagation)

When you use `Environment` + `link_params_to_impairments()`, the
chain is selected based on what the propagation model populates on
`PathLossResult`:

| `rms_delay_spread_s` | `k_factor_db` | Emitted fading stage |
|----------------------|---------------|----------------------|
| set | set | `TDLChannel` (TDL-D base, scaled, Rician K embedded) |
| set | None | `TDLChannel` (TDL-B base, scaled, Rayleigh) |
| None | set | `RicianFading(k_factor=10^(k_db/10))` |
| None | None | Falls back to `fading_suggestion` string if present |

3GPP TR 38.901 models (UMa, UMi, RMa, InH) populate both fields;
free-space, log-distance, Hata-family, and P.1411 models do not. See
[propagation.md](propagation.md) for details.
