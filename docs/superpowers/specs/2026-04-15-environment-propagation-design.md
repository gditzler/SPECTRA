# Environment & Propagation Modeling Layer

**Date:** 2026-04-15
**Status:** Draft
**Scope:** New `environment` module for physics-based scenario modeling

## Motivation

SPECTRA has strong waveform diversity (62 classes) and a mature impairment pipeline, but lacks a physical environment model. Users manually select SNR, fading type, and Doppler parameters rather than having them derived from transmitter/receiver geometry and propagation physics. This limits realism for users who want to simulate specific RF scenarios (urban cellular, indoor, suburban) rather than just generate labeled ML training data.

## Goals

1. Add a propagation-aware `Environment` layer that computes per-signal impairment parameters from physical geometry and link budgets.
2. Provide a pluggable `PropagationModel` abstraction with 3 initial terrestrial models and presets.
3. Integrate cleanly with the existing `Composer` and impairment pipeline without modifying them.
4. Support YAML serialization for reproducible scenario definitions.
5. Design for future extension to 3D geometry, satellite/radar domains, and mobility.

## Non-Goals

- Replacing or modifying the existing `Composer`, `SceneConfig`, or impairment classes.
- Time-varying propagation within a single `generate()` call (future mobility mixin).
- Full 3D geometry calculations (2D now, 3D-ready abstractions).
- Satellite, radar-specific, or atmospheric propagation models (future sub-projects).
- Ray tracing or terrain-aware propagation.

## Architecture

### Relationship to Existing Modules

```
Environment (geometry + propagation)        [NEW]
  -> derives per-signal LinkParams
  -> link_params_to_impairments()           [NEW]
  -> feeds into Composer (spectral placement + mixing)  [UNCHANGED]
  -> or feeds directly into single-signal pipeline      [UNCHANGED]
```

The `Environment` is a separate layer that sits upstream of the `Composer`. It produces impairment parameters; the `Composer` consumes them. Users who don't need physics-based propagation are unaffected.

### File Structure

```
python/spectra/environment/
    __init__.py          # public API exports
    core.py              # Environment, Emitter, ReceiverConfig, LinkParams
    position.py          # Position dataclass, geometry helpers
    propagation.py       # PropagationModel ABC, FreeSpacePathLoss, LogDistancePL, COST231HataPL
    presets.py           # propagation_presets dict
    integration.py       # link_params_to_impairments(), Composer bridge utilities
```

## Core Abstractions

### Position

Dataclass holding `(x, y)` with optional `z=None`. All distances in meters.

```python
@dataclass
class Position:
    x: float  # meters
    y: float  # meters
    z: float | None = None  # meters, optional for future 3D

    def distance_to(self, other: Position) -> float: ...
    def angle_to(self, other: Position) -> float: ...
    def bearing_to(self, other: Position) -> float: ...
```

When `z` is `None`, all geometry helpers use 2D calculations. When both positions have `z`, 3D Euclidean distance is used. The `angle_to` method returns azimuth in 2D; elevation angle is computed only when `z` is available on both positions.

### Emitter

Describes a transmitting source.

```python
@dataclass
class Emitter:
    waveform: Waveform
    position: Position
    power_dbm: float
    freq_hz: float
    velocity_mps: tuple[float, float] | None = None  # (vx, vy) for Doppler
    antenna_gain_dbi: float = 0.0
```

`velocity_mps` is optional. When provided, the `Environment` computes Doppler shift from the radial velocity component along the emitter-to-receiver bearing. When `None`, Doppler is zero.

### ReceiverConfig

Describes the receive side. Named `ReceiverConfig` to avoid collision with `spectra.receivers.Receiver`.

```python
@dataclass
class ReceiverConfig:
    position: Position
    noise_figure_db: float = 6.0
    bandwidth_hz: float = 1.0e6
    antenna_gain_dbi: float = 0.0
    temperature_k: float = 290.0  # standard ambient
```

### LinkParams

Output of the link budget computation for a single emitter.

```python
@dataclass
class LinkParams:
    emitter_index: int
    snr_db: float
    path_loss_db: float
    received_power_dbm: float
    delay_s: float
    doppler_hz: float
    distance_m: float
    fading_suggestion: str | None  # e.g., "rician_k6", "rayleigh", None
```

This is a plain dataclass. Users can modify any field after `compute()` returns, before passing it to the impairment bridge. This is the override mechanism — no special API needed.

### Environment

The orchestrator that ties everything together.

```python
class Environment:
    def __init__(
        self,
        propagation: PropagationModel,
        emitters: list[Emitter],
        receiver: ReceiverConfig,
    ): ...

    def compute(self, seed: int | None = None) -> list[LinkParams]:
        """Compute link parameters for each emitter."""
        ...

    @classmethod
    def from_yaml(cls, path: str) -> Environment: ...

    def to_yaml(self, path: str) -> None: ...
```

The `compute()` method:
1. For each emitter, computes distance from emitter to receiver via `Position.distance_to()`.
2. Calls `self.propagation(distance_m, freq_hz)` to get `PathLossResult`.
3. Computes received power: `power_dbm + antenna_gain_tx + antenna_gain_rx - path_loss_db`.
4. Computes noise floor: `10*log10(k_B * temperature_k * bandwidth_hz) + noise_figure_db` in dBm (which equals `-174 + 10*log10(bandwidth_hz) + noise_figure_db` at the standard 290 K).
5. Derives `snr_db = received_power_dbm - noise_power_dbm`.
6. Computes `delay_s = distance_m / c`.
7. If `velocity_mps` is set, computes radial velocity and `doppler_hz = (v_radial / c) * freq_hz`.
8. Sets `fading_suggestion` based on `PathLossResult` metadata (K-factor -> Rician, delay spread -> TDL, else None).

The optional `seed` parameter is passed to the propagation model for reproducible shadow fading samples.

## Propagation Models

### ABC

```python
@dataclass
class PathLossResult:
    path_loss_db: float
    shadow_fading_db: float = 0.0  # realized shadow fading sample
    rms_delay_spread_s: float | None = None
    k_factor_db: float | None = None
    angular_spread_deg: float | None = None  # future use

class PropagationModel(ABC):
    @abstractmethod
    def __call__(self, distance_m: float, freq_hz: float, **kwargs) -> PathLossResult:
        """Compute path loss for given distance and frequency."""
        ...
```

`PathLossResult` is deliberately extensible. Future models can populate additional fields without breaking existing code. The `**kwargs` on `__call__` allows models to accept additional context (e.g., antenna heights, environment type) without changing the ABC signature.

### FreeSpacePathLoss

Friis equation: `PL(dB) = 20*log10(d) + 20*log10(f) + 20*log10(4*pi/c)`

No shadowing, no multipath. Baseline for line-of-sight and satellite links.

```python
class FreeSpacePathLoss(PropagationModel):
    def __call__(self, distance_m: float, freq_hz: float, **kwargs) -> PathLossResult:
        ...
```

No configurable parameters. Returns `PathLossResult` with only `path_loss_db` populated.

### LogDistancePL

`PL(dB) = PL(d0) + 10*n*log10(d/d0) + X_sigma`

where `PL(d0)` is free-space path loss at reference distance `d0`, and `X_sigma` is a zero-mean Gaussian shadow fading sample with standard deviation `sigma_db`.

```python
class LogDistancePL(PropagationModel):
    def __init__(
        self,
        n: float = 3.0,           # path loss exponent
        sigma_db: float = 0.0,    # shadow fading std dev
        d0: float = 1.0,          # reference distance (m)
    ): ...
```

Typical `n` values: free-space=2.0, indoor=1.6-3.3, urban=2.7-3.5, suburban=3.0-5.0.

### COST231HataPL

COST-231 extension of Okumura-Hata model for 1500-2000 MHz urban/suburban/rural environments.

```python
class COST231HataPL(PropagationModel):
    def __init__(
        self,
        h_bs_m: float = 30.0,      # base station antenna height
        h_ms_m: float = 1.5,       # mobile station antenna height
        environment: str = "urban", # "urban", "suburban", "rural"
    ): ...
```

Implements the standard COST-231 Hata formulas with correction factors for mobile antenna height and environment type. Valid for: `fc` 1500-2000 MHz, `h_bs` 30-200 m, `h_ms` 1-10 m, distance 1-20 km.

### Presets

```python
propagation_presets: dict[str, PropagationModel] = {
    "free_space": FreeSpacePathLoss(),
    "urban_macro": LogDistancePL(n=3.5, sigma_db=8.0),
    "suburban": LogDistancePL(n=3.0, sigma_db=6.0),
    "indoor_office": LogDistancePL(n=2.0, sigma_db=4.0),
    "cost231_urban": COST231HataPL(h_bs_m=30, h_ms_m=1.5, environment="urban"),
}
```

## Integration with Existing Modules

### link_params_to_impairments()

Converts a `LinkParams` into an ordered list of existing SPECTRA `Transform` objects.

```python
def link_params_to_impairments(params: LinkParams) -> list[Transform]:
    """Convert derived link parameters to an impairment chain."""
    impairments = []

    # Doppler shift (if nonzero)
    if abs(params.doppler_hz) > 0.01:
        impairments.append(DopplerShift(shift_hz=params.doppler_hz))

    # Fading (if suggested by propagation model)
    if params.fading_suggestion is not None:
        impairments.append(_fading_from_suggestion(params.fading_suggestion))

    # AWGN at computed SNR (applied last)
    impairments.append(AWGN(snr_db=params.snr_db))

    return impairments
```

The function uses only existing impairment classes. No new impairments are created. The `_fading_from_suggestion()` helper maps string suggestions to configured `RayleighFading`, `RicianFading`, or `TDLChannel` instances.

### Wideband path (with Composer)

```python
env = Environment(
    propagation=LogDistancePL(n=3.5, sigma_db=8.0),
    emitters=[
        Emitter(waveform=QPSK(samples_per_symbol=8), position=Position(500, 200), power_dbm=30, freq_hz=2.4e9),
        Emitter(waveform=FMCW(bandwidth=5e6), position=Position(1200, 0), power_dbm=40, freq_hz=2.4e9),
    ],
    receiver=ReceiverConfig(position=Position(0, 0), noise_figure_db=6.0, bandwidth_hz=10e6),
)
params_list = env.compute(seed=42)

scene = SceneConfig(
    signals=[
        SignalConfig(
            waveform=emitter.waveform,
            impairments=Compose(link_params_to_impairments(params)),
        )
        for emitter, params in zip(env.emitters, params_list)
    ],
    bandwidth=10e6,
    sample_rate=20e6,
)
```

### Narrowband path (single signal)

```python
params = env.compute(seed=42)[0]
iq, desc = env.emitters[0].waveform.generate(num_samples=1024, sample_rate=1e6)
for transform in link_params_to_impairments(params):
    iq, desc = transform(iq, desc)
```

## YAML Serialization

`Environment` supports round-trip YAML serialization following existing SPECTRA benchmark config patterns.

```yaml
environment:
  propagation:
    type: log_distance
    n: 3.5
    sigma_db: 8.0
  receiver:
    position: [0, 0]
    noise_figure_db: 6.0
    bandwidth_hz: 1.0e6
  emitters:
    - waveform:
        type: QPSK
        samples_per_symbol: 8
      position: [500, 200]
      power_dbm: 30
      freq_hz: 2.4e9
    - waveform:
        type: FMCW
        bandwidth: 5.0e6
      position: [1200, 0]
      power_dbm: 40
      freq_hz: 2.4e9
```

The `from_yaml()` / `to_yaml()` methods on `Environment` handle serialization. Propagation model type is resolved from a registry mapping type strings to classes, following the same pattern used for waveform resolution in existing benchmark configs.

## Testing

### Unit Tests

- **`tests/test_position.py`** — 2D distance (3-4-5 triangle), angle, bearing. Verify z=None uses 2D math. Verify z present uses 3D Euclidean distance. Edge cases: zero distance, same position.
- **`tests/test_propagation.py`** — Each model against textbook values:
  - Free-space: 1 km at 2.4 GHz ≈ 100.0 dB
  - Log-distance with n=2 matches free-space at same distance
  - COST-231: compare against published Hata model reference tables
  - Verify sigma_db=0 produces zero shadow fading
- **`tests/test_link_budget.py`** — `Environment.compute()` with free-space model and known geometry. Hand-calculate expected SNR, delay, Doppler. Verify all `LinkParams` fields.
- **`tests/test_integration.py`** — `link_params_to_impairments()` returns correct impairment types. Verify the chain can be applied to IQ data without error. Verify SNR override works (modify `LinkParams.snr_db` before conversion).

### Integration Tests

- End-to-end: Environment → compute → impairments → apply to waveform → verify valid IQ output with approximately expected SNR (within 1-2 dB, measured from output).
- YAML round-trip: serialize Environment to YAML, deserialize, verify `compute()` produces identical `LinkParams`.

### No New Markers

All tests are standard pytest. No Rust FFI involved (pure Python module). No slow computations.

## Example

One new example notebook: **Example 20 — Environment-Driven Scene Generation**

Demonstrates:
1. Creating an `Environment` with multiple emitters at different positions
2. Computing link budgets and inspecting `LinkParams`
3. Overriding a computed parameter (e.g., forcing a specific SNR)
4. Feeding into `Composer` for a wideband scene
5. Single-signal narrowband path
6. Using presets for quick scenario setup
7. YAML serialization round-trip

## Future Extensions (Out of Scope)

These are natural follow-on sub-projects, each with its own spec/plan cycle:

- **Mobility mixin** — Integrate with `targets/` trajectory models for time-varying positions per dataset index
- **3D geometry** — Elevation angles, slant range for satellite/airborne scenarios
- **3GPP 38.901 channel models** — UMa, UMi, RMa, InH with full parameter tables
- **Satellite propagation** — Free-space + atmospheric attenuation, rain fade, orbital Doppler
- **Radar environment** — Radar range equation, RCS-driven SNR, jammer-to-signal ratio
- **Composite propagation** — Chain multiple models (e.g., path loss + atmospheric absorption)
- **Terrain/building awareness** — Obstruction modeling, diffraction losses
