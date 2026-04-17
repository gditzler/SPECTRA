# Propagation Models

SPECTRA provides terrestrial path-loss models spanning the major
standards used in RF simulation. All models implement the common
`PropagationModel.__call__(distance_m, freq_hz, **kwargs) -> PathLossResult`
interface and can be dropped directly into `Environment`, used standalone
for link-budget studies, or looked up via the YAML-backed
`_PROPAGATION_REGISTRY`.

## Selection Guidance

| Scenario | Recommended model | Frequency range | Distance range |
|----------|------------------|-----------------|----------------|
| Analytical free-space | `FreeSpacePathLoss` | Any | Any |
| Atmospheric-realistic LOS link | `ITU_R_P525(include_gaseous=True)` | 1 GHz – 100 GHz | Any |
| Parametric macro cell | `LogDistancePL` | Any | Any |
| Legacy 2G/3G urban macro | `OkumuraHataPL` | 150 MHz – 1.5 GHz | 1 – 20 km |
| DCS/PCS urban macro | `COST231HataPL` | 1.5 – 2 GHz | 1 – 20 km |
| 5G urban macro | `GPP38901UMa` | 0.5 – 100 GHz | 10 m – 5 km |
| 5G urban micro (street canyon) | `GPP38901UMi` | 0.5 – 100 GHz | 10 m – 5 km |
| 5G rural macro | `GPP38901RMa` | 0.5 – 30 GHz | 10 m – 10 km |
| 5G indoor hotspot | `GPP38901InH` | 0.5 – 100 GHz | 1 – 150 m |
| Short-range outdoor | `ITU_R_P1411` | 300 MHz – 100 GHz | 50 m – 3 km |

## `PathLossResult`

Every model returns a `PathLossResult`:

```python
@dataclass
class PathLossResult:
    path_loss_db: float
    shadow_fading_db: float = 0.0
    rms_delay_spread_s: float | None = None
    k_factor_db: float | None = None
    angular_spread_deg: float | None = None
```

The optional fields are populated only when the underlying standard
specifies them (currently, 38.901 populates all of them; other models
leave them `None`).

## LOS / NLOS Handling

Models that distinguish LOS from NLOS (`GPP38901*`, `ITU_R_P1411`)
accept a `los_mode` constructor argument:

- `"stochastic"` (default): sample LOS/NLOS from the standard's LOS
  probability at the given 2D distance using the per-call `seed`.
- `"force_los"`: always LOS — useful for best-case studies and unit tests.
- `"force_nlos"`: always NLOS — useful for worst-case studies.

## Automatic Impairment Chain

When a propagation model populates `rms_delay_spread_s` and/or
`k_factor_db`, `link_params_to_impairments()` uses those values:

- `delay_spread + k_factor` → `TDLChannel` (TDL-D base, scaled to the
  target delay spread, with the Rician K-factor embedded).
- `delay_spread` alone → `TDLChannel` (TDL-B base, Rayleigh-flavored).
- `k_factor` alone → `RicianFading(k_factor=...)`.
- Legacy `fading_suggestion` string → mapped as before (back-compat).

See the [impairments guide](impairments.md#auto-impairment-chain-from-propagation)
for the end-to-end chain.

## ITU-R P.525 and P.676

`ITU_R_P525(include_gaseous=True)` stacks free-space loss with a
simplified ITU-R P.676-13 Annex 2 gaseous-attenuation helper covering
1–100 GHz. The helper models horizontal terrestrial paths only; slant-
path support is out of scope. Below 1 GHz, gaseous attenuation is
negligible and the helper returns 0 dB with a one-time warning. Above
100 GHz the helper raises `ValueError`.

## 3GPP TR 38.901 Depth

The 38.901 models implement:

- Path loss (Table 7.4.1-1) including LOS/NLOS branches and scenario-
  specific distance breakpoints.
- LOS probability (Table 7.4.2-1).
- Shadow-fading σ (Table 7.5-6).
- RMS delay spread (lognormal per Table 7.5-6).
- Rician K-factor (LOS only, per Table 7.5-6).
- Azimuth arrival spread (ASA) median (Table 7.5-6).

Full stochastic small-scale parameters (cluster angles, XPR) are not
included — the models expose enough output to drive `TDLChannel` via
the auto-impairment chain, which is the intended integration point.

## ITU-R P.1411 Scope

Only the site-general model from P.1411-12 §4.1.1 is implemented for
the three standard environments: urban high-rise, urban low-rise /
suburban, and residential. Sub-models for street-canyon NLOS,
building-entry loss, and over-roof propagation are deferred to future
work.

## YAML Serialization

Every propagation model is registered in `_PROPAGATION_REGISTRY` and
round-trips through `Environment.to_yaml()` / `Environment.from_yaml()`:

```yaml
environment:
  propagation:
    type: gpp_38_901_uma
    h_bs_m: 25.0
    h_ut_m: 1.5
    los_mode: stochastic
    strict_range: true
  receiver: ...
  emitters: ...
```

## Examples

- `examples/environment/terrestrial_propagation_models.py` —
  comprehensive PL-vs-distance, PL-vs-frequency, LOS probability,
  and shadow-fading-histogram plots for every model.
- `examples/environment/propagation_and_links.py` — link budget with
  multiple emitters.
- `examples/environment/urban_5g_scene.py` — end-to-end 38.901 UMa
  scene driving the auto-impairment chain.
