# Terrestrial Propagation Models — Design Spec

**Date:** 2026-04-16
**Status:** Approved for implementation planning
**Preceded by:** `2026-04-15-environment-propagation-design.md` (established `PropagationModel` ABC, `PathLossResult`, `Environment`, and initial models FreeSpace / LogDistance / COST-231 Hata)

## Goal

Expand SPECTRA's terrestrial propagation model library from the current three models to a comprehensive set covering the major standards used in RF simulation: free-space, log-distance, Okumura-Hata, COST-231 Hata, 3GPP TR 38.901 (UMa, UMi, RMa, InH), ITU-R P.525 (with optional P.676 gaseous absorption), and ITU-R P.1411 (site-general short-range outdoor).

## Non-Goals

- Full stochastic 38.901 channel (cluster angles, XPR, small-scale multipath parameters beyond RMS delay spread, K-factor, and ASA).
- ITU-R P.530 terrestrial link design, P.838 rain-specific attenuation, or P.676 Annex 1 line-by-line model.
- P.1411 sub-models beyond the site-general model (e.g., street-canyon NLOS, building-entry loss, over-roof propagation).
- Rust implementations of propagation models (Python is sufficient; all models are closed-form scalar evaluations).

## Design Decisions (Q&A Summary)

1. **38.901 depth:** path loss + LOS probability + K-factor + RMS delay spread + azimuth arrival spread (ASA). Populate the currently-unused fields of `PathLossResult` so `link_params_to_impairments()` can auto-configure `RicianFading` and `TDLChannel`.
2. **LOS/NLOS handling:** `los_mode: Literal["stochastic", "force_los", "force_nlos"]` on every model that has an LOS/NLOS distinction. Default `stochastic`, seeded via the model's RNG.
3. **P.525 scope:** ITU-R citation wrapper around free-space, with optional P.676 gaseous absorption add-on.
4. **Okumura-Hata:** separate `OkumuraHataPL` class alongside existing `COST231HataPL`. Different validity envelopes → explicit errors for out-of-range use rather than silent model swaps.
5. **Examples:** one comprehensive PL-demo script + one dataset/scene integration script + docs page.

## Architecture

### Module Layout

Split `python/spectra/environment/propagation.py` into a subpackage:

```
python/spectra/environment/propagation/
├── __init__.py          # re-exports all public classes for back-compat
├── _base.py             # PropagationModel ABC, PathLossResult, shared helpers
├── free_space.py        # FreeSpacePathLoss, ITU_R_P525
├── atmospheric.py       # P.676 gaseous attenuation helper (module-level function)
├── empirical.py         # LogDistancePL, OkumuraHataPL, COST231HataPL
├── gpp_38_901.py        # _GPP38901Base, GPP38901UMa, UMi, RMa, InH
└── itu_r_p1411.py       # ITU_R_P1411
```

**Back-compatibility:** `propagation/__init__.py` re-exports every class. Existing imports (`from spectra.environment.propagation import COST231HataPL`) and YAML registry keys (`free_space`, `log_distance`, `cost231_hata`) continue to work.

### Public API

The `PropagationModel.__call__(distance_m, freq_hz, **kwargs) -> PathLossResult` contract and the `PathLossResult` dataclass are unchanged. New models simply populate more of the existing fields.

### Shared Infrastructure (`_base.py`)

- `PropagationModel` ABC (moved from `propagation.py`).
- `PathLossResult` dataclass (moved from `propagation.py`, unchanged).
- `_LOSMode = Literal["stochastic", "force_los", "force_nlos"]` type alias.
- `_resolve_los(los_mode, p_los, rng) -> bool` — single source of truth for LOS sampling.
- `_check_freq_range(freq_hz, lo_hz, hi_hz, model_name, strict=True)` — raises `ValueError` or warns based on `strict`.
- `_check_distance_range(distance_m, lo_m, hi_m, model_name, strict=True)` — same pattern for distance.

### Registry & Presets

Extend `_PROPAGATION_REGISTRY` in `environment/core.py`:

```python
_PROPAGATION_REGISTRY = {
    # existing:
    "free_space":       FreeSpacePathLoss,
    "log_distance":     LogDistancePL,
    "cost231_hata":     COST231HataPL,
    # new:
    "itu_r_p525":       ITU_R_P525,
    "okumura_hata":     OkumuraHataPL,
    "gpp_38_901_uma":   GPP38901UMa,
    "gpp_38_901_umi":   GPP38901UMi,
    "gpp_38_901_rma":   GPP38901RMa,
    "gpp_38_901_inh":   GPP38901InH,
    "itu_r_p1411":      ITU_R_P1411,
}
```

New presets in `environment/presets.py`: `urban_macro_5g` (38.901 UMa @ 3.5 GHz), `urban_micro_mmwave` (38.901 UMi @ 28 GHz), `rural_macro_5g` (38.901 RMa @ 700 MHz), `indoor_office_5g` (38.901 InH @ 3.5 GHz), `urban_hata_4g` (COST-231 urban @ 1.8 GHz), `short_range_urban` (P.1411 urban high-rise @ 2.4 GHz).

## Model Specifications

### 1. `FreeSpacePathLoss` (unchanged)

Existing implementation retained as-is.

### 2. `ITU_R_P525`

ITU-R P.525-4 free-space path loss with optional P.676 gaseous absorption.

```python
class ITU_R_P525(PropagationModel):
    def __init__(
        self,
        include_gaseous: bool = False,
        temperature_k: float = 288.15,
        pressure_hpa: float = 1013.25,
        water_vapor_density_g_m3: float = 7.5,
    ): ...
```

- Returns `PathLossResult(path_loss_db = fspl + gaseous_db)`; multipath fields `None`; shadow fading 0.
- Assumes horizontal terrestrial link (full `distance_m` at surface level). Elevated / slant links are out of scope.

### 3. `atmospheric.gaseous_attenuation_db()`

Module-level function implementing ITU-R P.676-13 Annex 2 simplified model.

```python
def gaseous_attenuation_db(
    distance_m: float,
    freq_hz: float,
    temperature_k: float = 288.15,
    pressure_hpa: float = 1013.25,
    water_vapor_density_g_m3: float = 7.5,
) -> float: ...
```

- Computes specific attenuation γ_o (oxygen) + γ_w (water vapor) in dB/km, then multiplies by path length.
- Valid 1–350 GHz; returns 0.0 below 1 GHz with a one-time `warnings.warn`.
- Standalone helper so 38.901 / P.1411 can consume it later (not wired in this round; YAGNI).

### 4. `OkumuraHataPL`

```python
class OkumuraHataPL(PropagationModel):
    def __init__(
        self,
        h_bs_m: float,
        h_ms_m: float,
        environment: Literal["urban_large", "urban_small_medium", "suburban", "rural"],
        sigma_db: float = 0.0,
        strict_range: bool = True,
        seed: int | None = None,
    ): ...
```

- Validity: 150–1500 MHz, 1–20 km, h_bs ∈ [30, 200] m, h_ms ∈ [1, 10] m.
- Implements Hata (1980) urban basic PL, mobile antenna correction `a(h_ms)` with large / small-medium city variants, and suburban / rural offset terms.
- Populates `path_loss_db` and `shadow_fading_db`; multipath fields `None`.
- `freq_hz > 1.5 GHz` raises `ValueError` pointing to `COST231HataPL`.

### 5. `COST231HataPL` (unchanged)

Existing implementation retained.

### 6. `LogDistancePL` (unchanged)

Existing implementation retained.

### 7. 3GPP TR 38.901 Scenarios

Shared base:

```python
class _GPP38901Base(PropagationModel, abc.ABC):
    def __init__(
        self,
        h_bs_m: float,
        h_ut_m: float,
        los_mode: _LOSMode = "stochastic",
        strict_range: bool = True,
        seed: int | None = None,
    ): ...

    @abc.abstractmethod
    def _los_probability(self, d_2d_m: float) -> float: ...

    @abc.abstractmethod
    def _path_loss_los(self, d_3d, d_2d, f_ghz, h_bs, h_ut) -> float: ...

    @abc.abstractmethod
    def _path_loss_nlos(self, d_3d, d_2d, f_ghz, h_bs, h_ut) -> float: ...

    @abc.abstractmethod
    def _large_scale_params(self, is_los: bool, f_ghz: float) -> tuple[float, float, float, float]:
        """Return (sigma_sf_db, mu_lgDS, sigma_lgDS, asa_deg_median)."""

    @abc.abstractmethod
    def _k_factor_params(self, f_ghz: float) -> tuple[float, float]:
        """Return (mu_k_db, sigma_k_db) for LOS. Called only when is_los=True."""
```

`__call__` orchestrates: compute 2D/3D distance → evaluate `_los_probability` → `_resolve_los` → dispatch LOS/NLOS PL → sample shadow fading → sample RMS delay spread (lognormal per Table 7.5-6) → sample K-factor (LOS only) → return populated `PathLossResult`.

Subclasses per TR 38.901 §7.4.1 Table 7.4.1-1 and §7.5 Table 7.5-6:

| Class | Freq | d_2D | h_bs | h_ut | Notes |
|---|---|---|---|---|---|
| `GPP38901UMa` | 0.5–100 GHz | 10 m – 5 km | 25 m | 1.5–22.5 m | Standard breakpoint `d_BP`; NLOS = max(PL_LOS, PL_NLOS') |
| `GPP38901UMi` | 0.5–100 GHz | 10 m – 5 km | 10 m | 1.5–22.5 m | Same form as UMa, different constants |
| `GPP38901RMa` | 0.5–30 GHz | 10 m – 10 km | 10–150 m | 1–10 m | Extra args: `h_building_m=5.0`, `w_street_m=20.0`; two-slope LOS |
| `GPP38901InH` | 0.5–100 GHz | 1–150 m | 3 m | 1 m | Extra arg: `variant: Literal["mixed_office", "open_office"]` |

Populated `PathLossResult` fields: `path_loss_db`, `shadow_fading_db`, `rms_delay_spread_s`, `k_factor_db` (LOS only; `None` for NLOS), `angular_spread_deg` (ASA median).

### 8. `ITU_R_P1411`

```python
class ITU_R_P1411(PropagationModel):
    def __init__(
        self,
        environment: Literal["urban_high_rise", "urban_low_rise_suburban", "residential"],
        los_mode: _LOSMode = "stochastic",
        strict_range: bool = True,
        seed: int | None = None,
    ): ...
```

- Implements P.1411-12 §4.1.1 site-general model: `L = α·log₁₀(d_m) + β + γ·log₁₀(f_GHz)` with (α, β, γ, σ) coefficients per environment and LOS/NLOS.
- LOS probability per P.1411 §4.3, environment-specific.
- Validity: 300 MHz – 100 GHz, 50 m – 3 km.
- Populates `path_loss_db` and `shadow_fading_db` only. Multipath fields `None` (P.1411 site-general does not specify delay spread or K-factor).
- Other P.1411 sub-models (street-canyon NLOS, building entry, over-roof) documented as future work.

## Integration with Impairment Chain

Today, `LinkParams` carries only a `fading_suggestion: str | None` (e.g., `"rayleigh"`, `"rician_k4"`), and `link_params_to_impairments()` maps that string onto a `RayleighFading()` or `RicianFading(k_factor=...)` instance. This is too lossy: it drops the `rms_delay_spread_s`, `k_factor_db`, and `shadow_fading_db` that 38.901 produces.

**Changes:**

1. **Extend `LinkParams`** in `environment/core.py` with optional fields populated from `PathLossResult`:
   ```python
   shadow_fading_db: float = 0.0
   rms_delay_spread_s: float | None = None
   k_factor_db: float | None = None
   angular_spread_deg: float | None = None
   ```
   `Environment.compute()` fills these directly from the `PathLossResult` returned by the propagation model. `shadow_fading_db` is added into the link budget so it affects `received_power_dbm` and `snr_db` (already handled implicitly by `path_loss_db` in the current code; this just exposes it separately for diagnostics).

2. **Extend `link_params_to_impairments()`** with this selection order (first match wins for the fading slot):
   - If `rms_delay_spread_s is not None`: emit `TDLChannel` sized to that delay spread. If `k_factor_db is not None`, use a Rician-flavored profile (TDL-D or TDL-E); otherwise Rayleigh-flavored (TDL-A/B/C).
   - Elif `k_factor_db is not None`: emit `RicianFading(k_factor=10**(k_factor_db/10))`.
   - Elif `fading_suggestion is not None`: fall back to current string-based path (back-compat for FreeSpace / LogDistance / Hata).
   - Else: no fading stage.

3. **Resulting chains:**
   - 38.901 LOS → `DopplerShift → TDLChannel(delay_spread=…, profile=TDL-D, k_factor=…) → AWGN`
   - 38.901 NLOS → `DopplerShift → TDLChannel(delay_spread=…, profile=TDL-B) → AWGN`
   - P.1411 / Okumura-Hata / COST-231 → `DopplerShift → RayleighFading | RicianFading → AWGN` (via `fading_suggestion`, unchanged)
   - Free-space / P.525 → `DopplerShift → AWGN` (no fading)

These are additive changes: existing model outputs and existing call sites continue to work without modification.

## Testing Strategy

Extend `tests/test_propagation.py` with per-model test classes. For each new model:

- **Reference values:** compare `path_loss_db` against hand-computed values at canonical (freq, distance, height) points matching the standard's formula.
- **Monotonicity:** PL monotonically increases with distance at fixed frequency.
- **Frequency scaling:** PL increases with frequency at fixed distance (where the standard specifies so).
- **LOS vs NLOS:** `force_nlos` PL ≥ `force_los` PL at the same evaluation point.
- **Seed reproducibility:** same seed → identical `PathLossResult`; different seeds → different.
- **Validity envelope:** `ValueError` raised outside `freq_hz` / `distance_m` range; `strict_range=False` emits `UserWarning` instead.
- **Populated fields:** 38.901 returns non-None `rms_delay_spread_s`, `k_factor_db` (LOS), `angular_spread_deg`.

New `tests/test_atmospheric.py` for `gaseous_attenuation_db`: check known lines at 22 GHz (water vapor), 60 GHz (oxygen complex), and 1 GHz (negligible).

Extend `tests/test_environment_integration.py`:

- YAML round-trip for each new model via `Environment.to_yaml` / `from_yaml`.
- `link_params_to_impairments()` produces `RicianFading` with the propagation model's `k_factor_db` when LOS.
- `link_params_to_impairments()` produces `TDLChannel` sized to `rms_delay_spread_s` when that field is populated.

All new tests use existing markers (`rust` N/A; no new markers).

## Examples

1. **`examples/environment/terrestrial_propagation_models.py`** *(new)* — comprehensive demo:
   - PL vs distance overlay for all 8 models (2.1 GHz, urban)
   - PL vs frequency for free-space / P.525 / P.525+P.676 (1–100 GHz, shows gaseous absorption lines)
   - 38.901 UMa LOS probability curve vs distance
   - Shadow-fading histograms (10 000 samples per model)

2. **`examples/environment/propagation_and_links.py`** *(update existing)* — extend the model-comparison section to include Okumura-Hata and at least one 38.901 scenario. Preserve existing link-budget narrative.

3. **`examples/environment/urban_5g_scene.py`** *(new)* — integration demo. Uses `Environment` with `GPP38901UMa` @ 3.5 GHz to drive a `WidebandDataset`, showing the auto-impairment chain picking up K-factor and delay spread. Plots spectrograms before and after propagation.

## Documentation

- **New:** `docs/user-guide/propagation.md` — model overview, selection guidance (frequency / environment / use case), parameter tables, standards references.
- **Update:** `docs/user-guide/impairments.md` — brief note on auto-chain behavior when propagation populates multipath fields.
- **Update:** `mkdocs.yml` navigation to include the new page.
- **API reference:** auto-generated via mkdocstrings (no manual updates).

## Out-of-Scope (Future Work)

- Full stochastic small-scale 38.901 (cluster angles, XPR).
- ITU-R P.530 terrestrial link design, P.838 rain attenuation, P.676 Annex 1 line-by-line.
- P.1411 sub-models (street-canyon NLOS, building-entry loss, over-roof propagation).
- Rust ports of any propagation model (not needed for performance).
- Slant-path / elevation-angle support in P.676 (currently horizontal only).
