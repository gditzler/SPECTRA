# Direction-Finding Dataset Generation — Design Spec

## Overview

Add direction-finding (DoA) dataset generation to SPECTRA. Users can generate multi-antenna IQ snapshots with known angle-of-arrival ground truth for training ML-based DoA estimators and benchmarking classical algorithms (MUSIC, ESPRIT, Capon). The architecture opens the door for future wideband joint detection+DoA.

## Goals

1. **Generic antenna modeling** — support arbitrary element patterns via an ABC, with built-in analytical patterns (isotropic, dipole, cosine-power) and MSI/Planet file import.
2. **Flexible array geometry** — arbitrary 2D planar element positions, with convenience constructors for ULA, UCA, and rectangular arrays.
3. **Realistic signal model** — multiple co-channel sources at different angles, per-element independent noise, per-element calibration errors.
4. **Composable architecture** — antenna and array modules are independent building blocks reusable beyond the dataset (beamforming, array analysis, future wideband DoA).

## Non-Goals (for initial implementation)

- 3D (non-planar) array geometries.
- Angular spread / spatially distributed sources (interface reserves fields for future addition).
- Wideband joint detection+DoA dataset (architecture supports it, not built yet).
- Rust acceleration of array manifold computation.
- Deprecation of existing `mimo_utils.py`.

---

## Architecture

Three new composable layers plus a dataset and a transform:

```
antennas/          arrays/              datasets/
AntennaElement --> AntennaArray ------> DirectionFindingDataset
  (pattern)        (positions +            (waveform pool +
                    elements +              array + impairments
                    steering_vector)        --> multi-antenna IQ)

                                        transforms/
                                        ToSnapshotMatrix
                                          ([n_elem, 2, T] -> complex [n_elem, T])
```

Existing modules (waveforms, impairments, `SignalDescription`) are reused without modification. AoA metadata is stored in `SignalDescription.modulation_params["doa"]`, following the established extension pattern.

---

## Module 1: Antenna Element Model

**Location:** `python/spectra/antennas/`

### AntennaElement ABC (`base.py`)

```python
class AntennaElement(ABC):
    @abstractmethod
    def pattern(self, azimuth: np.ndarray, elevation: np.ndarray) -> np.ndarray:
        """Return complex gain at query angles.

        Args:
            azimuth: Azimuth angles in radians.
            elevation: Elevation angles in radians.

        Returns:
            Complex-valued array (magnitude and phase) with shape
            matching the broadcast shape of inputs.
        """
        ...

    @property
    @abstractmethod
    def frequency(self) -> float:
        """Design frequency in Hz."""
        ...
```

All elements return complex gain so the interface supports phase patterns, even though initial built-ins are real-valued (phase = 0).

### Built-in Implementations

**`IsotropicElement`** (`isotropic.py`):
- Returns `1.0 + 0j` everywhere. No parameters beyond `frequency`.

**`ShortDipoleElement(axis='z', frequency=...)`** (`dipole.py`):
- Analytical `sin(theta)` pattern relative to dipole axis.

**`HalfWaveDipoleElement(axis='z', frequency=...)`** (`dipole.py`):
- Analytical `cos(pi/2 * cos(theta)) / sin(theta)` pattern.

**`CosinePowerElement(exponent=1.5, peak_gain_dbi=0.0, frequency=...)`** (`cosine_power.py`):
- `cos^n(theta_off_boresight)` pattern. Configurable exponent controls beamwidth.
- Approximates patch/microstrip antennas.

### MSI/Planet Import (`msi.py`)

**`parse_msi(path: str) -> MSIAntennaElement`**:

MSI/Planet format:
- Header: antenna name, frequency (MHz), gain (dBi), tilt, polarization.
- `HORIZONTAL` section: 360 lines of `(angle, gain_dB)` at 1-degree steps (azimuth cut at 0-degree elevation).
- `VERTICAL` section: 360 lines of `(angle, gain_dB)` (elevation cut at 0-degree azimuth).

Parsing:
- Extract header metadata (frequency, peak gain, name).
- Parse horizontal and vertical cuts into 1D arrays (360 points each).
- Construct 2D gain surface `(360, 181)` indexed by `(azimuth_deg, elevation_deg)` using the additive pattern method: `G(az, el) = G_h(az) + G_v(el) - G_peak` in dB.
- Store in dB internally, convert to linear in `pattern()`.

**`MSIAntennaElement`**:
- `pattern(azimuth, elevation)` — bilinear interpolation via `scipy.interpolate.RegularGridInterpolator`. Returns real-valued gain (MSI has no phase data).
- Properties: `frequency`, `peak_gain_dbi`, `name`.
- Validation: clear errors for malformed files (missing sections, wrong line counts, non-numeric data).

---

## Module 2: Array Model

**Location:** `python/spectra/arrays/`

### AntennaArray (`array.py`)

```python
class AntennaArray:
    positions: np.ndarray          # (N_elements, 2) in wavelengths
    elements: List[AntennaElement] # one per element, or single broadcast to all
    reference_frequency: float     # Hz, defines the wavelength for positions

    @property
    def num_elements(self) -> int: ...

    def steering_vector(self, azimuth, elevation) -> np.ndarray:
        """Compute array response vector.

        For a single angle: returns shape (N_elements,).
        For multiple angles: returns shape (N_elements, N_angles).

        Combines geometry phase and per-element pattern:
            a_i(az, el) = g_i(az, el) * exp(j*2*pi*(x_i*cos(el)*cos(az) + y_i*cos(el)*sin(az)))
        """
        ...
```

### Convenience Constructors

Module-level functions returning `AntennaArray`:

- **`ula(num_elements, spacing=0.5, element=None, frequency=...)`** — positions at `(n*d, 0)` for `n=0..N-1`. Defaults to `IsotropicElement`.
- **`uca(num_elements, radius=None, element=None, frequency=...)`** — positions on a circle. Default radius gives approximately lambda/2 inter-element spacing.
- **`rectangular(rows, cols, spacing_x=0.5, spacing_y=0.5, element=None, frequency=...)`** — grid array.

### Calibration Errors (`calibration.py`)

```python
@dataclass
class CalibrationErrors:
    gain_offsets_db: np.ndarray     # (N_elements,)
    phase_offsets_rad: np.ndarray   # (N_elements,)

    @classmethod
    def random(cls, num_elements, gain_std_db=0.5, phase_std_rad=0.05, rng=None): ...

    def apply(self, steering_vector: np.ndarray) -> np.ndarray:
        """Apply as diagonal: a_cal = diag(gain * exp(j*phase)) @ a"""
        ...
```

---

## Module 3: Direction-Finding Dataset

**Location:** `python/spectra/datasets/direction_finding.py`

### DirectionFindingDataset

```python
class DirectionFindingDataset(torch.utils.data.Dataset):
    def __init__(
        self,
        array: AntennaArray,
        signal_pool: List[Waveform],
        num_signals: Union[int, Tuple[int, int]],
        num_snapshots: int,
        sample_rate: float,
        snr_range: Tuple[float, float],
        azimuth_range: Tuple[float, float] = (0, 2 * np.pi),
        elevation_range: Tuple[float, float] = (-np.pi / 2, np.pi / 2),
        min_angular_separation: Optional[float] = None,
        calibration_errors: Optional[CalibrationErrors] = None,
        impairments: Optional[Compose] = None,
        transform: Optional[callable] = None,
        num_samples: int = 10000,
        seed: int = 0,
    ): ...
```

### `__getitem__(idx)` Pipeline

1. Seed RNG with `(base_seed, idx)` — deterministic, worker-safe.
2. Draw number of signals from `num_signals` range.
3. For each signal:
   - Pick random waveform from `signal_pool`, generate baseband IQ with derived seed.
   - Draw random `(azimuth, elevation)` within configured ranges, respecting `min_angular_separation`.
   - Draw random SNR from `snr_range`.
   - Apply per-signal impairments if provided.
4. Compute array manifold: for each signal, get steering vector `a(theta, phi)` from `AntennaArray`.
5. Apply calibration errors if provided.
6. Spatial mixing: `X = sum_k(a_k @ s_k^T) + N`, where `N` is per-element independent complex Gaussian noise scaled to achieve target per-signal SNR.
7. Output tensor shape: `[n_elements, 2, num_snapshots]` (I/Q separated per element).
8. Apply transform if provided.

Returns: `(tensor, DirectionFindingTarget)`

### DirectionFindingTarget

```python
@dataclass
class DirectionFindingTarget:
    azimuths: np.ndarray        # (num_sources,) radians
    elevations: np.ndarray      # (num_sources,) radians
    snrs: np.ndarray            # (num_sources,) dB
    num_sources: int
    labels: List[str]           # modulation label per source
    signal_descs: List[SignalDescription]
```

### SignalDescription AoA Metadata

Stored in the existing `modulation_params` dict:

```python
desc.modulation_params["doa"] = {
    "azimuth_rad": float,
    "elevation_rad": float,
    "azimuth_spread_rad": None,    # reserved for future
    "elevation_spread_rad": None,  # reserved for future
}
```

No changes to the `SignalDescription` dataclass definition.

---

## Module 4: Snapshot Matrix Transform

**Location:** `python/spectra/transforms/snapshot.py`

```python
class ToSnapshotMatrix:
    """Convert [n_elements, 2, num_snapshots] to complex [n_elements, num_snapshots].

    For classical DoA algorithms (MUSIC, ESPRIT, Capon).
    """
    def __call__(self, x: np.ndarray) -> np.ndarray:
        return x[:, 0, :] + 1j * x[:, 1, :]
```

---

## File Layout Summary

```
python/spectra/
    antennas/
        __init__.py
        base.py              # AntennaElement ABC
        isotropic.py         # IsotropicElement
        dipole.py            # ShortDipoleElement, HalfWaveDipoleElement
        cosine_power.py      # CosinePowerElement
        msi.py               # MSIAntennaElement, parse_msi

    arrays/
        __init__.py
        array.py             # AntennaArray, ula(), uca(), rectangular()
        calibration.py       # CalibrationErrors

    datasets/
        direction_finding.py # DirectionFindingDataset, DirectionFindingTarget

    transforms/
        snapshot.py          # ToSnapshotMatrix
```

## What Does NOT Change

- Rust layer — no modifications.
- Existing datasets (`NarrowbandDataset`, `WidebandDataset`, etc.).
- Existing impairments (reused as-is via `Compose`).
- Existing waveforms (reused as-is via `signal_pool`).
- `SignalDescription` dataclass definition.
- `mimo_utils.py` — left intact, not deprecated.

## Dependencies

No new dependencies. Uses `numpy` (array math), `scipy` (interpolation for MSI patterns — already a dependency).

## Future Extensions

- **Angular spread sources:** Add `azimuth_spread_rad` / `elevation_spread_rad` to the `doa` dict in `modulation_params`. Modify spatial mixing to use a spread model instead of a single steering vector.
- **Wideband DoA:** Plug `AntennaArray` into `Composer` to add spatial dimension to wideband scenes. `WidebandDataset` gains per-signal AoA labels.
- **3D arrays:** Extend `AntennaArray.positions` to `(N, 3)` and update the steering vector phase computation.
- **Rust acceleration:** Move `steering_vector()` and pattern interpolation to Rust if profiling shows a bottleneck.
- **Additional file formats:** NEC2, CSV import following the same `AntennaElement` interface.
