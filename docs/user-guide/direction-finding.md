# Direction Finding

SPECTRA provides a complete DoA (Direction of Arrival) estimation pipeline:
antenna elements and arrays for geometry, narrowband and wideband direction-finding
datasets for training data, and classical subspace estimators (MUSIC, ESPRIT,
Root-MUSIC, Capon) for inference.

## Overview

The DoA workflow has three stages:

1. **Geometry** — choose antenna elements and assemble an array using `ula`, `uca`,
   or `rectangular` from `spectra.arrays`.
2. **Snapshot** — collect an `(N_elements, N_snapshots)` complex IQ matrix from
   your receiver, or generate one synthetically with
   [`DirectionFindingDataset`](datasets.md#directionfindingdataset).
3. **Estimate** — apply one of the estimators in `spectra.algorithms.doa` to recover
   the direction of arrival.

## Building an Array

SPECTRA ships three factory functions for common array geometries:

| Factory | Geometry |
|---------|----------|
| `ula` | Uniform linear array along the x-axis |
| `uca` | Uniform circular array in the x-y plane |
| `rectangular` | 2-D rectangular grid |

```python
from spectra.antennas import HalfWaveDipoleElement
from spectra.arrays import ula

array = ula(
    num_elements=8,
    spacing=0.5,               # half-wavelength (in wavelengths)
    frequency=2.4e9,           # carrier frequency in Hz
    element=HalfWaveDipoleElement(),
)
```

Omitting `element` defaults to an isotropic radiator.  For a circular or
rectangular array:

```python
from spectra.arrays import uca, rectangular

circle = uca(num_elements=8, frequency=2.4e9)
grid   = rectangular(rows=4, cols=4, spacing_x=0.5, spacing_y=0.5, frequency=2.4e9)
```

See [`spectra.arrays`](../api/arrays.md) and [`spectra.antennas`](../api/antennas.md)
for the full API, including `CalibrationErrors` for per-element gain/phase offsets.

### Coordinate convention

SPECTRA uses a right-handed coordinate system where:

- **azimuth** is measured from the positive x-axis in the x-y plane (radians).
- **elevation** is the angle above the x-y plane (radians).

For a ULA along the x-axis at elevation 0 the inter-element phase shift is
`2π · spacing · cos(az)`, which means azimuth angles are defined on `[0, π]` —
values in that range are unambiguous, while `az` and `-az` produce identical phase
progressions (left-right ambiguity).  When scanning with MUSIC or Capon, restrict
`scan_angles` to `[0, π]` (or equivalently `[0°, 180°]`) to avoid spurious mirror
peaks.

## Estimators

All estimators accept the snapshot matrix `X` of shape `(N_elements, N_snapshots)`
and return estimated azimuth angles in radians.

### MUSIC

MUSIC (Multiple Signal Classification) projects the sample covariance matrix onto
the noise subspace and evaluates a pseudospectrum over a candidate angle grid.
Peaks correspond to source directions.

```python
import numpy as np
from spectra.algorithms.doa import music, find_peaks_doa

# scan_angles must cover the expected source range (0..pi for a ULA)
scan_angles = np.deg2rad(np.linspace(5, 175, 512))  # avoid exactly 0 and 180
spectrum    = music(X, num_sources=2, array=array, scan_angles=scan_angles)
peaks       = find_peaks_doa(spectrum, scan_angles, num_peaks=2)
print(np.rad2deg(peaks))  # estimated azimuths in degrees
```

`find_peaks_doa(spectrum, scan_angles, num_peaks)` returns the `num_peaks`
highest local-maximum angles from the pseudospectrum, sorted ascending.

### ESPRIT

ESPRIT (Estimation of Signal Parameters via Rotational Invariance Techniques)
exploits the shift-invariance of a ULA to estimate source angles in closed form —
no grid search is required.  Works only with ULAs.

```python
from spectra.algorithms.doa import esprit

angles = esprit(X, num_sources=2, spacing=0.5)
print(np.rad2deg(angles))  # values in [0, 180]
```

### Root-MUSIC

Root-MUSIC avoids the grid search by forming the MUSIC polynomial and finding its
roots.  The `num_sources` roots closest to the unit circle correspond to source
directions.  Works only with ULAs.

```python
from spectra.algorithms.doa import root_music

angles = root_music(X, num_sources=2, spacing=0.5)
print(np.rad2deg(angles))  # values in [0, 180]
```

### Capon (MVDR)

The Capon beamformer minimises output power subject to a distortionless-response
constraint and does not require knowing the number of sources.  It is more robust
to model errors than MUSIC but generally delivers lower spectral resolution.

```python
from spectra.algorithms.doa import capon, find_peaks_doa

spectrum = capon(X, array=array, scan_angles=scan_angles)
peaks    = find_peaks_doa(spectrum, scan_angles, num_peaks=2)
```

### Comparison

| Estimator | Requires source count | Grid search | Array constraint | Notes |
|-----------|----------------------|-------------|-----------------|-------|
| `music` | Yes | Yes | Any (uses steering vectors) | High resolution |
| `root_music` | Yes | No (polynomial roots) | ULA only | Closed-form, values in `[0, π]` |
| `esprit` | Yes | No (closed-form) | ULA only | Fast; values in `[0, π]` |
| `capon` | No | Yes | Any (uses steering vectors) | Robust to model error |

## End-to-End Example

The following snippet synthesises a single source at 30° and recovers it with all
four estimators.  Note the scan range `[5°, 175°]` — this avoids the endfire
singularities at 0° and 180° and restricts the search to the unambiguous half-space
of the ULA.

```python
import numpy as np
from spectra.antennas import HalfWaveDipoleElement
from spectra.arrays import ula
from spectra.algorithms.doa import music, esprit, root_music, capon, find_peaks_doa

# Build a half-wavelength ULA at 2.4 GHz
array = ula(num_elements=8, spacing=0.5, frequency=2.4e9,
            element=HalfWaveDipoleElement())

# Synthesise a source at 30 degrees azimuth
az_true = np.deg2rad(30.0)
a = array.steering_vector(azimuth=az_true, elevation=0.0)
rng = np.random.default_rng(0)
s = (rng.standard_normal((1, 100)) + 1j * rng.standard_normal((1, 100))) / np.sqrt(2)
X = a[:, None] @ s + 0.1 * (
    rng.standard_normal((8, 100)) + 1j * rng.standard_normal((8, 100))
)

# Scan angles — avoid 0 and 180 to stay clear of endfire
scan_angles = np.deg2rad(np.linspace(5, 175, 512))

# MUSIC
spectrum_music = music(X, num_sources=1, array=array, scan_angles=scan_angles)
peaks_music    = find_peaks_doa(spectrum_music, scan_angles, num_peaks=1)
print(f"MUSIC:      {np.rad2deg(peaks_music[0]):.1f} deg")

# Capon
spectrum_capon = capon(X, array=array, scan_angles=scan_angles)
peaks_capon    = find_peaks_doa(spectrum_capon, scan_angles, num_peaks=1)
print(f"Capon:      {np.rad2deg(peaks_capon[0]):.1f} deg")

# ESPRIT (ULA only, no grid)
angles_esprit = esprit(X, num_sources=1, spacing=0.5)
print(f"ESPRIT:     {np.rad2deg(angles_esprit[0]):.1f} deg")

# Root-MUSIC (ULA only, no grid)
angles_root = root_music(X, num_sources=1, spacing=0.5)
print(f"Root-MUSIC: {np.rad2deg(angles_root[0]):.1f} deg")
```

Expected output:

```
MUSIC:      30.0 deg
Capon:      30.0 deg
ESPRIT:     30.0 deg
Root-MUSIC: 30.0 deg
```

## Using DirectionFindingDataset

For large-scale training experiments SPECTRA provides
`DirectionFindingDataset`, which generates snapshots on the fly with
deterministic `(seed, idx)` seeding — safe for `DataLoader` with
`num_workers > 0`.

```python
import numpy as np
from torch.utils.data import DataLoader
from spectra.arrays import ula
from spectra.waveforms import BPSK, QPSK, QAM16
from spectra.datasets import DirectionFindingDataset

array = ula(num_elements=8, spacing=0.5, frequency=2.4e9)

ds = DirectionFindingDataset(
    array=array,
    signal_pool=[BPSK(samples_per_symbol=4), QPSK(samples_per_symbol=4), QAM16(samples_per_symbol=4)],
    num_signals=2,
    num_snapshots=256,
    sample_rate=1e6,
    snr_range=(15.0, 25.0),
    azimuth_range=(np.deg2rad(10), np.deg2rad(170)),   # avoid near-endfire
    elevation_range=(0.0, 0.0),                         # 2-D scenario
    min_angular_separation=np.deg2rad(15),
    num_samples=2000,
    seed=42,
)

# Output tensor: [N_elements, 2, N_snapshots]  (channel 0=I, 1=Q)
data, target = ds[0]
print(f"Tensor shape: {data.shape}")
print(f"True azimuths: {np.rad2deg(target.azimuths)} deg")
```

`default_collate` cannot batch `DirectionFindingTarget` directly.  Use a
custom `collate_fn`:

```python
def collate_fn(batch):
    return torch.stack([x for x, _ in batch]), [t for _, t in batch]

loader = DataLoader(ds, batch_size=8, collate_fn=collate_fn)
```

### Converting the tensor to a complex snapshot matrix

The dataset returns a real-valued tensor with separate I/Q channels.  To
feed the snapshot matrix into an estimator, reconstruct the complex matrix:

```python
import torch

data, target = ds[0]                           # data: [N_elem, 2, T]
X = data[:, 0, :].numpy() + 1j * data[:, 1, :].numpy()  # (N_elem, T) complex

scan_angles = np.deg2rad(np.linspace(5, 175, 512))
spectrum = music(X, num_sources=target.num_sources, array=array,
                 scan_angles=scan_angles)
peaks = find_peaks_doa(spectrum, scan_angles, num_peaks=target.num_sources)
print(f"Estimated: {np.rad2deg(peaks)} deg")
print(f"True:      {np.rad2deg(np.sort(target.azimuths))} deg")
```

## Wideband Direction Finding

For wideband sources where each source occupies a distinct sub-band, use
`WidebandDirectionFindingDataset` and per-band processing — see
[Datasets — Direction-Finding Datasets](datasets.md#direction-finding-datasets).

## See Also

- API reference: [`spectra.algorithms`](../api/algorithms.md),
  [`spectra.arrays`](../api/arrays.md),
  [`spectra.antennas`](../api/antennas.md)
- [Datasets — Direction-Finding Datasets](datasets.md#direction-finding-datasets)
- Examples: `examples/antenna_arrays/direction_finding.py`,
  `examples/antenna_arrays/wideband_direction_finding.py`
