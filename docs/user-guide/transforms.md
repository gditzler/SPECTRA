# Transforms & CSP

## Spectral Transforms

### STFT

Short-Time Fourier Transform. Returns a complex spectrogram tensor.

| Parameter | Default | Description |
|-----------|---------|-------------|
| `nfft` | 256 | FFT size |
| `hop_length` | 64 | Hop between frames in samples |
| `window` | `"hann"` | Window function name |

Output shape: `[1, nfft//2+1, T]` (magnitude) or `[2, nfft//2+1, T]` (real+imag).

### Spectrogram

Computes power spectrogram (|STFT|²) in dB. Same parameters as `STFT`.

Output shape: `[1, F, T]` — compatible with 2D CNN inputs.

### PSD

Power spectral density estimate via Welch's method.

| Parameter | Default | Description |
|-----------|---------|-------------|
| `nfft` | 512 | FFT size per segment |
| `overlap` | 0.5 | Fractional overlap between segments |

Output shape: `[nfft//2+1]`.

---

## Cyclostationary Signal Processing

### SCD

Spectral Correlation Density — the core CSP feature for cycle-frequency analysis.

Three `method` choices:

| Method | Description | When to prefer |
|--------|-------------|----------------|
| `"ssca"` | Strip Spectral Correlation Analyzer | General purpose; good spectral resolution |
| `"fam"` | FFT Accumulation Method | Fast for wide cycle-frequency range |
| `"s3ca"` | Sparse Spectral Correlation Analyzer | Best for signals with sparse cyclic structure |

```python
from spectra.transforms.scd import SCD

scd = SCD(method="ssca", nfft=256, num_cyclic_freqs=128)
feature = scd(iq)  # Tensor[F, alpha]
```

### SCF

Spectral Correlation Function evaluated at a single cycle frequency `alpha`.

```python
from spectra.transforms.scf import SCF

scf = SCF(alpha=1.0 / 8, nfft=256)  # alpha normalized by sample_rate
feature = scf(iq)
```

### CAF

Cyclic Autocorrelation Function — time-domain CSP feature.

```python
from spectra.transforms.caf import CAF

caf = CAF(alpha=1.0 / 8, max_lag=64)
feature = caf(iq)  # Tensor[2*max_lag+1]
```

### Cumulants

Higher-order statistical features: 2nd, 4th, 6th, and 8th-order complex cumulants.
Invariant to carrier phase; useful for order classification.

```python
from spectra.transforms.cumulants import Cumulants

cum = Cumulants(orders=[2, 4, 6, 8])
feature = cum(iq)  # Tensor[num_cumulants]
```

### EnergyDetector

Computes total signal energy. Simple baseline for signal presence detection.

```python
from spectra.transforms.energy import EnergyDetector

ed = EnergyDetector(threshold_db=-10.0)
energy = ed(iq)
```

---

## Normalization

`Normalize` standardizes the IQ tensor to zero mean and unit variance, or
scales to a fixed amplitude range.

```python
from spectra.transforms.normalize import Normalize

norm = Normalize(mode="zscore")  # or "minmax"
iq_norm = norm(iq)
```

---

## Representation Conversion

`ComplexTo2D` converts a 1D complex array to a `[2, N]` real tensor with
separate I and Q channels — the standard input format for 1D CNNs.

```python
from spectra.transforms.complex_to_2d import ComplexTo2D

c2d = ComplexTo2D()
tensor = c2d(iq)  # Tensor[2, N]
```

---

## Time-Frequency Representations

### WVD (Wigner-Ville Distribution)

The Wigner-Ville Distribution provides a high-resolution time-frequency representation:

```python
from spectra import WVD

wvd = WVD(nfft=256, output_format="magnitude")
feature = wvd(iq)  # Tensor[1, N, 256]
```

| Parameter | Default | Description |
|-----------|---------|-------------|
| `nfft` | 256 | FFT size for frequency axis |
| `n_time` | `None` | Number of time samples (subsampled); `None` uses all |
| `output_format` | `"magnitude"` | `"magnitude"` (C=1), `"log_magnitude"` (C=1), or `"real_imag"` (C=2) |

Output shape: `[C, n_time, nfft]`.

### AmbiguityFunction

The Ambiguity Function computes the delay-Doppler representation, useful for radar and comms analysis:

```python
from spectra import AmbiguityFunction

af = AmbiguityFunction(max_lag=128, n_doppler=256, output_format="magnitude")
feature = af(iq)  # Tensor[1, 256, 257]
```

| Parameter | Default | Description |
|-----------|---------|-------------|
| `max_lag` | 128 | Maximum delay in samples |
| `n_doppler` | 256 | Number of Doppler bins |
| `output_format` | `"magnitude"` | `"magnitude"` (C=1), `"mag_phase"` (C=2), or `"real_imag"` (C=2) |

Output shape: `[C, n_doppler, 2*max_lag+1]`.

---

## Data Augmentations

| Class | Description |
|-------|-------------|
| `CutOut` | Zeros out a random time segment (regularization) |
| `TimeReversal` | Reverses the IQ time series |
| `PatchShuffle` | Randomly permutes non-overlapping time patches |
| `MixUp` | Blends signal with a random permutation of itself (Beta-distributed lambda) |
| `CutMix` | Replaces a random time segment with shuffled samples |

These are applied to the IQ tensor (or spectrogram) as part of the `transform` argument.

For cross-sample MixUp/CutMix with soft labels, see [Dataset-Level Wrappers](datasets.md#dataset-level-wrappers).

---

## Target Transforms

| Class | Input | Output | Description |
|-------|-------|--------|-------------|
| `ClassIndex` | label string | `int` | Maps label to pool index |
| `FamilyIndex` | label string | `int` | Maps label to modulation family index |
| `FamilyName` | label string | `str` | Maps label to family name (e.g., `"PSK"`) |
| `BoxesNormalize` | boxes `Tensor[N,4]` | `Tensor[N,4]` | Normalizes boxes to [0, 1] by image size |
| `YOLOLabel` | COCO dict | YOLO tensor | Converts to `[class, cx, cy, w, h]` format |

**Modulation family groupings** (used by `FamilyIndex`/`FamilyName`):

| Family | Members |
|--------|---------|
| `PSK` | BPSK, QPSK, 8PSK, 16PSK, 32PSK, 64PSK |
| `QAM` | 16QAM – 1024QAM |
| `ASK` | OOK, 4ASK – 64ASK |
| `FSK` | FSK, GFSK, GMSK, MSK and multi-level variants |
| `AM` | AM-DSB, AM-DSB-SC, AM-LSB, AM-USB |
| `FM` | FM |
| `OFDM` | OFDM variants |

---

---

## Alignment & Domain Adaptation

When training a classifier on data from one capture and evaluating on
another, **alignment transforms** equalize the recordings so domain
shift does not masquerade as model error. Each is a stateless `Transform`
with the standard `(iq, desc, **kwargs) -> (iq, desc)` interface and is
composable via `Compose`.

| Transform | Purpose |
|-----------|---------|
| `DCRemove` | Subtract the mean to remove any DC residual |
| `Resample` | Polyphase resample to a target sample rate |
| `PowerNormalize` | Scale total power to a target dBFS level (default -20 dBFS) |
| `AGCNormalize` | Normalize gain to undo differences in hardware AGC settings |
| `ClipNormalize` | Clip outliers beyond N sigma and rescale to [-1, 1] |
| `BandpassAlign` | Frequency-shift and bandpass-filter to a target center and bandwidth |
| `NoiseFloorMatch` | Estimate noise floor and scale to match a target level |
| `NoiseProfileTransfer` | Transfer noise characteristics from a real capture *(planned)* |
| `SpectralWhitening` | Flatten PSD by dividing by the smoothed spectral envelope |
| `ReceiverEQ` | Equalize receiver frequency response using a reference PSD *(planned)* |

```python
import numpy as np
from spectra.impairments import Compose
from spectra.transforms.alignment import (
    DCRemove, PowerNormalize, ClipNormalize,
    BandpassAlign, SpectralWhitening,
)

iq = (np.random.randn(4096) + 1j * np.random.randn(4096)).astype(np.complex64)
desc = None  # or a SignalDescription instance

# Build a reusable alignment pipeline
align = Compose([
    DCRemove(),
    ClipNormalize(clip_sigma=3.0),
    BandpassAlign(center_freq=0.0, bandwidth=0.4),  # requires sample_rate kwarg at call time
    SpectralWhitening(smoothing_window=64),
    PowerNormalize(target_power_dbfs=-20.0),
])

# Call with sample_rate when BandpassAlign is in the chain
iq_aligned, desc_aligned = align(iq, desc, sample_rate=1.0)
```

`Resample` normalises the output length to a new sample rate:

```python
from spectra.transforms.alignment import Resample

resampler = Resample(target_sample_rate=2e6)
iq_resampled, _ = resampler(iq, desc, sample_rate=4e6)
# iq_resampled.shape == (2048,)  (half the original 4096 samples)
```

!!! note "Planned transforms"
    `NoiseProfileTransfer` and `ReceiverEQ` are designed but not yet
    implemented. Calling them raises `NotImplementedError`. Contributions
    are tracked in `docs/plans/2026-03-11-domain-adaptation-transforms.md`.

---

## Choi-Williams Distribution (CWD)

The Choi-Williams Distribution suppresses the cross-terms that the
Wigner-Ville Distribution introduces between multi-component signals.
Prefer CWD over WVD when the signal contains multiple tones or chirps
whose interference terms obscure real features.

```python
import numpy as np
from spectra.transforms import CWD

iq = (np.random.randn(2048) + 1j * np.random.randn(2048)).astype(np.complex64)

cwd = CWD(nfft=256, sigma=1.0, output_format="magnitude", db_scale=False)
tfd = cwd(iq)  # Tensor[1, 2048, 256]
```

| Parameter | Default | Description |
|-----------|---------|-------------|
| `nfft` | 256 | FFT size (frequency axis length) |
| `n_time` | `None` | Time samples to keep; `None` uses all |
| `sigma` | 1.0 | Cross-term suppression strength — larger values suppress more |
| `output_format` | `"magnitude"` | `"magnitude"` (C=1) or `"real_imag"` (C=2) |
| `db_scale` | `False` | Convert magnitude to dB |

Output shape: `[C, n_time, nfft]`.

---

## Reassigned Gabor Transform

The Reassigned Gabor Transform sharpens spectrogram features by
relocating each energy contribution to its time-frequency centroid.
This produces a more concentrated representation than a plain
`Spectrogram` when noise broadens spectral lines or thin chirp
tracks need to be resolved.

```python
import numpy as np
from spectra.transforms import ReassignedGabor

iq = (np.random.randn(2048) + 1j * np.random.randn(2048)).astype(np.complex64)

rg = ReassignedGabor(nfft=256, hop_length=64)
S = rg(iq)  # Tensor[1, 256, n_frames]
```

| Parameter | Default | Description |
|-----------|---------|-------------|
| `nfft` | 256 | FFT size (frequency resolution) |
| `hop_length` | 64 | Hop between successive frames |
| `sigma` | `None` | Gaussian window sigma; `None` derives from `nfft` |

Output shape: `[1, nfft, n_frames]` where `n_frames = ceil(N / hop_length)`.

---

## Instantaneous Frequency

`InstantaneousFrequency` computes the differential phase of a complex
IQ signal — equivalent to the Hilbert-domain frequency at each sample.
This is useful as an FM-domain feature or as the input to any pipeline
that reasons about instantaneous modulation.

```python
import numpy as np
from spectra.transforms import InstantaneousFrequency

iq = (np.random.randn(2048) + 1j * np.random.randn(2048)).astype(np.complex64)

inst = InstantaneousFrequency(sample_rate=1.0, normalize=False)
freq = inst(iq)  # Tensor[1, 2048]
```

| Parameter | Default | Description |
|-----------|---------|-------------|
| `sample_rate` | 1.0 | Sample rate in Hz; scales the output to Hz units |
| `normalize` | `False` | Normalize output to the range [-0.5, 0.5] |

Output shape: `[1, N]` (same length as the input signal).

---

## Snapshot-Matrix Conversion

`ToSnapshotMatrix` converts the `[n_elements, 2, num_snapshots]` real
tensor produced by `DirectionFindingDataset` into the complex
`[n_elements, num_snapshots]` snapshot matrix consumed by the classical
DoA estimators in `spectra.algorithms.doa` (MUSIC, ESPRIT, Capon).

```python
import torch
from spectra.transforms import ToSnapshotMatrix

to_snap = ToSnapshotMatrix()

# Typical DirectionFindingDataset output: [N_elements, 2 (I/Q), T_snapshots]
data = torch.randn(4, 2, 512)
X = to_snap(data)  # complex Tensor[4, 512]

# Build sample covariance matrix for MUSIC/Capon
R = (X @ X.conj().T) / X.shape[1]  # [4, 4] Hermitian
```

The constructor takes no arguments. The transform is deterministic and
operates on CPU or CUDA tensors.

---

## Multi-Representation Example

```python
from spectra import CyclostationaryDataset, QPSK, BPSK, AWGN, Compose, SCD, Cumulants
from torch.utils.data import DataLoader

dataset = CyclostationaryDataset(
    waveform_pool=[QPSK(), BPSK()],
    num_samples=3000,
    num_iq_samples=4096,
    sample_rate=1e6,
    representations={
        "scd": SCD(method="ssca", nfft=256),
        "cumulants": Cumulants(orders=[2, 4, 6, 8]),
    },
    impairments=Compose([AWGN(snr_range=(0.0, 20.0))]),
    seed=42,
)

loader = DataLoader(dataset, batch_size=32, num_workers=4)
for data, labels in loader:
    scd_feat = data["scd"]       # Tensor[B, F, alpha]
    cum_feat = data["cumulants"] # Tensor[B, num_cumulants]
```
