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
