# Scene Composition

## Overview

`Composer` builds a wideband composite IQ capture by mixing multiple independently
generated signals, each frequency-shifted to its assigned center frequency:

```
Signal pool (waveforms)
       |
       v
  [Generate baseband IQ] x N signals
       |
       v
  [Per-signal impairments (optional)]
       |
       v
  [Frequency shift to f_center]
       |
       v
  [Scale to target SNR]
       |
       v
  [Sum all signals -> composite IQ]
       |
       v
  composite IQ + List[SignalDescription]
```

---

## SceneConfig

`SceneConfig` is a dataclass that fully specifies the scene parameters:

| Field | Type | Description |
|-------|------|-------------|
| `capture_duration` | `float` | Total capture length in seconds |
| `capture_bandwidth` | `float` | Total bandwidth in Hz, centered at DC |
| `sample_rate` | `float` | Samples per second |
| `num_signals` | `int` or `(int, int)` | Fixed count or `(min, max)` uniform range |
| `signal_pool` | `List[Waveform]` | Waveforms to draw from uniformly at random |
| `snr_range` | `(float, float)` | Per-signal SNR in dB drawn uniformly |
| `allow_overlap` | `bool` | Allow spectral overlap between signals (default `True`) |

---

## Composer.generate()

```python
composite, descriptions = composer.generate(seed=42, impairments=pipeline)
```

For each signal, `generate()`:

1. Draws a random waveform from `signal_pool`
2. Picks a random center frequency within `capture_bandwidth`
3. Draws a random SNR from `snr_range`
4. Generates baseband IQ using a per-signal sub-seed
5. Optionally applies `impairments` (a `Compose` pipeline)
6. Frequency-shifts the signal to its center frequency
7. Scales to the target SNR relative to unit noise power
8. Sums the result into the composite buffer

The outer `seed` controls all random choices — same seed gives identical output.

---

## SignalDescription

Each signal produces a `SignalDescription` with physical-unit ground truth:

| Field | Description |
|-------|-------------|
| `t_start` | Signal start time in seconds |
| `t_stop` | Signal stop time in seconds |
| `f_low` | Lower frequency edge in Hz (relative to DC) |
| `f_high` | Upper frequency edge in Hz (relative to DC) |
| `label` | Modulation class string |
| `snr` | Signal-to-noise ratio in dB |

Derived properties: `f_center`, `bandwidth`, `duration`.

---

## COCO-Format Labels

`WidebandDataset` converts `SignalDescription` records to COCO-style bounding
boxes in STFT pixel space via `to_coco()`.

`STFTParams` defines the pixel grid:

```python
from spectra.scene.labels import STFTParams, to_coco

stft_params = STFTParams(
    nfft=256,
    hop_length=64,
    sample_rate=20e6,
    capture_duration=0.001,
)
```

`to_coco(descriptions, stft_params)` returns a dict with:

```python
{
    "boxes":        Tensor[N, 4],   # [x_min, y_min, x_max, y_max] in pixels
    "labels":       Tensor[N],      # integer class indices
    "signal_descs": List[SignalDescription],
}
```

Coordinate conversion maps `(f_low, t_start)` → `(x_min, y_min)` and
`(f_high, t_stop)` → `(x_max, y_max)` based on the STFT frequency and
time axes.

---

## End-to-End Example

```python
from spectra import QPSK, BPSK, QAM16, AWGN, Compose, WidebandDataset
from spectra.scene.composer import SceneConfig
from torch.utils.data import DataLoader

config = SceneConfig(
    capture_duration=0.001,      # 1 ms
    capture_bandwidth=10e6,      # 10 MHz
    sample_rate=20e6,            # 20 Msps
    num_signals=(1, 5),          # 1–5 signals per scene
    signal_pool=[QPSK(), BPSK(), QAM16()],
    snr_range=(5.0, 20.0),
)

dataset = WidebandDataset(
    config=config,
    num_samples=5000,
    seed=42,
    impairments=Compose([AWGN(snr_range=(5.0, 20.0))]),
)

loader = DataLoader(
    dataset,
    batch_size=16,
    num_workers=4,
    collate_fn=dataset.collate_fn,
)

for specs, targets in loader:
    # specs:   Tensor[16, 1, F, T]  — spectrogram batch
    # targets: list of 16 dicts, each with:
    #   "boxes":  Tensor[N_i, 4]   — bounding boxes
    #   "labels": Tensor[N_i]      — class indices
    pass
```
