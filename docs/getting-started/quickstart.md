# Quickstart

## Narrowband AMC (classification)

```python
from spectra import (
    NarrowbandDataset, QPSK, BPSK, QAM16, AWGN, FrequencyOffset, Compose
)
from torch.utils.data import DataLoader

dataset = NarrowbandDataset(
    waveform_pool=[QPSK(), BPSK(), QAM16()],
    num_samples=30_000,
    num_iq_samples=1024,
    sample_rate=1e6,
    impairments=Compose([
        AWGN(snr_range=(-5.0, 20.0)),
        FrequencyOffset(max_offset=5000.0),
    ]),
    seed=42,
)
loader = DataLoader(dataset, batch_size=64, num_workers=4)

for iq, labels in loader:
    # iq: [B, 2, 1024]  (float32, channels = I and Q)
    # labels: [B]       (int64, index into waveform_pool)
    pass
```

## Wideband Detection

```python
from spectra import (
    WidebandDataset, QPSK, BPSK, QAM16, AWGN, Compose
)
from spectra.scene.composer import SceneConfig
from torch.utils.data import DataLoader

config = SceneConfig(
    capture_duration=0.001,
    capture_bandwidth=10e6,
    sample_rate=20e6,
    num_signals=(1, 5),
    signal_pool=[QPSK(), BPSK(), QAM16()],
    snr_range=(5.0, 20.0),
)
dataset = WidebandDataset(config=config, num_samples=5000, seed=42)

loader = DataLoader(dataset, batch_size=16, collate_fn=dataset.collate_fn)
for specs, targets in loader:
    # specs: [B, 1, F, T]  spectrogram
    # targets: list of dicts with 'boxes' [N,4] and 'labels' [N]
    pass
```

## Cyclostationary Feature Classification

```python
from spectra import (
    CyclostationaryDataset, QPSK, BPSK, QAM16, AWGN, Compose,
    SCD, Cumulants
)
from spectra.classifiers.amc import CyclostationaryAMC

dataset = CyclostationaryDataset(
    waveform_pool=[QPSK(), BPSK(), QAM16()],
    num_samples=3000,
    num_iq_samples=4096,
    sample_rate=1e6,
    representations={"scd": SCD(), "cumulants": Cumulants()},
    impairments=Compose([AWGN(snr=10.0)]),
    seed=42,
)

clf = CyclostationaryAMC(classifier="random_forest")
clf.fit_from_dataset(dataset)
print(clf.score_from_dataset(dataset))
```
