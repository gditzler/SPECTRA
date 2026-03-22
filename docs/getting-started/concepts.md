# Key Concepts

## Two-Layer Architecture

SPECTRA separates concerns between two layers:

- **Rust layer** (`spectra._rust`) — stateless functions that accept and return
  NumPy arrays via PyO3. Handles compute-intensive DSP: modulator symbol generation,
  RRC filtering, CSP (SCD/SCF/CAF), and PSD estimation.
- **Python layer** (`spectra`) — orchestrates composition, impairments, and dataset
  construction. All randomness and state live here.

**Design rule:** Rust functions are pure and stateless. Pass a seed; get the same output.

## Waveforms

A `Waveform` is a generator with three responsibilities:

```python
wf.generate(num_symbols, sample_rate, seed)  # -> np.ndarray[complex64]
wf.bandwidth(sample_rate)                    # -> float (Hz)
wf.label                                     # -> str (e.g. "QPSK")
```

Most waveforms use RRC pulse shaping: symbols are generated in Rust, upsampled,
then convolved with an RRC filter. The `samples_per_symbol`, `rolloff`, and
`filter_span` constructor arguments control the filter.

## Impairments

A `Transform` takes `(iq, desc)` and returns `(iq, desc)` — it can modify both
the signal and its metadata:

```python
iq_out, desc_out = transform(iq_in, desc_in, sample_rate=1e6)
```

`Compose([t1, t2, t3])` chains transforms in order, forwarding `**kwargs`
(always include `sample_rate`) to each one.

## SignalDescription

Every signal carries a `SignalDescription` through the pipeline with physical-unit
ground truth: `t_start`, `t_stop`, `f_low`, `f_high` (Hz relative to DC), `label`,
and `snr`. Impairments that shift frequency (e.g., `FrequencyOffset`) update
`f_low` and `f_high` accordingly.

## Deterministic Seeding

Datasets use `np.random.default_rng(seed=(base_seed, idx))` for each sample.
This means every index maps to exactly one signal regardless of which DataLoader
worker processes it — no seed collisions, fully reproducible datasets.

## Dataset Types

| Class | Task | `__getitem__` output |
|-------|------|----------------------|
| `NarrowbandDataset` | AMC classification | `(Tensor[2,N], int)` |
| `NarrowbandDataset` (MIMO) | MIMO AMC | `(Tensor[n_rx*2,N], int)` |
| `WidebandDataset` | Signal detection | `(Tensor[1,F,T], Dict)` |
| `CyclostationaryDataset` | CSP-based AMC | `(Dict[str,Tensor], int)` |
| `DirectionFindingDataset` | DoA from array IQ | `(Tensor[n_el,2,N], DirectionFindingTarget)` |
| `WidebandDirectionFindingDataset` | Wideband + multi-source DoA | `(Tensor[n_el,2,N], WidebandDFTarget)` |
| `MixUpDataset` | Soft-label AMC | `(Tensor, (int, int, float))` |
| `CutMixDataset` | Soft-label AMC | `(Tensor, (int, int, float))` |
| `SignalFolderDataset` | Disk recordings | `(Tensor[2,N], int)` |
| `ManifestDataset` | Disk recordings | `(Tensor[2,N], int)` |
