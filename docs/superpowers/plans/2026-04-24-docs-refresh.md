# Documentation Refresh Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Bring `docs/` into sync with the current code — fix broken example signatures, expose all currently-undocumented public APIs, write user-guide pages for major features that have none, and re-enable `mkdocs build --strict` in CI.

**Architecture:** Three layers of work, easiest first: (1) inline signature-bug fixes to existing pages; (2) `:::` mkdocstrings additions to `docs/api/*.md` so the auto-generated reference covers the whole public API; (3) net-new user-guide prose pages for features that have no narrative coverage yet (5G NR, protocols, radar/DF datasets, LinkSimulator, Studio, alignment transforms). Final step re-enables `--strict` in the GitHub Actions workflow.

**Tech Stack:** mkdocs 1.6+, mkdocs-material 9.5+, mkdocstrings[python] 0.26+, pymdown-extensions 10+, Python 3.10+.

**Build verification:** After every task, run a non-strict local build:
```
uv run --with "mkdocs>=1.6" --with "mkdocs-material>=9.5" --with "mkdocstrings[python]>=0.26" --with "pymdown-extensions>=10.0" mkdocs build --site-dir /tmp/spectra-site
```
The build must succeed with **at most the same warnings** as before your edit. New warnings = your edit broke something.

After Phase 5 the build must pass with `--strict`.

---

## File Structure

| File | Phase | Type of change |
|---|---|---|
| `docs/user-guide/waveforms.md` | 1 + 3 | Replace `OFDM1024` row with `OFDM900` + `OFDM1200`; append new sections for radar (FMCW, PulsedRadar) and spread-spectrum extensions |
| `docs/user-guide/benchmarks.md` | 1 | Rewrite `evaluate_snr_sweep` example to match current API |
| `docs/user-guide/datasets.md` | 1 + 3 | Fix `SNRSweepDataset` example; append sections for `RadarDataset`, `RadarPipelineDataset`, `DirectionFindingDataset`, `WidebandDirectionFindingDataset`, `DirectionFindingSNRSweepDataset` |
| `docs/user-guide/transforms.md` | 3 | Append sections for alignment (`DCRemove`, `Resample`, `PowerNormalize`, `AGCNormalize`, `ClipNormalize`, `BandpassAlign`, `NoiseFloorMatch`, `NoiseProfileTransfer`, `SpectralWhitening`, `ReceiverEQ`), `CWD`, `ReassignedGabor`, `InstantaneousFrequency`, `ToSnapshotMatrix` |
| `docs/api/waveforms.md` | 2 | Add `:::` for NR, aviation/maritime, additional radar/spread-spectrum, multifunction |
| `docs/api/transforms.md` | 2 | Add `:::` for alignment, `CWD`, `ReassignedGabor`, `InstantaneousFrequency`, `ToSnapshotMatrix` |
| `docs/api/datasets.md` | 2 | Add `:::` for radar / DF datasets |
| Create: `docs/api/receivers.md` | 2 | New page — `CoherentReceiver`, `Decoder`, `PassthroughDecoder` |
| Create: `docs/api/link.md` | 2 | New page — `LinkSimulator`, `LinkResults` |
| Create: `docs/api/algorithms.md` | 2 | New page — DoA, beamforming, radar, MTI |
| Create: `docs/api/antennas.md` | 2 | New page — `IsotropicElement`, `ShortDipoleElement`, etc. |
| Create: `docs/api/arrays.md` | 2 | New page — `AntennaArray`, `ula`, `uca`, `rectangular` |
| Create: `docs/api/environment.md` | 2 | New page — `Environment`, `Position`, `Emitter`, propagation models |
| Create: `docs/user-guide/5g-nr.md` | 3 | NR_OFDM, NR_PDSCH, NR_PUSCH, NR_PRACH, NR_SSB |
| Create: `docs/user-guide/protocols.md` | 3 | ADSB, ModeS, AIS, ACARS, DME, ILS_Localizer |
| Create: `docs/user-guide/direction-finding.md` | 3 | DoA pipeline + DirectionFindingDataset family |
| Create: `docs/user-guide/link-simulation.md` | 3 | LinkSimulator BER/SER/PER curves |
| Create: `docs/user-guide/studio.md` | 3 | Gradio UI overview |
| `docs/index.md` | 4 | Update feature list to mention NR, protocols, alignment transforms, multifunction |
| `mkdocs.yml` | 4 | Add new nav entries; add `mkdocs-exclude` to drop `superpowers/` from public site |
| `pyproject.toml` | 4 | Add `mkdocs-exclude` to `docs` extras |
| `docs/superpowers/plans/2026-04-17-multifunction-emitter.md` | 5 | Fix the one broken link that blocks strict mode |
| `.github/workflows/docs.yml` | 5 | Re-add `--strict` flag |

---

## Phase 1 — Signature-bug fixes (3 tasks, ~20 min)

These are real bugs that mislead users today. Each fix is a small inline edit, verified by checking that the example actually matches the current code.

### Task 1.1: Replace `OFDM1024` with `OFDM900` and `OFDM1200` in waveforms.md

**Files:**
- Modify: `docs/user-guide/waveforms.md` lines 184-191 (the OFDM family table)

The actual classes in `python/spectra/waveforms/ofdm.py` are: `OFDM72, OFDM128, OFDM180, OFDM256, OFDM300, OFDM512, OFDM600, OFDM900, OFDM1200, OFDM2048`. The doc lists `OFDM1024` (does not exist), and is missing `OFDM256`, `OFDM900`, `OFDM1200`.

- [ ] **Step 1: Read the current OFDM table**

```bash
sed -n '180,200p' docs/user-guide/waveforms.md
```

- [ ] **Step 2: Apply Edit**

- old_string:
  ```
  | `OFDM72` | `"OFDM-72"` | 72 |
  | `OFDM128` | `"OFDM-128"` | 128 |
  | `OFDM180` | `"OFDM-180"` | 180 |
  | `OFDM300` | `"OFDM-300"` | 300 |
  | `OFDM512` | `"OFDM-512"` | 512 |
  | `OFDM600` | `"OFDM-600"` | 600 |
  | `OFDM1024` | `"OFDM-1024"` | 1024 |
  | `OFDM2048` | `"OFDM-2048"` | 2048 |
  ```
- new_string:
  ```
  | `OFDM72` | `"OFDM-72"` | 72 |
  | `OFDM128` | `"OFDM-128"` | 128 |
  | `OFDM180` | `"OFDM-180"` | 180 |
  | `OFDM256` | `"OFDM-256"` | 256 |
  | `OFDM300` | `"OFDM-300"` | 300 |
  | `OFDM512` | `"OFDM-512"` | 512 |
  | `OFDM600` | `"OFDM-600"` | 600 |
  | `OFDM900` | `"OFDM-900"` | 900 |
  | `OFDM1200` | `"OFDM-1200"` | 1200 |
  | `OFDM2048` | `"OFDM-2048"` | 2048 |
  ```

- [ ] **Step 3: Verify**

```bash
grep "OFDM1024" docs/user-guide/waveforms.md
```
Expected: no output.

- [ ] **Step 4: Commit**

```bash
git add docs/user-guide/waveforms.md
git commit -m "docs: replace OFDM1024 with OFDM900/OFDM1200/OFDM256 in waveforms guide

Co-Authored-By: Claude Opus 4.7 (1M context) <noreply@anthropic.com>"
```

---

### Task 1.2: Rewrite `evaluate_snr_sweep` example in benchmarks.md

**Files:**
- Modify: `docs/user-guide/benchmarks.md:81-105` (the `### evaluate_snr_sweep()` section)

Current code shows fake kwargs (`model=`, `benchmark_name=`, `snr_values=`, `num_samples_per_snr=`, `device=`). The actual signature in `python/spectra/benchmarks/evaluate.py:9` is:

```python
def evaluate_snr_sweep(
    predict_fn: Callable[[torch.Tensor], torch.Tensor],
    dataset: Any,
    batch_size: int = 64,
) -> Dict[float, Dict[str, Any]]:
```

It accepts a pre-built `SNRSweepDataset`, not a benchmark name. Return is `{snr_db: {"accuracy": float, "per_class": ...}}`.

- [ ] **Step 1: Read the section**

```bash
sed -n '78,105p' docs/user-guide/benchmarks.md
```

- [ ] **Step 2: Replace the entire example block**

Use Edit to replace from the line `### evaluate_snr_sweep()` through the closing fence ```` ``` ````.

- old_string (the full broken block):
  ````markdown
  ### evaluate_snr_sweep()

  Evaluates a trained model or classifier across a range of SNR values.

  ```python
  from spectra.benchmarks.evaluate import evaluate_snr_sweep

  results = evaluate_snr_sweep(
      model=my_model,           # callable: iq_batch -> label_batch
      benchmark_name="spectra-18",
      snr_values=[-10, -5, 0, 5, 10, 15, 20],
      num_samples_per_snr=500,
      device="cpu",
  )

  # results: {"snr_db": [...], "accuracy": [...]}
  for snr, acc in zip(results["snr_db"], results["accuracy"]):
      print(f"SNR {snr:+3d} dB: {acc:.1%}")
  ```
  ````

- new_string:
  ````markdown
  ### evaluate_snr_sweep()

  Evaluates a callable predictor over an `SNRSweepDataset` and returns
  per-SNR accuracy.

  ```python
  import torch
  from spectra.benchmarks.evaluate import evaluate_snr_sweep
  from spectra.benchmarks.loader import load_snr_sweep

  def predict_fn(batch: torch.Tensor) -> torch.Tensor:
      """Accepts Tensor[B, 2, N], returns Tensor[B] of predicted class indices."""
      return model(batch).argmax(dim=-1)

  dataset = load_snr_sweep("spectra-snr")
  results = evaluate_snr_sweep(predict_fn, dataset, batch_size=64)

  # results: {snr_db: {"accuracy": float, "per_class": Dict[int, float]}}
  for snr_db in sorted(results):
      acc = results[snr_db]["accuracy"]
      print(f"SNR {snr_db:+5.1f} dB: {acc:.1%}")
  ```
  ````

- [ ] **Step 3: Verify**

```bash
grep -E "model=my_model|benchmark_name=" docs/user-guide/benchmarks.md
```
Expected: no output.

- [ ] **Step 4: Commit**

```bash
git add docs/user-guide/benchmarks.md
git commit -m "docs: fix evaluate_snr_sweep example to match current API

Co-Authored-By: Claude Opus 4.7 (1M context) <noreply@anthropic.com>"
```

---

### Task 1.3: Fix `SNRSweepDataset` example in datasets.md

**Files:**
- Modify: `docs/user-guide/datasets.md:94-105`

The actual constructor in `python/spectra/datasets/snr_sweep.py:14`:

```python
class SNRSweepDataset(BaseIQDataset[Tuple[torch.Tensor, int, float]]):
    def __init__(
        self,
        waveform_pool: List[Waveform],
        snr_levels: List[float],
        samples_per_cell: int,
        num_iq_samples: int,
        sample_rate: float,
        impairments_fn: Callable[[float], Compose],
        seed: Optional[int] = None,
    ):
```

Note `snr_levels` (not `snr_values`), `samples_per_cell` (not `num_samples_per_snr`), and the **required** `impairments_fn` callable that takes the SNR in dB and returns a `Compose` chain. The doc omits this entirely.

- [ ] **Step 1: Read the section**

```bash
sed -n '88,115p' docs/user-guide/datasets.md
```

- [ ] **Step 2: Replace the example block**

- old_string:
  ````markdown
  ```python
  from spectra.datasets.snr_sweep import SNRSweepDataset

  dataset = SNRSweepDataset(
      waveform_pool=[QPSK(), BPSK()],
      num_samples_per_snr=500,
      snr_values=[-10, -5, 0, 5, 10, 15, 20],
      num_iq_samples=1024,
      sample_rate=1e6,
      seed=42,
  )
  ```
  ````

- new_string:
  ````markdown
  ```python
  from spectra.datasets.snr_sweep import SNRSweepDataset
  from spectra.impairments import AWGN, Compose
  from spectra.waveforms import BPSK, QPSK

  def impairments_fn(snr_db: float) -> Compose:
      """Build the impairment chain for a given SNR."""
      return Compose([AWGN(snr_db=snr_db)])

  dataset = SNRSweepDataset(
      waveform_pool=[QPSK(), BPSK()],
      snr_levels=[-10, -5, 0, 5, 10, 15, 20],
      samples_per_cell=500,
      num_iq_samples=1024,
      sample_rate=1e6,
      impairments_fn=impairments_fn,
      seed=42,
  )

  iq, class_idx, snr_db = dataset[0]   # __getitem__ returns (Tensor, int, float)
  ```
  ````

- [ ] **Step 3: Verify**

```bash
grep -E "num_samples_per_snr|snr_values=" docs/user-guide/datasets.md
```
Expected: no output.

- [ ] **Step 4: Commit**

```bash
git add docs/user-guide/datasets.md
git commit -m "docs: fix SNRSweepDataset example to match current API

Co-Authored-By: Claude Opus 4.7 (1M context) <noreply@anthropic.com>"
```

---

## Phase 2 — Complete the API reference (~30 min, mechanical)

These tasks add `:::` mkdocstrings imports so the auto-generated reference covers the full public API. No prose required — mkdocstrings reads the docstrings.

### Task 2.1: Extend `docs/api/waveforms.md` to cover NR, protocols, multifunction

**Files:**
- Modify: `docs/api/waveforms.md` (append new sections at the end)

Verify class names with: `grep "^class " python/spectra/waveforms/{nr,aviation_maritime,radar,spread_spectrum}.py python/spectra/waveforms/multifunction/*.py`.

- [ ] **Step 1: Read the file end**

```bash
tail -40 docs/api/waveforms.md
```

- [ ] **Step 2: Append the missing sections**

Use Edit to find the last `:::` line in the file and append (immediately after it):

```markdown

## 5G NR

::: spectra.waveforms.nr.NR_OFDM
::: spectra.waveforms.nr.NR_SSB
::: spectra.waveforms.nr.NR_PDSCH
::: spectra.waveforms.nr.NR_PUSCH
::: spectra.waveforms.nr.NR_PRACH

## Aviation & Maritime Protocols

::: spectra.waveforms.aviation_maritime.ADSB
::: spectra.waveforms.aviation_maritime.ModeS
::: spectra.waveforms.aviation_maritime.AIS
::: spectra.waveforms.aviation_maritime.ACARS
::: spectra.waveforms.aviation_maritime.DME
::: spectra.waveforms.aviation_maritime.ILS_Localizer

## Radar (Additional)

::: spectra.waveforms.radar.PulsedRadar
::: spectra.waveforms.radar.PulseDoppler
::: spectra.waveforms.radar.FMCW
::: spectra.waveforms.radar.SteppedFrequency
::: spectra.waveforms.radar.NonlinearFM
::: spectra.waveforms.radar.BarkerCodedPulse
::: spectra.waveforms.radar.PolyphaseCodedPulse

## Spread Spectrum (Additional)

::: spectra.waveforms.spread_spectrum.DSSS_QPSK
::: spectra.waveforms.spread_spectrum.FHSS
::: spectra.waveforms.spread_spectrum.THSS
::: spectra.waveforms.spread_spectrum.CDMA_Forward
::: spectra.waveforms.spread_spectrum.CDMA_Reverse

## Multi-Function Emitters

::: spectra.waveforms.multifunction.scheduled.ScheduledWaveform
::: spectra.waveforms.multifunction.schedule.StaticSchedule
::: spectra.waveforms.multifunction.schedule.StochasticSchedule
::: spectra.waveforms.multifunction.schedule.CognitiveSchedule
::: spectra.waveforms.multifunction.schedule.SegmentSpec
::: spectra.waveforms.multifunction.schedule.ModeSpec
```

**Important:** Before pasting, verify each `:::` target with grep. If the actual module path differs (e.g. `multifunction/scheduled.py` may not exist; the class might live in `schedule.py`), update the `:::` line. Run:

```bash
python3 -c "from spectra.waveforms.multifunction.schedule import StaticSchedule, StochasticSchedule, CognitiveSchedule, SegmentSpec, ModeSpec; print('OK')"
python3 -c "from spectra.waveforms.multifunction import ScheduledWaveform; print('OK')"
```

If the second import fails, use `python3 -c "import spectra.waveforms.multifunction as m; print(m.__file__); print([n for n in dir(m) if not n.startswith('_')])"` to find the right path and update the `:::` accordingly.

- [ ] **Step 3: Build locally**

```bash
uv run --with "mkdocs>=1.6" --with "mkdocs-material>=9.5" --with "mkdocstrings[python]>=0.26" --with "pymdown-extensions>=10.0" mkdocs build --site-dir /tmp/spectra-site 2>&1 | tail -20
```

Look for `Documentation built in N seconds`. New `griffe` warnings about classes you just added are OK *if* the warning is "no docstring" — the class will still render. New ERRORS are not OK; fix the `:::` path.

- [ ] **Step 4: Commit**

```bash
git add docs/api/waveforms.md
git commit -m "docs(api): cover NR, protocols, additional radar/spread-spectrum, multifunction

Co-Authored-By: Claude Opus 4.7 (1M context) <noreply@anthropic.com>"
```

---

### Task 2.2: Extend `docs/api/transforms.md` with alignment + missing transforms

**Files:**
- Modify: `docs/api/transforms.md` (append new sections)

- [ ] **Step 1: Append sections after the last existing one**

```markdown

## Alignment & Domain Adaptation

::: spectra.transforms.alignment.DCRemove
::: spectra.transforms.alignment.Resample
::: spectra.transforms.alignment.PowerNormalize
::: spectra.transforms.alignment.AGCNormalize
::: spectra.transforms.alignment.ClipNormalize
::: spectra.transforms.alignment.BandpassAlign
::: spectra.transforms.alignment.NoiseFloorMatch
::: spectra.transforms.alignment.NoiseProfileTransfer
::: spectra.transforms.alignment.SpectralWhitening
::: spectra.transforms.alignment.ReceiverEQ

## Time-Frequency (Additional)

::: spectra.transforms.cwd.CWD
::: spectra.transforms.reassigned_gabor.ReassignedGabor
::: spectra.transforms.instantaneous_frequency.InstantaneousFrequency

## Other Representations

::: spectra.transforms.snapshot.ToSnapshotMatrix
```

- [ ] **Step 2: Verify imports**

```bash
python3 -c "
from spectra.transforms.alignment import (DCRemove, Resample, PowerNormalize, AGCNormalize, ClipNormalize, BandpassAlign, NoiseFloorMatch, NoiseProfileTransfer, SpectralWhitening, ReceiverEQ)
from spectra.transforms.cwd import CWD
from spectra.transforms.reassigned_gabor import ReassignedGabor
from spectra.transforms.instantaneous_frequency import InstantaneousFrequency
from spectra.transforms.snapshot import ToSnapshotMatrix
print('all imports OK')
"
```
Expected: `all imports OK`

- [ ] **Step 3: Build locally and commit**

```bash
uv run --with "mkdocs>=1.6" --with "mkdocs-material>=9.5" --with "mkdocstrings[python]>=0.26" --with "pymdown-extensions>=10.0" mkdocs build --site-dir /tmp/spectra-site 2>&1 | tail -10
git add docs/api/transforms.md
git commit -m "docs(api): cover alignment transforms, CWD, ReassignedGabor, InstantaneousFrequency, ToSnapshotMatrix

Co-Authored-By: Claude Opus 4.7 (1M context) <noreply@anthropic.com>"
```

---

### Task 2.3: Extend `docs/api/datasets.md` with radar/DF datasets

**Files:**
- Modify: `docs/api/datasets.md` (insert into the `## Synthetic Datasets` section)

- [ ] **Step 1: Replace the synthetic-datasets block**

- old_string:
  ```markdown
  ## Synthetic Datasets

  ::: spectra.datasets.narrowband.NarrowbandDataset
  ::: spectra.datasets.wideband.WidebandDataset
  ::: spectra.datasets.cyclo.CyclostationaryDataset
  ::: spectra.datasets.snr_sweep.SNRSweepDataset
  ```

- new_string:
  ```markdown
  ## Synthetic Datasets

  ::: spectra.datasets.narrowband.NarrowbandDataset
  ::: spectra.datasets.wideband.WidebandDataset
  ::: spectra.datasets.cyclo.CyclostationaryDataset
  ::: spectra.datasets.snr_sweep.SNRSweepDataset

  ## Radar Datasets

  ::: spectra.datasets.radar.RadarDataset
  ::: spectra.datasets.radar.RadarTarget
  ::: spectra.datasets.radar_pipeline.RadarPipelineDataset
  ::: spectra.datasets.radar_pipeline.RadarPipelineTarget

  ## Direction-Finding Datasets

  ::: spectra.datasets.direction_finding.DirectionFindingDataset
  ::: spectra.datasets.direction_finding.DirectionFindingTarget
  ::: spectra.datasets.wideband_df.WidebandDirectionFindingDataset
  ::: spectra.datasets.wideband_df.WidebandDFTarget
  ::: spectra.datasets.df_snr_sweep.DirectionFindingSNRSweepDataset
  ```

- [ ] **Step 2: Verify, build, commit**

```bash
python3 -c "
from spectra.datasets.radar import RadarDataset, RadarTarget
from spectra.datasets.radar_pipeline import RadarPipelineDataset, RadarPipelineTarget
from spectra.datasets.direction_finding import DirectionFindingDataset, DirectionFindingTarget
from spectra.datasets.wideband_df import WidebandDirectionFindingDataset, WidebandDFTarget
from spectra.datasets.df_snr_sweep import DirectionFindingSNRSweepDataset
print('OK')
"
uv run --with "mkdocs>=1.6" --with "mkdocs-material>=9.5" --with "mkdocstrings[python]>=0.26" --with "pymdown-extensions>=10.0" mkdocs build --site-dir /tmp/spectra-site 2>&1 | tail -5
git add docs/api/datasets.md
git commit -m "docs(api): cover radar and direction-finding datasets

Co-Authored-By: Claude Opus 4.7 (1M context) <noreply@anthropic.com>"
```

---

### Task 2.4: Create new API pages for receivers, link, algorithms, antennas, arrays, environment

**Files:**
- Create: `docs/api/receivers.md`
- Create: `docs/api/link.md`
- Create: `docs/api/algorithms.md`
- Create: `docs/api/antennas.md`
- Create: `docs/api/arrays.md`
- Create: `docs/api/environment.md`

These are simple stub pages with `:::` imports. We'll wire them into nav in Task 4.

- [ ] **Step 1: Create `docs/api/receivers.md`**

```markdown
# spectra.receivers

Coherent demodulators and FEC decoder stubs used by `LinkSimulator`.

::: spectra.receivers.coherent.CoherentReceiver
::: spectra.receivers.coherent.constellation_to_bits
::: spectra.receivers.base.Decoder
::: spectra.receivers.base.PassthroughDecoder
```

- [ ] **Step 2: Create `docs/api/link.md`**

```markdown
# spectra.link

Link-level BER/SER/PER simulation.

::: spectra.link.simulator.LinkSimulator
::: spectra.link.results.LinkResults
```

- [ ] **Step 3: Create `docs/api/algorithms.md`**

```markdown
# spectra.algorithms

DoA estimation, beamforming, radar detection, and MTI processing.

## Direction of Arrival

::: spectra.algorithms.doa.music
::: spectra.algorithms.doa.esprit
::: spectra.algorithms.doa.root_music
::: spectra.algorithms.doa.capon
::: spectra.algorithms.doa.find_peaks_doa

## Beamforming

::: spectra.algorithms.beamforming.delay_and_sum
::: spectra.algorithms.beamforming.mvdr
::: spectra.algorithms.beamforming.lcmv
::: spectra.algorithms.beamforming.compute_beam_pattern

## Radar Detection

::: spectra.algorithms.radar.matched_filter
::: spectra.algorithms.radar.ca_cfar
::: spectra.algorithms.radar.os_cfar

## MTI

::: spectra.algorithms.mti.single_pulse_canceller
::: spectra.algorithms.mti.double_pulse_canceller
::: spectra.algorithms.mti.doppler_filter_bank
```

- [ ] **Step 4: Create `docs/api/antennas.md`**

```markdown
# spectra.antennas

Single-element antenna patterns used by `AntennaArray`.

::: spectra.antennas.base.AntennaElement
::: spectra.antennas.isotropic.IsotropicElement
::: spectra.antennas.dipole.ShortDipoleElement
::: spectra.antennas.dipole.HalfWaveDipoleElement
::: spectra.antennas.cosine_power.CosinePowerElement
::: spectra.antennas.yagi.YagiElement
::: spectra.antennas.msi.MSIAntennaElement
```

- [ ] **Step 5: Create `docs/api/arrays.md`**

```markdown
# spectra.arrays

Antenna array geometry and steering vectors.

::: spectra.arrays.array.AntennaArray
::: spectra.arrays.factories.ula
::: spectra.arrays.factories.uca
::: spectra.arrays.factories.rectangular
::: spectra.arrays.calibration.CalibrationErrors
```

Verify the factories module path before committing:

```bash
python3 -c "from spectra.arrays import ula, uca, rectangular; import spectra.arrays.factories; print(spectra.arrays.factories.__file__)"
```

If `factories.py` does not exist, find the actual module:

```bash
grep -rn "^def ula\|^def uca\|^def rectangular" python/spectra/arrays/
```

Update the `:::` lines to match.

- [ ] **Step 6: Create `docs/api/environment.md`**

```markdown
# spectra.environment

Geometry-driven link modeling and propagation models.

## Core

::: spectra.environment.position.Position
::: spectra.environment.emitter.Emitter
::: spectra.environment.receiver.ReceiverConfig
::: spectra.environment.environment.Environment
::: spectra.environment.environment.LinkParams
::: spectra.environment.environment.link_params_to_impairments

## Propagation Models

::: spectra.environment.propagation.free_space.FreeSpacePathLoss
::: spectra.environment.propagation.empirical.LogDistancePL
::: spectra.environment.propagation.empirical.COST231HataPL
::: spectra.environment.propagation.empirical.OkumuraHataPL
::: spectra.environment.propagation.itu_r_p525.ITU_R_P525
::: spectra.environment.propagation.itu_r_p1411.ITU_R_P1411
::: spectra.environment.propagation.gpp_38_901.GPP38901UMa
::: spectra.environment.propagation.gpp_38_901.GPP38901UMi
::: spectra.environment.propagation.gpp_38_901.GPP38901RMa
::: spectra.environment.propagation.gpp_38_901.GPP38901InH
::: spectra.environment.propagation._base.PathLossResult
```

Verify these paths before committing:

```bash
python3 -c "
import spectra.environment.position
import spectra.environment.emitter
import spectra.environment.receiver
import spectra.environment.environment
import spectra.environment.propagation.free_space
import spectra.environment.propagation.empirical
import spectra.environment.propagation.itu_r_p525
import spectra.environment.propagation.itu_r_p1411
import spectra.environment.propagation.gpp_38_901
import spectra.environment.propagation._base
print('all paths OK')
"
```

If any path differs, run:

```bash
find python/spectra/environment -name "*.py" | xargs grep -l "class FreeSpacePathLoss\|class LogDistancePL\|class COST231HataPL\|class OkumuraHataPL\|class ITU_R_P525\|class ITU_R_P1411\|class GPP38901\|class PathLossResult\|class Position\|class Emitter\|class ReceiverConfig\|class Environment\|class LinkParams\|^def link_params_to_impairments"
```

…to find the correct module for each symbol, then fix the `:::` lines.

- [ ] **Step 7: Build and commit all six new files together**

```bash
uv run --with "mkdocs>=1.6" --with "mkdocs-material>=9.5" --with "mkdocstrings[python]>=0.26" --with "pymdown-extensions>=10.0" mkdocs build --site-dir /tmp/spectra-site 2>&1 | tail -10
git add docs/api/receivers.md docs/api/link.md docs/api/algorithms.md docs/api/antennas.md docs/api/arrays.md docs/api/environment.md
git commit -m "docs(api): add reference pages for receivers, link, algorithms, antennas, arrays, environment

Co-Authored-By: Claude Opus 4.7 (1M context) <noreply@anthropic.com>"
```

---

## Phase 3 — Net-new user-guide pages (substantial writing)

Each task is one page or one chunk of prose. Aim for ~150-300 lines per page: a 2-3 sentence overview, a runnable code example, an explanation of key parameters, and a "see also" cross-reference.

### Task 3.1: Create `docs/user-guide/5g-nr.md`

**Files:**
- Create: `docs/user-guide/5g-nr.md`

Read these to ground the prose in real signatures:

```bash
sed -n '1,80p' python/spectra/waveforms/nr.py
grep -n "^class \|def __init__\|def generate" python/spectra/waveforms/nr.py | head -30
```

- [ ] **Step 1: Write the page**

Page outline (write the full content, not these placeholders):

```markdown
# 5G NR Waveforms

SPECTRA includes 5G New Radio primitives for synchronization signals, downlink
shared channel, uplink shared channel, and random access. All NR classes
generate baseband IQ following the **3GPP 38.211** specification at the
specified subcarrier spacing.

## Available classes

| Class | Purpose | Carrier signal type |
|-------|---------|---------------------|
| `NR_OFDM` | Generic NR OFDM symbol with cyclic prefix | Generic |
| `NR_SSB` | Synchronization Signal Block (PSS+SSS+PBCH+DMRS) | Cell sync |
| `NR_PDSCH` | Physical Downlink Shared Channel | Data DL |
| `NR_PUSCH` | Physical Uplink Shared Channel | Data UL |
| `NR_PRACH` | Physical Random Access Channel (preamble) | Random access |

## Quickstart

\```python
from spectra.waveforms import NR_SSB

ssb = NR_SSB(numerology=1)   # 30 kHz subcarrier spacing
iq = ssb.generate(num_symbols=4, sample_rate=30.72e6, seed=0)
print(iq.shape, iq.dtype)
\```

## Common parameters

(Document `numerology` (μ ∈ {0,1,2,3,4}), `num_subcarriers`, `cp_type`, etc. — read each class's `__init__` and copy actual kwargs.)

## See also

- [Waveforms guide](waveforms.md) — base `Waveform` interface
- API reference: [spectra.waveforms — 5G NR](../api/waveforms.md#5g-nr)
```

**Important:** before writing the prose, run the quickstart code yourself in a Python REPL to verify it executes without errors. If it doesn't, fix the example before committing.

- [ ] **Step 2: Verify the example runs**

```bash
python3 -c "
from spectra.waveforms import NR_SSB
ssb = NR_SSB(numerology=1)
iq = ssb.generate(num_symbols=4, sample_rate=30.72e6, seed=0)
assert iq.shape[0] > 0
print('quickstart OK,', iq.shape, iq.dtype)
"
```

If the example fails (wrong kwargs, etc.), inspect `NR_SSB.__init__` and fix the doc example.

- [ ] **Step 3: Build and commit**

```bash
uv run --with "mkdocs>=1.6" --with "mkdocs-material>=9.5" --with "mkdocstrings[python]>=0.26" --with "pymdown-extensions>=10.0" mkdocs build --site-dir /tmp/spectra-site 2>&1 | tail -5
git add docs/user-guide/5g-nr.md
git commit -m "docs: add 5G NR user-guide page

Co-Authored-By: Claude Opus 4.7 (1M context) <noreply@anthropic.com>"
```

---

### Task 3.2: Create `docs/user-guide/protocols.md`

**Files:**
- Create: `docs/user-guide/protocols.md`

Read first:

```bash
sed -n '1,80p' python/spectra/waveforms/aviation_maritime.py
grep -n "^class \|def __init__" python/spectra/waveforms/aviation_maritime.py
```

- [ ] **Step 1: Write the page**

```markdown
# Aviation & Maritime Protocols

SPECTRA implements baseband IQ generators for several civilian protocol
waveforms. These produce realistic frame-level signals with correct CRC,
preamble, and modulation, suitable for protocol-aware detector training.

## Available classes

| Class | Service | Modulation | Default carrier offset |
|-------|---------|------------|------------------------|
| `ADSB` | Aircraft surveillance (1090 MHz ES) | PPM | 0 (baseband) |
| `ModeS` | Secondary surveillance radar | PPM | 0 |
| `AIS` | Maritime AIS (Class A/B) | GMSK | 0 |
| `ACARS` | Aircraft text/data | MSK | 0 |
| `DME` | Distance Measuring Equipment | Pulsed | 0 |
| `ILS_Localizer` | Instrument Landing System | AM | 0 |

## Quickstart

\```python
from spectra.waveforms import ADSB

wf = ADSB()
iq = wf.generate(num_symbols=64, sample_rate=2e6, seed=0)
print(wf.label, iq.shape)
\```

## Frame contents

(Describe each class's framing — for ADSB, mention 56-bit Mode-S short and 112-bit extended squitter; for AIS mention HDLC bit-stuffing, etc. Read each class's docstring/source for accurate detail.)

## See also

- API reference: [spectra.waveforms — Aviation & Maritime](../api/waveforms.md#aviation-maritime-protocols)
```

- [ ] **Step 2: Verify runnable example**

```bash
python3 -c "
from spectra.waveforms import ADSB, ModeS, AIS, ACARS, DME, ILS_Localizer
for cls in [ADSB, ModeS, AIS, ACARS, DME, ILS_Localizer]:
    wf = cls()
    iq = wf.generate(num_symbols=8, sample_rate=2e6, seed=0)
    print(cls.__name__, iq.shape, iq.dtype)
"
```
Expected: each class prints a non-empty IQ array.

- [ ] **Step 3: Commit**

```bash
git add docs/user-guide/protocols.md
git commit -m "docs: add aviation/maritime protocols user-guide page

Co-Authored-By: Claude Opus 4.7 (1M context) <noreply@anthropic.com>"
```

---

### Task 3.3: Append radar + spread-spectrum sections to `waveforms.md`

**Files:**
- Modify: `docs/user-guide/waveforms.md` (append at the end, before the last `---` separator)

- [ ] **Step 1: Read the current end-of-file**

```bash
tail -30 docs/user-guide/waveforms.md
```

- [ ] **Step 2: Append new sections**

Use Edit to add (after the last existing section):

```markdown

## Radar — Pulsed & FMCW

| Class | Description | Key kwargs |
|-------|-------------|------------|
| `PulsedRadar` | Generic pulse train | `pulse_width`, `pri`, `num_pulses` |
| `PulseDoppler` | Coherent pulse-Doppler | `pulse_width`, `pri`, `num_pulses_per_cpi` |
| `FMCW` | Linear FM continuous-wave | `chirp_bandwidth`, `chirp_duration`, `num_chirps` |
| `SteppedFrequency` | Stepped-frequency waveform | `step_size_hz`, `num_steps` |
| `NonlinearFM` | Nonlinear FM sweep | `bandwidth`, `chirp_duration` |
| `BarkerCodedPulse` | Barker code (length 7/13) | `code_length` |
| `PolyphaseCodedPulse` | Frank/P1–P4 codes | `code_type`, `code_length` |

\```python
from spectra.waveforms import FMCW

wf = FMCW(chirp_bandwidth=20e6, chirp_duration=10e-6, num_chirps=8)
iq = wf.generate(num_symbols=1, sample_rate=50e6, seed=0)
print(wf.label, iq.shape)
\```

## Spread Spectrum — DSSS, FHSS, THSS, CDMA

| Class | Description |
|-------|-------------|
| `DSSS_BPSK`, `DSSS_QPSK` | Direct-sequence spread spectrum (Gold/Kasami codes) |
| `FHSS` | Frequency-hopping spread spectrum |
| `THSS` | Time-hopping spread spectrum |
| `CDMA_Forward`, `CDMA_Reverse` | CDMA with Walsh-Hadamard codes |
| `ChirpSS` | Chirp spread spectrum |

\```python
from spectra.waveforms import DSSS_QPSK

wf = DSSS_QPSK(spreading_code="gold", code_length=31)
iq = wf.generate(num_symbols=128, sample_rate=10e6, seed=0)
\```
```

Read each class's `__init__` to make sure the kwargs in the table and example match the actual signature. Adjust if needed.

- [ ] **Step 3: Run examples**

```bash
python3 -c "
from spectra.waveforms import FMCW, DSSS_QPSK
wf = FMCW(chirp_bandwidth=20e6, chirp_duration=10e-6, num_chirps=8)
iq = wf.generate(num_symbols=1, sample_rate=50e6, seed=0)
print('FMCW', iq.shape)
wf = DSSS_QPSK(spreading_code='gold', code_length=31)
iq = wf.generate(num_symbols=128, sample_rate=10e6, seed=0)
print('DSSS_QPSK', iq.shape)
"
```

If the FMCW kwargs don't match (the actual class may use `bandwidth=` not `chirp_bandwidth=`), inspect `python/spectra/waveforms/radar.py` and fix the doc.

- [ ] **Step 4: Commit**

```bash
git add docs/user-guide/waveforms.md
git commit -m "docs: extend waveforms guide with radar and spread-spectrum families

Co-Authored-By: Claude Opus 4.7 (1M context) <noreply@anthropic.com>"
```

---

### Task 3.4: Append radar / direction-finding dataset sections to `datasets.md`

**Files:**
- Modify: `docs/user-guide/datasets.md` (append at end)

- [ ] **Step 1: Append new sections**

```markdown

## RadarDataset

On-the-fly range-profile dataset for target-detection training. Each
`__getitem__` returns `(Tensor[num_range_bins], RadarTarget)` where
`RadarTarget` carries `range_bins`, `snrs`, `num_targets`, and `waveform_label`.

\```python
from spectra.datasets.radar import RadarDataset
from spectra.waveforms import LFM, BarkerCodedPulse, PolyphaseCodedPulse

ds = RadarDataset(
    waveform_pool=[LFM(), BarkerCodedPulse(), PolyphaseCodedPulse(code_type="p4")],
    num_range_bins=512,
    sample_rate=1e6,
    snr_range=(5.0, 25.0),
    num_targets_range=(1, 3),
    num_samples=1000,
    seed=42,
)
profile, target = ds[0]
\```

## RadarPipelineDataset

End-to-end radar pipeline producing multi-CPI training data: waveform → target
injection → clutter → matched filter → MTI → CFAR → Kalman tracker.

\```python
from spectra.datasets.radar_pipeline import RadarPipelineDataset
from spectra.targets.trajectory import ConstantVelocity
from spectra.waveforms import LFM

ds = RadarPipelineDataset(
    waveform_pool=[LFM()],
    trajectory_pool=[ConstantVelocity(initial_range=128, velocity=5, dt=1e-3)],
    num_range_bins=256,
    sample_rate=1e6,
    pulses_per_cpi=16,
    sequence_length=4,
    apply_mti=True,
    cfar_type="ca",
    num_samples=500,
    seed=0,
)
iq, target = ds[0]   # target.kf_states, target.detections, target.true_ranges, ...
\```

## DirectionFindingDataset

Snapshot-matrix dataset for DoA estimation training. Returns
`(Tensor[num_elements, num_snapshots, 2], DirectionFindingTarget)`.

\```python
from spectra.arrays import ula
from spectra.datasets.direction_finding import DirectionFindingDataset
from spectra.waveforms import QPSK

array = ula(num_elements=8, spacing=0.5, frequency=1e9)
ds = DirectionFindingDataset(
    array=array,
    signal_pool=[QPSK()],
    num_signals=2,
    num_snapshots=128,
    sample_rate=1e6,
    snr_range=(10.0, 20.0),
    num_samples=1000,
    seed=0,
)
\```

**Note:** `default_collate` cannot batch `DirectionFindingTarget` directly — pass
a custom `collate_fn` to your `DataLoader`:

\```python
def collate_fn(batch):
    return torch.stack([x for x, _ in batch]), [t for _, t in batch]
\```

## WidebandDirectionFindingDataset

Joint wideband spectrum + DoA dataset where each source occupies a distinct
sub-band. See its docstring for `min_freq_separation` and
`min_angular_separation` constraints.

\```python
from spectra.arrays import ula
from spectra.datasets.wideband_df import WidebandDirectionFindingDataset
from spectra.waveforms import BPSK, QPSK

ds = WidebandDirectionFindingDataset(
    array=ula(num_elements=8, spacing=0.5, frequency=1e9),
    signal_pool=[BPSK(), QPSK()],
    num_signals=3,
    num_snapshots=256,
    sample_rate=10e6,
    capture_bandwidth=8e6,
    snr_range=(5.0, 20.0),
    num_samples=500,
    seed=0,
)
\```
```

Verify each constructor's kwargs by reading the corresponding `__init__`. Update the example if anything differs.

- [ ] **Step 2: Run a smoke test for each example**

```bash
python3 << 'EOF'
import torch
from spectra.arrays import ula
from spectra.waveforms import LFM, BPSK, QPSK, BarkerCodedPulse, PolyphaseCodedPulse
from spectra.targets.trajectory import ConstantVelocity
from spectra.datasets.radar import RadarDataset
from spectra.datasets.radar_pipeline import RadarPipelineDataset
from spectra.datasets.direction_finding import DirectionFindingDataset
from spectra.datasets.wideband_df import WidebandDirectionFindingDataset

ds1 = RadarDataset(waveform_pool=[LFM(), BarkerCodedPulse(), PolyphaseCodedPulse(code_type="p4")], num_range_bins=512, sample_rate=1e6, snr_range=(5.0, 25.0), num_targets_range=(1, 3), num_samples=10, seed=42)
print('RadarDataset:', ds1[0][0].shape)

ds2 = RadarPipelineDataset(waveform_pool=[LFM()], trajectory_pool=[ConstantVelocity(initial_range=128, velocity=5, dt=1e-3)], num_range_bins=256, sample_rate=1e6, pulses_per_cpi=16, sequence_length=4, apply_mti=True, cfar_type="ca", num_samples=2, seed=0)
print('RadarPipelineDataset:', ds2[0][0].shape)

array = ula(num_elements=8, spacing=0.5, frequency=1e9)
ds3 = DirectionFindingDataset(array=array, signal_pool=[QPSK()], num_signals=2, num_snapshots=128, sample_rate=1e6, snr_range=(10.0, 20.0), num_samples=10, seed=0)
print('DirectionFindingDataset:', ds3[0][0].shape)

ds4 = WidebandDirectionFindingDataset(array=array, signal_pool=[BPSK(), QPSK()], num_signals=3, num_snapshots=256, sample_rate=10e6, capture_bandwidth=8e6, snr_range=(5.0, 20.0), num_samples=10, seed=0)
print('WidebandDirectionFindingDataset:', ds4[0][0].shape)
EOF
```

- [ ] **Step 3: Commit**

```bash
git add docs/user-guide/datasets.md
git commit -m "docs: add radar and direction-finding datasets to user guide

Co-Authored-By: Claude Opus 4.7 (1M context) <noreply@anthropic.com>"
```

---

### Task 3.5: Create `docs/user-guide/direction-finding.md`

**Files:**
- Create: `docs/user-guide/direction-finding.md`

This page is about the **algorithms** (MUSIC, ESPRIT, Capon, etc.), not just the dataset. The dataset gets full coverage in `datasets.md` (Task 3.4); here we explain the workflow.

- [ ] **Step 1: Write the page**

```markdown
# Direction Finding

SPECTRA provides a complete DoA pipeline: antenna elements + arrays for
geometry, narrowband and wideband DF datasets for training data, and
estimators (MUSIC, ESPRIT, Capon, Root-MUSIC) for inference.

## Building an array

\```python
from spectra.antennas import HalfWaveDipoleElement
from spectra.arrays import ula

array = ula(num_elements=8, spacing=0.5, frequency=1e9,
            element=HalfWaveDipoleElement())
\```

Factory functions: `ula`, `uca`, `rectangular`. See
[`spectra.arrays`](../api/arrays.md) for full options.

## Estimators

\```python
import numpy as np
from spectra.algorithms import music, find_peaks_doa

# X: snapshot matrix, shape (num_elements, num_snapshots), complex
spectrum = music(X, array, num_sources=2,
                 az_grid=np.linspace(-np.pi/2, np.pi/2, 361))
estimated_az = find_peaks_doa(spectrum, num_peaks=2)
\```

| Function | Strengths | Weakness |
|----------|-----------|----------|
| `music` | High resolution, many sources | Needs known source count |
| `root_music` | Closed-form, no peak search | ULA only |
| `esprit` | Closed-form, fast | ULA only |
| `capon` (MVDR) | Robust to model error | Lower resolution than MUSIC |

## End-to-end with `DirectionFindingDataset`

(Reference Task 3.4 for the dataset usage; show one example linking dataset
output to one of the estimators.)

## See also

- API reference: [spectra.algorithms](../api/algorithms.md), [spectra.arrays](../api/arrays.md)
- Examples: [`examples/antenna_arrays/direction_finding.py`](https://github.com/gditzler/SPECTRA/tree/main/examples/antenna_arrays)
```

- [ ] **Step 2: Verify estimator example**

```bash
python3 << 'EOF'
import numpy as np
from spectra.antennas import HalfWaveDipoleElement
from spectra.arrays import ula
from spectra.algorithms import music, find_peaks_doa

array = ula(num_elements=8, spacing=0.5, frequency=1e9, element=HalfWaveDipoleElement())
# Synthesize a snapshot matrix with one source at 30 deg
az_true = np.deg2rad(30.0)
a = array.steering_vector(azimuth=az_true, elevation=0.0)
X = a[:, None] @ (np.random.randn(1, 100) + 1j * np.random.randn(1, 100)) / np.sqrt(2)
spectrum = music(X, array, num_sources=1, az_grid=np.linspace(-np.pi/2, np.pi/2, 361))
peaks = find_peaks_doa(spectrum, num_peaks=1)
print('Estimated:', np.rad2deg(peaks))
EOF
```

If `music()` signature differs from the example, fix the doc to match. Common drift points: `az_grid` may be a `grid` argument; `num_sources` could be `K`; `find_peaks_doa` may need a `grid` argument too.

- [ ] **Step 3: Commit**

```bash
git add docs/user-guide/direction-finding.md
git commit -m "docs: add direction-finding user-guide page

Co-Authored-By: Claude Opus 4.7 (1M context) <noreply@anthropic.com>"
```

---

### Task 3.6: Create `docs/user-guide/link-simulation.md`

**Files:**
- Create: `docs/user-guide/link-simulation.md`

- [ ] **Step 1: Write the page**

```markdown
# Link Simulation

`LinkSimulator` produces BER/SER/PER curves vs. Eb/N0 for a chosen modulation
and optional channel impairments. Internally it uses Rust-backed
`*_with_indices` symbol generators to keep ground-truth bit/symbol streams
aligned with received samples through pulse shaping and channel effects.

## Quickstart

\```python
import numpy as np
from spectra.link.simulator import LinkSimulator
from spectra.waveforms import QPSK

sim = LinkSimulator(waveform=QPSK(), num_symbols=10000, seed=0)
results = sim.run(eb_n0_points=np.arange(0, 11, 2))   # 0..10 dB

print(results.eb_n0_db)   # array
print(results.ber)        # array, same length
print(results.ser, results.per)
\```

## Adding channel impairments

\```python
from spectra.impairments import AWGN, RayleighFading
from spectra.link.simulator import LinkSimulator
from spectra.waveforms import QPSK

# Note: AWGN is applied internally; channel impairments here run AFTER noise
sim = LinkSimulator(
    waveform=QPSK(),
    channel=[RayleighFading(doppler_hz=10.0)],
    num_symbols=10000,
    seed=0,
)
\```

## Supported waveforms

`LinkSimulator` supports modulations with constellation-based demodulation:
**BPSK, QPSK, 8PSK / M-PSK, M-QAM (square), OOK / M-ASK**. Other waveform
families raise `ValueError` at run time.

## See also

- API reference: [spectra.link](../api/link.md), [spectra.receivers](../api/receivers.md)
- Example: [`examples/communications/link_simulator.py`](https://github.com/gditzler/SPECTRA/tree/main/examples/communications)
```

- [ ] **Step 2: Verify**

```bash
python3 << 'EOF'
import numpy as np
from spectra.link.simulator import LinkSimulator
from spectra.waveforms import QPSK

sim = LinkSimulator(waveform=QPSK(), num_symbols=2000, seed=0)
results = sim.run(eb_n0_points=np.arange(0, 11, 5))
print('eb_n0:', results.eb_n0_db)
print('ber:', results.ber)
EOF
```

- [ ] **Step 3: Commit**

```bash
git add docs/user-guide/link-simulation.md
git commit -m "docs: add link-simulation user-guide page

Co-Authored-By: Claude Opus 4.7 (1M context) <noreply@anthropic.com>"
```

---

### Task 3.7: Create `docs/user-guide/studio.md`

**Files:**
- Create: `docs/user-guide/studio.md`

This page documents the Gradio-based UI. Read `python/spectra/studio/__init__.py` and `app.py` for the entry point.

- [ ] **Step 1: Write the page**

```markdown
# Studio (Interactive UI)

SPECTRA Studio is a Gradio-based browser UI for interactive waveform
generation, visualization, and SigMF export. It's the fastest way to explore
the waveform / impairment / scene spaces without writing code.

## Installation

\```
uv pip install -e ".[ui]"
\```

This pulls in `gradio>=6.0` and `scipy`.

## Launching

\```
spectra studio
spectra studio --port 7860 --dark
spectra studio --share          # public link via Gradio tunnel
\```

Or programmatically:

\```python
from spectra.studio import launch
launch(port=7860, share=False, dark=True)
\```

## Tabs

- **Generate** — pick a waveform family + class, set sample rate / SNR /
  impairment preset, see constellation + PSD live.
- **Visualize** — load any IQ file (SigMF / `.cf32` / `.npy` / raw) and view
  IQ scatter, FFT, waterfall, constellation, SCD, ambiguity, eye diagram.
- **Export** — render the current signal as SigMF / NumPy / raw IQ, or
  download the YAML config for headless reproduction with `spectra generate`.

## CLI complement

The companion `spectra` CLI is documented under
[CLI](../api/utils.md). For headless batch generation:

\```
spectra generate --config configs/spectra-18.yaml --output ./out
spectra viz path/to/signal.sigmf-meta --plot waterfall --save plot.png
\```
```

- [ ] **Step 2: Commit**

(No runtime check needed — the page is descriptive.)

```bash
git add docs/user-guide/studio.md
git commit -m "docs: add Studio user-guide page

Co-Authored-By: Claude Opus 4.7 (1M context) <noreply@anthropic.com>"
```

---

### Task 3.8: Append alignment + missing-transform sections to `transforms.md`

**Files:**
- Modify: `docs/user-guide/transforms.md` (append at end)

- [ ] **Step 1: Append**

```markdown

## Alignment & Domain Adaptation

Use these to bring two recordings into a comparable representation before
training a cross-domain classifier or running evaluation. Each transform is
a stateless `Transform` in the standard `(iq, desc, **kwargs) -> (iq, desc)`
shape.

| Transform | Purpose |
|-----------|---------|
| `DCRemove` | Subtract the mean (kill any DC residual) |
| `Resample` | Polyphase resample to a target sample rate |
| `PowerNormalize` | Normalize total power to 1.0 |
| `AGCNormalize` | Automatic gain control normalization |
| `ClipNormalize` | Clip to a percentile and rescale |
| `BandpassAlign` | Bandpass-filter to a fixed pass band |
| `NoiseFloorMatch` | Match the noise floor of two recordings |
| `NoiseProfileTransfer` | Transfer noise PSD from a reference |
| `SpectralWhitening` | Whiten via inverse PSD |
| `ReceiverEQ` | Apply a stored receiver equalization profile |

\```python
from spectra.impairments import Compose
from spectra.transforms import (
    DCRemove, PowerNormalize, BandpassAlign, SpectralWhitening,
)

align = Compose([
    DCRemove(),
    BandpassAlign(low_hz=-0.4, high_hz=0.4),    # in normalized fs
    SpectralWhitening(),
    PowerNormalize(),
])
\```

## Choi-Williams Distribution (CWD)

\```python
from spectra.transforms import CWD
import numpy as np

cwd = CWD(sigma=1.0, nfft=256)
tfd = cwd(np.asarray(my_iq, dtype=np.complex64))
\```

The Choi-Williams kernel suppresses cross terms that the Wigner-Ville
distribution introduces between multi-component signals; use this when WVD
is too noisy on multi-tone or chirp signals.

## Reassigned Gabor

\```python
from spectra.transforms import ReassignedGabor

rg = ReassignedGabor(window_size=256, hop_length=64)
S = rg(my_iq)
\```

Reassigns spectrogram energy to refined time-frequency centroids, sharpening
features under noise compared to a plain Spectrogram.

## Instantaneous Frequency

\```python
from spectra.transforms import InstantaneousFrequency
inst = InstantaneousFrequency()
freq = inst(my_iq)
\```

Returns the unwrapped phase derivative — useful as an FM-feature input or
for Hilbert-transform pipelines.

## Snapshot-matrix conversion

`ToSnapshotMatrix` converts an `(num_elements, num_iq_samples)` IQ array into
the snapshot matrix shape `(num_elements, num_snapshots, 2)` consumed by the
DoA estimators.

\```python
from spectra.transforms import ToSnapshotMatrix
to_snap = ToSnapshotMatrix(num_snapshots=128)
X = to_snap(iq_multichannel)
\```
```

Read each class's `__init__` to confirm kwargs match. Most likely points of drift: `CWD` may be `kernel_param` not `sigma`; `ReassignedGabor` may be `n_fft` not `window_size`. Fix any divergence before committing.

- [ ] **Step 2: Verify each example imports**

```bash
python3 -c "
from spectra.transforms import (
    CWD, ReassignedGabor, InstantaneousFrequency, ToSnapshotMatrix,
    DCRemove, Resample, PowerNormalize, AGCNormalize, ClipNormalize,
    BandpassAlign, NoiseFloorMatch, NoiseProfileTransfer, SpectralWhitening,
    ReceiverEQ,
)
print('all imports OK')
"
```

- [ ] **Step 3: Commit**

```bash
git add docs/user-guide/transforms.md
git commit -m "docs: extend transforms guide with alignment, CWD, ReassignedGabor, IF, snapshot

Co-Authored-By: Claude Opus 4.7 (1M context) <noreply@anthropic.com>"
```

---

## Phase 4 — Wire it together (nav, index, exclusions)

### Task 4.1: Update `mkdocs.yml` nav

**Files:**
- Modify: `mkdocs.yml` lines 101-129 (the `nav:` section)

- [ ] **Step 1: Replace the `nav:` section**

- old_string: the entire current `nav:` block (lines 101-129)
- new_string:
  ```yaml
  nav:
    - Home: index.md
    - Getting Started:
      - Installation: getting-started/installation.md
      - Quickstart: getting-started/quickstart.md
      - Key Concepts: getting-started/concepts.md
    - User Guide:
      - Waveforms: user-guide/waveforms.md
      - 5G NR: user-guide/5g-nr.md
      - Aviation & Maritime Protocols: user-guide/protocols.md
      - Multi-Function Emitters: user-guide/multifunction-emitters.md
      - Impairments: user-guide/impairments.md
      - Propagation: user-guide/propagation.md
      - Scene Composition: user-guide/scene-composition.md
      - Datasets: user-guide/datasets.md
      - Direction Finding: user-guide/direction-finding.md
      - Link Simulation: user-guide/link-simulation.md
      - Transforms & CSP: user-guide/transforms.md
      - File I/O: user-guide/file-io.md
      - Benchmarks: user-guide/benchmarks.md
      - Curriculum & Streaming: user-guide/curriculum-streaming.md
      - Studio: user-guide/studio.md
    - API Reference:
      - spectra.waveforms: api/waveforms.md
      - spectra.impairments: api/impairments.md
      - spectra.scene: api/scene.md
      - spectra.environment: api/environment.md
      - spectra.antennas: api/antennas.md
      - spectra.arrays: api/arrays.md
      - spectra.algorithms: api/algorithms.md
      - spectra.datasets: api/datasets.md
      - spectra.transforms: api/transforms.md
      - spectra.receivers: api/receivers.md
      - spectra.link: api/link.md
      - spectra.classifiers: api/classifiers.md
      - spectra.benchmarks: api/benchmarks.md
      - spectra.curriculum: api/curriculum.md
      - spectra.streaming: api/streaming.md
      - spectra.utils: api/utils.md
    - Contributing: contributing.md
  ```

- [ ] **Step 2: Add an exclude plugin so `superpowers/` doesn't appear on the public site**

In `mkdocs.yml`, find the `plugins:` section and add `exclude` entry. New `plugins:` block:

```yaml
plugins:
  - search
  - exclude:
      glob:
        - superpowers/**
        - plans/**
  - mkdocstrings:
      default_handler: python
      handlers:
        python:
          options:
            docstring_style: google
            show_source: true
            show_bases: true
            members_order: source
            separate_signature: true
            show_signature_annotations: true
            show_root_heading: true
            show_root_full_path: false
            heading_level: 2
```

Note both `superpowers/**` and `plans/**` (the audit found a top-level `docs/plans/` directory too).

- [ ] **Step 3: Add `mkdocs-exclude` to `pyproject.toml` docs extras**

- old_string:
  ```
  docs = [
      "mkdocs>=1.6",
      "mkdocs-material>=9.5",
      "mkdocstrings[python]>=0.26",
      "pymdown-extensions>=10.0",
  ]
  ```
- new_string:
  ```
  docs = [
      "mkdocs>=1.6",
      "mkdocs-material>=9.5",
      "mkdocstrings[python]>=0.26",
      "mkdocs-exclude>=1.0",
      "pymdown-extensions>=10.0",
  ]
  ```

- [ ] **Step 4: Add `mkdocs-exclude` to the GitHub Actions install line**

In `.github/workflows/docs.yml`, find the install step:

- old_string: `pip install maturin numpy`
- new_string: `pip install maturin numpy mkdocs-exclude`

(The `pip install -e ".[docs,all]"` line will pick up the package once `pyproject.toml` is updated, but adding it explicitly avoids ordering surprises.)

- [ ] **Step 5: Build locally and verify nav**

```bash
uv run --with "mkdocs>=1.6" --with "mkdocs-material>=9.5" --with "mkdocstrings[python]>=0.26" --with "mkdocs-exclude>=1.0" --with "pymdown-extensions>=10.0" mkdocs build --site-dir /tmp/spectra-site 2>&1 | tail -10
ls /tmp/spectra-site/user-guide/  # confirm new pages exist
ls /tmp/spectra-site/api/
```

Verify there's no `/superpowers/` directory in the built site.

- [ ] **Step 6: Commit**

```bash
git add mkdocs.yml pyproject.toml .github/workflows/docs.yml
git commit -m "docs: extend nav for new pages, exclude superpowers/plans from public site

Co-Authored-By: Claude Opus 4.7 (1M context) <noreply@anthropic.com>"
```

---

### Task 4.2: Update `docs/index.md` feature list

**Files:**
- Modify: `docs/index.md`

The current feature list is missing 5G NR, aviation/maritime protocols,
alignment transforms, multifunction emitters, and the radar/DF dataset family.

- [ ] **Step 1: Read the current feature list**

```bash
cat docs/index.md
```

- [ ] **Step 2: Update the feature list**

Find the bullet list of "What SPECTRA includes" (or similar). Add the missing categories. Use the same style as existing bullets. Example additions:

- "5G NR primitives — `NR_OFDM`, `NR_SSB`, `NR_PDSCH`, `NR_PUSCH`, `NR_PRACH` per 3GPP 38.211"
- "Aviation & maritime protocols — `ADSB`, `ModeS`, `AIS`, `ACARS`, `DME`, `ILS_Localizer` with correct framing/CRC"
- "Multi-function emitters — `ScheduledWaveform` with `StaticSchedule`, `StochasticSchedule`, `CognitiveSchedule` for time-multiplexed modes"
- "Radar pipelines — `RadarDataset`, `RadarPipelineDataset` with end-to-end MF → MTI → CFAR → Kalman tracking"
- "Direction finding — antenna arrays, MUSIC/ESPRIT/Capon/Root-MUSIC, narrowband + wideband DF datasets"
- "Link-level simulation — `LinkSimulator` for BER/SER/PER vs. Eb/N0 curves"
- "Domain-adaptation transforms — `DCRemove`, `BandpassAlign`, `NoiseFloorMatch`, `SpectralWhitening`, `NoiseProfileTransfer`, `ReceiverEQ`"

Match the formatting of the existing bullets exactly (some may use `**bold**`, others may not).

- [ ] **Step 3: Commit**

```bash
uv run --with "mkdocs>=1.6" --with "mkdocs-material>=9.5" --with "mkdocstrings[python]>=0.26" --with "mkdocs-exclude>=1.0" --with "pymdown-extensions>=10.0" mkdocs build --site-dir /tmp/spectra-site 2>&1 | tail -3
git add docs/index.md
git commit -m "docs: update homepage feature list to include NR, protocols, multifunction, alignment

Co-Authored-By: Claude Opus 4.7 (1M context) <noreply@anthropic.com>"
```

---

## Phase 5 — Re-enable strict mode

### Task 5.1: Fix the broken link blocking strict mode

**Files:**
- Modify: `docs/superpowers/plans/2026-04-17-multifunction-emitter.md`

Per the audit, this plan has a relative link `docs/user-guide/multifunction-emitters.md` that resolves wrong because the plan itself is already inside `docs/`.

- [ ] **Step 1: Find the broken link**

```bash
grep -n "docs/user-guide/multifunction-emitters.md" docs/superpowers/plans/2026-04-17-multifunction-emitter.md
```

- [ ] **Step 2: Fix the link**

Change `docs/user-guide/multifunction-emitters.md` to `../../user-guide/multifunction-emitters.md` (relative from `docs/superpowers/plans/`). Apply the Edit.

If the exclude plugin (Task 4.1) excluded this directory anyway, the broken link no longer triggers a warning — but fix it for correctness regardless.

- [ ] **Step 3: Verify**

```bash
uv run --with "mkdocs>=1.6" --with "mkdocs-material>=9.5" --with "mkdocstrings[python]>=0.26" --with "mkdocs-exclude>=1.0" --with "pymdown-extensions>=10.0" mkdocs build --strict --site-dir /tmp/spectra-site 2>&1 | tail -10
```

If strict mode reports more broken links or other errors:
- For genuine docs bugs (broken links, missing files referenced by `:::`), fix at the source.
- For mkdocstrings "no annotation" warnings on internal helpers, add a per-class `options: show_source: false` to suppress, OR add type annotations to the source.
- For `Doc file '...' not in nav` warnings, either add the file to nav or to the exclude pattern. The audit identified `superpowers/` and `plans/` already; check for new orphans.

- [ ] **Step 4: Commit**

```bash
git add docs/superpowers/plans/2026-04-17-multifunction-emitter.md
git commit -m "docs: fix relative link in multifunction-emitter plan

Co-Authored-By: Claude Opus 4.7 (1M context) <noreply@anthropic.com>"
```

---

### Task 5.2: Re-enable `--strict` in CI

**Files:**
- Modify: `.github/workflows/docs.yml`

- [ ] **Step 1: Apply Edit**

- old_string: `          mkdocs build --site-dir site`
- new_string: `          mkdocs build --strict --site-dir site`

- [ ] **Step 2: Local sanity-check**

```bash
uv run --with "mkdocs>=1.6" --with "mkdocs-material>=9.5" --with "mkdocstrings[python]>=0.26" --with "mkdocs-exclude>=1.0" --with "pymdown-extensions>=10.0" mkdocs build --strict --site-dir /tmp/spectra-site 2>&1 | tail -5
```
Expected: `Documentation built in N seconds` with no `Aborted with N warnings`.

- [ ] **Step 3: Commit and push**

```bash
git add .github/workflows/docs.yml
git commit -m "ci(docs): re-enable mkdocs build --strict

Co-Authored-By: Claude Opus 4.7 (1M context) <noreply@anthropic.com>"
git push origin <branch>
```

- [ ] **Step 4: Watch the workflow run**

```bash
gh run list --workflow="Deploy Docs" --limit 1
gh run view --log-failed 2>&1 | tail -20   # only if it failed
```

If CI fails under strict mode but local passes, the difference is usually a stale lockfile or a plugin version mismatch — read the log carefully. Most fixes amount to one more file added to the exclude pattern.

---

## Final verification

### Task 6: Confirm site-wide quality

- [ ] **Step 1: Build with `--strict` locally**

```bash
uv run --with "mkdocs>=1.6" --with "mkdocs-material>=9.5" --with "mkdocstrings[python]>=0.26" --with "mkdocs-exclude>=1.0" --with "pymdown-extensions>=10.0" mkdocs build --strict --site-dir /tmp/spectra-site
```
Expected: clean.

- [ ] **Step 2: Spot-check rendered pages**

```bash
python3 -m http.server --directory /tmp/spectra-site 8000 &
echo "Open http://localhost:8000 — verify:"
echo "  • Home page mentions NR / protocols / direction finding / link simulation"
echo "  • User Guide nav shows all new pages"
echo "  • API Reference shows the 6 new pages (receivers, link, algorithms, antennas, arrays, environment)"
echo "  • Click into a few API pages — mkdocstrings should render docstrings"
echo "  • No /superpowers/ pages visible in nav or via direct URL"
kill %1
```

- [ ] **Step 3: Push to main**

```bash
git push origin <branch>
```

- [ ] **Step 4: Wait for the deploy and confirm**

```bash
gh run watch
curl -sI https://gditzler.github.io/SPECTRA/ | head -1   # expect 200
```

---

## Notes for the executor

- **Verify every code example you put in the docs.** Run it in `python3 -c '...'` against the current code. If the code in the example doesn't import or doesn't run, fix the doc — don't ship broken examples.
- **Match constructor kwargs exactly.** A doc claiming `RadarDataset(snr_range=(5, 25))` is broken if the code uses `snr_db_range=`. Always read the `__init__` line.
- **Don't write speculative documentation.** If a class is in `python/spectra/` but you're unsure of the docstring or behavior, read the source. If still unclear, leave the section short and note it as DONE_WITH_CONCERNS rather than fabricate detail.
- **Prefer adding to existing pages over creating new ones** when the topic is small. Only create a new page when the audit listed it as a separate file.
- **Frequent commits, one per task.** Each commit should leave the docs in a buildable state.
- **Pre-existing broken docs warnings are out of scope.** If `griffe` warns about an internal helper missing annotations and that warning existed before your task started, leave it. Only fix what your edits change.
