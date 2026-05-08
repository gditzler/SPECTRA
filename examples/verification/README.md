# SPECTRA Signal Generation — Verification Suite

An evidence-based verification suite for SPECTRA's core waveform
generators, designed to convince an RF / communications expert that
generated signals are correct.

Every claim in every script:

1. carries a numbered ID (`P1`, `S2`, …),
2. is asserted with a literature- or standards-grounded tolerance,
3. is annotated with a citation key that resolves to [`REFERENCES.md`](REFERENCES.md).

If you find a citation that doesn't match the code, file an issue —
that's a bug.

## Layout

| File | Purpose |
|------|---------|
| `_verify_helpers.py` | Result accounting, theoretical formulas, measurement primitives, plotting. Example-local — not part of the public API. |
| `REFERENCES.md` | Canonical bibliography. Parsed at startup; unresolved keys raise. |
| `verify_<waveform>.py` | Per-waveform proof scripts. Each exposes `properties()` and `performance(full)`. |
| `verification_suite.ipynb` | Master narrative notebook. Imports every script. |

## Methodology

Two tiers per waveform:

- **Property checks (`P*`)** — deterministic, fast (< 1 s), always run in CI.
  These are exact equalities or inequalities that follow from the waveform's
  mathematical definition or from a published standard.
- **Performance checks (`S*`)** — statistical, slow-gated (`@pytest.mark.slow`).
  Monte-Carlo / sampling-bound checks: BER vs theory, EVM at fixed SNR,
  ACLR over long captures, PAPR percentiles.

Every numeric tolerance carries a citation. No "industry rule of thumb"
tolerances.

## Running

```bash
# Single waveform, fast mode
python examples/verification/verify_qpsk.py

# Single waveform, publication-grade sample sizes
python examples/verification/verify_qpsk.py --full

# Whole CI tier (property checks)
pytest -m verification tests/verification/

# Slow tier (performance checks)
pytest -m "verification and slow" tests/verification/

# Notebook smoke
pytest --nbmake examples/verification/verification_suite.ipynb
```

## Waveform coverage (first cut)

| Script | Class | Strongest evidence |
|--------|-------|--------------------|
| `verify_bpsk.py`     | Linear binary    | BER-vs-theory exact; constellation on real axis |
| `verify_qpsk.py`     | Linear M-ary     | SER-vs-theory; Gray constellation; ACLR |
| `verify_qam16.py`    | Linear high-order| SER-vs-theory; EVM; PAPR |
| `verify_gmsk.py`     | CPM              | Constant envelope; PSD vs Laurent |
| `verify_ofdm.py`     | Multicarrier     | Subcarrier orthogonality; CP correlation |
| `verify_nr_pss.py`   | Spec sequence    | Sample equality with 3GPP TS 38.211 |
| `verify_nr_sss.py`   | Spec sequence    | Gold-sequence equality |
| `verify_lfm.py`      | Radar FM         | IF linear ramp; matched-filter gain |
| `verify_barker13.py` | Radar code       | PSLR exactly = 13 |
| `verify_adsb.py`     | Protocol w/ CRC  | CRC-24 byte equality |

Future expansion (8PSK, M-PSK ≥ 16, FSK, NR DMRS/PRACH, FMCW, NLFM,
polyphase codes, Mode S, AIS, ACARS, spread spectrum, AM/FM) follows
the same pattern.
