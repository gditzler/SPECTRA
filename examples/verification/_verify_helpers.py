"""Verification helpers: result accounting, theoretical formulas, and
known-answer-tested measurement primitives.

This module is example-local. It is NOT part of the public ``spectra``
package surface — do not import it from library code.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any


@dataclass
class CheckResult:
    """One row in a result table."""

    test_id: str
    name: str
    measured: Any
    expected: Any
    tolerance: float
    passed: bool
    citation: str
    units: str = ""


class ResultTable:
    """Collects ``CheckResult`` rows and renders them as ASCII or HTML."""

    def __init__(self, title: str) -> None:
        self.title = title
        self._rows: list[CheckResult] = []

    def add(
        self,
        test_id: str,
        name: str,
        *,
        measured: Any,
        expected: Any,
        tol: float,
        cite: str,
        units: str = "",
    ) -> CheckResult:
        passed = self._evaluate(measured, expected, tol)
        row = CheckResult(
            test_id=test_id,
            name=name,
            measured=measured,
            expected=expected,
            tolerance=tol,
            passed=passed,
            citation=cite,
            units=units,
        )
        self._rows.append(row)
        return row

    @staticmethod
    def _evaluate(measured: Any, expected: Any, tol: float) -> bool:
        import numpy as _np

        m = _np.asarray(measured)
        e = _np.asarray(expected)
        if m.shape != e.shape:
            return False
        return bool(_np.all(_np.abs(m - e) <= tol))

    @property
    def all_passed(self) -> bool:
        return all(r.passed for r in self._rows)

    def render(self) -> str:
        lines = [f"=== {self.title} ==="]
        header = f"{'ID':<5}{'Check':<40}{'Measured':<15}{'Expected':<15}{'|Δ|':<10}{'':<6}"
        lines.append(header)
        lines.append("-" * len(header))
        for r in self._rows:
            try:
                import numpy as _np

                delta = float(_np.max(_np.abs(_np.asarray(r.measured) - _np.asarray(r.expected))))
                delta_s = f"{delta:<10.4g}"
            except Exception:
                delta_s = "n/a       "
            mark = "[PASS]" if r.passed else "[FAIL]"
            meas_s = self._scalar_str(r.measured)
            exp_s = self._scalar_str(r.expected)
            lines.append(
                f"{r.test_id:<5}{r.name[:39]:<40}{meas_s:<15}{exp_s:<15}{delta_s}{mark:<6}"
            )
            lines.append(f"     ↳ {r.citation}")
        return "\n".join(lines)

    @staticmethod
    def _scalar_str(v: Any) -> str:
        import numpy as _np

        a = _np.asarray(v)
        if a.shape == ():
            return f"{float(a):.4g}"
        return f"<{a.shape}>"

    def render_html(self) -> str:
        rows_html = []
        for r in self._rows:
            colour = "#dff5d8" if r.passed else "#f8d7da"
            mark = "✓" if r.passed else "✗"
            rows_html.append(
                f"<tr style='background:{colour}'>"
                f"<td>{r.test_id}</td>"
                f"<td>{r.name}</td>"
                f"<td>{self._scalar_str(r.measured)}</td>"
                f"<td>{self._scalar_str(r.expected)}</td>"
                f"<td>{r.tolerance:g}</td>"
                f"<td>{mark}</td>"
                f"<td><code>{r.citation}</code></td>"
                f"</tr>"
            )
        return (
            f"<h4>{self.title}</h4>"
            "<table border='1' cellpadding='4' style='border-collapse:collapse'>"
            "<thead><tr><th>ID</th><th>Check</th><th>Measured</th>"
            "<th>Expected</th><th>Tol</th><th>Pass</th><th>Cite</th></tr></thead>"
            f"<tbody>{''.join(rows_html)}</tbody></table>"
        )


import re
from pathlib import Path


def _parse_references_md(path: Path) -> dict[str, dict]:
    """Parse REFERENCES.md into ``{key: {"raw": str, "loci": {locus: text, ...}}}``."""
    text = path.read_text()
    entries: dict[str, dict] = {}
    current_key: str | None = None
    current_block: list[str] = []
    in_loci = False
    loci: dict[str, str] = {}

    def _flush() -> None:
        nonlocal current_key, current_block, loci
        if current_key is not None:
            entries[current_key] = {
                "raw": "\n".join(current_block).strip(),
                "loci": dict(loci),
            }
        current_block = []
        loci = {}

    for line in text.splitlines():
        m = re.match(r"^##\s+\[([^\]]+)\]\s*$", line)
        if m:
            _flush()
            current_key = m.group(1)
            in_loci = False
            continue
        if current_key is None:
            continue
        current_block.append(line)
        if re.match(r"^- Loci used:\s*$", line):
            in_loci = True
            continue
        if in_loci:
            mloc = re.match(r"^\s+- ([^\s].*?)\s+—\s+(.+)$", line)
            if mloc:
                loci[mloc.group(1).rstrip(",")] = mloc.group(2)
    _flush()
    return entries


_REF_PATH = Path(__file__).resolve().parent / "REFERENCES.md"
REFERENCES: dict[str, dict] = _parse_references_md(_REF_PATH)


def cite(key: str) -> str:
    """Resolve a citation key like ``proakis2008:eq4.3-13`` to a short string."""
    if ":" in key:
        ref_key, locus = key.split(":", 1)
    else:
        ref_key, locus = key, ""
    if ref_key not in REFERENCES:
        raise KeyError(f"Unknown citation key '{ref_key}'. See REFERENCES.md.")
    raw = REFERENCES[ref_key]["raw"]
    short = ref_key
    for line in raw.splitlines():
        if line.startswith("- Authors:"):
            short = line.split(":", 1)[1].strip().split(",")[0].strip()
            break
        if line.startswith("- Org:"):
            short = line.split(":", 1)[1].strip()
            break
    year = ""
    for line in raw.splitlines():
        if line.startswith("- Year:"):
            year = line.split(":", 1)[1].strip()
            break
    return f"{short} {year}{(', ' + locus) if locus else ''}".strip()


import numpy as np
from scipy.special import erfc


def _q(x: np.ndarray) -> np.ndarray:
    """Q-function: Q(x) = 0.5·erfc(x/sqrt(2))."""
    return 0.5 * erfc(np.asarray(x) / np.sqrt(2.0))


def ber_bpsk_awgn(ebn0_db: np.ndarray | float) -> np.ndarray:
    """BER for coherent BPSK over AWGN.

    Reference: ``proakis2008:eq4.3-13`` — P_b = Q(sqrt(2·Eb/N0)).
    """
    ebn0 = 10.0 ** (np.asarray(ebn0_db, dtype=float) / 10.0)
    return _q(np.sqrt(2.0 * ebn0))


def ser_mpsk_awgn(M: int, ebn0_db: np.ndarray | float) -> np.ndarray:
    """SER for coherent M-PSK over AWGN (high-SNR approximation).

    Reference: ``proakis2008:eq4.3-15`` —
        P_s ≈ 2·Q(sqrt(2·Es/N0)·sin(π/M)),
    where Es/N0 = log2(M)·Eb/N0.
    """
    ebn0 = 10.0 ** (np.asarray(ebn0_db, dtype=float) / 10.0)
    esn0 = np.log2(M) * ebn0
    if M == 2:
        return _q(np.sqrt(2.0 * ebn0))
    return 2.0 * _q(np.sqrt(2.0 * esn0) * np.sin(np.pi / M))


def ser_mqam_awgn(M: int, ebn0_db: np.ndarray | float) -> np.ndarray:
    """SER for square M-QAM over AWGN.

    Reference: ``proakis2008:eq4.3-30`` —
        P_s ≈ 4·(1 − 1/sqrt(M))·Q(sqrt(3·log2(M)·Eb/N0 / (M−1))).
    """
    ebn0 = 10.0 ** (np.asarray(ebn0_db, dtype=float) / 10.0)
    arg = np.sqrt(3.0 * np.log2(M) * ebn0 / (M - 1.0))
    return 4.0 * (1.0 - 1.0 / np.sqrt(M)) * _q(arg)


def psd_rrc_squared(f: np.ndarray, Rs: float, alpha: float) -> np.ndarray:
    """One-sided PSD shape of a unit-energy root-raised-cosine pulse.

    Reference: ``proakis2008:eq9.2-37`` — squared-magnitude RRC frequency
    response. Returns an unnormalised shape; tests / measurements compare
    *correlation*, not absolute level.
    """
    f = np.abs(np.asarray(f, dtype=float))
    T = 1.0 / Rs
    psd = np.zeros_like(f)
    inner = f <= (1.0 - alpha) / (2.0 * T)
    outer = (f > (1.0 - alpha) / (2.0 * T)) & (f <= (1.0 + alpha) / (2.0 * T))
    psd[inner] = T
    if alpha > 0.0:
        x = (np.pi * T / alpha) * (f[outer] - (1.0 - alpha) / (2.0 * T))
        psd[outer] = 0.5 * T * (1.0 + np.cos(x))
    return psd


def matched_filter_gain_db(tbp: float) -> float:
    """LFM matched-filter compression gain.

    Reference: ``levanon2004:eq5.5`` — gain_dB = 10·log10(TBP).
    """
    return 10.0 * np.log10(float(tbp))


def simulate_ber_awgn(
    modulation: str,
    ebn0_db: np.ndarray,
    n_bits: int,
    seed: int = 0,
) -> np.ndarray:
    """Simulate BER over AWGN for a given modulation.

    Currently supports ``"bpsk"`` (used by ``verify_bpsk.py`` for S1).
    QPSK / QAM scripts use higher-level helpers that re-use this function
    via bit-to-symbol mappings.
    """
    rng = np.random.default_rng(seed)
    ebn0_db = np.atleast_1d(np.asarray(ebn0_db, dtype=float))
    bers = np.zeros_like(ebn0_db)

    if modulation.lower() == "bpsk":
        bits = rng.integers(0, 2, size=n_bits, endpoint=False)
        tx = (2.0 * bits - 1.0).astype(np.float64)  # 0→-1, 1→+1
        for i, ebn0 in enumerate(ebn0_db):
            ebn0_lin = 10 ** (ebn0 / 10.0)
            sigma = np.sqrt(1.0 / (2.0 * ebn0_lin))  # bit energy = 1
            noise = sigma * rng.standard_normal(n_bits)
            rx = tx + noise
            bits_hat = (rx > 0).astype(int)
            errors = int(np.sum(bits_hat != bits))
            bers[i] = max(errors / n_bits, 1.0 / n_bits)  # floor at 1/N
        return bers
    raise NotImplementedError(f"simulate_ber_awgn(modulation={modulation!r})")


def measure_evm_rms(rx_symbols: np.ndarray, tx_ref: np.ndarray) -> float:
    """RMS EVM relative to the reference constellation power.

    Reference: ``3gpp_38_104:§B.2``. Definition:
        EVM_RMS = sqrt(mean(|rx - tx|²)) / sqrt(mean(|tx|²))
    """
    rx = np.asarray(rx_symbols)
    tx = np.asarray(tx_ref)
    err = rx - tx
    num = np.sqrt(np.mean(np.abs(err) ** 2))
    den = np.sqrt(np.mean(np.abs(tx) ** 2))
    return float(num / den)


def _welch_psd(iq: np.ndarray, fs: float, nperseg: int = 4096) -> tuple[np.ndarray, np.ndarray]:
    from scipy.signal import welch

    f, p = welch(iq, fs=fs, nperseg=min(nperseg, len(iq)),
                 return_onesided=False, scaling="density")
    order = np.argsort(f)
    return f[order], p[order]


def measure_obw(iq: np.ndarray, fs: float, fraction: float = 0.99) -> float:
    """Occupied bandwidth containing ``fraction`` of total spectral power.

    Reference: ``itu_sm_328:§3``.
    """
    f, p = _welch_psd(iq, fs=fs)
    cum = np.cumsum(p)
    cum /= cum[-1]
    lo_target = (1.0 - fraction) / 2.0
    hi_target = 1.0 - lo_target
    lo_idx = int(np.searchsorted(cum, lo_target))
    hi_idx = int(np.searchsorted(cum, hi_target))
    return float(f[hi_idx] - f[lo_idx])


def measure_papr_db(iq: np.ndarray, percentile: float = 99.9) -> float:
    """Peak-to-Average Power Ratio at the given amplitude percentile."""
    p = np.abs(iq) ** 2
    peak = np.percentile(p, percentile)
    avg = np.mean(p)
    return 10.0 * np.log10(peak / avg)


def measure_psd_shape_correlation(measured_psd: np.ndarray, theory_psd: np.ndarray) -> float:
    """Pearson correlation between two PSD shapes (length-matched)."""
    a = np.asarray(measured_psd, dtype=float)
    b = np.asarray(theory_psd, dtype=float)
    if a.shape != b.shape:
        raise ValueError(f"shape mismatch {a.shape} vs {b.shape}")
    a = a - a.mean()
    b = b - b.mean()
    denom = np.sqrt((a * a).sum() * (b * b).sum())
    if denom == 0:
        return 0.0
    return float((a * b).sum() / denom)


def measure_acpr_db(
    iq: np.ndarray,
    fs: float,
    channel_bw: float,
    offsets: tuple[float, ...],
) -> dict[float, float]:
    """ACPR (in dB) for adjacent channels at the given offsets from DC.

    Returns a dict ``{offset: acpr_db}`` where ACPR = 10·log10(P_main / P_adj).
    """
    f, p = _welch_psd(iq, fs=fs)
    half = channel_bw / 2.0
    main_mask = (f >= -half) & (f <= half)
    main_power = float(np.trapezoid(p[main_mask], f[main_mask]))
    out: dict[float, float] = {}
    for off in offsets:
        adj_mask = ((f >= off - half) & (f <= off + half)) | (
            (f >= -off - half) & (f <= -off + half)
        )
        adj_power = float(np.trapezoid(p[adj_mask], f[adj_mask]))
        out[off] = 10.0 * np.log10(main_power / max(adj_power, 1e-30))
    return out


def autocorr_peak_to_sidelobe(seq: np.ndarray) -> float:
    """Aperiodic autocorrelation peak / max-sidelobe ratio.

    Reference: ``levanon2004:eq3.32`` — for a length-N Barker code, this
    ratio equals N exactly. Defining property of Barker codes.
    """
    a = np.asarray(seq, dtype=float)
    full = np.correlate(a, a, mode="full")
    centre = len(full) // 2
    peak = float(full[centre])
    sidelobes = np.delete(full, centre)
    max_side = float(np.max(np.abs(sidelobes)))
    if max_side == 0:
        return float("inf")
    return abs(peak) / max_side


def measure_cp_correlation_peak(
    ofdm_iq: np.ndarray,
    n_fft: int,
    n_cp: int,
) -> tuple[int, float]:
    """Argmax of the autocorrelation between ``x[n]`` and ``x[n+n_fft]``.

    Reference: ``vandeBeek1997:§III``. Returns ``(lag_at_peak, normalised_peak)``.
    A correctly-built OFDM sequence with cyclic prefix of length ``n_cp``
    yields a peak at lag = ``n_fft``.
    """
    x = np.asarray(ofdm_iq).astype(np.complex128)
    max_lag = min(2 * n_fft, len(x) - 1)
    corr = np.zeros(max_lag, dtype=float)
    for k in range(1, max_lag):
        a = x[: len(x) - k]
        b = x[k:]
        win = min(n_cp, len(a))
        num = np.abs(np.sum(a[:win] * np.conj(b[:win])))
        den = np.sqrt(np.sum(np.abs(a[:win]) ** 2) * np.sum(np.abs(b[:win]) ** 2))
        corr[k] = num / max(den, 1e-30)
    lag = int(np.argmax(corr))
    return lag, float(corr[lag])
