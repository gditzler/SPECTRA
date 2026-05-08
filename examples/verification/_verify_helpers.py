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
