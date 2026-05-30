"""Cross-platform validation framework.

A *cross-platform benchmark* runs a single model through the
femsolver pipeline and compares the result against **multiple
documented references** -- a textbook closed form, a NAFEMS-table
value, a vendor verification manual entry, etc. Each reference
carries its own source attribution and acceptance tolerance.

Use cases
---------

* Establishing that femsolver agrees with closed-form Timoshenko /
  Cook / Scordelis benchmarks within the same tolerance that
  commercial codes report.
* Producing a publication-style validation table that an engineering
  manager or QA officer can sign off.
* Tracking which references the implementation has been tested
  against (essential audit trail for safety-critical work).

Three classes:

* :class:`CrossPlatformReference` -- one source's value + tolerance.
* :class:`CrossPlatformBenchmark` -- the runner with a list of refs.
* :class:`CrossPlatformResult` -- per-source pass/fail breakdown.
"""
from __future__ import annotations

import csv
import html
import math
from dataclasses import dataclass, field
from typing import Callable


@dataclass
class CrossPlatformReference:
    """One documented reference value for a benchmark.

    Attributes
    ----------
    source : str
        Citation, e.g. ``"NAFEMS LE3"``, ``"Timoshenko 1959 p.169"``,
        ``"SAP2000 V&V Manual Ex 1-001"``, ``"ETABS V18 V&V doc"``.
    value : float
        The published / closed-form scalar to compare against.
    tolerance : float, default 0.01
        Acceptance tolerance on relative error.
    notes : str
        Free-text annotation (mesh density used, FE software version).
    """

    source: str
    value: float
    tolerance: float = 0.01
    notes: str = ""


@dataclass
class CrossPlatformResult:
    """Outcome of running one cross-platform benchmark.

    Each reference is compared independently; the benchmark passes
    overall when **all** references pass.
    """

    benchmark: "CrossPlatformBenchmark"
    computed_value: float
    per_reference: dict          # source -> (rel_err, passes)
    overall_passed: bool
    error_message: str | None = None


@dataclass
class CrossPlatformBenchmark:
    """A benchmark with multiple documented references.

    Attributes
    ----------
    name : str
    category : str
        ``"linear-static"``, ``"modal"``, ``"buckling"``, etc.
    units : str
        Result units for display.
    runner : Callable[[], float]
        Zero-argument callable returning the computed scalar.
    references : list[CrossPlatformReference]
    description : str
        One-line problem description.
    """

    name: str
    category: str
    units: str
    runner: Callable[[], float]
    references: list = field(default_factory=list)
    description: str = ""

    def run(self) -> CrossPlatformResult:
        try:
            computed = float(self.runner())
        except Exception as e:
            return CrossPlatformResult(
                benchmark=self,
                computed_value=float("nan"),
                per_reference={},
                overall_passed=False,
                error_message=str(e),
            )
        per_ref = {}
        all_pass = True
        for ref in self.references:
            if ref.value == 0.0:
                rel = abs(computed)
            else:
                rel = abs(computed - ref.value) / abs(ref.value)
            passes = bool(rel <= ref.tolerance)
            all_pass = all_pass and passes
            per_ref[ref.source] = (rel, passes)
        return CrossPlatformResult(
            benchmark=self,
            computed_value=computed,
            per_reference=per_ref,
            overall_passed=all_pass,
        )


# ============================================================ reporters

def format_validation_table(
    results: list[CrossPlatformResult],
    *,
    width: int = 110,
) -> str:
    """Format the multi-source validation table as a plain-text grid.

    One row per benchmark, one column per source, with the relative
    error in each cell and an overall pass / fail at the right.
    """
    if not results:
        return "(no benchmarks)\n"
    # Collect distinct sources in encountered order
    sources: list[str] = []
    seen = set()
    for r in results:
        for s in r.per_reference:
            if s not in seen:
                seen.add(s)
                sources.append(s)
    n_pass = sum(1 for r in results if r.overall_passed)
    lines = []
    lines.append("=" * width)
    lines.append(
        f"Cross-platform validation -- {n_pass}/{len(results)} passed"
    )
    lines.append("=" * width)
    name_w = max(len(r.benchmark.name) for r in results)
    name_w = max(name_w, 12)
    src_w = max(8, max((len(s) for s in sources), default=8))
    header = f"{'Benchmark':<{name_w}} " + " ".join(
        f"{s:>{src_w}}" for s in sources
    ) + "  Overall"
    lines.append(header)
    lines.append("-" * len(header))
    for r in results:
        cells = []
        for s in sources:
            if s in r.per_reference:
                err, passes = r.per_reference[s]
                tag = "" if passes else "*"
                cells.append(f"{err*100:>{src_w-1}.2f}%{tag}")
            else:
                cells.append(f"{'-':>{src_w}}")
        status = "PASS" if r.overall_passed else "FAIL"
        line = (f"{r.benchmark.name[:name_w]:<{name_w}} "
                + " ".join(cells) + f"  {status}")
        lines.append(line)
    lines.append("-" * len(header))
    lines.append("(*) indicates that reference exceeded tolerance")
    return "\n".join(lines) + "\n"


def render_validation_html(
    results: list[CrossPlatformResult],
    *,
    title: str = "Cross-platform validation",
) -> str:
    """Render the validation table as a self-contained HTML page."""
    sources: list[str] = []
    seen = set()
    for r in results:
        for s in r.per_reference:
            if s not in seen:
                seen.add(s)
                sources.append(s)
    rows = []
    n_pass = sum(1 for r in results if r.overall_passed)
    rows.append(
        f"<h1>{html.escape(title)}</h1>"
        f"<p><b>{n_pass}/{len(results)}</b> benchmarks passed all "
        f"reference checks.</p>"
    )
    rows.append("<table border='1' cellpadding='6' cellspacing='0' "
                "style='border-collapse:collapse;'>")
    rows.append("<tr><th>Benchmark</th><th>Category</th><th>Units</th>"
                "<th>Computed</th>")
    for s in sources:
        rows.append(f"<th>{html.escape(s)}</th>")
    rows.append("<th>Overall</th></tr>")
    for r in results:
        b = r.benchmark
        rows.append("<tr>")
        rows.append(f"<td>{html.escape(b.name)}</td>")
        rows.append(f"<td>{html.escape(b.category)}</td>")
        rows.append(f"<td>{html.escape(b.units)}</td>")
        rows.append(f"<td>{r.computed_value:.4g}</td>")
        for s in sources:
            if s in r.per_reference:
                err, passes = r.per_reference[s]
                color = "#d8f3d8" if passes else "#f7c8c8"
                rows.append(
                    f"<td style='background:{color};'>"
                    f"{err*100:.2f}%</td>"
                )
            else:
                rows.append("<td>-</td>")
        color = "#d8f3d8" if r.overall_passed else "#f7c8c8"
        status = "PASS" if r.overall_passed else "FAIL"
        rows.append(
            f"<td style='background:{color};'><b>{status}</b></td>"
        )
        rows.append("</tr>")
    rows.append("</table>")
    body = "\n".join(rows)
    return ("<!DOCTYPE html>\n<html><head><meta charset='utf-8'>"
            f"<title>{html.escape(title)}</title>"
            "<style>body{font-family:Helvetica,Arial,sans-serif;"
            "margin:24px;color:#222} h1{border-bottom:2px solid #333;"
            "padding-bottom:6px} th{background:#eef}</style></head>"
            f"<body>{body}</body></html>")


def export_validation_csv(
    results: list[CrossPlatformResult], path: str,
) -> None:
    """Long-form CSV: one row per (benchmark, reference) pair."""
    with open(path, "w", newline="", encoding="utf-8") as fp:
        w = csv.writer(fp)
        w.writerow([
            "benchmark", "category", "units", "computed",
            "source", "reference_value", "tolerance",
            "rel_error", "passes",
        ])
        for r in results:
            b = r.benchmark
            for ref in b.references:
                if ref.source in r.per_reference:
                    err, passes = r.per_reference[ref.source]
                else:
                    err, passes = float("nan"), False
                w.writerow([
                    b.name, b.category, b.units,
                    f"{r.computed_value:.6g}",
                    ref.source, f"{ref.value:.6g}",
                    f"{ref.tolerance:.6g}",
                    f"{err:.6g}", "yes" if passes else "no",
                ])
