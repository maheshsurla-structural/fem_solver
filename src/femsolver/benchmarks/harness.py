"""V&V benchmark harness.

A *benchmark* is a problem with a known reference value (analytical
solution, NAFEMS table, or established commercial-code result). The
harness runs the benchmark, captures the computed value, compares to
the reference, and reports pass/fail with a relative error.

The result objects support:

* a clean Python report (printed table) suitable for CI logs;
* CSV export for an auditable record;
* batch execution of an entire suite via :func:`run_benchmarks`.

Each benchmark is a simple :class:`Benchmark` instance built around a
zero-argument callable returning a scalar (the computed value). The
benchmark library in :mod:`femsolver.benchmarks` provides the standard
NAFEMS, Scordelis-Lo, pinched-cylinder, etc., problems pre-packaged.

Example
-------
.. code-block:: python

    from femsolver.benchmarks import all_benchmarks
    from femsolver.benchmarks.harness import run_benchmarks, format_report

    results = run_benchmarks(all_benchmarks())
    print(format_report(results))
"""
from __future__ import annotations

import csv
import math
from dataclasses import dataclass, field
from typing import Callable


@dataclass
class Benchmark:
    """A single V&V benchmark.

    Attributes
    ----------
    name : str
        Short identifier (e.g. ``"Cantilever tip load (Bernoulli)"``).
    category : str
        Grouping (``"linear-static"``, ``"modal"``, ``"buckling"``,
        ``"nonlinear"``, ``"transient"``).
    reference_value : float
        Known reference (analytical / NAFEMS / textbook).
    reference_source : str
        Citation (e.g. ``"Timoshenko & Gere 1961, p.27"``).
    units : str
        For pretty printing (``"m"``, ``"Hz"``, ``"N"``, ``""``).
    tolerance : float, default 0.01
        Pass tolerance on relative error (1% by default).
    runner : Callable
        Zero-argument callable returning the computed scalar.
    note : str
        Free-text note shown in the report (e.g. element type used).
    """

    name: str
    category: str
    reference_value: float
    reference_source: str
    units: str
    tolerance: float
    runner: Callable[[], float]
    note: str = ""

    def run(self) -> "BenchmarkResult":
        """Execute the runner and produce a result."""
        try:
            computed = float(self.runner())
            err = None
        except Exception as e:    # any runtime failure -> failed run
            return BenchmarkResult(
                benchmark=self,
                computed_value=float("nan"),
                relative_error=float("nan"),
                passed=False,
                error_message=str(e),
            )
        ref = self.reference_value
        if ref == 0.0:
            rel = abs(computed)
        else:
            rel = abs(computed - ref) / abs(ref)
        return BenchmarkResult(
            benchmark=self,
            computed_value=computed,
            relative_error=rel,
            passed=bool(rel <= self.tolerance),
            error_message=None,
        )


@dataclass
class BenchmarkResult:
    """Outcome of running one benchmark."""

    benchmark: Benchmark
    computed_value: float
    relative_error: float
    passed: bool
    error_message: str | None = None


# ============================================================ suite runner

def run_benchmarks(benchmarks: list[Benchmark]) -> list[BenchmarkResult]:
    """Run every benchmark in turn, collecting results."""
    return [b.run() for b in benchmarks]


# ============================================================ reporters

def format_report(
    results: list[BenchmarkResult],
    *,
    width: int = 96,
) -> str:
    """Format the results as a printable table.

    Groups by category, shows name / reference / computed / relative
    error / pass-fail / source. Sized for an 80-100 column terminal.
    """
    if not results:
        return "(no benchmarks)\n"
    lines: list[str] = []
    # Group by category
    by_cat: dict[str, list[BenchmarkResult]] = {}
    for r in results:
        by_cat.setdefault(r.benchmark.category, []).append(r)

    lines.append("=" * width)
    lines.append("V&V Benchmark Report")
    lines.append("=" * width)
    total = len(results)
    passed = sum(1 for r in results if r.passed)
    lines.append(f"Total: {total}, Passed: {passed}, "
                 f"Failed: {total - passed}")
    lines.append("")

    for cat in sorted(by_cat.keys()):
        cat_results = by_cat[cat]
        n_pass = sum(1 for r in cat_results if r.passed)
        lines.append(f"-- {cat} ({n_pass}/{len(cat_results)} passed) "
                     + "-" * (width - 18 - len(cat)))
        lines.append(f"  {'Benchmark':<38}{'Ref':>14}{'Computed':>14}"
                     f"{'Err':>10}{'Status':>8}")
        for r in cat_results:
            b = r.benchmark
            err_str = (f"{r.relative_error*100:.2f}%"
                       if math.isfinite(r.relative_error)
                       else "N/A")
            status = "PASS" if r.passed else "FAIL"
            ref_str = (f"{b.reference_value:.4g}"
                       if abs(b.reference_value) >= 1.0e-4
                       else f"{b.reference_value:.3e}")
            comp_str = (f"{r.computed_value:.4g}"
                        if math.isfinite(r.computed_value)
                        and abs(r.computed_value) >= 1.0e-4
                        else f"{r.computed_value:.3e}")
            lines.append(f"  {b.name[:38]:<38}{ref_str:>14}"
                         f"{comp_str:>14}{err_str:>10}{status:>8}")
            if r.error_message:
                lines.append(f"    ERROR: {r.error_message}")
        lines.append("")

    lines.append("=" * width)
    return "\n".join(lines) + "\n"


def export_csv(results: list[BenchmarkResult], path: str) -> None:
    """Write the results to a CSV file.

    Columns: category, name, units, reference, computed, rel_error,
    tolerance, passed, source, note.
    """
    with open(path, "w", newline="", encoding="utf-8") as fp:
        writer = csv.writer(fp)
        writer.writerow([
            "category", "name", "units",
            "reference", "computed", "rel_error",
            "tolerance", "passed", "source", "note",
        ])
        for r in results:
            b = r.benchmark
            writer.writerow([
                b.category, b.name, b.units,
                f"{b.reference_value:.6g}",
                f"{r.computed_value:.6g}",
                f"{r.relative_error:.6g}",
                f"{b.tolerance:.6g}",
                "yes" if r.passed else "no",
                b.reference_source, b.note,
            ])


def summary_stats(results: list[BenchmarkResult]) -> dict:
    """Return aggregated stats (count, passed, max error, etc.)."""
    total = len(results)
    passed = sum(1 for r in results if r.passed)
    errors = [r.relative_error for r in results
              if math.isfinite(r.relative_error)]
    return {
        "total": total,
        "passed": passed,
        "failed": total - passed,
        "max_error": max(errors) if errors else float("nan"),
        "mean_error": sum(errors) / len(errors) if errors else float("nan"),
    }
