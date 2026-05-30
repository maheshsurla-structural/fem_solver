"""V&V benchmark suite.

Curated set of FE benchmarks with closed-form or
widely-accepted reference values, organised by analysis category.
Run individually via :func:`Benchmark.run` or in bulk via
:func:`run_benchmarks` from :mod:`femsolver.benchmarks.harness`.

Includes:

* :func:`linear_static_benchmarks` -- cantilever / SS beam / Cook's
  membrane / SS plate / Hex8 cantilever.
* :func:`modal_buckling_benchmarks` -- cantilever frequency, Euler
  columns (pin-pin / cantilever / fixed-fixed).
* :func:`nonlinear_benchmarks` -- EP cantilever shape factor,
  yielding bar.
* :func:`all_benchmarks` -- union of the above.
"""
from femsolver.benchmarks.harness import (
    Benchmark,
    BenchmarkResult,
    export_csv,
    format_report,
    run_benchmarks,
    summary_stats,
)
from femsolver.benchmarks.linear_static import linear_static_benchmarks
from femsolver.benchmarks.modal_buckling import modal_buckling_benchmarks
from femsolver.benchmarks.nonlinear import nonlinear_benchmarks


def all_benchmarks() -> list[Benchmark]:
    """All V&V benchmarks in this distribution."""
    return (
        linear_static_benchmarks()
        + modal_buckling_benchmarks()
        + nonlinear_benchmarks()
    )


__all__ = [
    "Benchmark",
    "BenchmarkResult",
    "all_benchmarks",
    "export_csv",
    "format_report",
    "linear_static_benchmarks",
    "modal_buckling_benchmarks",
    "nonlinear_benchmarks",
    "run_benchmarks",
    "summary_stats",
]
