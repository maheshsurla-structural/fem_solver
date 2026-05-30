"""Phase 35 tests -- the V&V benchmark suite itself must pass.

This test module:

1. Runs every benchmark via the harness.
2. Asserts each individual benchmark passes its tolerance.
3. Tests the framework primitives (Benchmark, run_benchmarks,
   format_report, summary_stats, export_csv).
"""
from __future__ import annotations

import math
import os
import tempfile

import pytest

from femsolver.benchmarks import (
    Benchmark,
    BenchmarkResult,
    all_benchmarks,
    export_csv,
    format_report,
    linear_static_benchmarks,
    modal_buckling_benchmarks,
    nonlinear_benchmarks,
    run_benchmarks,
    summary_stats,
)


# ============================================================ Benchmark dataclass

def test_benchmark_run_returns_result():
    bm = Benchmark(
        name="trivial", category="linear-static",
        reference_value=10.0, reference_source="hand", units="",
        tolerance=0.01, runner=lambda: 10.0,
    )
    r = bm.run()
    assert isinstance(r, BenchmarkResult)
    assert r.passed
    assert r.relative_error == pytest.approx(0.0, abs=1.0e-12)


def test_benchmark_handles_failing_runner():
    """If the runner raises, the result should be marked failed."""
    def bad():
        raise RuntimeError("intentional")
    bm = Benchmark(
        name="bad", category="linear-static",
        reference_value=1.0, reference_source="", units="",
        tolerance=0.01, runner=bad,
    )
    r = bm.run()
    assert not r.passed
    assert math.isnan(r.computed_value)
    assert r.error_message is not None


def test_benchmark_tolerance_fail():
    bm = Benchmark(
        name="off", category="linear-static",
        reference_value=10.0, reference_source="", units="",
        tolerance=0.01, runner=lambda: 11.0,    # 10% off
    )
    r = bm.run()
    assert not r.passed
    assert r.relative_error == pytest.approx(0.1, rel=1.0e-12)


def test_benchmark_zero_reference():
    """If reference is exactly zero, the harness uses |computed|."""
    bm = Benchmark(
        name="zero", category="linear-static",
        reference_value=0.0, reference_source="", units="",
        tolerance=1.0e-6, runner=lambda: 1.0e-10,
    )
    r = bm.run()
    assert r.passed


# ============================================================ aggregate harness

def test_run_benchmarks_empty():
    assert run_benchmarks([]) == []


def test_run_benchmarks_runs_all():
    bms = [
        Benchmark("a", "linear-static", 1.0, "", "", 0.01, lambda: 1.0),
        Benchmark("b", "modal", 2.0, "", "", 0.01, lambda: 2.0),
    ]
    results = run_benchmarks(bms)
    assert len(results) == 2
    assert all(r.passed for r in results)


def test_summary_stats():
    bms = [
        Benchmark("a", "linear-static", 1.0, "", "", 0.01, lambda: 1.0),
        Benchmark("b", "modal", 2.0, "", "", 0.01, lambda: 2.5),  # 25% off
    ]
    results = run_benchmarks(bms)
    stats = summary_stats(results)
    assert stats["total"] == 2
    assert stats["passed"] == 1
    assert stats["failed"] == 1
    assert stats["max_error"] == pytest.approx(0.25, rel=1.0e-12)


# ============================================================ report formatters

def test_format_report_contains_pass_count():
    bms = [
        Benchmark("a", "linear-static", 1.0, "src", "m",
                  0.01, lambda: 1.0, note="note"),
    ]
    rep = format_report(run_benchmarks(bms))
    assert "1, Passed: 1" in rep
    assert "PASS" in rep
    assert "linear-static" in rep


def test_export_csv_round_trip(tmp_path):
    bms = [
        Benchmark("a", "linear-static", 1.0, "src", "m",
                  0.01, lambda: 1.0, note="note"),
        Benchmark("b", "modal", 2.0, "src", "Hz",
                  0.05, lambda: 2.1, note="note2"),
    ]
    results = run_benchmarks(bms)
    path = tmp_path / "out.csv"
    export_csv(results, str(path))
    content = path.read_text()
    assert "name" in content
    assert "linear-static" in content
    assert "yes" in content    # at least one pass row


# ============================================================ each benchmark passes

@pytest.mark.parametrize(
    "bm", all_benchmarks(), ids=lambda b: b.name,
)
def test_individual_benchmark_passes(bm):
    """Every shipped V&V benchmark must pass its declared tolerance."""
    r = bm.run()
    if not r.passed:
        msg = (
            f"{bm.name} failed: computed={r.computed_value}, "
            f"reference={bm.reference_value}, "
            f"rel_err={r.relative_error:.2%} (tol {bm.tolerance:.2%})"
        )
        if r.error_message:
            msg += f"; error: {r.error_message}"
        pytest.fail(msg)


def test_category_counts():
    """Each category contributes at least one benchmark."""
    bms = all_benchmarks()
    cats = {b.category for b in bms}
    assert "linear-static" in cats
    assert "modal" in cats
    assert "buckling" in cats
    assert "nonlinear" in cats


def test_benchmark_factory_lengths():
    assert len(linear_static_benchmarks()) >= 5
    assert len(modal_buckling_benchmarks()) >= 4
    assert len(nonlinear_benchmarks()) >= 2
