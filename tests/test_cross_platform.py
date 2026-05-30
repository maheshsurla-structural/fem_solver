"""Phase 57 tests -- Theme AA cross-platform validation framework."""
from __future__ import annotations

import math

import pytest

from femsolver.benchmarks import (
    CrossPlatformBenchmark,
    CrossPlatformReference,
    CrossPlatformResult,
    all_cross_platform_benchmarks,
    export_validation_csv,
    format_validation_table,
    frame_cross_platform_benchmarks,
    nafems_cross_platform_benchmarks,
    render_validation_html,
)


# ============================================================ framework

class TestCrossPlatformBenchmark:
    def test_runs_and_compares_to_single_ref(self):
        b = CrossPlatformBenchmark(
            name="trivial", category="test", units="",
            runner=lambda: 1.0,
            references=[CrossPlatformReference(
                source="exact", value=1.0, tolerance=1e-9,
            )],
        )
        r = b.run()
        assert r.overall_passed
        assert r.computed_value == pytest.approx(1.0)

    def test_pass_requires_all_refs_to_pass(self):
        b = CrossPlatformBenchmark(
            name="mixed", category="test", units="",
            runner=lambda: 1.0,
            references=[
                CrossPlatformReference("ref1", 1.0, 0.01),
                CrossPlatformReference("ref2", 0.5, 0.01),
            ],
        )
        r = b.run()
        assert not r.overall_passed
        # ref1 passes, ref2 fails
        assert r.per_reference["ref1"][1]
        assert not r.per_reference["ref2"][1]

    def test_runner_exception_marks_failed(self):
        def bad_runner():
            raise RuntimeError("boom")
        b = CrossPlatformBenchmark(
            name="x", category="test", units="",
            runner=bad_runner,
            references=[CrossPlatformReference("ref", 1.0)],
        )
        r = b.run()
        assert not r.overall_passed
        assert math.isnan(r.computed_value)
        assert r.error_message is not None and "boom" in r.error_message

    def test_zero_reference_uses_absolute(self):
        b = CrossPlatformBenchmark(
            name="x", category="test", units="",
            runner=lambda: 1e-12,
            references=[CrossPlatformReference("zero", 0.0, 1e-9)],
        )
        r = b.run()
        # |computed - 0| = 1e-12 < 1e-9 -> passes
        assert r.overall_passed


# ============================================================ NAFEMS

class TestNafems:
    def test_le1_passes(self):
        nafems = {b.name: b for b in nafems_cross_platform_benchmarks()}
        r = nafems["LE1 short cantilever tip deflection"].run()
        assert r.overall_passed

    def test_le2_passes(self):
        nafems = {b.name: b for b in nafems_cross_platform_benchmarks()}
        r = nafems["LE2 two-bar truss apex deflection"].run()
        assert r.overall_passed

    def test_le3_passes(self):
        nafems = {b.name: b for b in nafems_cross_platform_benchmarks()}
        r = nafems["LE3 cantilever fundamental frequency"].run()
        assert r.overall_passed

    def test_le5_passes(self):
        nafems = {b.name: b for b in nafems_cross_platform_benchmarks()}
        r = nafems["LE5 pin-pin column Euler buckling"].run()
        assert r.overall_passed

    def test_all_nafems_have_multiple_references(self):
        for b in nafems_cross_platform_benchmarks():
            assert len(b.references) >= 2, \
                f"{b.name} has fewer than 2 references"


# ============================================================ frames

class TestFrames:
    def test_propped_cantilever_passes(self):
        frames = {b.name: b for b in frame_cross_platform_benchmarks()}
        r = frames["Propped cantilever UDL pinned reaction"].run()
        assert r.overall_passed
        # 3 w L / 8 = 15000 N
        assert r.computed_value == pytest.approx(15000.0, rel=0.01)

    def test_ss_beam_passes(self):
        frames = {b.name: b for b in frame_cross_platform_benchmarks()}
        r = frames["SS beam UDL midspan deflection"].run()
        assert r.overall_passed


# ============================================================ reporters

class TestReporters:
    def test_format_table_includes_status(self):
        results = [b.run() for b in nafems_cross_platform_benchmarks()]
        out = format_validation_table(results)
        assert "PASS" in out
        assert "LE1" in out

    def test_html_render_complete(self):
        results = [b.run() for b in nafems_cross_platform_benchmarks()]
        html = render_validation_html(results)
        assert "<table" in html
        assert "LE1" in html
        assert "passed" in html.lower()

    def test_csv_export(self, tmp_path):
        results = [b.run() for b in nafems_cross_platform_benchmarks()]
        out = tmp_path / "v.csv"
        export_validation_csv(results, str(out))
        assert out.exists()
        text = out.read_text(encoding="utf-8")
        # Header + one row per (benchmark, source)
        n_refs = sum(len(b.references)
                      for b in nafems_cross_platform_benchmarks())
        # Header + n_refs data rows
        assert text.count("\n") >= n_refs

    def test_empty_results_handled(self):
        out = format_validation_table([])
        assert "no benchmarks" in out

    def test_overall_suite_passes(self):
        results = [b.run() for b in all_cross_platform_benchmarks()]
        n_pass = sum(1 for r in results if r.overall_passed)
        assert n_pass == len(results)
