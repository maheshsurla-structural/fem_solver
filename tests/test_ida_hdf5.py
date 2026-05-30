"""Phase 48.1 tests -- HDF5 persistence for IDA results."""
from __future__ import annotations

import math
import os

import numpy as np
import pytest

from femsolver.analysis.ida import IDAPoint, IDARecord
from femsolver.analysis.ida_collapse import CollapseResult, IDASummary

h5py = pytest.importorskip("h5py")
from femsolver.io import (   # noqa: E402  (after importorskip)
    load_ida_record,
    load_ida_summary,
    save_ida_record,
    save_ida_summary,
)


def _make_record(name: str = "rec_a", n_levels: int = 5) -> IDARecord:
    pts = []
    for i in range(n_levels):
        im = 0.1 * (i + 1)
        edp = 0.001 * (i + 1) ** 1.6        # mock IM-EDP curve
        pts.append(IDAPoint(
            IM=im,
            scale_factor=im / 0.5,
            EDPs={"max_drift_ratio": edp, "max_roof_drift": 0.01 * (i + 1)},
            converged=(i < n_levels - 1),       # last one diverged
            n_steps_completed=200 - 10 * i,
        ))
    return IDARecord(record_name=name, points=pts)


# ============================================================ round-trip

class TestRecordRoundTrip:
    def test_round_trip_single(self, tmp_path):
        rec = _make_record("rec_a", n_levels=5)
        path = tmp_path / "ida.h5"
        save_ida_record(rec, str(path))
        assert os.path.exists(path)
        loaded = load_ida_record(str(path))
        assert loaded.record_name == "rec_a"
        assert len(loaded.points) == 5
        for p_in, p_out in zip(rec.points, loaded.points):
            assert p_out.IM == pytest.approx(p_in.IM)
            assert p_out.scale_factor == pytest.approx(p_in.scale_factor)
            assert p_out.converged == p_in.converged
            assert p_out.n_steps_completed == p_in.n_steps_completed
            for k in p_in.EDPs:
                assert p_out.EDPs[k] == pytest.approx(p_in.EDPs[k])

    def test_record_name_required_when_multiple(self, tmp_path):
        # Build a 2-record summary to exercise the "name required" branch
        rec_a = _make_record("alpha", 3)
        rec_b = _make_record("beta", 3)
        summary = IDASummary(
            records=[rec_a, rec_b],
            collapse_results=[
                CollapseResult(0.5, "drift_limit", 2),
                CollapseResult(0.4, "non_convergence", 2),
            ],
            collapse_IMs=np.array([0.5, 0.4]),
            n_collapsed=2, median_collapse_IM=0.45,
        )
        path = tmp_path / "ida2.h5"
        save_ida_summary(summary, str(path))
        with pytest.raises(ValueError, match="record_name required"):
            load_ida_record(str(path))
        # Loading by name should work
        rec = load_ida_record(str(path), record_name="alpha")
        assert rec.record_name == "alpha"


class TestSummaryRoundTrip:
    def test_collapse_block_preserved(self, tmp_path):
        rec_a = _make_record("a", 4)
        rec_b = _make_record("b", 4)
        summary = IDASummary(
            records=[rec_a, rec_b],
            collapse_results=[
                CollapseResult(0.6, "drift_limit", 3),
                CollapseResult(float("inf"), "no_collapse", None),
            ],
            collapse_IMs=np.array([0.6, math.inf]),
            n_collapsed=1, median_collapse_IM=0.6,
        )
        path = tmp_path / "summary.h5"
        save_ida_summary(summary, str(path))
        loaded = load_ida_summary(str(path))
        assert len(loaded.records) == 2
        assert loaded.n_collapsed == 1
        assert loaded.median_collapse_IM == pytest.approx(0.6)
        # First record collapsed at 0.6 with drift_limit
        cr0 = loaded.collapse_results[0]
        assert cr0.collapse_IM == pytest.approx(0.6)
        assert cr0.cause == "drift_limit"
        assert cr0.collapse_point_index == 3
        # Second did not collapse
        cr1 = loaded.collapse_results[1]
        assert math.isinf(cr1.collapse_IM)
        assert cr1.cause == "no_collapse"
        assert cr1.collapse_point_index is None


class TestEdpHandling:
    def test_missing_edp_becomes_nan(self, tmp_path):
        # One record has a third EDP only in some points -- missing
        # values must round-trip as NaN.
        pts = [
            IDAPoint(IM=0.1, scale_factor=0.2,
                       EDPs={"max_drift_ratio": 0.001, "extra": 1.0},
                       converged=True),
            IDAPoint(IM=0.2, scale_factor=0.4,
                       EDPs={"max_drift_ratio": 0.003},
                       converged=True),
        ]
        rec = IDARecord(record_name="x", points=pts)
        path = tmp_path / "missing.h5"
        save_ida_record(rec, str(path))
        loaded = load_ida_record(str(path))
        assert loaded.points[0].EDPs["extra"] == pytest.approx(1.0)
        assert math.isnan(loaded.points[1].EDPs["extra"])
