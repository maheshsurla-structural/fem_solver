"""Step-indexed nonlinear-results model (plan GUI-I1).

Unit tests for ``desktop.nl_results.NonlinearResults`` -- the pure, Qt-free
model that consolidates a run's per-step arrays so views read "state at step k"
uniformly. Round-trips the run dict, handles the no-capture case, and reports
summary quantities.
"""
from __future__ import annotations

import os
import sys

import pytest

sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", "desktop"))

from nl_results import NonlinearResults, StepState   # noqa: E402


def test_accept_frames_and_milestones_roundtrip():
    """Acceptance levels are exposed per step (member_state) and milestones are
    held + round-tripped (§16 C5)."""
    res = {"protocol": "monotonic", "disp": [0.0, 0.02, 0.05],
           "shear": [0.0, 200.0, 300.0],
           "shape_frames": [{1: (0.0, 0.0)}, {1: (0.0, 0.02)}, {1: (0.0, 0.05)}],
           "damage_frames": [{1: 0.0}, {1: 0.004}, {1: 0.012}],
           "accept_frames": [{1: 0}, {1: 1}, {1: 2}],
           "accept_milestones": {"IO": {"step": 2, "disp": 0.02}}}
    r = NonlinearResults.from_run(res)
    assert r.step(0).member_state == {1: 0}
    assert r.step(2).member_state == {1: 2}
    assert r.accept_milestones["IO"]["disp"] == 0.02
    d = r.to_dict()
    assert d["accept_frames"] == res["accept_frames"]
    assert d["accept_milestones"] == res["accept_milestones"]


def _run_dict(with_capture: bool) -> dict:
    d = {"disp": [0.0, 0.5, 1.0], "shear": [0.0, 80.0, -60.0],
         "protocol": "cyclic"}
    if with_capture:
        d["fiber_frames"] = [
            [(0.1, 0.0, 1.0e6, 1.0e-4)],
            [(0.1, 0.0, 2.0e6, 2.0e-4)],
            [(0.1, 0.0, -3.0e6, -3.0e-4)]]
        d["shape_frames"] = [{1: (0.0, 0.0)}, {1: (0.0, 0.5)}, {1: (0.0, 1.0)}]
        d["damage_frames"] = [{1: 0.0}, {1: 2.0e-4}, {1: 3.0e-4}]
    return d


def test_from_run_and_curve():
    r = NonlinearResults.from_run(_run_dict(False))
    assert r.protocol == "cyclic"
    assert r.n_curve == 3
    assert r.n_steps == 0                         # no captures
    assert not r.has_fibers and not r.has_shape
    disp, shear = r.curve()
    assert disp == [0.0, 0.5, 1.0]
    assert shear == [0.0, 80.0, -60.0]


def test_step_with_captures():
    r = NonlinearResults.from_run(_run_dict(True))
    assert r.n_steps == 3 and r.has_fibers and r.has_shape
    st = r.step(2)
    assert isinstance(st, StepState)
    assert st.index == 2
    assert st.disp == 1.0 and st.shear == -60.0
    assert st.node_disp == {1: (0.0, 1.0)}
    assert st.member_damage == {1: 3.0e-4}
    assert st.fibers[0] == (0.1, 0.0, -3.0e6, -3.0e-4)


def test_step_without_captures_returns_none_fields():
    r = NonlinearResults.from_run(_run_dict(False))
    st = r.step(1)
    assert st.disp == 0.5 and st.shear == 80.0
    assert st.node_disp is None and st.member_damage is None
    assert st.fibers is None


def test_step_bounds():
    r = NonlinearResults.from_run(_run_dict(True))
    with pytest.raises(IndexError):
        r.step(3)
    with pytest.raises(IndexError):
        r.step(-1)
    with pytest.raises(IndexError):
        NonlinearResults([], []).step(0)


def test_to_dict_roundtrip():
    src = _run_dict(True)
    r = NonlinearResults.from_run(src)
    d = r.to_dict()
    assert d["disp"] == src["disp"] and d["shear"] == src["shear"]
    assert d["protocol"] == "cyclic"
    assert d["fiber_frames"] == src["fiber_frames"]
    assert d["shape_frames"] == src["shape_frames"]
    assert d["damage_frames"] == src["damage_frames"]
    # no-capture: the frame keys are omitted, not empty
    d0 = NonlinearResults.from_run(_run_dict(False)).to_dict()
    assert "fiber_frames" not in d0 and "shape_frames" not in d0


def test_summary_quantities():
    r = NonlinearResults.from_run(_run_dict(True))
    assert r.peak_shear() == 80.0                 # largest magnitude, signed
    assert r.disp_at_peak() == 0.5
    assert r.max_abs_disp() == 1.0
    assert len(r) == 3


def test_empty_results():
    r = NonlinearResults([], [])
    assert len(r) == 0 and r.n_steps == 0
    assert r.peak_shear() == 0.0 and r.max_abs_disp() == 0.0
