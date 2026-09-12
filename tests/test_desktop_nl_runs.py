"""Nonlinear run-history persistence + browser (plan §16 G-S2).

A completed pushover/cyclic/time-history run is saved to the project as a lean
``RunRecord`` (curve + summary + ASCE 41 milestones); it survives save/load and
is re-viewable in the Run-history dialog. Headless (offscreen).
"""
from __future__ import annotations

import math
import os
import sys
from pathlib import Path

import numpy as np
import pytest

os.environ.setdefault("QT_QPA_PLATFORM", "offscreen")
_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(_ROOT))
sys.path.insert(0, str(_ROOT / "desktop"))

from nl_results import NonlinearResults            # noqa: E402
from nl_runs import RunRecord, next_run_id         # noqa: E402
from project import Project                        # noqa: E402


def _results():
    r = NonlinearResults(
        disp=[0.0, 0.01, 0.02, 0.03], shear=[0.0, 100.0, 180.0, 150.0],
        protocol="monotonic")
    r.accept_milestones = {"IO": {"disp": 0.01}, "LS": {"disp": 0.02}}
    return r


# ------------------------------------------------------ RunRecord

def test_from_results_curve_summary_and_milestones():
    rec = RunRecord.from_results(_results(), id=1, name="PO", kind="pushover",
                                 meta={"Control": "node 2 Uy"})
    assert rec.x == [0.0, 0.01, 0.02, 0.03]
    assert rec.peak == pytest.approx(180.0)        # signed largest |shear|
    assert rec.peak_x == pytest.approx(0.02)
    assert rec.milestones == {"IO": 0.01, "LS": 0.02}
    assert rec.meta["Control"] == "node 2 Uy"
    # a curve-only NonlinearResults is reconstructable for re-plotting
    back = rec.results()
    assert back.curve()[1][2] == pytest.approx(180.0)


def test_from_results_is_json_safe_with_numpy_inputs():
    import json
    r = NonlinearResults(disp=np.linspace(0, 0.03, 4),
                         shear=np.array([0.0, 100.0, 180.0, 150.0]))
    rec = RunRecord.from_results(r, id=2, name="np")
    # every stored number is a plain float -> json.dumps must not raise
    json.dumps(rec.to_dict())
    assert all(isinstance(v, float) for v in rec.x + rec.y)


def test_from_time_history_record():
    res = {"times": [0.0, 0.1, 0.2, 0.3], "disp": [0.0, 0.5, -0.8, 0.3]}
    rec = RunRecord.from_time_history(res, id=1, name="TH")
    assert rec.kind == "time_history" and rec.x_label == "time"
    assert rec.peak == pytest.approx(-0.8) and rec.peak_x == pytest.approx(0.2)


# ------------------------------------------------------ project round-trip

def test_project_runs_survive_save_load(tmp_path):
    p = Project()
    p.runs.append(RunRecord.from_results(_results(), id=next_run_id(p.runs),
                                         name="PO", meta={"Target": "0.03 m"}))
    path = tmp_path / "m.fsproj"
    p.save(path)
    q = Project.load(path)
    assert len(q.runs) == 1
    r = q.runs[0]
    assert isinstance(r, RunRecord)
    assert r.name == "PO"
    assert r.x == [0.0, 0.01, 0.02, 0.03]
    assert r.milestones == {"IO": 0.01, "LS": 0.02}
    assert r.meta["Target"] == "0.03 m"


def test_from_dict_ignores_unknown_keys():
    rec = RunRecord.from_dict({"id": 5, "name": "x", "x": [1], "y": [2],
                               "bogus": 99})
    assert rec.id == 5 and rec.x == [1.0]


# ------------------------------------------------------ dialog

@pytest.fixture(scope="module")
def qapp():
    pytest.importorskip("PySide6")
    pytest.importorskip("matplotlib")
    from PySide6.QtWidgets import QApplication
    yield QApplication.instance() or QApplication([])


def _project_with_runs():
    p = Project()
    p.runs = [
        RunRecord.from_results(_results(), id=1, name="PO-1", kind="pushover"),
        RunRecord.from_time_history({"times": [0, 0.1, 0.2],
                                     "disp": [0, 0.3, -0.2]}, id=2, name="TH-1"),
    ]
    return p


def test_dialog_lists_plots_and_edits(qapp):
    from run_history_dialog import RunHistoryDialog
    p = _project_with_runs()
    dlg = RunHistoryDialog(None, p)
    assert dlg.list.count() == 2
    dlg.list.setCurrentRow(0)
    dlg._draw()                                    # plots the pushover, no crash
    # rename + delete mutate the working copy, not the project until accepted
    dlg._runs[0].name = "renamed"
    dlg._refresh(0)
    assert "renamed" in dlg.list.item(0).text()
    dlg.list.setCurrentRow(1)
    dlg._delete()
    assert len(dlg.result_runs()) == 1
    assert p.runs[0].name == "PO-1"                # original untouched


# ------------------------------------------------------ main-window integration

def _fiber_column():
    import dataclasses
    import section_gui_core as core
    from project import Material, Member, Node, Section
    spec = core.Spec(kind="Circular", D=0.6, fc=35e6, fy=500e6)
    gsd = dataclasses.asdict(spec)
    p = Project()
    p.nodes = [Node(1, 0.0, 0.0, supports=(1, 1, 1)), Node(2, 3.0, 0.0)]
    p.sections = [Section(id=1, name="col", A=math.pi * 0.09,
                          Iz=math.pi * 0.6**4 / 64.0, gsd_spec=gsd)]
    p.materials = [Material(1, "conc", E=30e9, nu=0.2)]
    p.members = [Member(1, 1, 2, 1, 1)]
    return p


def test_main_window_records_run_and_it_persists(qapp, tmp_path):
    import nonlinear as NL
    from main_window import MainWindow
    from nl_results import NonlinearResults
    w = MainWindow()
    p = _fiber_column()
    w.load_project(p)
    res = NonlinearResults.from_run(
        NL.run_pushover(p, control_node=2, control_dof=1, target=0.03,
                        n_steps=6))
    w._record_run(("Pushover", "pushover", {"Control": "node 2 Uy"}), res)
    assert len(w._project.runs) == 1               # recorded as an undoable edit
    assert not w._undo_stack.isClean()             # marks the project dirty
    # persists through save/load
    path = tmp_path / "rec.fsproj"
    w._project.save(path)
    reopened = Project.load(path)
    assert len(reopened.runs) == 1
    assert reopened.runs[0].name == "Pushover"
    assert reopened.runs[0].peak != 0.0
    # undo removes it
    w._undo_stack.undo()
    assert len(w._project.runs) == 0


def test_show_run_history_without_runs_is_guarded(qapp):
    from main_window import MainWindow
    w = MainWindow()
    w.load_project(_fiber_column())
    w.show_run_history()                           # no runs -> status msg, no dialog
    assert "No saved runs" in w.statusBar().currentMessage()
