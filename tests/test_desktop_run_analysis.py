"""Run-analysis control (plan A4) — ``desktop/run_analysis_dialog.py``.

Headless coverage of the case table, the Run/Do-not-run actions, deferred
queuing of every flagged case (saved analysis cases — incl. Linear Static,
E3b — and nonlinear cases), and the main-window wiring.
"""
from __future__ import annotations

import os
import sys
from pathlib import Path

import pytest

os.environ.setdefault("QT_QPA_PLATFORM", "offscreen")
_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(_ROOT))
sys.path.insert(0, str(_ROOT / "desktop"))

from project import (AnalysisCase, LoadCase, Material, Member,  # noqa: E402
                     NonlinearCase, Node, Project, Section)


@pytest.fixture(scope="module")
def qapp():
    pytest.importorskip("PySide6")
    from PySide6.QtWidgets import QApplication
    yield QApplication.instance() or QApplication([])


def _project():
    p = Project(ndm=2, ndf=3)
    p.nodes = [Node(1, 0, 0, supports=(1, 1, 1)), Node(2, 0, 3)]
    p.sections = [Section(id=1, name="s", A=1e-3, Iz=1e-5)]
    p.materials = [Material(1, "m", E=2e11, nu=0.3)]
    p.members = [Member(1, 1, 2, 1, 1)]
    p.load_cases = [LoadCase(1, "Dead", "dead")]
    p.nonlinear_cases = [NonlinearCase(id=7, name="Pushover", control_node=2,
                                       control_dof=1)]
    p.analysis_cases = [
        AnalysisCase(id=3, name="Modal-A", type="modal",
                     params={"num_modes": 4, "lumped": False}),
        AnalysisCase(id=5, name="Static-1", type="linstatic",
                     params={"loads_applied": [[1, 1.0]]}),
    ]
    return p


def test_rows_are_runnable_cases(qapp):
    from run_analysis_dialog import RunAnalysisDialog
    dlg = RunAnalysisDialog(None, _project())
    kinds = [m["kind"] for m in dlg._rows]
    # E3b: Linear Static is a saved analysis case now, not a built-in launcher
    # row; every saved analysis case appears (E5a), none as a special "linear".
    assert "linear" not in kinds
    assert kinds == ["nonlinear", "analysis", "analysis"]
    # nothing auto-runs — the user flags what to run
    assert all(not m["action"].currentData() for m in dlg._rows)


def test_analysis_case_flagged_is_queued(qapp):
    from run_analysis_dialog import RunAnalysisDialog
    dlg = RunAnalysisDialog(None, _project())
    r = next(i for i, m in enumerate(dlg._rows) if m.get("case_id") == 3)
    act = dlg._rows[r]["action"]
    act.setCurrentIndex(act.findData(True))
    dlg._run_now()
    assert ("case", 3) in dlg.deferred_requests()
    assert dlg.table.item(r, 3).text().startswith("Queued")


def test_linstatic_case_flagged_is_queued(qapp):
    from run_analysis_dialog import RunAnalysisDialog
    dlg = RunAnalysisDialog(None, _project())
    r = next(i for i, m in enumerate(dlg._rows) if m.get("case_id") == 5)
    act = dlg._rows[r]["action"]
    act.setCurrentIndex(act.findData(True))
    dlg._run_now()
    assert ("case", 5) in dlg.deferred_requests()      # runs headlessly after


def test_nonlinear_flagged_is_queued(qapp):
    from run_analysis_dialog import RunAnalysisDialog
    dlg = RunAnalysisDialog(None, _project())
    r = next(i for i, m in enumerate(dlg._rows) if m["kind"] == "nonlinear")
    act = dlg._rows[r]["action"]
    act.setCurrentIndex(act.findData(True))
    dlg._run_now()
    assert ("nonlinear", 7) in dlg.deferred_requests()
    assert dlg.table.item(r, 3).text().startswith("Queued")


def test_main_window_wires_run_analysis(qapp):
    from main_window import MainWindow
    w = MainWindow()
    w.load_project(_project())
    assert hasattr(w, "act_run")                       # Ctrl+R opens the chooser
    assert callable(w.run_analysis)
