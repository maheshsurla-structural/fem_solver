"""Run-analysis control (plan A4) — ``desktop/run_analysis_dialog.py``.

Headless coverage of the case table, the Run/Do-not-run actions, inline
linear-static status, deferred nonlinear/time-history queuing, and wiring.
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
    p.analysis_cases = [AnalysisCase(id=3, name="Modal-A", type="modal",
                                     params={"num_modes": 4, "lumped": False})]
    return p


def test_rows_are_runnable_cases(qapp):
    from run_analysis_dialog import RunAnalysisDialog
    dlg = RunAnalysisDialog(None, _project())
    kinds = [m["kind"] for m in dlg._rows]
    # E5a: every saved analysis case now appears; the standalone Time-History
    # launcher is gone (Time History is a saved case).
    assert kinds == ["linear", "nonlinear", "analysis"]
    assert "timehistory" not in kinds
    # linear defaults to Run, the others to Do-not-run
    assert dlg._rows[0]["action"].currentData() is True
    assert dlg._rows[1]["action"].currentData() is False


def test_analysis_case_flagged_is_queued(qapp):
    from run_analysis_dialog import RunAnalysisDialog
    dlg = RunAnalysisDialog(None, _project(), run_linear=lambda: {"neq": 1})
    r = next(i for i, m in enumerate(dlg._rows) if m["kind"] == "analysis")
    act = dlg._rows[r]["action"]
    act.setCurrentIndex(act.findData(True))
    dlg._run_now()
    assert ("case", 3) in dlg.deferred_requests()
    assert dlg.table.item(r, 3).text().startswith("Queued")


def test_run_now_runs_linear_inline_done(qapp):
    from run_analysis_dialog import RunAnalysisDialog
    called = []
    dlg = RunAnalysisDialog(None, _project(),
                            run_linear=lambda: called.append(1) or {"neq": 3})
    dlg._run_now()
    assert called == [1]
    assert dlg.table.item(0, 3).text() == "Done"
    assert dlg.deferred_requests() == []          # nothing queued by default


def test_run_now_linear_failure_shows_failed(qapp):
    from run_analysis_dialog import RunAnalysisDialog
    def boom():
        raise RuntimeError("singular")
    dlg = RunAnalysisDialog(None, _project(), run_linear=boom)
    dlg._run_now()
    assert dlg.table.item(0, 3).text().startswith("Failed")


def test_run_now_no_model_status(qapp):
    from run_analysis_dialog import RunAnalysisDialog
    dlg = RunAnalysisDialog(None, _project(), run_linear=lambda: None)
    dlg._run_now()
    assert dlg.table.item(0, 3).text() == "No model"


def test_nonlinear_flagged_is_queued(qapp):
    from run_analysis_dialog import RunAnalysisDialog
    dlg = RunAnalysisDialog(None, _project(), run_linear=lambda: {"neq": 1})
    # flag the nonlinear row to Run
    act = dlg._rows[1]["action"]
    act.setCurrentIndex(act.findData(True))
    dlg._run_now()
    assert ("nonlinear", 7) in dlg.deferred_requests()
    assert dlg.table.item(1, 3).text().startswith("Queued")


def test_main_window_wires_run_analysis(qapp):
    from main_window import MainWindow
    w = MainWindow()
    w.load_project(_project())
    assert hasattr(w, "act_runanalysis")
    assert callable(w.run_analysis)
