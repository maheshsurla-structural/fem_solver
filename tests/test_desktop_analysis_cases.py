"""Unified analysis-cases home (plan A1) — ``desktop/analysis_cases_dialog.py``.

Headless coverage of the combined list (built-in + nonlinear + planned rows),
the add / modify / delete of nonlinear cases, the Run request dispatch, and the
main-window wiring.
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
    p.nonlinear_cases = [NonlinearCase(id=1, name="Pushover-X", control_node=2,
                                       control_dof=1)]
    return p


def test_lists_builtin_nonlinear_and_planned(qapp):
    from analysis_cases_dialog import AnalysisCasesDialog, _PLANNED
    dlg = AnalysisCasesDialog(None, _project())
    kinds = [m["kind"] for m in dlg._row_meta]
    assert kinds[0] == "linear"
    assert "nonlinear" in kinds and "timehistory" in kinds
    # un-migrated types are still launcher rows
    for k in ("responsespectrum", "stages", "cabletuning"):
        assert k in kinds
    # migrated types are no longer fixed launcher rows — they are saved
    # AnalysisCase types offered in the Add ▾ menu
    import case_types
    for t in ("modal", "buckling", "movingload", "tempgradient", "loadrating"):
        assert t not in kinds
        assert case_types.get(t)
    assert kinds.count("planned") == len(_PLANNED)
    if _PLANNED:                                   # any remaining planned rows
        planned_row = kinds.index("planned")       # are greyed / not selectable
        from PySide6.QtCore import Qt
        it = dlg.table.item(planned_row, 0)
        assert not (it.flags() & Qt.ItemFlag.ItemIsSelectable)


def test_buttons_gate_on_row_kind(qapp):
    from analysis_cases_dialog import AnalysisCasesDialog
    dlg = AnalysisCasesDialog(None, _project())
    kinds = [m["kind"] for m in dlg._row_meta]
    dlg.table.setCurrentCell(kinds.index("linear"), 0)      # linear: run only
    assert dlg._run_btn.isEnabled()
    assert not dlg._mod_btn.isEnabled() and not dlg._del_btn.isEnabled()
    dlg.table.setCurrentCell(kinds.index("nonlinear"), 0)   # nl: all enabled
    assert dlg._run_btn.isEnabled()
    assert dlg._mod_btn.isEnabled() and dlg._del_btn.isEnabled()


def test_run_request_linear(qapp):
    from analysis_cases_dialog import AnalysisCasesDialog
    dlg = AnalysisCasesDialog(None, _project())
    kinds = [m["kind"] for m in dlg._row_meta]
    dlg.table.setCurrentCell(kinds.index("linear"), 0)
    dlg._run()                                # sets request + accepts
    assert dlg._run_request == ("linear",)


def test_run_request_nonlinear_carries_case_id(qapp):
    from analysis_cases_dialog import AnalysisCasesDialog
    dlg = AnalysisCasesDialog(None, _project())
    kinds = [m["kind"] for m in dlg._row_meta]
    dlg.table.setCurrentCell(kinds.index("nonlinear"), 0)
    dlg._run()
    assert dlg._run_request == ("nonlinear", 1)


def test_delete_blocks_when_continued_from(qapp, monkeypatch):
    import analysis_cases_dialog as mod
    from analysis_cases_dialog import AnalysisCasesDialog
    # the in-use guard pops a modal warning — stub it so the test never blocks
    monkeypatch.setattr(mod.QMessageBox, "warning",
                        staticmethod(lambda *a, **k: None))
    p = _project()
    p.nonlinear_cases = [NonlinearCase(id=1, name="A", control_node=2),
                         NonlinearCase(id=2, name="B", control_node=2,
                                       continue_from=1)]
    dlg = AnalysisCasesDialog(None, p)
    # select the first nonlinear row (row 1: after the linear row); id 1 is
    # continued-from by id 2, so delete must refuse
    dlg.table.setCurrentCell(1, 0)
    dlg._delete()
    assert len(dlg._cases) == 2               # refused, nothing deleted


def test_main_window_wires_analysis_cases(qapp):
    from main_window import MainWindow
    w = MainWindow()
    w.load_project(_project())
    assert hasattr(w, "act_analysiscases")
    assert callable(w.manage_analysis_cases)


# ------------------------------------------------ saved AnalysisCase CRUD (ACM)
def test_add_case_appends_and_selects(qapp, monkeypatch):
    import case_types
    from analysis_cases_dialog import AnalysisCasesDialog
    dlg = AnalysisCasesDialog(None, _project())
    monkeypatch.setattr(case_types.TYPES["modal"], "edit",
                        lambda parent, project, case=None:
                        AnalysisCase(id=0, name="Modal-A", type="modal",
                                     params={"num_modes": 4, "lumped": True}))
    dlg._add_case("modal")
    assert len(dlg._acases) == 1
    new = dlg._acases[-1]
    assert new.id >= 1 and new.name == "Modal-A"
    m = dlg._row_meta[dlg.table.currentRow()]     # the new row is selected
    assert m["kind"] == "analysis" and m["case_id"] == new.id


def test_modify_case_replaces_params(qapp, monkeypatch):
    import case_types
    from analysis_cases_dialog import AnalysisCasesDialog
    p = _project()
    p.analysis_cases = [AnalysisCase(id=7, name="Modal-7", type="modal",
                                     params={"num_modes": 6, "lumped": False})]
    dlg = AnalysisCasesDialog(None, p)
    monkeypatch.setattr(case_types.TYPES["modal"], "edit",
                        lambda parent, project, case=None:
                        AnalysisCase(id=case.id, name="Modal-7b", type="modal",
                                     params={"num_modes": 9, "lumped": True}))
    kinds = [m["kind"] for m in dlg._row_meta]
    dlg.table.setCurrentCell(kinds.index("analysis"), 0)
    dlg._modify()
    assert dlg._acases[0].params["num_modes"] == 9
    assert dlg._acases[0].name == "Modal-7b" and dlg._acases[0].id == 7


def test_delete_case(qapp):
    from analysis_cases_dialog import AnalysisCasesDialog
    p = _project()
    p.analysis_cases = [AnalysisCase(id=1, name="M", type="modal",
                                     params={"num_modes": 4, "lumped": False})]
    dlg = AnalysisCasesDialog(None, p)
    kinds = [m["kind"] for m in dlg._row_meta]
    dlg.table.setCurrentCell(kinds.index("analysis"), 0)
    dlg._delete()
    assert dlg._acases == []


def test_manage_returns_three_tuple(qapp, monkeypatch):
    from analysis_cases_dialog import AnalysisCasesDialog
    monkeypatch.setattr(AnalysisCasesDialog, "exec", lambda self: True)
    res = AnalysisCasesDialog.manage(None, _project())
    assert res is not None and len(res) == 3
    cases, acases, run = res
    assert isinstance(cases, list) and isinstance(acases, list) and run is None


def test_run_saved_case_dispatches(qapp, monkeypatch):
    import case_types
    from main_window import MainWindow
    p = _project()
    p.analysis_cases = [AnalysisCase(id=3, name="Modal-3", type="modal",
                                     params={"num_modes": 5, "lumped": True})]
    w = MainWindow()
    w.load_project(p)
    seen = {}
    monkeypatch.setattr(case_types.TYPES["modal"], "dispatch",
                        lambda win, config: seen.setdefault("config", config))
    w._run_saved_case(3)
    assert seen["config"] == (5, True)      # build_config(params) → runtime cfg


def test_analysis_case_serialization_round_trip():
    p = Project(ndm=2, ndf=3)
    p.analysis_cases = [
        AnalysisCase(id=1, name="Modal-6", type="modal",
                     params={"num_modes": 6, "lumped": False}, notes="n"),
        AnalysisCase(id=2, name="Buckling", type="buckling",
                     params={"selection": ["case", 2], "num_modes": 4,
                             "subdivisions": 8}),
    ]
    back = Project.from_json(p.to_json())
    assert len(back.analysis_cases) == 2
    a, b = back.analysis_cases
    assert a.id == 1 and a.type == "modal" and a.params["num_modes"] == 6
    assert a.notes == "n"
    assert b.params["selection"] == ["case", 2] and b.params["subdivisions"] == 8
