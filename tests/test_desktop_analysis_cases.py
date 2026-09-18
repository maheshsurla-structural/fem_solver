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
    assert "linear" not in kinds              # E3b: no built-in launcher row
    assert "nonlinear" in kinds
    # Construction Stages stays a launcher row (operates on shared project.stages)
    assert "stages" in kinds
    # built-in types are saved AnalysisCase types offered in the Add ▾ menu, not
    # fixed launcher rows — including Linear Static (linstatic, E3b)
    import case_types
    for t in ("linstatic", "modal", "buckling", "movingload", "tempgradient",
              "loadrating", "responsespectrum", "vehicledynamics",
              "influencesurface", "cabletuning", "timehistory"):
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
    dlg.table.setCurrentCell(kinds.index("stages"), 0)      # stages: run only
    assert dlg._run_btn.isEnabled()
    assert not dlg._mod_btn.isEnabled() and not dlg._del_btn.isEnabled()
    dlg.table.setCurrentCell(kinds.index("nonlinear"), 0)   # nl: all enabled
    assert dlg._run_btn.isEnabled()
    assert dlg._mod_btn.isEnabled() and dlg._del_btn.isEnabled()


def test_run_request_stages(qapp):
    from analysis_cases_dialog import AnalysisCasesDialog
    dlg = AnalysisCasesDialog(None, _project())
    kinds = [m["kind"] for m in dlg._row_meta]
    dlg.table.setCurrentCell(kinds.index("stages"), 0)
    dlg._run()                                # sets request + accepts
    assert dlg._run_request == ("stages",)


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
    # select the first nonlinear row (row 0, now the launcher row is gone);
    # id 1 is continued-from by id 2, so delete must refuse
    dlg.table.setCurrentCell(0, 0)
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
    # build_config(params) → runtime cfg; now carries the E2 initial condition
    assert seen["config"] == (5, True, ("zero",))
    assert "Modal-3" in w.log.toPlainText()  # run log names the case


def test_add_case_auto_suffixes_duplicate_name(qapp, monkeypatch):
    import case_types
    from analysis_cases_dialog import AnalysisCasesDialog
    dlg = AnalysisCasesDialog(None, _project())
    monkeypatch.setattr(case_types.TYPES["modal"], "edit",
                        lambda parent, project, case=None:
                        AnalysisCase(id=0, name="Dup", type="modal",
                                     params={"num_modes": 4, "lumped": False}))
    dlg._add_case("modal")
    dlg._add_case("modal")
    dlg._add_case("modal")
    assert [c.name for c in dlg._acases] == ["Dup", "Dup (2)", "Dup (3)"]


def test_saved_case_notes_shown_as_tooltip(qapp):
    from analysis_cases_dialog import AnalysisCasesDialog
    p = _project()
    p.analysis_cases = [AnalysisCase(id=1, name="Modal-DBE", type="modal",
                                     params={"num_modes": 4, "lumped": False},
                                     notes="design basis event")]
    dlg = AnalysisCasesDialog(None, p)
    kinds = [m["kind"] for m in dlg._row_meta]
    r = kinds.index("analysis")
    assert "design basis event" in dlg.table.item(r, 0).toolTip()


def _two_modal(p):
    p.analysis_cases = [
        AnalysisCase(id=1, name="A", type="modal",
                     params={"num_modes": 2, "lumped": False}),
        AnalysisCase(id=2, name="B", type="modal",
                     params={"num_modes": 3, "lumped": False}),
    ]
    return p


def test_duplicate_analysis_case(qapp):
    from analysis_cases_dialog import AnalysisCasesDialog
    p = _project()
    p.analysis_cases = [AnalysisCase(id=1, name="RS-X", type="responsespectrum",
                                     params={"source": "asce7", "num_modes": 6})]
    dlg = AnalysisCasesDialog(None, p)
    kinds = [m["kind"] for m in dlg._row_meta]
    dlg.table.setCurrentCell(kinds.index("analysis"), 0)
    dlg._duplicate()
    assert len(dlg._acases) == 2
    clone = dlg._acases[-1]
    assert clone.name == "RS-X (copy)" and clone.id != 1
    assert clone.params == dlg._acases[0].params          # copied content
    assert clone.params is not dlg._acases[0].params      # but a distinct object
    m = dlg._row_meta[dlg.table.currentRow()]             # clone is selected
    assert m["kind"] == "analysis" and m["case_id"] == clone.id


def test_move_reorders_within_list(qapp):
    from analysis_cases_dialog import AnalysisCasesDialog
    dlg = AnalysisCasesDialog(None, _two_modal(_project()))
    kinds = [m["kind"] for m in dlg._row_meta]
    dlg.table.setCurrentCell(kinds.index("analysis"), 0)  # case A (id 1)
    dlg._move(1)                                           # down
    assert [c.id for c in dlg._acases] == [2, 1]
    assert dlg._row_meta[dlg.table.currentRow()]["case_id"] == 1  # follows
    dlg._move(-1)                                          # back up
    assert [c.id for c in dlg._acases] == [1, 2]


def test_analysis_cases_lists_saved_timehistory_case(qapp):
    from analysis_cases_dialog import AnalysisCasesDialog
    from project import TimeHistoryFunction
    p = _project()
    p.th_functions = [TimeHistoryFunction(id=1, name="EC", dt=0.02,
                                          values=[0.1, 0.2, 0.3])]
    p.analysis_cases = [AnalysisCase(id=1, name="TH-EC", type="timehistory",
                                     params={"function_id": 1, "control_node": 2,
                                             "direction": "y", "scale": 1.0})]
    dlg = AnalysisCasesDialog(None, p)
    kinds = [m["kind"] for m in dlg._row_meta]
    r = kinds.index("analysis")
    assert "EC" in dlg._row_meta[r]["detail"]       # detail resolves the function
    dlg.table.setCurrentCell(r, 0)
    assert dlg._mod_btn.isEnabled() and dlg._del_btn.isEnabled()
    dlg._run()
    assert dlg._run_request == ("case", 1)


def test_move_and_dup_buttons_gate(qapp):
    from analysis_cases_dialog import AnalysisCasesDialog
    dlg = AnalysisCasesDialog(None, _two_modal(_project()))
    kinds = [m["kind"] for m in dlg._row_meta]
    rows = [i for i, k in enumerate(kinds) if k == "analysis"]
    dlg.table.setCurrentCell(rows[0], 0)                  # first: up off, down on
    assert not dlg._up_btn.isEnabled() and dlg._down_btn.isEnabled()
    assert dlg._dup_btn.isEnabled()
    dlg.table.setCurrentCell(rows[1], 0)                  # last: up on, down off
    assert dlg._up_btn.isEnabled() and not dlg._down_btn.isEnabled()
    dlg.table.setCurrentCell(kinds.index("stages"), 0)    # launcher: none apply
    assert not dlg._dup_btn.isEnabled()
    assert not dlg._up_btn.isEnabled() and not dlg._down_btn.isEnabled()


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


# --------------------------------------------------------- E6 manager chrome

def test_filter_hides_nonmatching_rows(qapp):
    from analysis_cases_dialog import AnalysisCasesDialog
    p = _project()
    p.analysis_cases = [AnalysisCase(id=1, name="Modal-6", type="modal",
                                     params={"num_modes": 6, "lumped": False})]
    dlg = AnalysisCasesDialog(None, p)
    rows = {m["name"]: r for r, m in enumerate(dlg._row_meta)}
    dlg._filter.setText("modal")
    assert not dlg.table.isRowHidden(rows["Modal-6"])       # matches
    assert dlg.table.isRowHidden(rows["Construction Stages"])   # hidden
    dlg._filter.setText("")                                  # cleared -> all show
    assert not dlg.table.isRowHidden(rows["Construction Stages"])


def test_context_menu_offers_actions_per_row(qapp):
    from analysis_cases_dialog import AnalysisCasesDialog
    p = _project()
    p.analysis_cases = [AnalysisCase(id=1, name="Modal-6", type="modal",
                                     params={"num_modes": 6, "lumped": False})]
    dlg = AnalysisCasesDialog(None, p)
    # a saved analysis case row: Run/Modify/Duplicate/Delete enabled
    r = next(i for i, m in enumerate(dlg._row_meta) if m["kind"] == "analysis")
    dlg.table.setCurrentCell(r, 0)
    labels = {a.text(): a.isEnabled() for a in dlg._build_context_menu().actions()
              if a.text()}
    assert labels.get("Run") and labels.get("Modify…") and labels.get("Delete")
    assert "Show tree…" in labels
    # a launcher row (Construction Stages): Run enabled, Modify/Delete disabled
    lr = next(i for i, m in enumerate(dlg._row_meta) if m["kind"] == "stages")
    dlg.table.setCurrentCell(lr, 0)
    labels = {a.text(): a.isEnabled() for a in dlg._build_context_menu().actions()
              if a.text()}
    assert labels.get("Run") and not labels.get("Modify…")
