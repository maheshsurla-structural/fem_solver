"""Session run status (E5c) — the transient per-case status registry, the manager
Status column, the Run-Analysis dialog seeding, dispatch updates, and the E4
tree "stale" flag.
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
    p.analysis_cases = [AnalysisCase(id=1, name="Modal-6", type="modal",
                                     params={"num_modes": 6, "lumped": False})]
    return p


# ------------------------------------------------------- registry

def test_case_status_registry_is_transient():
    p = _project()
    assert p.case_status(("analysis", 1)) is None            # unset
    p.set_case_status(("analysis", 1), "Finished", when=123.0)
    st = p.case_status(("analysis", 1))
    assert st["status"] == "Finished" and st["when"] == 123.0
    # NOT serialized (results are recomputed each session)
    assert "_case_status" not in p.to_dict()
    assert Project.from_json(p.to_json()).case_status(("analysis", 1)) is None


# ------------------------------------------------------- manager column

def test_manager_status_column_shows_status(qapp):
    from analysis_cases_dialog import AnalysisCasesDialog
    p = _project()
    p.set_case_status(("analysis", 1), "Finished")
    dlg = AnalysisCasesDialog(None, p)
    r = next(i for i, m in enumerate(dlg._row_meta) if m["kind"] == "analysis")
    assert dlg.table.item(r, 3).text() == "Finished"
    lr = next(i for i, m in enumerate(dlg._row_meta) if m["kind"] == "stages")
    assert dlg.table.item(lr, 3).text() == "—"               # not run


def test_run_dialog_seeds_status(qapp):
    from run_analysis_dialog import RunAnalysisDialog
    p = _project()
    p.set_case_status(("analysis", 1), "Finished")
    dlg = RunAnalysisDialog(None, p)
    r = next(i for i, m in enumerate(dlg._rows) if m["kind"] == "analysis")
    assert dlg.table.item(r, 3).text() == "Finished"


# ------------------------------------------------------- dispatch updates

def test_run_saved_case_sets_finished(qapp, monkeypatch):
    import case_types
    from main_window import MainWindow
    monkeypatch.setattr(case_types.TYPES["modal"], "dispatch",
                        lambda win, config: None)
    w = MainWindow()
    w.load_project(_project())
    w._run_saved_case(1)
    assert w._project.case_status(("analysis", 1))["status"] == "Finished"


def test_run_saved_case_sets_could_not_start(qapp, monkeypatch):
    import case_types
    import main_window as mw
    from main_window import MainWindow

    def _boom(*a, **k):
        raise ValueError("bad config")
    monkeypatch.setattr(case_types.TYPES["modal"], "build_config", _boom)
    monkeypatch.setattr(mw.QMessageBox, "warning",
                        staticmethod(lambda *a, **k: None))
    w = MainWindow()
    w.load_project(_project())
    w._run_saved_case(1)
    assert w._project.case_status(("analysis", 1))["status"] \
        == "Could not start"


# ------------------------------------------------------- tree stale flag

def test_case_graph_stale_when_source_ran_later():
    import case_graph
    p = Project()
    p.nonlinear_cases = [NonlinearCase(id=1, name="PRELOAD", control_node=2)]
    p.analysis_cases = [AnalysisCase(id=1, name="Modal-after", type="modal",
                                     params={},
                                     initial_condition=("state", 1))]
    p.set_case_status(("analysis", 1), "Finished", when=100.0)   # ran first
    p.set_case_status(("nonlinear", 1), "Finished", when=200.0)  # source later
    nodes = {n["key"]: n for n in case_graph.case_nodes(p)}
    assert nodes[("analysis", 1)]["stale"] is True
    assert nodes[("nonlinear", 1)]["stale"] is False            # a root
    # if the case ran after its source, it is fresh
    p.set_case_status(("analysis", 1), "Finished", when=300.0)
    nodes = {n["key"]: n for n in case_graph.case_nodes(p)}
    assert nodes[("analysis", 1)]["stale"] is False
