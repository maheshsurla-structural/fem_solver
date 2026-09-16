"""Load Case Tree (E4) — dependency graph (`case_graph`) + tree dialog +
Analysis-cases-home "Tree…" button.
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

from project import AnalysisCase, NonlinearCase, Project  # noqa: E402


@pytest.fixture(scope="module")
def qapp():
    pytest.importorskip("PySide6")
    from PySide6.QtWidgets import QApplication
    yield QApplication.instance() or QApplication([])


def _chained_project():
    """PRELOAD (root) ← MONO (continue_from) and ← TH (initial_condition)."""
    p = Project()
    p.nonlinear_cases = [
        NonlinearCase(id=1, name="PRELOAD", control_node=2),
        NonlinearCase(id=2, name="MONO", control_node=2, continue_from=1)]
    p.analysis_cases = [
        AnalysisCase(id=1, name="TH-0.25", type="timehistory",
                     params={}, initial_condition=("state", 1))]
    return p


# ----------------------------------------------------------- case_graph

def test_case_nodes_report_parents():
    import case_graph
    nodes = {n["key"]: n for n in case_graph.case_nodes(_chained_project())}
    assert nodes[("nonlinear", 1)]["parent"] is None          # root
    assert nodes[("nonlinear", 2)]["parent"] == ("nonlinear", 1)   # continue_from
    assert nodes[("analysis", 1)]["parent"] == ("nonlinear", 1)    # state IC
    assert not any(n["dangling"] for n in nodes.values())


def test_case_forest_nests_dependents_under_source():
    import case_graph
    roots, cycles = case_graph.case_forest(_chained_project())
    assert cycles == []
    assert len(roots) == 1 and roots[0]["name"] == "PRELOAD"
    kids = {(c["kind"], c["name"]) for c in roots[0]["children"]}
    assert kids == {("nonlinear", "MONO"), ("analysis", "TH-0.25")}


def test_case_forest_flags_dangling_source():
    import case_graph
    p = Project()
    p.analysis_cases = [AnalysisCase(id=1, name="orphan", type="modal",
                                     params={},
                                     initial_condition=("state", 99))]
    roots, cycles = case_graph.case_forest(p)
    assert cycles == []
    assert len(roots) == 1 and roots[0]["dangling"] is True


def test_case_forest_breaks_cycle():
    import case_graph
    p = Project()
    p.nonlinear_cases = [
        NonlinearCase(id=1, name="A", control_node=2, continue_from=2),
        NonlinearCase(id=2, name="B", control_node=2, continue_from=1)]
    roots, cycles = case_graph.case_forest(p)     # must not recurse forever
    assert set(cycles) == {("nonlinear", 1), ("nonlinear", 2)}
    assert len(roots) == 2 and all(r["children"] == [] for r in roots)


# ----------------------------------------------------------- dialog / button

def test_case_tree_dialog_renders(qapp):
    from case_tree_dialog import CaseTreeDialog
    dlg = CaseTreeDialog(None, _chained_project())
    assert dlg.tree.topLevelItemCount() == 1                  # PRELOAD
    assert dlg.tree.topLevelItem(0).childCount() == 2         # MONO + TH


def test_manager_has_tree_button_and_opens(qapp, monkeypatch):
    import case_tree_dialog
    from analysis_cases_dialog import AnalysisCasesDialog
    opened = []
    monkeypatch.setattr(case_tree_dialog.CaseTreeDialog, "show_tree",
                        classmethod(lambda cls, parent, project:
                                    opened.append(project)))
    dlg = AnalysisCasesDialog(None, _chained_project())
    assert dlg._tree_btn.isEnabled()
    dlg._show_tree()
    assert opened and len(opened[0].analysis_cases) == 1      # proxy carries edits
