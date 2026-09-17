"""Unified Load Case Data editor (E1) — `case_editor.CaseEditorDialog` + the
Modal/Buckling adapters routing through it.
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
    return p


def test_case_editor_builds_modal_case(qapp):
    from case_editor import CaseEditorDialog
    dlg = CaseEditorDialog(None, _project(), "modal")
    dlg._bodies["modal"].modes.setValue(5)     # within the ndf*nodes cap
    case = dlg._result_case(0)
    assert case.type == "modal"
    assert case.params == {"num_modes": 5, "lumped": False}
    assert case.initial_condition == ("zero",)


def test_case_editor_switches_type_to_buckling(qapp):
    from case_editor import CaseEditorDialog
    dlg = CaseEditorDialog(None, _project(), "modal")
    dlg.type_combo.setCurrentIndex(dlg.type_combo.findData("buckling"))
    case = dlg._result_case(0)
    assert case.type == "buckling"
    assert case.params["selection"] == ["all", None]
    assert "subdivisions" in case.params


def test_case_editor_seeds_from_existing_case(qapp):
    from case_editor import CaseEditorDialog
    c = AnalysisCase(id=5, name="MyModal", type="modal",
                     params={"num_modes": 5, "lumped": True})
    dlg = CaseEditorDialog(None, _project(), "modal", c)
    assert dlg._name.text() == "MyModal"
    assert dlg._bodies["modal"].modes.value() == 5
    out = dlg._result_case(5)
    assert out.id == 5 and out.params == {"num_modes": 5, "lumped": True}


def test_case_editor_carries_initial_condition(qapp):
    from case_editor import CaseEditorDialog
    p = _project()
    p.nonlinear_cases = [NonlinearCase(id=1, name="PRELOAD", control_node=2)]
    c = AnalysisCase(id=1, name="m", type="modal",
                     params={"num_modes": 4, "lumped": False},
                     initial_condition=("state", 1))
    dlg = CaseEditorDialog(None, p, "modal", c)
    assert dlg.initial.value() == ("state", 1)
    assert dlg._result_case(1).initial_condition == ("state", 1)


def test_modal_and_buckling_adapters_route_to_editor(qapp, monkeypatch):
    import case_editor
    import case_types
    seen = []
    monkeypatch.setattr(
        case_editor.CaseEditorDialog, "edit",
        classmethod(lambda cls, parent, project, type_id, case=None:
                    seen.append(type_id)))
    case_types.get("modal").edit(None, _project())
    case_types.get("buckling").edit(None, _project())
    assert seen == ["modal", "buckling"]


def test_case_editor_response_spectrum_body(qapp):
    import case_types
    from case_editor import CaseEditorDialog
    from femsolver import ResponseSpectrum
    dlg = CaseEditorDialog(None, _project(), "responsespectrum")
    body = dlg._bodies["responsespectrum"]
    body.modes.setValue(4)
    body.source.setCurrentIndex(body.source.findData("asce7"))
    case = dlg._result_case(0)
    assert case.type == "responsespectrum"
    assert case.params["source"] == "asce7" and case.params["num_modes"] == 4
    # build_config rebuilds the derived spectrum from the body's params
    cfg = case_types.get("responsespectrum").build_config(
        _project(), case.params, initial_condition=("zero",))
    assert isinstance(cfg[0], ResponseSpectrum)


def test_rs_adapter_routes_to_editor(qapp, monkeypatch):
    import case_editor
    import case_types
    seen = []
    monkeypatch.setattr(
        case_editor.CaseEditorDialog, "edit",
        classmethod(lambda cls, parent, project, type_id, case=None:
                    seen.append(type_id)))
    case_types.get("responsespectrum").edit(None, _project())
    assert seen == ["responsespectrum"]
