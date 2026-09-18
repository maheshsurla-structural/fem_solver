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


def test_case_editor_temperature_gradient(qapp):
    from case_editor import CaseEditorDialog
    dlg = CaseEditorDialog(None, _project(), "tempgradient")
    body = dlg._bodies["tempgradient"]
    body.source.setCurrentIndex(body.source.findData("aashto"))
    case = dlg._result_case(0)
    assert case.type == "tempgradient"
    assert case.params["source"] == "aashto"
    assert case.params["alpha"] == 1e-5          # stored SI
    assert case.params["members"] == [1]         # default: all members
    assert case.initial_condition == ("zero",)   # non-IC type


def test_case_editor_cable_tuning(qapp):
    from case_editor import CaseEditorDialog
    dlg = CaseEditorDialog(None, _project(), "cabletuning")
    body = dlg._bodies["cabletuning"]
    body.cables.item(0).setSelected(True)        # member 1
    body.targets.item(1).setSelected(True)       # node 2
    case = dlg._result_case(0)
    assert case.type == "cabletuning"
    assert case.params == {"cables": [1], "targets": [2]}


def test_case_editor_hides_ic_for_non_ic_types(qapp):
    from case_editor import CaseEditorDialog
    dlg = CaseEditorDialog(None, _project(), "modal")
    assert not dlg.initial.isHidden()            # modal continues from a state
    dlg.type_combo.setCurrentIndex(dlg.type_combo.findData("tempgradient"))
    assert dlg.initial.isHidden()                # temp gradient does not


def test_batch2_adapters_route_to_editor(qapp, monkeypatch):
    import case_editor
    import case_types
    seen = []
    monkeypatch.setattr(
        case_editor.CaseEditorDialog, "edit",
        classmethod(lambda cls, parent, project, type_id, case=None:
                    seen.append(type_id)))
    case_types.get("tempgradient").edit(None, _project())
    case_types.get("cabletuning").edit(None, _project())
    assert seen == ["tempgradient", "cabletuning"]


def test_case_editor_moving_load(qapp):
    from case_editor import CaseEditorDialog
    dlg = CaseEditorDialog(None, _project(), "movingload")
    case = dlg._result_case(0)
    assert case.type == "movingload"
    assert set(case.params["lane"]) == {1, 2}       # default: whole model
    assert case.params["vehicle"] == "hl93"
    assert case.params["response"][0] == "M"        # default bending @ member
    assert case.initial_condition == ("zero",)      # non-IC type


def test_case_editor_influence_surface(qapp):
    from case_editor import CaseEditorDialog
    dlg = CaseEditorDialog(None, _project(), "influencesurface")
    case = dlg._result_case(0)
    assert case.type == "influencesurface"
    assert set(case.params["deck"]) == {1, 2}
    assert case.params["response"][0] == "disp"
    assert case.params["multi_presence"] is True


def test_batch3_adapters_route_to_editor(qapp, monkeypatch):
    import case_editor
    import case_types
    seen = []
    monkeypatch.setattr(
        case_editor.CaseEditorDialog, "edit",
        classmethod(lambda cls, parent, project, type_id, case=None:
                    seen.append(type_id)))
    case_types.get("movingload").edit(None, _project())
    case_types.get("influencesurface").edit(None, _project())
    assert seen == ["movingload", "influencesurface"]


def test_case_editor_vehicle_dynamics_roundtrip(qapp):
    from case_editor import CaseEditorDialog
    params = {"lane": [1, 2], "kind": "vbi", "mass": 25000.0, "bounce": 2.5,
              "susp_damp": 0.15, "speed": 20.0, "zeta": 0.03, "node": 2}
    c = AnalysisCase(id=1, name="vd", type="vehicledynamics", params=params)
    dlg = CaseEditorDialog(None, _project(), "vehicledynamics", c)
    out = dlg._result_case(1)
    assert out.type == "vehicledynamics" and out.params["kind"] == "vbi"
    for k in ("mass", "bounce", "susp_damp", "speed", "zeta"):
        assert out.params[k] == pytest.approx(params[k])   # display↔SI round-trip
    assert out.params["node"] == 2 and out.initial_condition == ("zero",)


def test_case_editor_load_rating_roundtrip(qapp):
    from case_editor import CaseEditorDialog
    params = {"lane": [1, 2], "response": ["M", 1, "i"], "Rn": 1000.0, "DC": 200.0,
              "DW": 50.0, "P": 0.0, "phi": 0.9, "phi_c": 0.95, "phi_s": 1.0,
              "im": 0.33, "adtt": 5000, "permit_gamma_LL": None}
    c = AnalysisCase(id=1, name="lr", type="loadrating", params=params)
    dlg = CaseEditorDialog(None, _project(), "loadrating", c)
    out = dlg._result_case(1)
    assert out.params["response"][0] == "M"
    for k in ("Rn", "DC", "DW", "phi", "im"):
        assert out.params[k] == pytest.approx(params[k])
    assert out.params["phi_c"] == 0.95 and out.params["adtt"] == 5000
    assert out.params["permit_gamma_LL"] is None


def test_case_editor_time_history_with_ic_and_hold(qapp):
    from case_editor import CaseEditorDialog
    from project import TimeHistoryFunction
    p = _project()
    p.th_functions = [TimeHistoryFunction(id=1, name="EC", dt=0.02,
                                          values=[0.1, -0.2, 0.3])]
    p.nonlinear_cases = [NonlinearCase(id=5, name="PRELOAD", control_node=2)]
    c = AnalysisCase(id=1, name="th", type="timehistory",
                     params={"function_id": 1, "control_node": 2,
                             "direction": "y", "scale": 1.5, "zeta": 0.03,
                             "density": 2500.0, "hold_source_loads": True},
                     initial_condition=("state", 5))
    dlg = CaseEditorDialog(None, p, "timehistory", c)
    out = dlg._result_case(1)
    assert out.type == "timehistory"
    assert out.params["function_id"] == 1
    assert out.params["scale"] == pytest.approx(1.5)
    assert out.params["hold_source_loads"] is True     # injected by the shell
    assert out.initial_condition == ("state", 5)


def test_all_types_migrated_to_editor(qapp):
    import case_bodies
    import case_types
    reg_ids = {tid for tid, _, _ in case_bodies.REGISTRY}
    assert reg_ids == set(case_types.TYPES)            # all 10 types unified
