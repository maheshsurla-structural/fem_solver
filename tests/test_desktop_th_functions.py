"""Time-history function library — ``desktop/th_functions.py`` +
``project.TimeHistoryFunction`` (analysis-cases-manager plan, TH-1).

Headless coverage of the entity + serialization, the manager (add / delete,
in-use guard), the editor's data / short-record guard, and the main-window
wiring.
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

from project import (AnalysisCase, Material, Member,  # noqa: E402
                     Node, Project, Section, TimeHistoryFunction)


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
    return p


def test_function_properties():
    f = TimeHistoryFunction(id=1, name="EC", dt=0.02, values=[0.1, 0.2, 0.3])
    assert f.npts == 3 and abs(f.duration - 0.04) < 1e-9


def test_th_function_serialization_round_trip():
    p = Project(ndm=2, ndf=3)
    p.th_functions = [TimeHistoryFunction(id=1, name="EC", dt=0.02,
                                          values=[0.1, -0.2, 0.3], in_g=True,
                                          source="elcentro.txt")]
    back = Project.from_json(p.to_json())
    f = back.th_functions[0]
    assert f.id == 1 and f.name == "EC" and f.dt == 0.02 and f.in_g is True
    assert f.values == [0.1, -0.2, 0.3] and f.source == "elcentro.txt"
    assert f.npts == 3 and abs(f.duration - 0.04) < 1e-9


def test_editor_data_round_trips(qapp):
    from th_functions import TimeHistoryFunctionDialog
    src = TimeHistoryFunction(id=7, name="Kobe", dt=0.005,
                              values=[0.0, 0.5, -0.5, 0.2], in_g=True,
                              source="kobe.txt")
    dlg = TimeHistoryFunctionDialog(None, _project(), src)
    out = dlg.data()
    assert out.id == 7 and out.name == "Kobe" and out.dt == 0.005
    assert out.in_g is True and out.values == [0.0, 0.5, -0.5, 0.2]


def test_editor_refuses_short_record(qapp, monkeypatch):
    import th_functions as thm
    monkeypatch.setattr(thm.QMessageBox, "warning",
                        staticmethod(lambda *a, **k: None))
    dlg = thm.TimeHistoryFunctionDialog(None, _project())   # no record imported
    dlg.accept()
    assert dlg.result() == 0                                 # not accepted


def test_manager_add_assigns_id_and_delete(qapp, monkeypatch):
    import th_functions as thm
    dlg = thm.TimeHistoryFunctionManagerDialog(None, _project())
    monkeypatch.setattr(
        thm.TimeHistoryFunctionDialog, "edit",
        staticmethod(lambda parent, project, func=None:
                     TimeHistoryFunction(id=0, name="EC", dt=0.02,
                                         values=[0.1, 0.2, 0.3], in_g=True)))
    dlg._add()
    assert len(dlg._funcs) == 1
    assert dlg._funcs[0].id == 1 and dlg._funcs[0].npts == 3
    dlg.table.setCurrentCell(0, 0)
    dlg._delete()
    assert dlg._funcs == []


def test_manager_delete_blocked_when_referenced(qapp, monkeypatch):
    import th_functions as thm
    monkeypatch.setattr(thm.QMessageBox, "warning",
                        staticmethod(lambda *a, **k: None))
    p = _project()
    p.th_functions = [TimeHistoryFunction(id=1, name="EC", values=[0.1, 0.2])]
    p.analysis_cases = [AnalysisCase(id=1, name="TH", type="timehistory",
                                     params={"function_id": 1})]
    dlg = thm.TimeHistoryFunctionManagerDialog(None, p)
    dlg.table.setCurrentCell(0, 0)
    dlg._delete()
    assert len(dlg._funcs) == 1                              # refused (in use)


def test_main_window_wires_th_functions(qapp):
    from main_window import MainWindow
    w = MainWindow()
    w.load_project(_project())
    assert hasattr(w, "act_th_functions")
    assert callable(w.manage_th_functions)


# --------------------------------------------- TH-2: Time History as a saved case
def _project_with_fn():
    p = _project()
    p.th_functions = [TimeHistoryFunction(id=1, name="EC", dt=0.02,
                                          values=[0.1, -0.2, 0.3, 0.0],
                                          in_g=True)]
    return p


def test_timehistory_case_dialog_params_round_trip(qapp):
    from timehistory_dialog import TimeHistoryCaseDialog
    initial = {"function_id": 1, "control_node": 2, "direction": "y",
               "scale": 1.5, "zeta": 0.03, "density": 2500.0}
    dlg = TimeHistoryCaseDialog(None, _project_with_fn(), initial=initial,
                                name="TH-Y")
    assert dlg.params() == initial
    assert dlg.header.name() == "TH-Y"


def test_timehistory_case_dialog_accept_requires_function(qapp, monkeypatch):
    import timehistory_dialog as thd
    monkeypatch.setattr(thd.QMessageBox, "warning",
                        staticmethod(lambda *a, **k: None))
    dlg = thd.TimeHistoryCaseDialog(None, _project())       # no functions
    dlg.accept()
    assert dlg.result() == 0                                 # refused


def test_timehistory_adapter_build_config_and_detail(qapp):
    import case_types
    p = _project_with_fn()
    ct = case_types.get("timehistory")
    assert ct is not None and ct.type_label == "Time History"
    params = {"function_id": 1, "control_node": 2, "direction": "y",
              "scale": 2.0, "zeta": 0.05, "density": 2400.0}
    cfg = ct.build_config(p, params)
    assert cfg["values"] == [0.1, -0.2, 0.3, 0.0] and cfg["dt"] == 0.02
    assert cfg["in_g"] is True and cfg["scale"] == 2.0 and cfg["name"] == "EC"
    assert "EC" in ct.detail(p, params)


def test_timehistory_build_config_missing_function_raises(qapp):
    import case_types
    ct = case_types.get("timehistory")
    with pytest.raises(ValueError):
        ct.build_config(_project(), {"function_id": 999})


def test_timehistory_runner_apply_seed(qapp):
    from timehistory_dialog import TimeHistoryDialog
    seed = {"values": [0.1, 0.2, 0.3], "dt": 0.02, "in_g": True, "name": "EC",
            "control_node": 2, "direction": "y", "scale": 1.0, "zeta": 0.05,
            "density": 2400.0}
    dlg = TimeHistoryDialog(None, _project(), seed=seed)
    assert dlg._accel is not None and dlg._accel.size == 3
    assert dlg.run_btn.isEnabled()
    assert dlg.dt.value() == 0.02 and dlg.in_g.currentIndex() == 1
