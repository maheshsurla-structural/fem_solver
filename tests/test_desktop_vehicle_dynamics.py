"""Vehicle dynamics / moving-load time-history — desktop wiring (GUI G5).

Headless coverage of the setup dialog, the Analysis-cases row, and the
``MainWindow.run_vehicle_dynamics`` runner (moving-force + sprung-mass VBI).
"""
from __future__ import annotations

import os
import sys
from pathlib import Path

import pytest

os.environ.setdefault("QT_QPA_PLATFORM", "offscreen")
_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(_ROOT))
sys.path.insert(0, str(_ROOT / "src"))
sys.path.insert(0, str(_ROOT / "desktop"))

from project import Material, Member, Node, Project, Section  # noqa: E402


@pytest.fixture(scope="module")
def qapp():
    pytest.importorskip("PySide6")
    from PySide6.QtWidgets import QApplication
    yield QApplication.instance() or QApplication([])


def _girder(n=8, dx=3.0, rho=2500.0):
    p = Project(ndm=2, ndf=3)
    p.nodes = [Node(i + 1, i * dx, 0.0) for i in range(n + 1)]
    p.nodes[0].supports = (1, 1, 0)
    p.nodes[-1].supports = (0, 1, 0)
    p.sections = [Section(id=1, name="g", A=0.5, Iz=0.05)]
    p.materials = [Material(1, "conc", E=3e10, nu=0.2, rho=rho)]
    p.members = [Member(i + 1, i + 1, i + 2, 1, 1) for i in range(n)]
    return p


def _cfg(p, kind="force", **over):
    cfg = {"lane": [nd.id for nd in p.nodes], "kind": kind,
           "vehicle": "hl93_truck", "mass": 20000.0, "bounce": 2.0,
           "susp_damp": 0.1, "speed": 60 / 3.6, "zeta": 0.02,
           "node": len(p.nodes) // 2 + 1}
    cfg.update(over)
    return cfg


# ---------------------------------------------------------------- dialog
def test_dialog_defaults_and_conversions(qapp):
    from vehicle_dynamics_dialog import VehicleDynamicsDialog
    d = VehicleDynamicsDialog(None, _girder())
    assert len(d.lane_nodes()) == 9                    # all nodes
    r = d.result()
    assert r["speed"] == pytest.approx(60 / 3.6)       # km/h → m/s
    assert r["zeta"] == pytest.approx(0.02)            # % → fraction
    assert r["mass"] == pytest.approx(20000.0)         # t → kg


def test_dialog_kind_switch(qapp):
    from vehicle_dynamics_dialog import VehicleDynamicsDialog
    d = VehicleDynamicsDialog(None, _girder())
    d.kind.setCurrentIndex(1)
    assert d.stack.currentIndex() == 1 and d.result()["kind"] == "vbi"


# ---------------------------------------------------- analysis-cases row
def test_analysis_cases_lists_vehicle_dynamics(qapp):
    from analysis_cases_dialog import AnalysisCasesDialog
    dlg = AnalysisCasesDialog(None, _girder())
    kinds = [m["kind"] for m in dlg._row_meta]
    assert "vehicledynamics" in kinds
    dlg.table.setCurrentCell(kinds.index("vehicledynamics"), 0)
    dlg._run()
    assert dlg._run_request == ("vehicledynamics",)


# ------------------------------------------------- run_vehicle_dynamics
def test_run_moving_force(qapp):
    from main_window import MainWindow
    w = MainWindow()
    p = _girder()
    w.load_project(p)
    res = w.run_vehicle_dynamics(config=_cfg(p, "force"))
    assert res is not None
    assert res["DAF"] > 1.0                            # dynamic amplification
    assert len(res["times"]) == len(res["dynamic_disp"])
    assert hasattr(w, "_vehicle_dyn_results_dlg")


def test_run_vbi_has_contact_force(qapp):
    from main_window import MainWindow
    w = MainWindow()
    p = _girder()
    w.load_project(p)
    res = w.run_vehicle_dynamics(config=_cfg(p, "vbi"))
    assert res is not None and res["DAF"] > 0.0
    cf = res["contact_force"]
    assert len(cf) == len(res["times"])                # per-step contact force


def test_run_guards(qapp, monkeypatch):
    import main_window as MW
    monkeypatch.setattr(MW.QMessageBox, "information",
                        staticmethod(lambda *a, **k: None))
    w = MW.MainWindow()
    p = _girder()
    w.load_project(p)
    # too few lane nodes
    assert w.run_vehicle_dynamics(config=_cfg(p, "force", lane=[1])) is None
    # zero mass
    p0 = _girder(rho=0.0)
    w.load_project(p0)
    assert w.run_vehicle_dynamics(config=_cfg(p0, "force")) is None
    # 3-D gated
    p3 = Project(ndm=3, ndf=6)
    p3.nodes = [Node(1, 0, 0, z=0), Node(2, 3, 0, z=0)]
    p3.sections = [Section(id=1, name="g", A=0.5, Iz=0.05, Iy=0.05, J=1e-3)]
    p3.materials = [Material(1, "c", E=3e10, nu=0.2, rho=2500.0)]
    p3.members = [Member(1, 1, 2, 1, 1)]
    w.load_project(p3)
    assert w.run_vehicle_dynamics(config=_cfg(p3, "force", lane=[1, 2])) is None
