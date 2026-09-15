"""Moving-load / influence-line — desktop wiring (first bridge GUI increment).

Headless coverage of the setup dialog, the Analysis-cases row, and the
``MainWindow.run_moving_load`` runner.
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


def _girder(nspan=8, dx=3.0):
    p = Project(ndm=2, ndf=3)
    p.nodes = [Node(i + 1, i * dx, 0.0) for i in range(nspan + 1)]
    p.nodes[0].supports = (1, 1, 0)
    p.nodes[-1].supports = (0, 1, 0)
    p.sections = [Section(id=1, name="g", A=0.5, Iz=0.05)]
    p.materials = [Material(1, "steel", E=2e11, nu=0.3, rho=7850.0)]
    p.members = [Member(i + 1, i + 1, i + 2, 1, 1) for i in range(nspan)]
    return p


# --------------------------------------------------------------- dialog
def test_dialog_defaults(qapp):
    from moving_load_dialog import MovingLoadDialog
    d = MovingLoadDialog(None, _girder())
    assert d.lane_nodes() == [1, 2, 3, 4, 5, 6, 7, 8, 9]   # all, X-ordered
    assert d.vehicle.count() == 5
    cfg = d.result()
    assert cfg["lane"] and "response" in cfg and "vehicle" in cfg


def test_dialog_response_target_sync(qapp):
    from moving_load_dialog import MovingLoadDialog
    d = MovingLoadDialog(None, _girder())
    # moment → member/end enabled, node disabled
    d.resp.setCurrentIndex(0)
    assert d.member.isEnabled() and not d.node.isEnabled()
    # displacement → node enabled, member disabled
    d.resp.setCurrentIndex(2)
    assert d.node.isEnabled() and not d.member.isEnabled()


# ---------------------------------------------------- analysis-cases row
def test_analysis_cases_lists_moving_load(qapp):
    from analysis_cases_dialog import AnalysisCasesDialog, _PLANNED
    assert _PLANNED == []                              # every row now live
    dlg = AnalysisCasesDialog(None, _girder())
    kinds = [m["kind"] for m in dlg._row_meta]
    assert "movingload" in kinds
    dlg.table.setCurrentCell(kinds.index("movingload"), 0)
    assert dlg._run_btn.isEnabled()
    dlg._run()
    assert dlg._run_request == ("movingload",)


# --------------------------------------------------------- run_moving_load
def test_run_moving_load_moment_envelope(qapp):
    from main_window import MainWindow
    w = MainWindow()
    p = _girder()
    w.load_project(p)
    lane = [nd.id for nd in p.nodes]
    res = w.run_moving_load(config={"lane": lane, "vehicle": "hl93",
                                    "response": ("M", 4, "j")})
    assert res is not None
    assert len(res["il"].values) == len(lane)
    assert res["env"]["max"] > 0.0                    # sagging moment envelope
    assert hasattr(w, "_moving_load_results_dlg")
    assert "Moving load" in w.statusBar().currentMessage()


def test_run_moving_load_displacement(qapp):
    from main_window import MainWindow
    w = MainWindow()
    p = _girder()
    w.load_project(p)
    lane = [nd.id for nd in p.nodes]
    res = w.run_moving_load(config={"lane": lane, "vehicle": "hl93_truck",
                                    "response": ("disp", 5, None)})
    assert res is not None and res["env"]["min"] < 0.0   # downward deflection


def test_run_moving_load_guards(qapp, monkeypatch):
    import main_window as MW
    monkeypatch.setattr(MW.QMessageBox, "information",
                        staticmethod(lambda *a, **k: None))
    w = MW.MainWindow()
    w.load_project(_girder())
    # too few lane nodes
    assert w.run_moving_load(config={"lane": [1], "vehicle": "hl93",
                                     "response": ("M", 4, "j")}) is None


def test_run_moving_load_3d(qapp):
    """A3.2: 3-D girder moving load — member moment envelope works via the
    vertical (uz) DOF; a displacement response uses uz too."""
    from main_window import MainWindow
    n, L = 8, 24.0
    p = Project(ndm=3, ndf=6)
    p.nodes = [Node(i + 1, i * L / n, 0.0, z=0.0) for i in range(n + 1)]
    p.nodes[0].supports = (1, 1, 1, 1, 0, 0)
    p.nodes[-1].supports = (0, 1, 1, 1, 0, 0)
    p.sections = [Section(id=1, name="g", A=0.5, Iz=0.05, Iy=0.05, J=0.02)]
    p.materials = [Material(1, "steel", E=2e11, nu=0.3, rho=7850.0)]
    p.members = [Member(i + 1, i + 1, i + 2, 1, 1) for i in range(n)]
    w = MainWindow()
    w.load_project(p)
    lane = [nd.id for nd in p.nodes]
    res = w.run_moving_load(config={"lane": lane, "vehicle": "hl93",
                                    "response": ("M", n // 2, "j")})
    assert res is not None and res["env"]["max"] > 0.0
    resd = w.run_moving_load(config={"lane": lane, "vehicle": "hl93_truck",
                                     "response": ("disp", n // 2 + 1, None)})
    assert resd["env"]["min"] < 0.0
