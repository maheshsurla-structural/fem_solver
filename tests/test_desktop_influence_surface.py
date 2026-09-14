"""Influence surface / multi-lane — desktop wiring (bridge GUI increment G4).

Headless coverage of the setup dialog, the Analysis-cases row, and the
``MainWindow.run_influence_surface`` runner (on a 3-D grillage deck).
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


def _grillage(nx=9, ny=5, dx=3.0, dy=1.875):
    """A 3-D grillage deck (~7.5 m wide → 2 AASHTO lanes), fixed at both ends."""
    p = Project(ndm=3, ndf=6)
    nid, tag = {}, 1
    for i in range(nx):
        for j in range(ny):
            p.nodes.append(Node(id=tag, x=i * dx, y=j * dy, z=0.0))
            nid[(i, j)] = tag
            tag += 1
    for j in range(ny):
        for k in (0, nx - 1):
            nd = next(n for n in p.nodes if n.id == nid[(k, j)])
            nd.supports = (1, 1, 1, 1, 1, 1)
    p.sections = [Section(id=1, name="g", A=0.4, Iz=0.05, Iy=0.03, J=0.02)]
    p.materials = [Material(1, "conc", E=3e10, nu=0.2, rho=2500.0)]
    eid = 1
    for j in range(ny):
        for i in range(nx - 1):
            p.members.append(Member(eid, nid[(i, j)], nid[(i + 1, j)], 1, 1))
            eid += 1
    for i in range(nx):
        for j in range(ny - 1):
            p.members.append(Member(eid, nid[(i, j)], nid[(i, j + 1)], 1, 1))
            eid += 1
    p._nid = nid
    return p


# ---------------------------------------------------------------- dialog
def test_dialog_defaults(qapp):
    from influence_surface_dialog import InfluenceSurfaceDialog
    p = _grillage()
    d = InfluenceSurfaceDialog(None, p)
    assert len(d.deck_nodes()) == len(p.nodes)         # all selected
    cfg = d.result()
    assert cfg["multi_presence"] is True and cfg["vehicle"] == "hl93_truck"


# ---------------------------------------------------- analysis-cases row
def test_analysis_cases_lists_influence_surface(qapp):
    from analysis_cases_dialog import AnalysisCasesDialog
    dlg = AnalysisCasesDialog(None, _grillage())
    kinds = [m["kind"] for m in dlg._row_meta]
    assert "influencesurface" in kinds
    dlg.table.setCurrentCell(kinds.index("influencesurface"), 0)
    dlg._run()
    assert dlg._run_request == ("influencesurface",)


# ------------------------------------------------- run_influence_surface
def test_run_influence_surface_multi_lane(qapp):
    from main_window import MainWindow
    p = _grillage()
    center = p._nid[(4, 2)]
    w = MainWindow()
    w.load_project(p)
    res = w.run_influence_surface(config={
        "deck": [nd.id for nd in p.nodes], "response": ("disp", center),
        "vehicle": "hl93_truck", "multi_presence": True})
    assert res is not None
    assert len(res["surface"].values) == len(p.nodes)
    assert res["lanes"] == 2                            # ~7.5 m → 2 lanes
    assert res["envelope"]["min"] < 0.0                # downward deflection
    assert res["envelope"]["min_num_lanes"] >= 1
    assert hasattr(w, "_influence_surface_results_dlg")


def test_run_guards(qapp, monkeypatch):
    import main_window as MW
    monkeypatch.setattr(MW.QMessageBox, "information",
                        staticmethod(lambda *a, **k: None))
    # 2-D model gated
    w = MW.MainWindow()
    p2 = Project(ndm=2, ndf=3)
    p2.nodes = [Node(1, 0, 0), Node(2, 3, 0), Node(3, 6, 0)]
    p2.sections = [Section(id=1, name="g", A=0.4, Iz=0.05)]
    p2.materials = [Material(1, "c", E=3e10, nu=0.2, rho=2500.0)]
    p2.members = [Member(1, 1, 2, 1, 1), Member(2, 2, 3, 1, 1)]
    w.load_project(p2)
    assert w.run_influence_surface(config={
        "deck": [1, 2, 3], "response": ("disp", 2),
        "vehicle": "hl93_truck", "multi_presence": True}) is None
    # too few deck nodes (3-D)
    p = _grillage()
    w.load_project(p)
    assert w.run_influence_surface(config={
        "deck": [1, 2], "response": ("disp", 1),
        "vehicle": "hl93_truck", "multi_presence": True}) is None
