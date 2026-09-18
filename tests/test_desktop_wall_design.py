"""Wall-modeling W3 — the desktop wall-design dialog + pier geometry helper.

Covers piers.pier_geometry and the WallDesignDialog wiring (populates the
per-cut table + summary from the solved model), plus the MainWindow guard.
"""
from __future__ import annotations

import os
import sys
from pathlib import Path

import pytest

os.environ.setdefault("QT_QPA_PLATFORM", "offscreen")
_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(_ROOT / "src"))
sys.path.insert(0, str(_ROOT))
sys.path.insert(0, str(_ROOT / "desktop"))

from femsolver.analysis.linear_static import LinearStaticAnalysis  # noqa: E402

import piers  # noqa: E402
import walls  # noqa: E402
from project import Material, Node, Project, ShellSection  # noqa: E402

L, H = 3.0, 4.0


def _wall_model(mesh=(3, 4)):
    p = Project(ndm=3, ndf=6)
    p.materials.append(Material(id=1, name="C30", E=30e9, nu=0.2, rho=2400.0,
                               params={"fc": 30e6}))
    p.shell_sections.append(ShellSection(id=1, name="W250", thickness=0.25))
    p.nodes.extend([Node(id=1, x=0.0, y=0.0, z=0.0),
                    Node(id=2, x=L, y=0.0, z=0.0)])
    walls.build_wall_line(p, [1, 2], height=H, shell_section=1, material=1,
                          mesh=mesh, pier="P1")
    m = p.build_model(with_loads=False)
    return p, m


def _solved(mesh=(3, 4)):
    p, m = _wall_model(mesh)
    for tag, nd in m.nodes.items():
        if abs(nd.coords[2]) < 1e-9:
            m.fix(tag, [1, 1, 1, 1, 1, 1])
    tops = [t for t, nd in m.nodes.items() if abs(nd.coords[2] - H) < 1e-9]
    for t in tops:
        m.add_nodal_load(t, [60_000.0 / len(tops), 0, 0,
                             -400_000.0 / len(tops), 0, 0])
    LinearStaticAnalysis(m).run()
    return p, m


# --------------------------------------------------------- geometry helper

def test_pier_geometry():
    p, m = _wall_model(mesh=(3, 4))
    lw, t, hw = piers.pier_geometry(p, m, "P1")
    assert lw == pytest.approx(L, rel=1e-6)
    assert t == pytest.approx(0.25, rel=1e-6)
    assert hw == pytest.approx(H, rel=1e-6)
    assert piers.pier_geometry(p, m, "NOPE") is None


# ------------------------------------------------------------- dialog

@pytest.fixture(scope="module")
def qapp():
    pytest.importorskip("PySide6")
    from PySide6.QtWidgets import QApplication
    yield QApplication.instance() or QApplication([])


def test_wall_design_dialog_populates(qapp):
    from wall_design_dialog import WallDesignDialog
    p, m = _solved()
    dlg = WallDesignDialog(None, p, m)
    assert dlg.tbl.rowCount() == 4                 # one row per mesh level
    # f'c prefilled from the material params (30e6 Pa → 30 MPa)
    assert dlg.fc.value() == pytest.approx(30.0)
    # summary carries a governing DCR + verdict
    txt = dlg.summary.text()
    assert "governing DCR" in txt
    assert "PASS" in txt or "FAIL" in txt


def test_wall_design_dialog_flags_overload(qapp):
    from wall_design_dialog import WallDesignDialog
    p, m = _solved()
    dlg = WallDesignDialog(None, p, m)
    # starve the wall of reinforcement and shrink boundary bars → must FAIL
    dlg.rho_t.setValue(0.0)
    dlg.rho_l.setValue(0.0)
    dlg.as_be.setValue(0.0)
    dlg._compute()
    assert "FAIL" in dlg.summary.text()


def test_wall_design_dialog_no_piers(qapp):
    from wall_design_dialog import WallDesignDialog
    p, m = _wall_model(mesh=(2, 2))
    for a in p.areas:
        a.pier = None
    dlg = WallDesignDialog(None, p, m)
    assert dlg.tbl.rowCount() == 0
    assert "No piers" in dlg.summary.text()


# ------------------------------------------------- MainWindow action guard

@pytest.fixture(scope="module")
def qapp_vtk():
    pytest.importorskip("PySide6")
    pytest.importorskip("pyvistaqt")
    from PySide6.QtWidgets import QApplication
    yield QApplication.instance() or QApplication([])


def test_show_wall_design_guarded_without_piers(qapp_vtk, monkeypatch):
    from main_window import MainWindow
    import main_window as mw
    p, _m = _wall_model(mesh=(2, 2))
    for a in p.areas:
        a.pier = None
    w = MainWindow()
    w.load_project(p)
    seen = {}
    monkeypatch.setattr(mw.QMessageBox, "information",
                        lambda *a, **k: seen.setdefault("info", a))
    w.show_wall_design()
    assert "info" in seen
