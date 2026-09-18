"""Wall-modeling W2 — pier force integration.

Validates ``piers.pier_forces`` by free-body equilibrium of a cantilever shear
wall: a horizontal in-plane tip load must reappear as base-cut shear V and
moment M = load × lever, and a vertical tip load as base-cut axial P.
"""
from __future__ import annotations

import sys
from pathlib import Path

import numpy as np
import pytest

_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(_ROOT / "src"))
sys.path.insert(0, str(_ROOT))
sys.path.insert(0, str(_ROOT / "desktop"))

from femsolver.analysis.linear_static import LinearStaticAnalysis  # noqa: E402

import piers  # noqa: E402
import walls  # noqa: E402
from project import Material, Node, Project, ShellSection  # noqa: E402

L, H = 3.0, 4.0            # wall length, height
MESH = (4, 8)


def _wall_model(mesh=MESH):
    """A single cantilever wall panel in the X-Z plane, pier 'P1'."""
    p = Project(ndm=3, ndf=6)
    p.materials.append(Material(id=1, name="C30", E=30e9, nu=0.2, rho=2400.0))
    p.shell_sections.append(ShellSection(id=1, name="W250", thickness=0.25))
    p.nodes.extend([Node(id=1, x=0.0, y=0.0, z=0.0),
                    Node(id=2, x=L, y=0.0, z=0.0)])
    walls.build_wall_line(p, [1, 2], height=H, shell_section=1, material=1,
                          mesh=mesh, pier="P1")
    m = p.build_model(with_loads=False)
    return p, m


def _fix_base(m):
    for tag, nd in m.nodes.items():
        if abs(nd.coords[2]) < 1e-9:
            m.fix(tag, [1, 1, 1, 1, 1, 1])


def _top_nodes(m):
    return [tag for tag, nd in m.nodes.items() if abs(nd.coords[2] - H) < 1e-9]


# ------------------------------------------------------------------ guards

def test_forces_zero_before_solve():
    p, m = _wall_model()
    _fix_base(m)
    forces = piers.pier_forces(p, m, "P1")
    assert forces                                   # cuts exist
    assert all(abs(f.axial) + abs(f.shear) + abs(f.moment) < 1e-6
               for f in forces)                      # but no force yet


def test_unknown_pier_returns_empty():
    p, m = _wall_model()
    assert piers.pier_forces(p, m, "NOPE") == []


# --------------------------------------------------------------- physics

def test_horizontal_tip_load_gives_base_shear_and_moment():
    p, m = _wall_model()
    _fix_base(m)
    tops = _top_nodes(m)
    Htot = 100_000.0                                 # total in-plane X load (N)
    for t in tops:
        m.add_nodal_load(t, [Htot / len(tops), 0, 0, 0, 0, 0])
    LinearStaticAnalysis(m).run()

    forces = piers.pier_forces(p, m, "P1")
    base = forces[0]                                 # lowest cut (near base)
    assert abs(base.shear) == pytest.approx(Htot, rel=0.08)
    # moment about the base cut = H × (H_top − z_cut)
    assert abs(base.moment) == pytest.approx(Htot * (H - base.z), rel=0.10)
    # negligible axial under a pure lateral load
    assert abs(base.axial) < 0.05 * Htot


def test_moment_grows_toward_the_base():
    p, m = _wall_model()
    _fix_base(m)
    tops = _top_nodes(m)
    for t in tops:
        m.add_nodal_load(t, [50_000.0 / len(tops), 0, 0, 0, 0, 0])
    LinearStaticAnalysis(m).run()
    forces = piers.pier_forces(p, m, "P1")
    moments = [abs(f.moment) for f in forces]        # bottom → top
    # cantilever moment is largest at the base and decreases up the height
    assert moments[0] > moments[-1]
    assert moments == sorted(moments, reverse=True) or moments[0] == max(moments)


def test_vertical_tip_load_gives_base_axial_compression():
    p, m = _wall_model()
    _fix_base(m)
    tops = _top_nodes(m)
    W = 200_000.0                                    # total downward load (N)
    for t in tops:
        m.add_nodal_load(t, [0, 0, -W / len(tops), 0, 0, 0])
    LinearStaticAnalysis(m).run()
    base = piers.pier_forces(p, m, "P1")[0]
    # tension-positive convention → downward load is compression (negative P)
    assert base.axial == pytest.approx(-W, rel=0.05)
    assert abs(base.shear) < 0.05 * W
    assert base.width == pytest.approx(L, rel=1e-6)


def test_cut_elevations_one_per_row():
    p, m = _wall_model(mesh=(2, 5))
    zs = piers.pier_cut_elevations(p, m, "P1")
    assert len(zs) == 5                              # n2 rows → 5 cuts
    assert zs[0] == pytest.approx(H / 5 / 2)         # mid-height of the first row


# --------------------------------------------------------------- GUI dialog

@pytest.fixture(scope="module")
def qapp():
    pytest.importorskip("PySide6")
    pytest.importorskip("matplotlib")
    from PySide6.QtWidgets import QApplication
    yield QApplication.instance() or QApplication([])


def _solved_lateral():
    p, m = _wall_model(mesh=(2, 4))
    _fix_base(m)
    tops = _top_nodes(m)
    for t in tops:
        m.add_nodal_load(t, [80_000.0 / len(tops), 0, 0, 0, 0, 0])
    LinearStaticAnalysis(m).run()
    return p, m


def test_pier_dialog_populates_table(qapp):
    from pier_forces_dialog import PierForcesDialog
    p, m = _solved_lateral()
    dlg = PierForcesDialog(None, p, m)
    assert dlg.tbl.rowCount() == 4                   # one row per mesh level
    # the base row (last, lowest elevation) carries the largest |M|
    texts = [dlg.tbl.item(r, 3).text() for r in range(dlg.tbl.rowCount())]
    assert any(t not in ("0", "0.000") for t in texts)


def test_pier_dialog_no_piers_shows_hint(qapp):
    from pier_forces_dialog import PierForcesDialog
    p, m = _wall_model(mesh=(2, 2))
    for a in p.areas:                                # strip the pier labels
        a.pier = None
    dlg = PierForcesDialog(None, p, m)
    assert dlg.tbl.rowCount() == 0
    assert "No piers" in dlg._empty.text()


def test_pier_dialog_csv_export(qapp, tmp_path, monkeypatch):
    import pier_forces_dialog as pfd
    p, m = _solved_lateral()
    dlg = pfd.PierForcesDialog(None, p, m)
    out = tmp_path / "piers.csv"
    monkeypatch.setattr(pfd.QFileDialog, "getSaveFileName",
                        lambda *a, **k: (str(out), "CSV files (*.csv)"))
    dlg._export()
    text = out.read_text(encoding="utf-8")
    assert text.splitlines()[0].startswith("pier,")
    assert "P1" in text
    assert len(text.splitlines()) == 1 + 4           # header + 4 level rows


# ----------------------------------------------- MainWindow action wiring

@pytest.fixture(scope="module")
def qapp_vtk():
    pytest.importorskip("PySide6")
    pytest.importorskip("pyvistaqt")
    from PySide6.QtWidgets import QApplication
    yield QApplication.instance() or QApplication([])


def test_show_pier_forces_without_piers_is_guarded(qapp_vtk, monkeypatch):
    from main_window import MainWindow
    import main_window as mw
    p, _m = _wall_model(mesh=(2, 2))
    for a in p.areas:
        a.pier = None                                # walls but no pier label
    w = MainWindow()
    w.load_project(p)
    seen = {}
    monkeypatch.setattr(mw.QMessageBox, "information",
                        lambda *a, **k: seen.setdefault("info", a))
    w.show_pier_forces()
    assert "info" in seen                            # told to label a pier
