"""Wall stream (optional) — macro (fibre) wall bridge to wall_section_2d."""
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

import macro_wall  # noqa: E402
import walls  # noqa: E402
from project import Material, Node, Project, ShellSection  # noqa: E402


def _wall_project():
    p = Project(ndm=3, ndf=6)
    p.materials.append(Material(id=1, name="C40", E=30e9, nu=0.2, rho=2400.0,
                               params={"fc": 40e6}))
    p.shell_sections.append(ShellSection(id=1, name="W300", thickness=0.30))
    p.nodes.extend([Node(id=1, x=0.0, y=0.0, z=0.0),
                    Node(id=2, x=4.0, y=0.0, z=0.0)])
    aid = walls.build_wall_line(p, [1, 2], height=3.0, shell_section=1,
                                material=1, mesh=(2, 2), pier="P1")[0]
    return p, aid


# ------------------------------------------------------------- builder

def test_build_macro_wall_section_gross_props():
    sec = macro_wall.build_macro_wall_section(
        lw=4.0, t=0.30, fc_web=30e6, fc_boundary=39e6, fy=420e6)
    props = macro_wall.macro_wall_properties(sec)
    # gross area ≈ ℓw·t (fibre discretization → within a few %)
    assert props["area"] == pytest.approx(4.0 * 0.30, rel=0.05)
    # gross Iz ≈ t·ℓw³/12
    assert props["Iz"] == pytest.approx(0.30 * 4.0 ** 3 / 12.0, rel=0.05)
    # symmetric section → centroid at mid-length
    assert abs(props["centroid_y"]) < 1e-6


def test_macro_wall_from_area_uses_geometry_and_fc():
    p, aid = _wall_project()
    sec, geom = macro_wall.macro_wall_from_area(p, p.area(aid))
    assert geom["lw"] == pytest.approx(4.0)
    assert geom["t"] == pytest.approx(0.30)
    assert geom["fc_web"] == pytest.approx(40e6)      # from material params
    assert geom["fc_boundary"] == pytest.approx(1.3 * 40e6)
    assert macro_wall.macro_wall_properties(sec)["area"] > 0


def test_thin_boundary_clamped_when_wall_short():
    # lbe default 0.15·ℓw is fine; a too-large lbe is clamped so 2·lbe < ℓw
    sec = macro_wall.build_macro_wall_section(
        lw=2.0, t=0.2, fc_web=30e6, fc_boundary=39e6, fy=420e6, lbe=1.5)
    assert macro_wall.macro_wall_properties(sec)["area"] > 0


# ------------------------------------------------------------- action

@pytest.fixture(scope="module")
def qapp_vtk():
    pytest.importorskip("PySide6")
    pytest.importorskip("pyvistaqt")
    from PySide6.QtWidgets import QApplication
    yield QApplication.instance() or QApplication([])


def test_show_macro_wall_reports_props(qapp_vtk):
    from main_window import MainWindow
    p, aid = _wall_project()
    w = MainWindow()
    w.load_project(p)
    w._set_selection([("area", aid)])
    w.show_macro_wall()
    log = w.log.toPlainText()
    assert "Macro fibre wall" in log
    assert "gross A" in log


def test_show_macro_wall_guarded(qapp_vtk, monkeypatch):
    from main_window import MainWindow
    import main_window as mw
    p, _aid = _wall_project()
    w = MainWindow()
    w.load_project(p)
    w._set_selection([])
    seen = {}
    monkeypatch.setattr(mw.QMessageBox, "information",
                        lambda *a, **k: seen.setdefault("info", a))
    w.show_macro_wall()
    assert "info" in seen
