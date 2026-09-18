"""Wall-modeling W4b — grid/story overlay geometry + snapping.

Covers the pure model_geometry helpers (grid_story_mesh, snap_targets,
snap_to_grid) headless, then the ModelView overlay + snap-target wiring and the
View toggle under a QApplication.
"""
from __future__ import annotations

import os
import sys
from pathlib import Path

import numpy as np
import pytest

os.environ.setdefault("QT_QPA_PLATFORM", "offscreen")
_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(_ROOT / "src"))
sys.path.insert(0, str(_ROOT))
sys.path.insert(0, str(_ROOT / "desktop"))

import model_geometry as mg  # noqa: E402
from project import GridLine, Material, Node, Project, Section, Story  # noqa: E402


def _proj_model():
    p = Project(ndm=3, ndf=6)
    p.materials.append(Material(id=1, name="S", E=200e9, nu=0.3))
    p.sections.append(Section(id=1, name="W", A=1e-2, Iz=1e-4, Iy=1e-4, J=1e-5))
    p.nodes.extend([Node(id=1, x=0.0, y=0.0, z=0.0),
                    Node(id=2, x=6.0, y=0.0, z=0.0),
                    Node(id=3, x=6.0, y=4.0, z=3.0)])
    p.stories.extend([Story(id=1, name="Base", elev=0.0),
                      Story(id=2, name="L1", elev=3.0, height=3.0)])
    p.grid_lines.extend([GridLine(id=1, name="A", axis="x", coord=0.0),
                         GridLine(id=2, name="B", axis="x", coord=6.0),
                         GridLine(id=3, name="1", axis="y", coord=0.0)])
    return p, p.build_model(with_loads=False)


# --------------------------------------------------------- geometry helpers

def test_grid_story_mesh_builds_lines():
    p, m = _proj_model()
    poly = mg.grid_story_mesh(p, m)
    assert poly is not None
    # 3 grid lines (1 seg each) + 2 stories (4 segs each) = 11 segments
    assert poly.n_lines == 3 + 2 * 4
    assert poly.n_points == poly.n_lines * 2


def test_grid_story_mesh_none_without_grid_or_stories():
    p = Project(ndm=3, ndf=6)
    p.nodes.append(Node(id=1, x=0.0, y=0.0, z=0.0))
    assert mg.grid_story_mesh(p, p.build_model(with_loads=False)) is None


def test_snap_targets():
    p, _m = _proj_model()
    xs, ys, zs = mg.snap_targets(p)
    assert sorted(xs) == [0.0, 6.0]      # const-X grid coords
    assert ys == [0.0]                   # const-Y grid coord
    assert sorted(zs) == [0.0, 3.0]      # story elevations


def test_snap_to_grid_within_tolerance():
    # x → 6.0 (0.2 away, in tol); y stays (0.9 away); z → 3.0 (0.15 away, in tol)
    x, y, z = mg.snap_to_grid(5.8, 0.9, 2.85, [0.0, 6.0], [0.0], [0.0, 3.0],
                              tol=0.3)
    assert x == 6.0
    assert y == 0.9
    assert z == 3.0


def test_snap_to_grid_leaves_far_points():
    x, y, z = mg.snap_to_grid(3.0, 2.0, 1.5, [0.0, 6.0], [0.0], [0.0, 3.0],
                              tol=0.3)
    assert (x, y, z) == (3.0, 2.0, 1.5)  # all beyond tol → unchanged


# --------------------------------------------------------- viewport wiring

@pytest.fixture(scope="module")
def qapp_vtk():
    pytest.importorskip("PySide6")
    pytest.importorskip("pyvistaqt")
    from PySide6.QtWidgets import QApplication
    yield QApplication.instance() or QApplication([])


def test_view_registers_snap_targets_and_overlay(qapp_vtk):
    from main_window import MainWindow
    p, _m = _proj_model()
    w = MainWindow()
    w.load_project(p)                    # triggers _render_model → set_story_grid
    assert sorted(w.view._snap_xs) == [0.0, 6.0]
    assert w.view._snap_ys == [0.0]
    # the overlay actor is present while the toggle is on
    assert "storygrid" in w.view.renderer.actors
    w.view.show_story_grid(False)
    assert "storygrid" not in w.view.renderer.actors
    w.view.show_story_grid(True)
    assert "storygrid" in w.view.renderer.actors


def test_story_grid_toggle_action_wired(qapp_vtk):
    from main_window import MainWindow
    p, _m = _proj_model()
    w = MainWindow()
    w.load_project(p)
    assert w.act_story_grid.isChecked()
    w.act_story_grid.setChecked(False)
    assert "storygrid" not in w.view.renderer.actors
