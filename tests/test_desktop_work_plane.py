"""Wall-modeling W1b — the draw work-plane (elevation / XZ-YZ planes).

Covers ModelView.set_work_plane state, that a draw-node pick lands on the chosen
plane with its full 3-D coordinates (not forced to z=0), and the MainWindow
plane-selector wiring.
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

from project import Material, Node, Project, Section  # noqa: E402


def _proj():
    p = Project(ndm=3, ndf=6)
    p.materials.append(Material(id=1, name="S", E=200e9, nu=0.3))
    p.sections.append(Section(id=1, name="W", A=1e-2, Iz=1e-4, Iy=1e-4, J=1e-5))
    p.nodes.extend([Node(id=1, x=0.0, y=0.0, z=0.0),
                    Node(id=2, x=4.0, y=0.0, z=0.0)])
    return p


@pytest.fixture(scope="module")
def qapp_vtk():
    pytest.importorskip("PySide6")
    pytest.importorskip("pyvistaqt")
    from PySide6.QtWidgets import QApplication
    yield QApplication.instance() or QApplication([])


def _window(qapp_vtk):
    from main_window import MainWindow
    w = MainWindow()
    w.load_project(_proj())
    return w


def test_set_work_plane_state(qapp_vtk):
    w = _window(qapp_vtk)
    w.view.set_work_plane("xz", 2.5)
    assert w.view._work_plane == "xz"
    assert w.view._work_offset == 2.5
    # a bad kind falls back to xy
    w.view.set_work_plane("nope", 0.0)
    assert w.view._work_plane == "xy"


def test_draw_pick_on_xz_plane_keeps_full_coords(qapp_vtk):
    w = _window(qapp_vtk)
    captured = []
    w.view._add_node_cb = lambda x, y, z: captured.append((x, y, z))
    w.view._mode = "draw_node"
    w.view._snap_on = False
    w.view.set_work_plane("xz", 2.0)        # elevation plane at y = 2.0
    w.view._on_point_picked(np.array([1.3, 5.0, 2.7]))
    x, y, z = captured[-1]
    assert y == 2.0                          # pinned to the plane, not 0
    assert x == pytest.approx(1.3)
    assert z == pytest.approx(2.7)           # z preserved (was forced to 0)


def test_draw_pick_on_yz_plane(qapp_vtk):
    w = _window(qapp_vtk)
    captured = []
    w.view._add_node_cb = lambda x, y, z: captured.append((x, y, z))
    w.view._mode = "draw_node"
    w.view._snap_on = False
    w.view.set_work_plane("yz", 3.0)        # elevation plane at x = 3.0
    w.view._on_point_picked(np.array([9.0, 1.1, 2.2]))
    x, y, z = captured[-1]
    assert x == 3.0
    assert (y, z) == pytest.approx((1.1, 2.2))


def test_default_plane_is_xy_ground(qapp_vtk):
    w = _window(qapp_vtk)
    captured = []
    w.view._add_node_cb = lambda x, y, z: captured.append((x, y, z))
    w.view._mode = "draw_node"
    w.view._snap_on = False
    w.view._on_point_picked(np.array([2.0, 3.0, 7.7]))
    x, y, z = captured[-1]
    assert z == 0.0                          # default XY ground plane pins z
    assert (x, y) == pytest.approx((2.0, 3.0))


def test_plane_selector_pushes_to_view(qapp_vtk):
    w = _window(qapp_vtk)
    i = w.plane_combo.findData("xz")
    w.plane_combo.setCurrentIndex(i)
    w.plane_offset.setValue(3.5)
    assert w.view._work_plane == "xz"
    assert w.view._work_offset == pytest.approx(3.5)


# ----------------------------------------------------- W1c: 3-point work plane

def test_set_work_plane_3pt_and_projection(qapp_vtk):
    w = _window(qapp_vtk)
    # a tilted plane through 3 points; normal = (p1-p0)x(p2-p0)
    ok = w.view.set_work_plane_3pt((0, 0, 0), (1, 0, 0), (0, 1, 1))
    assert ok and w.view._work_plane == "3pt"
    captured = []
    w.view._add_node_cb = lambda x, y, z: captured.append((x, y, z))
    w.view._mode = "draw_node"
    # pick a point off the plane → it must be projected ONTO the plane
    w.view._on_point_picked(np.array([0.5, 0.5, 0.5]))
    x, y, z = captured[-1]
    import numpy as _np
    o, n = w.view._work_plane_origin_normal()
    # the returned point lies on the plane: (q - o)·n ≈ 0
    assert abs(float(_np.dot(_np.array([x, y, z]) - o, n))) < 1e-6


def test_set_work_plane_3pt_rejects_collinear(qapp_vtk):
    w = _window(qapp_vtk)
    assert not w.view.set_work_plane_3pt((0, 0, 0), (1, 0, 0), (2, 0, 0))


def test_work_plane_from_nodes_action(qapp_vtk):
    from main_window import MainWindow
    p = _proj()
    p.nodes.append(Node(id=3, x=0.0, y=3.0, z=2.0))
    w = MainWindow()
    w.load_project(p)
    w._set_selection([("node", 1), ("node", 2), ("node", 3)])
    w.set_work_plane_from_nodes()
    assert w.view._work_plane == "3pt"


def test_work_plane_from_nodes_needs_three(qapp_vtk, monkeypatch):
    from main_window import MainWindow
    import main_window as mw
    w = MainWindow()
    w.load_project(_proj())
    w._set_selection([("node", 1)])
    seen = {}
    monkeypatch.setattr(mw.QMessageBox, "information",
                        lambda *a, **k: seen.setdefault("info", a))
    w.set_work_plane_from_nodes()
    assert "info" in seen
