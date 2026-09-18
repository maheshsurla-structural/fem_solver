"""Slab-modeling S2/S6 polish — click-to-draw areas + local-axis triads.

Pure-geometry coverage of ``model_geometry.area_local_axes`` and, via a headless
``ModelView``, the draw-area click accumulation (quad auto-close at 4 corners,
triangle close by re-clicking the first) and the local-axes toggle.
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
from project import (Area, Material, Node, Project, ShellSection)  # noqa: E402


def _model_with_quad():
    p = Project(ndm=3, ndf=6)
    p.materials.append(Material(id=1, name="C", E=30e9, nu=0.2))
    p.shell_sections.append(ShellSection(id=1, name="T", thickness=0.2))
    p.nodes.extend([
        Node(id=1, x=0.0, y=0.0, z=0.0),
        Node(id=2, x=4.0, y=0.0, z=0.0),
        Node(id=3, x=4.0, y=3.0, z=0.0),
        Node(id=4, x=0.0, y=3.0, z=0.0),
    ])
    p.areas.append(Area(id=1, nodes=[1, 2, 3, 4], shell_section=1, material=1,
                        mesh=(1, 1)))
    return p.build_model(with_loads=False)


# ---------------------------------------------------- local axes (S6)

def test_area_local_axes_triads():
    m = _model_with_quad()                         # 1 element (1×1 mesh)
    tri = mg.area_local_axes(m, scale=1.0)
    assert tri is not None and len(tri) == 3
    e1, e2, e3 = tri
    for poly in (e1, e2, e3):
        assert poly is not None and poly.n_cells == 1     # one triad per element
    # the XY quad's normal (e3) must point along ±Z
    pts = e3.points
    axis = pts[1] - pts[0]
    axis /= np.linalg.norm(axis)
    assert abs(abs(axis[2]) - 1.0) < 1e-9


def test_area_local_axes_none_without_areas():
    p = Project(ndm=3, ndf=6)
    p.nodes.append(Node(id=1, x=0.0, y=0.0, z=0.0))
    tri = mg.area_local_axes(p.build_model(with_loads=False))
    assert tri == (None, None, None)


# --------------------------------------------- draw-area mode (S2)

@pytest.fixture(scope="module")
def qapp():
    pytest.importorskip("PySide6")
    from PySide6.QtWidgets import QApplication
    yield QApplication.instance() or QApplication([])


def _view(qapp):
    from model_view import ModelView
    v = ModelView()
    v.set_model(_model_with_quad())
    return v


def test_draw_area_accumulates_quad(qapp):
    v = _view(qapp)
    got = []
    v.set_add_area_callback(lambda nodes: got.append(list(nodes)))
    v.set_mode("draw_area")
    for tag in (1, 2, 3, 4):                        # click the 4 corners
        v._on_point_picked(np.asarray(mg.to_xyz(v._model.nodes[tag].coords)))
    assert got == []                               # not closed yet (no auto-4)
    v._on_point_picked(np.asarray(mg.to_xyz(v._model.nodes[1].coords)))  # close
    assert got == [[1, 2, 3, 4]]
    assert v._area_pick == []                       # reset after close


def test_draw_area_triangle_closes_on_first_click(qapp):
    v = _view(qapp)
    got = []
    v.set_add_area_callback(lambda nodes: got.append(list(nodes)))
    v.set_mode("draw_area")
    for tag in (1, 2, 3):
        v._on_point_picked(np.asarray(mg.to_xyz(v._model.nodes[tag].coords)))
    # click the first corner again → close as a triangle
    v._on_point_picked(np.asarray(mg.to_xyz(v._model.nodes[1].coords)))
    assert got == [[1, 2, 3]]


def test_set_area_axes_toggle(qapp):
    v = _view(qapp)
    assert v._show_area_axes is False
    v.set_area_axes(True)
    assert v._show_area_axes is True
