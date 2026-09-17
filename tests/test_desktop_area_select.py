"""Slab-modeling S10 — click-to-select areas (hit-test, faces, properties).

Covers model_geometry.nearest_item picking an area by clicking its interior,
the mesh-tag → project-Area decode, area_faces_mesh (selection highlight), and
the read-only area properties form.
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

import model_geometry as mg  # noqa: E402
from project import (AREA_TAG_BASE, Area, Material, Node,  # noqa: E402
                     Project, ShellSection, area_element_tag)


def _slab_model(mesh=(2, 2)):
    p = Project(ndm=3, ndf=6)
    p.materials.append(Material(id=1, name="C30", E=30e9, nu=0.2))
    p.shell_sections.append(ShellSection(id=1, name="SLAB200", thickness=0.20))
    p.nodes.extend([
        Node(id=1, x=0.0, y=0.0, z=0.0), Node(id=2, x=4.0, y=0.0, z=0.0),
        Node(id=3, x=4.0, y=3.0, z=0.0), Node(id=4, x=0.0, y=3.0, z=0.0),
    ])
    p.areas.append(Area(id=1, nodes=[1, 2, 3, 4], shell_section=1, material=1,
                        mesh=mesh))
    return p, p.build_model(with_loads=False)


# ------------------------------------------------------------- decode

def test_decode_area_id():
    assert mg._decode_area_id(area_element_tag(7, 3)) == 7
    assert mg._decode_area_id(area_element_tag(1, 0)) == 1
    assert mg._decode_area_id(5) is None            # a member tag, not an area


# --------------------------------------------------------- hit-testing

def test_click_interior_selects_area():
    _p, m = _slab_model()
    tol = max(mg.model_span(m) * 0.05, 0.15)
    # a point well inside the first sub-quad, away from nodes/edges
    sel = mg.nearest_item(m, (1.0, 0.75, 0.0), tol)
    assert sel == ("area", 1)


def test_click_on_node_wins_over_area():
    _p, m = _slab_model()
    tol = max(mg.model_span(m) * 0.05, 0.15)
    assert mg.nearest_item(m, (0.0, 0.0, 0.0), tol) == ("node", 1)


def test_click_outside_area_is_none():
    _p, m = _slab_model()
    tol = max(mg.model_span(m) * 0.05, 0.15)
    assert mg.nearest_item(m, (10.0, 10.0, 0.0), tol) is None


# ------------------------------------------------------- faces mesh

def test_area_faces_mesh_covers_the_mesh():
    _p, m = _slab_model(mesh=(2, 2))
    faces = mg.area_faces_mesh(m, 1)
    assert faces is not None and faces.n_cells == 4     # 2×2 sub-elements
    assert mg.area_faces_mesh(m, 99) is None


# ------------------------------------------------------- properties

def test_area_properties_form():
    pytest.importorskip("PySide6")
    from PySide6.QtWidgets import QApplication
    QApplication.instance() or QApplication([])
    from properties import PropertiesPanel
    p, _m = _slab_model()
    panel = PropertiesPanel(on_apply=lambda *a: None, on_bulk=lambda *a: None)
    panel.show_item(p, "area", 1)                   # must not raise
    assert p.area(1) is not None
