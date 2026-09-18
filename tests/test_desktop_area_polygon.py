"""Slab-modeling — polygon areas (5+ sides via centroid-fan triangulation)."""
from __future__ import annotations

import math
import sys
from pathlib import Path

import numpy as np
import pytest

_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(_ROOT / "src"))
sys.path.insert(0, str(_ROOT))
sys.path.insert(0, str(_ROOT / "desktop"))

from femsolver import ShellTri3  # noqa: E402

import model_geometry as mg  # noqa: E402
from project import (Area, AreaLoad, Material, Node, Project,  # noqa: E402
                     ShellSection, area_element_tag)


def _hexagon(R=2.0, w=1000.0, kind="gravity"):
    p = Project(ndm=3, ndf=6)
    p.materials.append(Material(id=1, name="C", E=30e9, nu=0.2, rho=2400.0))
    p.shell_sections.append(ShellSection(id=1, name="T", thickness=0.2))
    for k in range(6):
        ang = math.radians(60 * k)
        p.nodes.append(Node(id=k + 1, x=R * math.cos(ang), y=R * math.sin(ang),
                            z=0.0))
    p.areas.append(Area(id=1, nodes=[1, 2, 3, 4, 5, 6], shell_section=1,
                        material=1))
    if w:
        p.area_loads.append(AreaLoad(area=1, w=w, kind=kind, case=1))
    return p, 3.0 * math.sqrt(3.0) / 2.0 * R ** 2   # regular-hexagon area


# ----------------------------------------------------------- meshing

def test_polygon_meshes_into_fan_of_triangles():
    p, _A = _hexagon(w=0.0)
    m = p.build_model(with_loads=False)
    els = list(m.elements.values())
    assert len(els) == 6                            # one tri per edge
    assert all(isinstance(e, ShellTri3) for e in els)
    assert len(m.nodes) == 7                        # 6 corners + centroid
    for k in range(6):
        assert area_element_tag(1, k) in m.elements


def test_polygon_area_element_tags():
    p, _A = _hexagon(w=0.0)
    assert len(p._area_element_tags(p.areas[0])) == 6


# ---------------------------------------------- load over the polygon

def test_gravity_total_over_polygon_area():
    p, A = _hexagon(w=1500.0, kind="gravity")
    m = p.build_model(with_loads=True)
    total = 0.0
    for tag in p._area_element_tags(p.areas[0]):
        total += m.element(tag).f_eq_global().reshape(3, 6)[:, 2].sum()
    assert total == pytest.approx(-1500.0 * A, rel=1e-6)   # tiles the polygon


# ------------------------------------------------------- selection

def test_click_selects_polygon_area():
    p, _A = _hexagon(w=0.0)
    m = p.build_model(with_loads=False)
    tol = max(mg.model_span(m) * 0.05, 0.15)
    # a point inside one of the fan triangles, off the centre/corners
    sel = mg.nearest_item(m, (0.9, 0.2, 0.0), tol)
    assert sel == ("area", 1)
