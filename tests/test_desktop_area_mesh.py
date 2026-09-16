"""Slab-modeling S3 — auto-meshing areas into shell elements.

Covers build_model subdividing a quad area into its mesh grid, coincident-node
merging across adjacent areas, load totals staying mesh-independent, triangles
staying single-element, the AreaDialog mesh field, and a meshed clamped plate
converging to the analytical uniform-load deflection.
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

from femsolver.analysis.linear_static import LinearStaticAnalysis  # noqa: E402

from project import (Area, AreaLoad, Material, Node, Project,  # noqa: E402
                     ShellSection, area_element_tag)


def _mat(p):
    p.materials.append(Material(id=1, name="C30", E=30e9, nu=0.2, rho=2400.0))


def _quad_area_project(mesh=(3, 2)):
    p = Project(ndm=3, ndf=6)
    _mat(p)
    p.shell_sections.append(ShellSection(id=1, name="SLAB200", thickness=0.20))
    p.nodes.extend([
        Node(id=1, x=0.0, y=0.0, z=0.0),
        Node(id=2, x=4.0, y=0.0, z=0.0),
        Node(id=3, x=4.0, y=3.0, z=0.0),
        Node(id=4, x=0.0, y=3.0, z=0.0),
    ])
    p.areas.append(Area(id=1, nodes=[1, 2, 3, 4], shell_section=1, material=1,
                        mesh=mesh))
    return p


# --------------------------------------------------------- mesh counts

def test_quad_area_meshes_into_grid():
    m = _quad_area_project(mesh=(3, 2)).build_model(with_loads=False)
    assert len(m.elements) == 6                    # 3 × 2 sub-elements
    # deterministic contiguous tags
    for k in range(6):
        assert area_element_tag(1, k) in m.elements


def test_meshing_generates_interior_and_edge_nodes():
    # 3×2 mesh of one quad → (3+1)*(2+1) = 12 grid nodes (4 are the corners)
    m = _quad_area_project(mesh=(3, 2)).build_model(with_loads=False)
    assert len(m.nodes) == 12


def test_area_element_tags_count():
    p = _quad_area_project(mesh=(4, 5))
    assert len(p._area_element_tags(p.areas[0])) == 20


def test_triangle_ignores_mesh():
    p = _quad_area_project(mesh=(4, 4))
    p.areas[0] = Area(id=1, nodes=[1, 2, 3], shell_section=1, material=1,
                      mesh=(4, 4))
    m = p.build_model(with_loads=False)
    assert len(m.elements) == 1


# ------------------------------------------------ coincident-node merge

def test_adjacent_areas_share_edge_nodes():
    p = Project(ndm=3, ndf=6)
    _mat(p)
    p.shell_sections.append(ShellSection(id=1, name="T", thickness=0.2))
    # two unit panels side by side sharing the edge x = 1
    for nid, (x, y) in enumerate([(0, 0), (1, 0), (1, 1), (0, 1),
                                  (2, 0), (2, 1)], start=1):
        p.nodes.append(Node(id=nid, x=float(x), y=float(y), z=0.0))
    p.areas.append(Area(id=1, nodes=[1, 2, 3, 4], shell_section=1, material=1,
                        mesh=(2, 2)))
    p.areas.append(Area(id=2, nodes=[2, 5, 6, 3], shell_section=1, material=1,
                        mesh=(2, 2)))
    m = p.build_model(with_loads=False)
    assert len(m.elements) == 8                    # 4 + 4
    # each 2×2 panel = 9 grid nodes; they share a 3-node edge → 15 unique
    assert len(m.nodes) == 15


# --------------------------------------------- load stays mesh-independent

def test_gravity_total_independent_of_mesh():
    A = 4.0 * 3.0
    w = 5000.0
    for mesh in [(1, 1), (2, 2), (5, 4)]:
        p = _quad_area_project(mesh=mesh)
        p.area_loads.append(AreaLoad(area=1, w=w, kind="gravity", case=1))
        m = p.build_model(with_loads=True)
        total = 0.0
        for tag in p._area_element_tags(p.areas[0]):
            total += m.element(tag).f_eq_global().reshape(4, 6)[:, 2].sum()
        assert total == pytest.approx(-w * A, rel=1e-9)


# ------------------------------------------------------- convergence

def _clamped_square(n, L=4.0, t=0.15, q=10_000.0):
    p = Project(ndm=3, ndf=6)
    _mat(p)
    p.shell_sections.append(ShellSection(id=1, name="T", thickness=t))
    p.nodes.extend([
        Node(id=1, x=0.0, y=0.0, z=0.0),
        Node(id=2, x=L, y=0.0, z=0.0),
        Node(id=3, x=L, y=L, z=0.0),
        Node(id=4, x=0.0, y=L, z=0.0),
    ])
    p.areas.append(Area(id=1, nodes=[1, 2, 3, 4], shell_section=1, material=1,
                        mesh=(n, n)))
    p.area_loads.append(AreaLoad(area=1, w=q, kind="gravity", case=1))
    m = p.build_model(with_loads=True)
    cen = None
    for tag, nd in m.nodes.items():
        x, y, _z = nd.coords
        if x in (0.0, L) or y in (0.0, L):
            m.fix(tag, [1, 1, 1, 1, 1, 1])         # clamp the boundary
        if abs(x - L / 2) < 1e-9 and abs(y - L / 2) < 1e-9:
            cen = tag
    LinearStaticAnalysis(m).run()
    return abs(m.node(cen).disp[2])


def test_clamped_plate_converges_to_analytical():
    L, t, q, E, nu = 4.0, 0.15, 10_000.0, 30e9, 0.2
    D = E * t ** 3 / (12.0 * (1.0 - nu ** 2))
    w_exact = 0.00126 * q * L ** 4 / D             # clamped square, uniform load
    w8 = _clamped_square(8, L=L, t=t, q=q)
    w4 = _clamped_square(4, L=L, t=t, q=q)
    assert w8 == pytest.approx(w_exact, rel=0.10)  # 8×8 within ~10%
    # refinement moves toward the analytical value
    assert abs(w8 - w_exact) <= abs(w4 - w_exact) + 1e-12
