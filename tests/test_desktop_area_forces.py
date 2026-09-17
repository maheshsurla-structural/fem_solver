"""Slab-modeling S7 (rich) — shell force / moment contours.

Covers model_geometry.areas_result_mesh: the unified resultant accessor across
element types, None/zero guards, square-slab symmetry, the transverse-shear
magnitude, and a one-way slab whose recovered midspan moment matches wL²/8.
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

import model_geometry as mg  # noqa: E402
from project import (Area, AreaLoad, Material, Node, Project,  # noqa: E402
                     ShellSection)


def _slab(L=5.0, B=1.0, t=0.20, mesh=(10, 2), w=10_000.0, E=30e9, nu=0.2):
    p = Project(ndm=3, ndf=6)
    p.materials.append(Material(id=1, name="C", E=E, nu=nu, rho=2400.0))
    p.shell_sections.append(ShellSection(id=1, name="T", thickness=t))
    p.nodes.extend([
        Node(id=1, x=0.0, y=0.0, z=0.0),
        Node(id=2, x=L, y=0.0, z=0.0),
        Node(id=3, x=L, y=B, z=0.0),
        Node(id=4, x=0.0, y=B, z=0.0),
    ])
    p.areas.append(Area(id=1, nodes=[1, 2, 3, 4], shell_section=1, material=1,
                        mesh=mesh))
    p.area_loads.append(AreaLoad(area=1, w=w, kind="gravity", case=1))
    return p


def _fix_where(model, pred, mask):
    for tag, nd in model.nodes.items():
        x, y, _z = nd.coords
        if pred(x, y):
            model.fix(tag, mask)


# ------------------------------------------------------------ guards

def test_result_mesh_none_without_areas():
    p = Project(ndm=3, ndf=6)
    p.nodes.append(Node(id=1, x=0.0, y=0.0, z=0.0))
    assert mg.areas_result_mesh(p.build_model(with_loads=False), "M11") is None


def test_result_mesh_zero_before_solve():
    m = _slab().build_model(with_loads=True)   # built but not solved
    poly = mg.areas_result_mesh(m, "M11")
    assert poly is not None
    assert np.allclose(poly.point_data["value"], 0.0)


# ------------------------------------------------- unified accessor

def test_element_resultant_shapes_after_solve():
    from femsolver import ShellMITC4
    m = _slab(mesh=(4, 2)).build_model(with_loads=True)
    _fix_where(m, lambda x, y: x in (0.0, 5.0), [1, 1, 1, 0, 0, 0])
    LinearStaticAnalysis(m).run()
    quads = [e for e in m.elements.values() if isinstance(e, ShellMITC4)]
    assert quads
    r = mg._element_resultant(quads[0])
    assert r is not None and r.shape == (8,)


# --------------------------------------------------------- physics

def test_one_way_slab_midspan_moment_matches_wL2_over_8():
    L, B, t, w = 5.0, 1.0, 0.20, 10_000.0
    p = _slab(L=L, B=B, t=t, mesh=(12, 2), w=w)
    m = p.build_model(with_loads=True)
    _fix_where(m, lambda x, y: abs(x) < 1e-9 or abs(x - L) < 1e-9,
               [1, 1, 1, 0, 0, 0])                 # simple supports on x-edges
    LinearStaticAnalysis(m).run()
    poly = mg.areas_result_mesh(m, "M11")
    m11_peak = float(np.max(np.abs(poly.point_data["value"])))
    assert m11_peak == pytest.approx(w * L ** 2 / 8.0, rel=0.10)   # 31250 N·m/m


def test_square_slab_symmetry_m11_equals_m22():
    L, t, w = 4.0, 0.15, 8_000.0
    p = _slab(L=L, B=L, t=t, mesh=(6, 6), w=w)
    m = p.build_model(with_loads=True)
    _fix_where(m, lambda x, y: x in (0.0, L) or y in (0.0, L),
               [1, 1, 1, 0, 0, 0])
    LinearStaticAnalysis(m).run()
    m11 = float(np.max(np.abs(mg.areas_result_mesh(m, "M11").point_data["value"])))
    m22 = float(np.max(np.abs(mg.areas_result_mesh(m, "M22").point_data["value"])))
    assert m11 == pytest.approx(m22, rel=0.05)      # symmetry


def test_vmax_is_shear_magnitude():
    res = np.array([0, 0, 0, 0, 0, 0, 3.0, 4.0], dtype=float)
    assert mg._resultant_scalar(res, "Vmax") == pytest.approx(5.0)
    assert mg._resultant_scalar(res, "M11") == 0.0


# ------------------------------------------------- Wood-Armer (S9)

def test_wood_armer_quantities_available():
    for q in ("WAx_bot", "WAy_bot", "WAx_top", "WAy_top"):
        assert q in mg.AREA_RESULT_QUANTITIES


def test_wood_armer_scalar_adds_twist():
    # FE convention: negative M = sagging (bottom tension). A sagging Mx=-10
    # with twist Mxy=3 → bottom design moment Mx* = |−10| + |3| = 13.
    res = np.array([0, 0, 0, -10.0, -4.0, 3.0, 0, 0], dtype=float)
    assert mg._resultant_scalar(res, "WAx_bot") == pytest.approx(13.0)
    assert mg._resultant_scalar(res, "WAy_bot") == pytest.approx(7.0)
    assert mg._resultant_scalar(res, "WAx_top") == pytest.approx(0.0)   # no hogging


def test_gp2node_extrapolation_partition_of_unity():
    # each corner's extrapolation weights must sum to 1 (reproduces a constant)
    assert np.allclose(mg._GP2NODE_Q4.sum(axis=1), 1.0)


def test_nodal_resultants_extrapolated_shape():
    from femsolver import ShellMITC4
    m = _slab(mesh=(3, 2)).build_model(with_loads=True)
    _fix_where(m, lambda x, y: x in (0.0, 5.0), [1, 1, 1, 0, 0, 0])
    LinearStaticAnalysis(m).run()
    quad = next(e for e in m.elements.values() if isinstance(e, ShellMITC4))
    nodal = mg._element_nodal_resultants(quad)
    assert nodal.shape == (4, 8)                 # a value per corner, extrapolated


def _clamped_square(n, L=6.0, t=0.25, q=12_000.0):
    p = Project(ndm=3, ndf=6)
    p.materials.append(Material(id=1, name="C", E=30e9, nu=0.2))
    p.shell_sections.append(ShellSection(id=1, name="T", thickness=t))
    p.nodes.extend([
        Node(id=1, x=0, y=0, z=0), Node(id=2, x=L, y=0, z=0),
        Node(id=3, x=L, y=L, z=0), Node(id=4, x=0, y=L, z=0),
    ])
    p.areas.append(Area(id=1, nodes=[1, 2, 3, 4], shell_section=1, material=1,
                        mesh=(n, n)))
    p.area_loads.append(AreaLoad(area=1, w=q, kind="gravity", case=1))
    m = p.build_model(with_loads=True)
    _fix_where(m, lambda x, y: x in (0.0, L) or y in (0.0, L),
               [1, 1, 1, 1, 1, 1])
    LinearStaticAnalysis(m).run()
    return m


def test_extrapolation_sharpens_coarse_edge_peak():
    # on a coarse clamped plate, GP→node extrapolation recovers a sharper
    # support-moment peak than scattering the element mean.
    m = _clamped_square(4)
    extrap = float(np.max(np.abs(mg.areas_result_mesh(m, "M11").point_data["value"])))
    # element-mean peak (the pre-refinement recovery) for comparison
    tags, _p, idx = mg.node_points(m)
    acc = np.zeros(len(tags))
    cnt = np.zeros(len(tags))
    for e in m.elements.values():
        nt = getattr(e, "node_tags", ())
        if len(nt) not in (3, 4):
            continue
        r = mg._element_resultant(e)
        if r is None:
            continue
        for t in nt:
            j = idx.get(t)
            if j is not None:
                acc[j] += r[3]
                cnt[j] += 1
    mean = float(np.max(np.abs(
        np.divide(acc, cnt, out=np.zeros_like(acc), where=cnt > 0))))
    assert extrap > 1.15 * mean                  # meaningfully sharper


def test_wood_armer_bottom_contour_matches_m11_without_twist():
    # a one-way slab has ~zero twist at midspan, so WAx_bot ≈ peak M11
    L, B, t, w = 5.0, 1.0, 0.20, 10_000.0
    p = _slab(L=L, B=B, t=t, mesh=(12, 2), w=w)
    m = p.build_model(with_loads=True)
    _fix_where(m, lambda x, y: abs(x) < 1e-9 or abs(x - L) < 1e-9,
               [1, 1, 1, 0, 0, 0])
    LinearStaticAnalysis(m).run()
    m11 = float(np.max(np.abs(mg.areas_result_mesh(m, "M11").point_data["value"])))
    wax = float(np.max(mg.areas_result_mesh(m, "WAx_bot").point_data["value"]))
    assert wax == pytest.approx(m11, rel=0.05)
