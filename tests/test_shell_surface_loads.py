"""Consistent surface loads on shell elements (slab plan S5a).

Validates the equivalent nodal load vectors ShellMITC4 / ShellTri3 build for a
uniform normal pressure and a uniform global traction (e.g. gravity area load),
plus that a pressure-loaded plate solves and deflects linearly with pressure.
"""
from __future__ import annotations

import sys
from pathlib import Path

import numpy as np
import pytest

_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(_ROOT / "src"))

from femsolver import ElasticIsotropic, Model, ShellMITC4, ShellTri3  # noqa: E402
from femsolver.analysis.linear_static import LinearStaticAnalysis  # noqa: E402


def _mat(tag=1):
    return ElasticIsotropic(tag, E=2.0e11, nu=0.3, rho=7850.0)


def _quad(a=2.0, b=3.0, z=0.0, plane="xy"):
    """One MITC4 quad of plan size a×b in the requested plane, bound to a
    model so node_coords() works. Returns (model, element, area)."""
    m = Model(ndm=3, ndf=6)
    m.add_material(_mat())
    if plane == "xy":                       # normal +z
        pts = [(0, 0, z), (a, 0, z), (a, b, z), (0, b, z)]
    else:                                    # "xz": in-plane x-z, normal ±y
        pts = [(0, z, 0), (a, z, 0), (a, z, b), (0, z, b)]
    for i, p in enumerate(pts, start=1):
        m.add_node(i, *p)
    e = ShellMITC4(1, (1, 2, 3, 4), m.material(1), 0.05)
    m.add_element(e)
    return m, e, a * b


# ------------------------------------------------------- MITC4 pressure

def test_mitc4_pressure_totals_qA_in_normal():
    m, e, A = _quad()
    q = 1000.0
    e.add_pressure(q)
    f = e.f_eq_global()
    assert f.shape == (24,)
    fz = f.reshape(4, 6)[:, 2]
    assert np.allclose(fz, q * A / 4.0)          # rectangle → equal quarters
    assert f.reshape(4, 6)[:, 2].sum() == pytest.approx(q * A)
    # no in-plane force, no nodal moments
    assert np.allclose(f.reshape(4, 6)[:, [0, 1]], 0.0)
    assert np.allclose(f.reshape(4, 6)[:, 3:], 0.0)


def test_mitc4_pressure_follows_normal_direction():
    # a vertical panel (x-z plane) has normal ±y, so pressure loads global y
    m, e, A = _quad(plane="xz")
    e.add_pressure(500.0)
    f = e.f_eq_global().reshape(4, 6)
    total = f[:, :3].sum(axis=0)
    assert abs(total[1]) == pytest.approx(500.0 * A)     # all load in ±y
    assert total[0] == pytest.approx(0.0, abs=1e-6)
    assert total[2] == pytest.approx(0.0, abs=1e-6)


def test_mitc4_gravity_traction_totals():
    m, e, A = _quad()
    w = 2500.0
    e.add_surface_load(0.0, 0.0, -w)             # downward area load
    total = e.f_eq_global().reshape(4, 6)[:, :3].sum(axis=0)
    assert total[2] == pytest.approx(-w * A)


def test_mitc4_clear_distributed_loads():
    m, e, A = _quad()
    e.add_pressure(1000.0)
    e.add_surface_load(0.0, 0.0, -100.0)
    e.clear_distributed_loads()
    assert not np.any(e.f_eq_global())


# --------------------------------------------------------- Tri3 pressure

def test_tri3_pressure_thirds():
    m = Model(ndm=3, ndf=6)
    m.add_material(_mat())
    for i, p in enumerate([(0, 0, 0), (3.0, 0, 0), (0, 4.0, 0)], start=1):
        m.add_node(i, *p)
    e = ShellTri3(1, (1, 2, 3), m.material(1), 0.05)
    m.add_element(e)
    A = 0.5 * 3.0 * 4.0
    e.add_pressure(900.0)
    f = e.f_eq_global().reshape(3, 6)
    assert np.allclose(f[:, 2], 900.0 * A / 3.0)
    assert f[:, 2].sum() == pytest.approx(900.0 * A)


# ------------------------------------------------ end-to-end plate solve

def _clamped_plate(n=2, L=2.0, t=0.02, q=0.0):
    """n×n MITC4 mesh of an L×L plate, all boundary nodes clamped, uniform
    pressure q on every element. Returns (model, centre_node_tag)."""
    m = Model(ndm=3, ndf=6)
    m.add_material(_mat())
    nL = n + 1
    coords = {}
    tag = 1
    for j in range(nL):
        for i in range(nL):
            x, y = L * i / n, L * j / n
            m.add_node(tag, x, y, 0.0)
            coords[(i, j)] = tag
            tag += 1
    eid = 1
    for j in range(n):
        for i in range(n):
            nodes = (coords[(i, j)], coords[(i + 1, j)],
                     coords[(i + 1, j + 1)], coords[(i, j + 1)])
            e = ShellMITC4(eid, nodes, m.material(1), t)
            if q:
                e.add_pressure(-q)            # push down (−z)
            m.add_element(e)
            eid += 1
    for (i, j), tg in coords.items():
        if i in (0, n) or j in (0, n):
            m.fix(tg, [1, 1, 1, 1, 1, 1])
    return m, coords[(n // 2, n // 2)]


def test_pressure_deflection_scales_linearly():
    m1, c = _clamped_plate(n=2, q=1000.0)
    LinearStaticAnalysis(m1).run()
    w1 = m1.node(c).disp[2]
    m2, c2 = _clamped_plate(n=2, q=2000.0)
    LinearStaticAnalysis(m2).run()
    w2 = m2.node(c2).disp[2]
    assert w1 < 0.0                          # pressed down → deflects down
    assert w2 == pytest.approx(2.0 * w1, rel=1e-6)   # linear in pressure
