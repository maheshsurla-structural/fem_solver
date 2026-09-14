"""Influence surfaces — 2-D moving load on decks / grillages (bridge plan T1.1).

Validates the influence-surface engine against direct solves (Betti/Maxwell
reciprocity → machine precision), plus the 2-D vehicle placement and envelope.
"""
from __future__ import annotations

import os
import sys
from pathlib import Path

import numpy as np
import pytest

_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(_ROOT / "src"))

from femsolver import (BeamColumn3D, ElasticIsotropic,  # noqa: E402
                       LinearStaticAnalysis, Model)
from femsolver.bridges import (DeckSurface, Displacement,  # noqa: E402
                               InfluenceLineEngine, InfluenceSurface, Reaction,
                               Vehicle2D, moving_load_surface_envelope)


def _grillage(nx=5, ny=3, dx=2.0, dy=2.0):
    """A fixed-ended 3-D grillage deck (vertical DOF = uz index 2)."""
    E, nu = 3.0e10, 0.2
    A, Iy, Iz, J = 0.5, 0.02, 0.04, 0.01
    m = Model(ndm=3, ndf=6)
    mat = ElasticIsotropic(1, E=E, nu=nu)
    m.add_material(mat)
    nid, tag = {}, 1
    for i in range(nx):
        for j in range(ny):
            m.add_node(tag, i * dx, j * dy, 0.0)
            nid[(i, j)] = tag
            tag += 1
    eid = 1
    for j in range(ny):                       # longitudinal beams
        for i in range(nx - 1):
            m.add_element(BeamColumn3D(eid, (nid[(i, j)], nid[(i + 1, j)]),
                                       mat, A, Iy, Iz, J)); eid += 1
    for i in range(nx):                       # transverse beams
        for j in range(ny - 1):
            m.add_element(BeamColumn3D(eid, (nid[(i, j)], nid[(i, j + 1)]),
                                       mat, A, Iy, Iz, J)); eid += 1
    for j in range(ny):                       # fix both x-ends
        m.fix(nid[(0, j)], [1, 1, 1, 1, 1, 1])
        m.fix(nid[(nx - 1, j)], [1, 1, 1, 1, 1, 1])
    return m, nid, (nx, ny)


def _direct_disp(m, load_node, target_node, dof=2):
    m.reset_results(); m.number_dofs(); m.clear_loads()
    m.add_nodal_load(load_node, [0, 0, -1.0, 0, 0, 0])
    LinearStaticAnalysis(m).run()
    return float(m.node(target_node).disp[dof])


# ---------------------------------------------------------- reciprocity
def test_influence_surface_matches_direct_solve():
    m, nid, (nx, ny) = _grillage()
    deck = [nid[(i, j)] for i in range(nx) for j in range(ny)]
    center = nid[(2, 1)]
    IS = InfluenceLineEngine(m).influence_surface(
        DeckSurface(node_tags=deck, load_dof=2), Displacement(center, 2))
    assert isinstance(IS, InfluenceSurface)
    for (i, j) in [(2, 1), (1, 0), (3, 2), (2, 0), (1, 1)]:
        p = m.node(nid[(i, j)]).coords
        assert IS((p[0], p[1])) == pytest.approx(
            _direct_disp(m, nid[(i, j)], center), abs=1e-18, rel=1e-9)


def test_influence_surface_reaction_reciprocity():
    m, nid, (nx, ny) = _grillage()
    deck = [nid[(i, j)] for i in range(nx) for j in range(ny)
            if 0 < i < nx - 1]                       # interior only
    corner = nid[(0, 0)]
    IS = InfluenceLineEngine(m).influence_surface(
        DeckSurface(node_tags=deck, load_dof=2), Reaction(corner, 2))
    # reaction IL is exact at each node too
    for (i, j) in [(1, 0), (2, 1), (3, 2)]:
        p = m.node(nid[(i, j)]).coords
        m.reset_results(); m.number_dofs(); m.clear_loads()
        m.add_nodal_load(nid[(i, j)], [0, 0, -1.0, 0, 0, 0])
        LinearStaticAnalysis(m).run()
        assert IS((p[0], p[1])) == pytest.approx(
            float(m.node(corner).reaction[2]), abs=1e-12, rel=1e-9)


def test_zero_outside_deck():
    m, nid, (nx, ny) = _grillage()
    deck = [nid[(i, j)] for i in range(nx) for j in range(ny)]
    IS = InfluenceLineEngine(m).influence_surface(
        DeckSurface(node_tags=deck, load_dof=2), Displacement(nid[(2, 1)], 2))
    assert IS((-50.0, -50.0)) == 0.0             # far outside the convex hull


# ------------------------------------------------------------- vehicles
def test_vehicle2d_from_axle_train_splits_wheels():
    from femsolver.bridges import MovingLoad
    train = MovingLoad(axle_loads=[100.0, 200.0], axle_offsets=[0.0, 4.0])
    veh = Vehicle2D.from_axle_train(train, track_width=1.8)
    assert veh.wheel_loads.size == 4
    assert veh.wheel_loads.sum() == pytest.approx(300.0)
    assert set(np.round(veh.wheel_xy[:, 1], 3)) == {0.9, -0.9}


def test_surface_envelope_matches_multiwheel_direct_solve():
    m, nid, (nx, ny) = _grillage()
    deck = [nid[(i, j)] for i in range(nx) for j in range(ny)]
    center = nid[(2, 1)]
    IS = InfluenceLineEngine(m).influence_surface(
        DeckSurface(node_tags=deck, load_dof=2), Displacement(center, 2))
    # two wheels landing on nodes (2,2) and (4,2) i.e. grid (1,1),(2,1)
    veh = Vehicle2D(wheel_loads=[1.0, 2.0], wheel_xy=[(0.0, 0.0), (2.0, 0.0)])
    pred = 1.0 * IS((2.0, 2.0)) + 2.0 * IS((4.0, 2.0))
    m.reset_results(); m.number_dofs(); m.clear_loads()
    m.add_nodal_load(nid[(1, 1)], [0, 0, -1.0, 0, 0, 0])
    m.add_nodal_load(nid[(2, 1)], [0, 0, -2.0, 0, 0, 0])
    LinearStaticAnalysis(m).run()
    assert pred == pytest.approx(float(m.node(center).disp[2]), rel=1e-9)


def test_surface_envelope_single_wheel_equals_extreme_ordinate():
    m, nid, (nx, ny) = _grillage()
    deck = [nid[(i, j)] for i in range(nx) for j in range(ny)]
    IS = InfluenceLineEngine(m).influence_surface(
        DeckSurface(node_tags=deck, load_dof=2), Displacement(nid[(2, 1)], 2))
    veh = Vehicle2D(wheel_loads=[1.0], wheel_xy=[(0.0, 0.0)])
    xs = sorted({m.node(t).coords[0] for t in deck})
    ys = sorted({m.node(t).coords[1] for t in deck})
    env = moving_load_surface_envelope(IS, veh, x_positions=xs, y_positions=ys)
    assert env["min"] == pytest.approx(IS.min_value(), rel=1e-9)


# --------------------------------------------------------------- guards
def test_decksurface_needs_three_nodes():
    with pytest.raises(ValueError, match="at least 3"):
        DeckSurface(node_tags=[1, 2])


def test_influence_surface_collinear_points_raise():
    IS = InfluenceSurface(points=[[0, 0], [1, 0], [2, 0]], values=[0, 1, 0])
    with pytest.raises(RuntimeError, match="triangulate"):
        IS((0.5, 0.0))
