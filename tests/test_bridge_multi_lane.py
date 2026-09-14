"""Multi-lane placement + AASHTO multiple presence (bridge plan T1.2).

Covers the multiple-presence factors, design-lane generation, the governing
k-lane combination, and physical superposition of lane contributions
(machine precision vs a direct multi-load solve).
"""
from __future__ import annotations

import sys
from pathlib import Path

import numpy as np
import pytest

_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(_ROOT / "src"))

from femsolver import (BeamColumn3D, ElasticIsotropic,  # noqa: E402
                       LinearStaticAnalysis, Model)
from femsolver.bridges import (DeckSurface, DesignLane,  # noqa: E402
                               Displacement, InfluenceLineEngine, Vehicle2D,
                               aashto_hl93_truck, generate_design_lanes,
                               moving_load_surface_envelope,
                               multi_lane_envelope, multi_presence_factor,
                               number_of_design_lanes)
from femsolver.bridges.influence import _lane_center_positions  # noqa: E402


def _grillage(nx=9, ny=6, dx=2.0, dy=1.5):
    E, nu = 3.0e10, 0.2
    A, Iy, Iz, J = 0.6, 0.03, 0.06, 0.02
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
    for j in range(ny):
        for i in range(nx - 1):
            m.add_element(BeamColumn3D(eid, (nid[(i, j)], nid[(i + 1, j)]),
                                       mat, A, Iy, Iz, J)); eid += 1
    for i in range(nx):
        for j in range(ny - 1):
            m.add_element(BeamColumn3D(eid, (nid[(i, j)], nid[(i, j + 1)]),
                                       mat, A, Iy, Iz, J)); eid += 1
    for j in range(ny):
        m.fix(nid[(0, j)], [1, 1, 1, 1, 1, 1])
        m.fix(nid[(nx - 1, j)], [1, 1, 1, 1, 1, 1])
    return m, nid, (nx, ny, dx, dy)


# --------------------------------------------------- multiple presence
def test_multi_presence_factors_match_aashto():
    assert [multi_presence_factor(k) for k in (1, 2, 3, 4, 5)] == \
        [1.20, 1.00, 0.85, 0.65, 0.65]


def test_multi_presence_rejects_zero():
    with pytest.raises(ValueError):
        multi_presence_factor(0)


def test_number_of_design_lanes():
    assert number_of_design_lanes(3.6) == 1
    assert number_of_design_lanes(7.0) == 2        # 6.0–7.2 special rule
    assert number_of_design_lanes(10.8) == 3
    assert number_of_design_lanes(14.5) == 4
    assert number_of_design_lanes(0.0) == 0


def test_generate_design_lanes_tiles_and_centres():
    lanes = generate_design_lanes(0.0, 7.5)         # 2 lanes of 3.6 in 7.5
    assert len(lanes) == 2
    assert all(abs(l.width - 3.6) < 1e-12 for l in lanes)
    # centred within the roadway → symmetric about 3.75
    assert (lanes[0].y_center + lanes[1].y_center) / 2 == pytest.approx(3.75)
    assert lanes[0].y_lo >= -1e-9 and lanes[-1].y_hi <= 7.5 + 1e-9


def test_design_lane_edges():
    lane = DesignLane(y_center=2.0, width=3.6)
    assert lane.y_lo == pytest.approx(0.2)
    assert lane.y_hi == pytest.approx(3.8)


# ------------------------------------------------ governing combination
def test_multi_lane_governing_matches_formula():
    m, nid, (nx, ny, dx, dy) = _grillage()
    deck = [nid[(i, j)] for i in range(nx) for j in range(ny)]
    IS = InfluenceLineEngine(m).influence_surface(
        DeckSurface(deck, load_dof=2), Displacement(nid[(4, 2)], 2))
    lanes = generate_design_lanes(0.0, (ny - 1) * dy)
    veh = Vehicle2D.from_axle_train(aashto_hl93_truck())
    res = multi_lane_envelope(IS, veh, lanes, n_x=41, n_y=7)
    mins = sorted(d["min"] for d in res["per_lane"])
    cand = [multi_presence_factor(k) * sum(mins[:k])
            for k in range(1, len(mins) + 1)]
    assert res["min"] == pytest.approx(min(cand), rel=1e-12)
    assert res["min_num_lanes"] == (int(np.argmin(cand)) + 1)


def test_single_lane_applies_1_2_factor():
    m, nid, (nx, ny, dx, dy) = _grillage()
    deck = [nid[(i, j)] for i in range(nx) for j in range(ny)]
    IS = InfluenceLineEngine(m).influence_surface(
        DeckSurface(deck, load_dof=2), Displacement(nid[(4, 2)], 2))
    lane = generate_design_lanes(0.0, (ny - 1) * dy)[0]
    veh = Vehicle2D.from_axle_train(aashto_hl93_truck())
    ys = _lane_center_positions(lane, veh, 0.6, 7)
    env = moving_load_surface_envelope(IS, veh, y_positions=ys, n_x=41)
    res = multi_lane_envelope(IS, veh, [lane], n_x=41, n_y=7)
    assert res["min"] == pytest.approx(1.20 * env["min"], rel=1e-12)


# --------------------------------------------- physical superposition
def test_lane_contributions_superpose_to_direct_solve():
    """Two unit wheels, one in each lane, landing on nodes: the sum of the
    two influence-surface ordinates equals the direct two-load solve."""
    m, nid, (nx, ny, dx, dy) = _grillage()
    deck = [nid[(i, j)] for i in range(nx) for j in range(ny)]
    center = nid[(4, 2)]
    IS = InfluenceLineEngine(m).influence_surface(
        DeckSurface(deck, load_dof=2), Displacement(center, 2))
    a, b = nid[(4, 1)], nid[(4, 4)]              # one node in each half
    pa, pb = m.node(a).coords, m.node(b).coords
    pred = IS((pa[0], pa[1])) + IS((pb[0], pb[1]))
    m.reset_results(); m.number_dofs(); m.clear_loads()
    m.add_nodal_load(a, [0, 0, -1.0, 0, 0, 0])
    m.add_nodal_load(b, [0, 0, -1.0, 0, 0, 0])
    LinearStaticAnalysis(m).run()
    assert pred == pytest.approx(float(m.node(center).disp[2]), rel=1e-9)


def test_lane_center_positions_keep_wheels_inside():
    lane = DesignLane(y_center=2.0, width=3.6)
    veh = Vehicle2D(wheel_loads=[1.0, 1.0], wheel_xy=[(0.0, 0.9), (0.0, -0.9)])
    ys = _lane_center_positions(lane, veh, lane_margin=0.6, n_y=5)
    for Y in ys:                                  # outer wheels stay in lane
        assert Y - 0.9 >= lane.y_lo + 0.6 - 1e-9
        assert Y + 0.9 <= lane.y_hi - 0.6 + 1e-9
