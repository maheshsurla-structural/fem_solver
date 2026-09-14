"""Cable-stayed initial-force optimisation — unknown load factor (T2.2).

Validates the ULF solver: targets met to machine precision on a determined
system, physically positive stay tensions, symmetry, least-squares for an
over-determined system, member-force targets, and independent verification.
"""
from __future__ import annotations

import sys
from pathlib import Path

import numpy as np
import pytest

_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(_ROOT / "src"))

from femsolver import (BeamColumn2D, ElasticIsotropic,  # noqa: E402
                       LinearStaticAnalysis, Model, Truss2D)
from femsolver.bridges import (Cable, apply_cable_tensions,  # noqa: E402
                               unknown_load_factors)
from femsolver.bridges.moving_load import BeamForce, Displacement  # noqa: E402

E = 2.0e11


def _bridge():
    """A symmetric cable-stayed model with a **ground-anchored** pylon."""
    m = Model(ndm=2, ndf=3)
    mat = ElasticIsotropic(1, E=E, nu=0.3)
    m.add_material(mat)
    for i, x in enumerate([0, 5, 10, 15, 20, 25, 30]):
        m.add_node(i + 1, x, 0.0)                  # deck 1..7
    m.add_node(8, 15.0, 12.0)                      # pylon top
    m.add_node(9, 15.0, 0.0)                       # pylon base (on ground)
    eid = 1
    for i in range(6):
        m.add_element(BeamColumn2D(eid, (i + 1, i + 2), mat, 0.5, 0.05)); eid += 1
    m.add_element(BeamColumn2D(eid, (9, 8), mat, 1.2, 0.2)); eid += 1
    m.add_element(Truss2D(eid, (8, 3), mat, 0.01)); eid += 1
    m.add_element(Truss2D(eid, (8, 5), mat, 0.01)); eid += 1
    # both deck ends pinned → the model is left/right symmetric
    m.fix(1, [1, 1, 0]); m.fix(7, [1, 1, 0]); m.fix(9, [1, 1, 1])
    return m


def _with_dead_load(m):
    for nd in (2, 3, 4, 5, 6):
        m.add_nodal_load(nd, [0, -50e3, 0])
    return m


_CABLES = [Cable(8, 3, "cL"), Cable(8, 5, "cR")]


def test_targets_met_and_tensions_positive():
    m = _with_dead_load(_bridge())
    res = unknown_load_factors(
        m, _CABLES, [(Displacement(3, 1), 0.0), (Displacement(5, 1), 0.0)])
    assert np.max(np.abs(res.residual)) < 1e-9        # targets met
    assert np.all(res.tensions > 0)                    # cables in tension
    assert res.dead_load_response[0] < 0               # dead load sags


def test_symmetry_gives_equal_tensions():
    m = _with_dead_load(_bridge())
    res = unknown_load_factors(
        m, _CABLES, [(Displacement(3, 1), 0.0), (Displacement(5, 1), 0.0)])
    assert res.tensions[0] == pytest.approx(res.tensions[1], rel=1e-6)


def test_independent_verification():
    m = _with_dead_load(_bridge())
    res = unknown_load_factors(
        m, _CABLES, [(Displacement(3, 1), 0.0), (Displacement(5, 1), 0.0)])
    # fresh model: dead load + tuned tensions → anchors reach the target
    m2 = _with_dead_load(_bridge())
    m2.number_dofs()
    apply_cable_tensions(m2, _CABLES, res.tensions)
    LinearStaticAnalysis(m2).run()
    assert abs(m2.node(3).disp[1]) < 1e-9
    assert abs(m2.node(5).disp[1]) < 1e-9


def test_member_force_target_uses_element_recovery():
    m = _with_dead_load(_bridge())
    # a member-moment target exercises the element-recovery path; one target
    # with two cables is well-posed (under-determined → min-norm, met exactly)
    res = unknown_load_factors(m, _CABLES, [(BeamForce(3, "M", end="j"), 0.0)])
    assert np.max(np.abs(res.residual)) < 1e-6


def test_overdetermined_is_least_squares():
    m = _with_dead_load(_bridge())
    # 3 targets, 2 cables → cannot meet all exactly; lstsq minimises
    targets = [(Displacement(3, 1), 0.0), (Displacement(4, 1), 0.0),
               (Displacement(5, 1), 0.0)]
    res = unknown_load_factors(m, _CABLES, targets)
    # achieved is closer to target than doing nothing (dead load alone)
    assert np.linalg.norm(res.residual) < np.linalg.norm(
        res.dead_load_response - res.target)


def test_apply_cable_tensions_adds_loads():
    m = _bridge()
    m.number_dofs()
    apply_cable_tensions(m, _CABLES, [100e3, 100e3])
    total = sum(float(np.any(nd._load != 0.0)) for nd in m.nodes.values())
    assert total >= 3                                  # pylon top + 2 anchors


def test_guards():
    m = _with_dead_load(_bridge())
    with pytest.raises(ValueError):
        unknown_load_factors(m, [], [(Displacement(3, 1), 0.0)])
    with pytest.raises(ValueError):
        unknown_load_factors(m, _CABLES, [])
