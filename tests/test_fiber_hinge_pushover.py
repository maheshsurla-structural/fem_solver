"""Fiber-hinge plan P6 — staged analysis (G6), protocols (G7), pushover.

Covers the displacement/load protocol generators, the ``StagedAnalysis``
continuation (hold a prior stage's load constant while a later stage runs), and
the benchmark monotonic pushover: hold P = 2400 kip axial, then push the tip
laterally and trace base shear vs tip displacement (plan §7.2).
"""
from __future__ import annotations

import math

import numpy as np
import pytest

from femsolver import (
    ConcreteMander,
    ElasticIsotropic,
    ForceBeamColumn2DCorotational,
    Model,
    NonlinearStaticAnalysis,
    StagedAnalysis,
    Truss2D,
    from_time_function,
    monotonic,
    rc_circular_column_section,
    stepped_cyclic,
)
from femsolver.analysis.static_integrator import DisplacementControl
from femsolver.materials.uniaxial import UniaxialReinforcingSteel
from femsolver.sections.analysis import mander_confined_circular


# ----------------------------------------------------------- G7 protocols

def test_monotonic():
    r = monotonic(10.0, 5)
    assert len(r) == 6
    assert r[0] == 0.0 and r[-1] == 10.0
    assert np.allclose(np.diff(r), 2.0)


def test_stepped_cyclic():
    r = stepped_cyclic(amplitudes=(0.5, 1.0), scale=2.0, cycles=1,
                       pts_per_cycle=8)
    assert r[0] == 0.0
    assert abs(r[-1]) < 1e-12                       # ends back at zero
    assert r.max() == pytest.approx(2.0)           # peak = 1.0 * scale 2.0
    assert r.min() == pytest.approx(-2.0)
    # growing amplitude: the first cycle peaks at 1.0, the second at 2.0
    first_half = r[: len(r) // 2]
    assert first_half.max() == pytest.approx(1.0)   # 0.5 * scale


def test_from_time_function():
    r = from_time_function([0.0, 1.0, 2.0], [0.0, 10.0, 0.0], dt=0.5)
    assert np.allclose(r, [0.0, 5.0, 10.0, 5.0, 0.0])


# ------------------------------------------------- G6 staged continuation

def test_staged_holds_prior_load():
    """A two-stage load-control run on an elastic truss: stage 2 continues
    from stage 1's state AND holds stage 1's load, so the final displacement
    reflects the SUM of both loads (not just stage 2's)."""
    E, A, L = 30000.0, 10.0, 100.0
    k = E * A / L
    F1, F2 = 500.0, 300.0
    m = Model(ndm=2, ndf=2)
    mat = ElasticIsotropic(1, E=E, nu=0.3)
    m.add_material(mat)
    m.add_node(1, 0.0, 0.0)
    m.add_node(2, L, 0.0)
    m.add_element(Truss2D(1, (1, 2), mat, A))
    m.fix(1, [1, 1])
    m.fix(2, [0, 1])                                # only node-2 x is free

    sa = StagedAnalysis(m)
    sa.add_stage("a", lambda mm: (
        mm.add_nodal_load(2, [F1, 0.0]),
        NonlinearStaticAnalysis(mm, num_steps=1, dlambda=1.0))[1])
    sa.add_stage("b", lambda mm: (
        mm.add_nodal_load(2, [F2, 0.0]),
        NonlinearStaticAnalysis(mm, num_steps=1, dlambda=1.0))[1])
    sa.run()

    # both loads are applied at the end -> u = (F1 + F2)/k
    assert m.nodes[2].disp[0] == pytest.approx((F1 + F2) / k, rel=1e-8)
    assert m.nodes[1].reaction[0] == pytest.approx(-(F1 + F2), rel=1e-8)


def test_const_force_default_is_noop():
    """A single load-control stage with StagedAnalysis matches a bare
    NonlinearStaticAnalysis (the const-force/keep-state defaults change
    nothing)."""
    E, A, L = 30000.0, 10.0, 100.0
    F = 800.0

    def build():
        m = Model(ndm=2, ndf=2)
        mat = ElasticIsotropic(1, E=E, nu=0.3)
        m.add_material(mat)
        m.add_node(1, 0.0, 0.0)
        m.add_node(2, L, 0.0)
        m.add_element(Truss2D(1, (1, 2), mat, A))
        m.fix(1, [1, 1])
        m.fix(2, [0, 1])
        return m

    m1 = build()
    m1.add_nodal_load(2, [F, 0.0])
    NonlinearStaticAnalysis(m1, num_steps=1, dlambda=1.0).run()

    m2 = build()
    StagedAnalysis(m2).add_stage("only", lambda mm: (
        mm.add_nodal_load(2, [F, 0.0]),
        NonlinearStaticAnalysis(mm, num_steps=1, dlambda=1.0))[1]).run()

    assert m2.nodes[2].disp[0] == pytest.approx(m1.nodes[2].disp[0], rel=1e-12)


# --------------------------------------------- benchmark staged pushover

def _caltrans_model():
    D, cover, Ec, fc, eps_c0, Es, fy, L = (
        84.0, 2.0, 3605.0, 5.0, 0.002219, 29000.0, 68.0, 49.0)
    conf = mander_confined_circular(
        fco=fc, eps_co=eps_c0, D_core=79.0, hoop_area=0.79, hoop_spacing=6.0,
        fyh=68.0, rho_long=0.0257, hoop_type="hoop", clear_spacing=5.0)
    sec = rc_circular_column_section(
        diameter=D, cover=cover, core_concrete=conf.to_material(Ec=Ec),
        cover_concrete=ConcreteMander(fpc=fc, eps_c0=eps_c0, Ec=Ec),
        n_bars=56, bar_area=2.25,
        steel=UniaxialReinforcingSteel(Es, fy, 95.0, 0.0075, 0.09),
        n_core_rings=8, n_cover_rings=2, n_wedges=24)
    mat = ElasticIsotropic(1, E=Ec, nu=0.2)
    m = Model(ndm=2, ndf=3)
    m.add_material(mat)
    m.add_node(1, 0.0, 0.0)
    m.add_node(2, L, 0.0)
    m.add_element(ForceBeamColumn2DCorotational(1, (1, 2), mat, section=sec))
    m.fix(1, [1, 1, 1])
    return m, L


def test_benchmark_pushover_holds_axial_and_yields():
    """Stage 1 applies P = 2400 kip; stage 2 holds it and pushes the tip
    laterally (displacement control). The axial load stays applied, the base
    shear grows with a yielding (sub-linear) curve, and base-shear * L is
    consistent with the section's moment capacity."""
    m, L = _caltrans_model()
    sa = StagedAnalysis(m)
    sa.add_stage("axial", lambda mm: (
        mm.add_nodal_load(2, [-2400.0, 0.0, 0.0]),
        NonlinearStaticAnalysis(mm, num_steps=8, dlambda=0.125,
                                integrator="load_control", tol=1e-8))[1])
    sa.add_stage("push", lambda mm: (
        mm.add_nodal_load(2, [0.0, -1.0, 0.0]),
        NonlinearStaticAnalysis(
            mm, num_steps=20, integrator=DisplacementControl(2, 1, -0.006),
            track=(2, 1), tol=1e-6, max_iter=60))[1])
    out = sa.run()

    # axial preload held throughout the lateral push
    assert abs(m.nodes[1].reaction[0]) == pytest.approx(2400.0, rel=1e-3)

    u = np.abs(np.array(out["push"]["tracked"]))
    V = np.abs(np.array(out["push"]["lambdas"]))     # ref lateral load = -1 kip
    assert len(u) == 20
    assert u[-1] == pytest.approx(0.12, rel=1e-6)
    # base shear rises monotonically
    assert np.all(np.diff(V) > 0)
    # yielding: the secant stiffness late in the push is well below the
    # initial elastic secant (the curve bends over)
    k_init = V[0] / u[0]
    k_late = (V[-1] - V[-3]) / (u[-1] - u[-3])
    assert k_late < 0.5 * k_init
    # base shear * L is within ~20% of the section M-phi peak (~33k kip-ft)
    M_from_V = V[-1] * L / 12.0                       # kip-ft
    assert 25000.0 < M_from_V < 40000.0
