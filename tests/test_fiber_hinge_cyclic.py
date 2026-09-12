"""Fiber-hinge plan P8 — cyclic benchmark + energy dissipation.

Hold P = 2400 kip axial, then apply a stepped reversed-cyclic lateral
displacement protocol and trace the base-shear vs tip-displacement hysteresis
(plan §7.3). Uses the cyclic ``ReinforcingSteelKinematic`` steel (P3) and the
``stepped_cyclic`` protocol (P6), driven through one displacement-control
analysis via a per-step increment schedule.

Displacement-based fiber elements are used for the cyclic run: the force-based
element's flexibility goes singular at cyclic reversals (documented in
``beam_force.py`` / plan §8), whereas displacement-based elements are robust
there and converge to the same capacity with a modest mesh.
"""
from __future__ import annotations

import numpy as np
import pytest

from femsolver import (
    BeamColumn2DCorotational,
    ConcreteMander,
    ElasticIsotropic,
    Model,
    NonlinearStaticAnalysis,
    StagedAnalysis,
    rc_circular_column_section,
    stepped_cyclic,
)
from femsolver.analysis.static_integrator import DisplacementControl
from femsolver.materials.uniaxial import ReinforcingSteelKinematic
from femsolver.sections.analysis import mander_confined_circular

D, COVER, EC, FC, EPS_C0, ES, FY, L = (
    84.0, 2.0, 3605.0, 5.0, 0.002219, 29000.0, 68.0, 49.0)


def _section():
    conf = mander_confined_circular(
        fco=FC, eps_co=EPS_C0, D_core=79.0, hoop_area=0.79, hoop_spacing=6.0,
        fyh=68.0, rho_long=0.0257, hoop_type="hoop", clear_spacing=5.0)
    return rc_circular_column_section(
        diameter=D, cover=COVER, core_concrete=conf.to_material(Ec=EC),
        cover_concrete=ConcreteMander(fpc=FC, eps_c0=EPS_C0, Ec=EC),
        n_bars=56, bar_area=2.25,
        steel=ReinforcingSteelKinematic(ES, FY, 95.0, 0.0075, 0.09),
        n_core_rings=8, n_cover_rings=2, n_wedges=24)


def _run_cyclic(n_el=2, amplitudes=(0.5, 1.0), scale=0.10, ppc=12):
    mat = ElasticIsotropic(1, E=EC, nu=0.2)
    m = Model(ndm=2, ndf=3)
    m.add_material(mat)
    for i in range(n_el + 1):
        m.add_node(i + 1, i * L / n_el, 0.0)
    for i in range(n_el):
        m.add_element(BeamColumn2DCorotational(
            i + 1, (i + 1, i + 2), mat, section=_section()))
    m.fix(1, [1, 1, 1])
    tip = n_el + 1

    targets = stepped_cyclic(amplitudes=amplitudes, scale=scale, cycles=1,
                             pts_per_cycle=ppc)
    du = -np.diff(targets)                    # lateral increments, -y direction

    sa = StagedAnalysis(m)
    sa.add_stage("axial", lambda mm: (
        mm.add_nodal_load(tip, [-2400.0, 0.0, 0.0]),
        NonlinearStaticAnalysis(mm, num_steps=8, dlambda=0.125,
                                integrator="load_control", tol=1e-8))[1])
    sa.add_stage("cyclic", lambda mm: (
        mm.add_nodal_load(tip, [0.0, -1.0, 0.0]),
        NonlinearStaticAnalysis(
            mm, num_steps=len(du),
            integrator=DisplacementControl(tip, 1, du),
            track=(tip, 1), tol=1e-6, max_iter=80))[1])
    out = sa.run()
    u = np.array(out["cyclic"]["tracked"])
    V = -np.array(out["cyclic"]["lambdas"])   # base shear (ref load = -1 kip)
    return m, u, V, ppc


def _loop_energy(u, V, cyc, ppc):
    """Dissipated energy of cycle ``cyc`` (0-indexed) = area of its loop."""
    lo, hi = cyc * ppc, (cyc + 1) * ppc
    uu = np.concatenate([[0.0 if cyc == 0 else u[lo - 1]], u[lo:hi]])
    vv = np.concatenate([[0.0 if cyc == 0 else V[lo - 1]], V[lo:hi]])
    return abs(np.sum(0.5 * (vv[1:] + vv[:-1]) * np.diff(uu)))


def test_cyclic_hysteresis_and_energy():
    m, u, V, ppc = _run_cyclic()
    # axial preload held through the whole cyclic run
    assert abs(m.nodes[1].reaction[0]) == pytest.approx(2400.0, rel=1e-3)
    # genuine reversals: shear swings both ways
    assert V.min() < 0.0 < V.max()
    # symmetric-ish protocol reaches both displacement extremes
    assert u.max() > 0.0 > u.min()

    # peak shear grows from the small-amplitude loop to the large one
    peak0 = np.max(np.abs(V[0:ppc]))
    peak1 = np.max(np.abs(V[ppc:2 * ppc]))
    assert peak1 > peak0

    # dissipated energy is positive and larger for the larger amplitude
    e0 = _loop_energy(u, V, 0, ppc)
    e1 = _loop_energy(u, V, 1, ppc)
    assert e0 > 0.0 and e1 > 0.0
    assert e1 > e0


def test_schedule_drives_displacement_path():
    """A DisplacementControl increment schedule steps the control DOF through
    the requested targets (unit check of the P8 driver mechanism)."""
    E, A, Lb = 30000.0, 10.0, 100.0
    from femsolver import Truss2D
    m = Model(ndm=2, ndf=2)
    mat = ElasticIsotropic(1, E=E, nu=0.3)
    m.add_material(mat)
    m.add_node(1, 0.0, 0.0)
    m.add_node(2, Lb, 0.0)
    m.add_element(Truss2D(1, (1, 2), mat, A))
    m.fix(1, [1, 1])
    m.fix(2, [0, 1])
    m.add_nodal_load(2, [1.0, 0.0])
    targets = np.array([0.0, 0.02, -0.01, 0.03])
    du = np.diff(targets)
    r = NonlinearStaticAnalysis(
        m, num_steps=len(du), integrator=DisplacementControl(2, 0, du),
        track=(2, 0), tol=1e-10).run()
    assert np.allclose(r["tracked"], targets[1:], atol=1e-10)
