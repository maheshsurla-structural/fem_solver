"""Engine convergence-robustness (plan §8).

Force-based / fiber-hinge elements leave an irreducible residual "noise floor"
in the assembled unbalance (their inner state-determination converges only to
its own tolerance). With an absolute ``NormUnbalance`` tolerance below that
floor the outer Newton residual chatters forever while the displacement
correction has already collapsed to machine zero. These tests lock:

1. the displacement-stagnation safety in ``NormUnbalance`` (unit level), and
2. a deep force-based pushover that used to fail with a sub-noise-floor
   tolerance now converging (integration level).
"""
from __future__ import annotations

import numpy as np
import pytest

from femsolver import (
    ConcreteMander,
    ElasticIsotropic,
    ForceBeamColumn2DCorotational,
    Model,
    NonlinearStaticAnalysis,
    NormUnbalance,
    rc_circular_column_section,
)
from femsolver.analysis.static_integrator import DisplacementControl
from femsolver.materials.uniaxial import UniaxialReinforcingSteel
from femsolver.sections.analysis import mander_confined_circular


# ------------------------------------------------------------ unit level

def test_stagnation_converges_below_noise_floor():
    """A collapsed correction with a residual far below the step's initial
    imbalance is accepted (the noise-floor chatter case)."""
    t = NormUnbalance(tol=1e-10, max_iter=50)
    assert t.check(np.array([1e-2]), np.array([1e-3]), 1) is False   # refs set
    # du collapsed to ~1e-11 of du_ref, residual to ~1e-7 of r_ref -> converged
    assert t.check(np.array([1e-9]), np.array([1e-14]), 2) is True


def test_stagnation_guard_rejects_stuck_iteration():
    """A collapsed correction with a still-large residual (stuck / singular
    tangent) is NOT accepted -- the residual guard holds."""
    t = NormUnbalance(tol=1e-10, max_iter=50)
    t.check(np.array([1e-2]), np.array([1e-3]), 1)
    assert t.check(np.array([5e-3]), np.array([1e-14]), 2) is False


def test_stagnation_can_be_disabled():
    """With ``stag_du=0`` the test is the pure ``||R|| < tol`` criterion."""
    t = NormUnbalance(tol=1e-10, max_iter=50, stag_du=0.0)
    t.check(np.array([1e-2]), np.array([1e-3]), 1)
    assert t.check(np.array([1e-9]), np.array([1e-14]), 2) is False   # 1e-9 > tol
    assert t.check(np.array([1e-11]), np.array([1e-14]), 3) is True   # < tol


def test_normal_convergence_unaffected():
    """When the residual genuinely reaches tol the test converges as before,
    regardless of the correction size."""
    t = NormUnbalance(tol=1e-6, max_iter=50)
    assert t.check(np.array([1e-7]), np.array([1.0]), 1) is True


# ------------------------------------------------------ integration level

def test_force_based_pushover_converges_below_noise_floor():
    """A force-based fiber pushover with an outer tolerance below the element's
    state-determination noise floor used to chatter to NotConvergedError; with
    the stagnation safety it converges the whole run."""
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
    m.add_nodal_load(2, [0.0, -1.0, 0.0])
    # tol = 1e-10 is below the ~1e-9 kip state-determination noise floor
    r = NonlinearStaticAnalysis(
        m, num_steps=25, integrator=DisplacementControl(2, 1, -0.006),
        track=(2, 1), tol=1e-10, max_iter=30).run()
    u = np.abs(np.array(r["tracked"]))
    assert u[-1] == pytest.approx(0.15, rel=1e-6)     # full run completed
    assert np.max(np.abs(np.array(r["lambdas"]))) > 5000.0   # yielded
