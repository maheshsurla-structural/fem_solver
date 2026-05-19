"""Newton-Raphson nonlinear static analysis.

Validates the iteration machinery against:

- Linear elements: NR converges in 1 iteration to the LinearStaticAnalysis answer.
- Corotational truss: at small displacements, identical to linear truss.
- Mises shallow truss: traced equilibrium path matches the analytical
  ``P(w) = 2 E A eps (h - w) / L`` curve before the limit point.
- Convergence failure: insufficient iterations / mechanism raise the right
  error.
"""
from __future__ import annotations

import math

import numpy as np
import pytest

from femsolver import (
    BeamColumn2D,
    ElasticIsotropic,
    LinearStaticAnalysis,
    Model,
    NonlinearStaticAnalysis,
    NormDispIncr,
    NormUnbalance,
    NotConvergedError,
    Truss2D,
    Truss2DCorotational,
)


# ---------------------------------------------------------------------------
# linear elements via NR — should converge in one iteration


def _three_bar_model() -> Model:
    """Three-bar pin-jointed truss (textbook problem). Equilateral triangle,
    pin/roller bottom chord, vertical load at apex. Symmetric, statically
    determinate, well-conditioned — the same configuration used in
    ``test_truss2d_three_bar``."""
    E, A, L_chord, P = 2.0e11, 1.0e-4, 2.0, 1.0e4
    m = Model(ndm=2, ndf=2)
    m.add_node(1, 0.0, 0.0)
    m.add_node(2, L_chord, 0.0)
    m.add_node(3, L_chord / 2.0, L_chord * math.sqrt(3.0) / 2.0)
    mat = ElasticIsotropic(1, E=E, nu=0.3)
    m.add_material(mat)
    m.add_element(Truss2D(1, (1, 3), mat, A))
    m.add_element(Truss2D(2, (2, 3), mat, A))
    m.add_element(Truss2D(3, (1, 2), mat, A))
    m.fix(1, [1, 1])
    m.fix(2, [0, 1])
    m.add_nodal_load(3, [0.0, -P])
    return m


def test_nr_linear_truss_converges_in_one_iteration():
    """Three-bar truss — NR with linear elements converges in exactly one
    iteration per step and matches the LinearStaticAnalysis answer."""
    m = _three_bar_model()
    a = NonlinearStaticAnalysis(
        m, num_steps=5, dlambda=0.2,
        convergence="unbalance", tol=1e-6,
    )
    info = a.run()
    assert info["final_lambda"] == pytest.approx(1.0)
    # one iteration per step — the linear system is solved exactly on the
    # first solve, so the next residual is at machine precision and passes
    # the test on iteration 1.
    assert all(c == 1 for c in info["iter_counts"])

    m2 = _three_bar_model()
    LinearStaticAnalysis(m2).run()
    np.testing.assert_allclose(m.node(3).disp, m2.node(3).disp, rtol=1e-10)


def test_nr_linear_beam_matches_linear_static():
    """Cantilever beam end load — NR with BeamColumn2D matches linear static."""
    m = Model(ndm=2, ndf=3)
    mat = ElasticIsotropic(1, E=2.0e11, nu=0.3)
    m.add_material(mat)
    L = 2.0
    m.add_node(1, 0.0, 0.0)
    m.add_node(2, L, 0.0)
    m.fix(1, [1, 1, 1])
    m.add_element(BeamColumn2D(1, (1, 2), mat, area=1e-2, Iz=8.333e-5))
    m.add_nodal_load(2, [0.0, -1e3, 0.0])
    a = NonlinearStaticAnalysis(m, num_steps=4, dlambda=0.25, tol=1e-10)
    a.run()

    m2 = Model(ndm=2, ndf=3)
    m2.add_material(mat)
    m2.add_node(1, 0.0, 0.0)
    m2.add_node(2, L, 0.0)
    m2.fix(1, [1, 1, 1])
    m2.add_element(BeamColumn2D(1, (1, 2), mat, area=1e-2, Iz=8.333e-5))
    m2.add_nodal_load(2, [0.0, -1e3, 0.0])
    LinearStaticAnalysis(m2).run()
    np.testing.assert_allclose(m.node(2).disp, m2.node(2).disp, rtol=1e-12)


# ---------------------------------------------------------------------------
# corotational truss — small displacement matches linear


def test_corotational_truss_small_disp_matches_linear():
    """At infinitesimal axial load, the corotational truss should agree
    with the linear truss to many digits."""
    m = Model(ndm=2, ndf=2)
    mat = ElasticIsotropic(1, E=2.0e11, nu=0.3)
    m.add_material(mat)
    m.add_node(1, 0.0, 0.0)
    m.add_node(2, 1.0, 0.0)
    m.fix(1, [1, 1])
    m.fix(2, [0, 1])  # only horizontal motion at node 2
    m.add_element(Truss2DCorotational(1, (1, 2), mat, area=1e-4))
    m.add_nodal_load(2, [10.0, 0.0])  # tiny load -> infinitesimal displacement

    # tol limited by floating-point cancellation in (L - L0)/L0 — for this
    # geometry, machine precision in the residual is ~ EA * eps_mach * eps,
    # which works out to a few times 1e-9 for the chosen scales.
    NonlinearStaticAnalysis(m, num_steps=1, dlambda=1.0, tol=1e-6).run()
    expected = 10.0 * 1.0 / (2.0e11 * 1.0e-4)  # u = P L / (E A)
    assert m.node(2).disp[0] == pytest.approx(expected, rel=1e-8)


# ---------------------------------------------------------------------------
# Mises shallow truss — equilibrium path before the limit point


def _mises_analytical_load(w: float, *, B: float, h: float, EA: float) -> float:
    """Apex load for a Mises (shallow) truss with two members of initial
    half-span B and rise h, at downward apex displacement w (engineering
    strain, downward-positive load)."""
    L0 = math.sqrt(B * B + h * h)
    L = math.sqrt(B * B + (h - w) ** 2)
    eps = (L - L0) / L0
    N = EA * eps
    # P_ext (down) = -2 N sin theta_current; sin theta = (h-w)/L
    return -2.0 * N * (h - w) / L


def test_mises_truss_traces_analytical_curve():
    """Apex displacement at each load step matches the analytical Mises
    truss equilibrium relation. We trace up to ~50% of the limit-point
    load to stay well in the stable region of the path."""
    B = 10.0
    h = 1.0
    EA = 1.0e6
    # peak (limit-point) load for these parameters is roughly 290 N at
    # w ~= 0.4 h. Pick a max load ~120 N (well below the peak).
    P_max = 120.0
    n_steps = 12
    dlambda = 1.0 / n_steps

    m = Model(ndm=2, ndf=2)
    mat = ElasticIsotropic(1, E=EA, nu=0.0)  # fold E*A into E by setting A=1
    m.add_material(mat)
    m.add_node(1, 0.0, 0.0)         # left support
    m.add_node(2, 2.0 * B, 0.0)     # right support
    m.add_node(3, B, h)             # apex
    m.fix(1, [1, 1])
    m.fix(2, [1, 1])
    # apex constrained to vertical motion (the structure is symmetric anyway,
    # but pinning u_x removes a small unphysical lateral drift due to FP)
    m.fix(3, [1, 0])
    m.add_element(Truss2DCorotational(1, (1, 3), mat, area=1.0))
    m.add_element(Truss2DCorotational(2, (3, 2), mat, area=1.0))
    m.add_nodal_load(3, [0.0, -P_max])  # downward

    a = NonlinearStaticAnalysis(
        m, num_steps=n_steps, dlambda=dlambda,
        convergence="unbalance", tol=1e-8, max_iter=30,
        track=(3, 1),
    )
    a.run()

    # at each converged lambda, compute the analytical P required for the
    # apex displacement reached, and compare it to the applied load
    for lam, w_observed in zip(a.lambdas, a.tracked):
        w = -w_observed  # tracked is uy, downward is negative
        P_applied = lam * P_max
        P_predicted = _mises_analytical_load(w, B=B, h=h, EA=EA)
        # ratio test (absolute may be small early on)
        assert P_applied == pytest.approx(P_predicted, rel=5e-4, abs=1e-6)


def test_modified_newton_also_converges_on_linear():
    m = _three_bar_model()
    a = NonlinearStaticAnalysis(
        m, num_steps=3, dlambda=1.0 / 3.0,
        algorithm="modified_newton", tol=1e-6,
    )
    info = a.run()
    assert info["final_lambda"] == pytest.approx(1.0)


# ---------------------------------------------------------------------------
# error paths


def test_nr_max_iter_too_small_raises():
    """A tiny ``max_iter`` cannot satisfy a stringent ``tol`` on the
    Mises truss — analysis should raise ``NotConvergedError``."""
    B = 10.0
    h = 1.0
    EA = 1.0e6
    m = Model(ndm=2, ndf=2)
    mat = ElasticIsotropic(1, E=EA, nu=0.0)
    m.add_material(mat)
    m.add_node(1, 0.0, 0.0)
    m.add_node(2, 2.0 * B, 0.0)
    m.add_node(3, B, h)
    m.fix(1, [1, 1])
    m.fix(2, [1, 1])
    m.fix(3, [1, 0])
    m.add_element(Truss2DCorotational(1, (1, 3), mat, area=1.0))
    m.add_element(Truss2DCorotational(2, (3, 2), mat, area=1.0))
    m.add_nodal_load(3, [0.0, -120.0])

    a = NonlinearStaticAnalysis(
        m, num_steps=10, dlambda=0.1,
        convergence=NormUnbalance(tol=1e-12, max_iter=2),
    )
    with pytest.raises(NotConvergedError):
        a.run()


def test_unknown_algorithm_raises():
    m = Model(ndm=2, ndf=2)
    mat = ElasticIsotropic(1, E=1.0, nu=0.0)
    m.add_material(mat)
    m.add_node(1, 0.0, 0.0)
    with pytest.raises(ValueError, match="unknown algorithm"):
        NonlinearStaticAnalysis(m, num_steps=1, algorithm="quasi_newton")


def test_unknown_convergence_raises():
    m = Model(ndm=2, ndf=2)
    mat = ElasticIsotropic(1, E=1.0, nu=0.0)
    m.add_material(mat)
    m.add_node(1, 0.0, 0.0)
    with pytest.raises(ValueError, match="unknown convergence"):
        NonlinearStaticAnalysis(m, num_steps=1, convergence="random")


# ---------------------------------------------------------------------------
# convergence-test variants


def test_norm_disp_incr_test():
    """Same problem, NormDispIncr should also work."""
    m = _three_bar_model()
    a = NonlinearStaticAnalysis(
        m, num_steps=2, dlambda=0.5,
        convergence=NormDispIncr(tol=1e-12, max_iter=10),
    )
    a.run()
    assert m.node(3).disp[1] != 0.0
