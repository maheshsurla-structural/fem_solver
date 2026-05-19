"""Tests for LineSearchNewton.

A backtracking line search wraps each Newton step in a "halve until
the residual decreases" loop. The benefit shows up on stiff problems
where the full Newton step would overshoot — particularly elastic-
plastic sections at the elastic-plastic boundary, where the residual
function is piecewise-linear with a slope discontinuity.

Tests:

1. **Linear-elastic regression** — for a problem where vanilla Newton
   converges in 1–2 iterations, LineSearchNewton converges to the
   same answer (perhaps with extra residual evaluations from the
   backtracking step).
2. **Constructor validation** — out-of-range parameters raise.
3. **Pushover with EPP hinge** — the same problem that Phase 4
   originally needed an internal line search for. With
   ``LineSearchNewton`` at the *algorithm* level (in addition to the
   internal Newton in ``HingedBeamColumn2D``), the analysis is
   robust over a wider range of loads.
"""
import numpy as np
import pytest

from femsolver import (
    BeamColumn2D,
    BilinearMomentRotationSpring,
    DisplacementControl,
    ElasticIsotropic,
    HingedBeamColumn2D,
    LineSearchNewton,
    Model,
    NonlinearStaticAnalysis,
)


# ====================================================== constructor ==

def test_line_search_rejects_invalid_max_backtracks():
    with pytest.raises(ValueError):
        LineSearchNewton(max_backtracks=0)


def test_line_search_rejects_invalid_descent_factor():
    with pytest.raises(ValueError):
        LineSearchNewton(descent_factor=0.0)
    with pytest.raises(ValueError):
        LineSearchNewton(descent_factor=1.0)


# ====================================================== regression ==

def test_line_search_matches_newton_on_linear_problem():
    """For a linear elastic problem, line search must converge to the
    same state as standard Newton — the full Newton step is already a
    descent direction so no backtracking happens."""
    E, A, Iz, L, P = 2.0e11, 1.0e-2, 8.333e-6, 3.0, 1.0e3
    mat = ElasticIsotropic(1, E=E, nu=0.3)

    def build():
        m = Model(ndm=2, ndf=3); m.add_material(mat)
        m.add_node(1, 0.0, 0.0); m.add_node(2, L, 0.0)
        m.add_element(BeamColumn2D(1, (1, 2), mat, A, Iz))
        m.fix(1, [1, 1, 1])
        m.add_nodal_load(2, [0.0, -P, 0.0])
        return m

    m_newton = build()
    NonlinearStaticAnalysis(
        m_newton, num_steps=3, dlambda=1.0 / 3,
        algorithm="newton", tol=1e-8, max_iter=10,
    ).run()
    v_newton = m_newton.node(2).disp[1]

    m_ls = build()
    NonlinearStaticAnalysis(
        m_ls, num_steps=3, dlambda=1.0 / 3,
        algorithm="line_search", tol=1e-8, max_iter=10,
    ).run()
    v_ls = m_ls.node(2).disp[1]

    assert v_ls == pytest.approx(v_newton, rel=1e-10)


# ====================================================== EPP pushover ==

def test_line_search_handles_bilinear_hinge_pushover_under_load_control():
    """Bilinear hinge pushover with load control. The standard Newton
    algorithm converges fine here; LineSearchNewton must also converge,
    and produce the same final state.

    Background — line search is *incompatible* with path-following
    integrators (DisplacementControl, ArcLength) because scaling
    ``du`` would violate the constraint that already updated
    ``lambda``. Those integrators advertise
    ``supports_line_search = False``; the algorithm respects the flag
    and falls back to the full Newton step. So we exercise line search
    with load control here.
    """
    E, A, Iz, L = 2.0e11, 1.0e-2, 8.333e-6, 3.0
    K_h = 4.0 * E * Iz / L
    My = 5.0e3
    b_post = 0.1
    mat = ElasticIsotropic(1, E=E, nu=0.3)

    def build():
        m = Model(ndm=2, ndf=3); m.add_material(mat)
        m.add_node(1, 0.0, 0.0); m.add_node(2, L, 0.0)
        h = BilinearMomentRotationSpring(K0=K_h, My=My, b=b_post)
        elem = HingedBeamColumn2D(1, (1, 2), mat, A, Iz, hinge_i=h)
        m.add_element(elem)
        m.fix(1, [1, 1, 1])
        P_yield = My / L
        P_max = 1.3 * P_yield
        m.add_nodal_load(2, [0.0, -P_max, 0.0])
        return m, elem

    m_newton, elem_n = build()
    NonlinearStaticAnalysis(
        m_newton, num_steps=10, dlambda=0.1, algorithm="newton",
        tol=1e-6, max_iter=30,
    ).run()

    m_ls, elem_ls = build()
    NonlinearStaticAnalysis(
        m_ls, num_steps=10, dlambda=0.1, algorithm="line_search",
        tol=1e-6, max_iter=30,
    ).run()

    # Same final state under both algorithms
    np.testing.assert_allclose(
        m_newton.node(2).disp, m_ls.node(2).disp, rtol=1e-6, atol=1e-9
    )
    assert elem_n.hinge_i.theta_p_committed == pytest.approx(
        elem_ls.hinge_i.theta_p_committed, rel=1e-6
    )


def test_line_search_skipped_for_displacement_control():
    """Line search must NOT engage when the integrator is
    displacement-controlled, even if the algorithm is LineSearchNewton.
    The analysis should still complete and reach the prescribed tip
    displacement — i.e. the line-search no-op fall-back is correct.
    """
    E, A, Iz, L = 2.0e11, 1.0e-2, 8.333e-6, 3.0
    mat = ElasticIsotropic(1, E=E, nu=0.3)
    m = Model(ndm=2, ndf=3); m.add_material(mat)
    m.add_node(1, 0.0, 0.0); m.add_node(2, L, 0.0)
    m.add_element(BeamColumn2D(1, (1, 2), mat, A, Iz))
    m.fix(1, [1, 1, 1])
    m.add_nodal_load(2, [0.0, -1.0, 0.0])

    integrator = DisplacementControl(node_tag=2, dof_index=1, du_step=-1.0e-4)
    NonlinearStaticAnalysis(
        m, num_steps=5, integrator=integrator,
        algorithm="line_search", tol=1e-8, max_iter=10,
    ).run()
    # The prescribed displacement (5 steps x -1e-4) must be exactly met.
    assert m.node(2).disp[1] == pytest.approx(-5.0e-4, rel=1e-10)
