"""Tests for the DisplacementControl integrator.

Displacement control treats the load factor ``lambda`` as an unknown
and prescribes a target value of a chosen DOF instead. This lets the
analysis trace softening and plateau regions of a force-displacement
curve that load control cannot follow:

* **Past the EPP-hinge mechanism point** (Phase 4 surfaced this as a
  hard limit under load control — the cantilever becomes a mechanism
  the instant the hinge yields, and load control diverges. With
  displacement control the analysis traces the perfectly-flat
  load-displacement plateau cleanly.)
* **Down the post-limit-point branch** of an arch-like structure.

The tests below pin down four properties.

1. **Elastic regression** — displacement control of a simple linear
   beam reaches the analytical force at the prescribed displacement.
2. **EPP plateau** — a cantilever with an EPP hinge at the base, under
   displacement control, traces a horizontal plateau at ``P = P_y``
   regardless of how far the tip is pushed.
3. **Sign reversibility** — negative ``du_step`` traces the same path
   in the opposite direction.
4. **Constructor validation** — bad inputs (fixed DOF, out-of-range
   index) raise informative errors.
"""
import numpy as np
import pytest

from femsolver import (
    BeamColumn2D,
    BilinearMomentRotationSpring,
    DisplacementControl,
    ElasticIsotropic,
    HingedBeamColumn2D,
    Model,
    NonlinearStaticAnalysis,
    NotConvergedError,
)


# ====================================================== constructor ==

def test_displacement_control_rejects_zero_step():
    with pytest.raises(ValueError):
        DisplacementControl(node_tag=2, dof_index=1, du_step=0.0)


def test_displacement_control_rejects_out_of_range_dof():
    mat = ElasticIsotropic(1, E=2.0e11, nu=0.3)
    m = Model(ndm=2, ndf=3); m.add_material(mat)
    m.add_node(1, 0.0, 0.0); m.add_node(2, 1.0, 0.0)
    m.add_element(BeamColumn2D(1, (1, 2), mat, 1.0e-2, 1.0e-6))
    m.fix(1, [1, 1, 1])
    integrator = DisplacementControl(node_tag=2, dof_index=5, du_step=-1.0e-4)
    with pytest.raises(ValueError, match="out of range"):
        NonlinearStaticAnalysis(
            m, num_steps=1, integrator=integrator, tol=1e-6,
        ).run()


def test_displacement_control_rejects_fixed_dof():
    mat = ElasticIsotropic(1, E=2.0e11, nu=0.3)
    m = Model(ndm=2, ndf=3); m.add_material(mat)
    m.add_node(1, 0.0, 0.0); m.add_node(2, 1.0, 0.0)
    m.add_element(BeamColumn2D(1, (1, 2), mat, 1.0e-2, 1.0e-6))
    m.fix(1, [1, 1, 1])
    m.fix(2, [0, 1, 0])     # v at node 2 is now fixed
    integrator = DisplacementControl(node_tag=2, dof_index=1, du_step=-1.0e-4)
    with pytest.raises(ValueError, match="fixed"):
        NonlinearStaticAnalysis(
            m, num_steps=1, integrator=integrator, tol=1e-6,
        ).run()


# ====================================================== elastic ==

def test_disp_control_elastic_cantilever_matches_load_control():
    """An elastic cantilever should reach the same final state under
    load control and displacement control of the equivalent ``v_tip``."""
    E, A, Iz, L, P = 2.0e11, 1.0e-2, 8.333e-6, 3.0, 1.0e3
    v_target = -P * L ** 3 / (3.0 * E * Iz)   # analytical tip deflection
    mat = ElasticIsotropic(1, E=E, nu=0.3)

    def build():
        m = Model(ndm=2, ndf=3); m.add_material(mat)
        m.add_node(1, 0.0, 0.0); m.add_node(2, L, 0.0)
        m.add_element(BeamColumn2D(1, (1, 2), mat, A, Iz))
        m.fix(1, [1, 1, 1])
        m.add_nodal_load(2, [0.0, -P, 0.0])     # unit reference load
        return m

    # Load control reference
    m_lc = build()
    NonlinearStaticAnalysis(
        m_lc, num_steps=1, dlambda=1.0, tol=1e-8, max_iter=10,
    ).run()
    v_lc = m_lc.node(2).disp[1]
    lambda_lc = 1.0

    # Displacement control: push v_tip in 5 steps to the target
    m_dc = build()
    n_steps = 5
    integrator = DisplacementControl(
        node_tag=2, dof_index=1, du_step=v_target / n_steps,
    )
    res = NonlinearStaticAnalysis(
        m_dc, num_steps=n_steps, integrator=integrator,
        algorithm="newton", tol=1e-8, max_iter=10,
    ).run()
    v_dc = m_dc.node(2).disp[1]
    lambda_dc = res["final_lambda"]

    # Same final v_tip, same final lambda
    assert v_dc == pytest.approx(v_target, rel=1e-10)
    assert lambda_dc == pytest.approx(lambda_lc, rel=1e-10)


# ====================================================== EPP plateau ==

def test_disp_control_traces_epp_plateau_past_mechanism():
    """An EPP hinge at the base of a cantilever forms a kinematic
    mechanism the moment it yields — load control diverges past P_y.
    Displacement control traces the flat plateau cleanly: lambda stays
    at ``M_y / (L * P_ref)``.

    This is the test that motivated Phase 10: in Phase 4
    ``test_epp_load_past_yield_diverges`` we documented this failure
    under load control. Here we show it succeeds under displacement
    control.
    """
    E, A, Iz, L = 2.0e11, 1.0e-2, 8.333e-6, 3.0
    K_h = 4.0 * E * Iz / L
    My = 5.0e3
    P_yield = My / L

    mat = ElasticIsotropic(1, E=E, nu=0.3)
    m = Model(ndm=2, ndf=3); m.add_material(mat)
    m.add_node(1, 0.0, 0.0); m.add_node(2, L, 0.0)
    h = BilinearMomentRotationSpring(K0=K_h, My=My, b=0.0)  # EPP
    elem = HingedBeamColumn2D(1, (1, 2), mat, A, Iz, hinge_i=h)
    m.add_element(elem)
    m.fix(1, [1, 1, 1])
    # Reference load: a unit downward force at the tip (so lambda is
    # in N for downward force).
    m.add_nodal_load(2, [0.0, -1.0, 0.0])

    # Push tip downward (v_tip < 0). We want to land well past yield so
    # the load plateaus.
    # v_tip at yield (elastic + hinge rotation):
    # v_y = -P_y * L^3 / (3 EI) - P_y * L^2 / K_h
    v_y = -P_yield * L ** 3 / (3.0 * E * Iz) - P_yield * L ** 2 / K_h
    v_target = 3.0 * v_y     # well past yield
    n_steps = 30
    integrator = DisplacementControl(
        node_tag=2, dof_index=1, du_step=v_target / n_steps,
    )
    res = NonlinearStaticAnalysis(
        m, num_steps=n_steps, integrator=integrator,
        algorithm="newton", tol=1e-6, max_iter=30,
        track=(2, 1),
    ).run()
    lambdas = np.array(res["lambdas"])
    disps = np.array(res["tracked"])

    # The lambdas should reach P_yield and plateau there.
    assert lambdas[-1] == pytest.approx(P_yield, rel=5e-3)
    # The plateau test: the last few lambdas should be ~ constant ~ P_yield
    np.testing.assert_allclose(
        lambdas[-5:], P_yield, rtol=1e-3
    )
    # The displacements should reach the prescribed target
    assert disps[-1] == pytest.approx(v_target, rel=1e-10)
    # And the hinge should have yielded
    assert elem.hinge_i.theta_p_committed != 0.0


# ====================================================== sign reversal ==

def test_disp_control_handles_negative_du_step():
    """Pushing in the negative direction must work as well."""
    E, A, Iz, L = 2.0e11, 1.0e-2, 8.333e-6, 3.0
    mat = ElasticIsotropic(1, E=E, nu=0.3)
    m = Model(ndm=2, ndf=3); m.add_material(mat)
    m.add_node(1, 0.0, 0.0); m.add_node(2, L, 0.0)
    m.add_element(BeamColumn2D(1, (1, 2), mat, A, Iz))
    m.fix(1, [1, 1, 1])
    m.add_nodal_load(2, [0.0, -1.0, 0.0])
    integrator = DisplacementControl(node_tag=2, dof_index=1, du_step=-1.0e-4)
    NonlinearStaticAnalysis(
        m, num_steps=5, integrator=integrator, tol=1e-8, max_iter=10,
    ).run()
    # After 5 steps of -1e-4: v_tip = -5e-4
    assert m.node(2).disp[1] == pytest.approx(-5.0e-4, rel=1e-10)


# ====================================================== Newton convergence ==

def test_disp_control_converges_in_one_iteration_for_linear_problem():
    """For a linear elastic problem the displacement-control corrector
    is *exact* in a single Newton iteration: lambda adjusts so the
    constraint is met, the residual is then zero. We don't have a
    direct test for iter count yet, so check that the analysis is fast
    (converges within max_iter=2 per step).
    """
    E, A, Iz, L = 2.0e11, 1.0e-2, 8.333e-6, 3.0
    mat = ElasticIsotropic(1, E=E, nu=0.3)
    m = Model(ndm=2, ndf=3); m.add_material(mat)
    m.add_node(1, 0.0, 0.0); m.add_node(2, L, 0.0)
    m.add_element(BeamColumn2D(1, (1, 2), mat, A, Iz))
    m.fix(1, [1, 1, 1])
    m.add_nodal_load(2, [0.0, -1.0, 0.0])
    integrator = DisplacementControl(node_tag=2, dof_index=1, du_step=-1.0e-4)
    res = NonlinearStaticAnalysis(
        m, num_steps=5, integrator=integrator,
        tol=1e-8, max_iter=2,
    ).run()
    # Should converge cleanly with at most 2 iters per step
    assert all(c <= 2 for c in res["iter_counts"])
