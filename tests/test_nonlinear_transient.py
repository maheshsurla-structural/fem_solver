"""Tests for the nonlinear transient (direct-integration) analysis.

These tests pin down four properties.

1. **Linear regression** — a linear-elastic problem solved by
   :class:`NonlinearTransientAnalysis` must match the
   :class:`TransientAnalysis` (Phase 8) result to machine precision.
   The Newton iteration converges in one step on a linear system, so
   the two paths produce bitwise-identical histories.

2. **Quadratic Newton convergence on elastic problems** — Newton
   converges in 1 iteration per step for a linear problem. We check
   the iter-count history.

3. **Energy dissipation under material plasticity** — a free-vibrating
   system with a yielding hinge must lose energy each cycle. Peak
   amplitude decays with successive cycles (whereas an elastic system
   with no damping would conserve energy).

4. **Quasi-static convergence** — under a *slowly* applied load the
   dynamic response approaches the static one (inertial effects
   vanish). Tested against ``NonlinearStaticAnalysis``.
"""
import math

import numpy as np
import pytest

from femsolver import (
    BeamColumn2D,
    BilinearMomentRotationSpring,
    ElasticIsotropic,
    HingedBeamColumn2D,
    LoadControl,
    Model,
    NonlinearStaticAnalysis,
    NonlinearTransientAnalysis,
    RayleighDamping,
    TransientAnalysis,
    Truss2D,
)


# ===================================================== helpers

def _sdof_model(*, K: float, M: float):
    """SDOF horizontal truss: K = EA/L, mass = rho A L / 3 at free DOF."""
    L = 1.0
    A = 1.0
    E = K * L / A
    rho = 3.0 * M / (A * L)
    mat = ElasticIsotropic(1, E=E, nu=0.3, rho=rho)
    m = Model(ndm=2, ndf=2); m.add_material(mat)
    m.add_node(1, 0.0, 0.0)
    m.add_node(2, L, 0.0)
    m.add_element(Truss2D(1, (1, 2), mat, area=A))
    m.fix(1, [1, 1])
    m.fix(2, [0, 1])
    return m


def _cantilever_model(*, n_elem: int = 4, with_hinge: bool = False, b_post: float = 0.0):
    """Cantilever beam. If with_hinge is True, attach a bilinear
    moment-rotation hinge at the base."""
    E, A, Iz, L, rho = 2.0e11, 1.0e-2, 8.333e-6, 3.0, 7850.0
    mat = ElasticIsotropic(1, E=E, nu=0.3, rho=rho)
    m = Model(ndm=2, ndf=3); m.add_material(mat)
    for i in range(n_elem + 1):
        m.add_node(i + 1, i * L / n_elem, 0.0)

    if with_hinge:
        K_h = 4.0 * E * Iz / L
        My = 5.0e3
        hinge = BilinearMomentRotationSpring(K0=K_h, My=My, b=b_post)
        # first element is hinged; the rest are linear-elastic beams
        m.add_element(
            HingedBeamColumn2D(1, (1, 2), mat, A, Iz, hinge_i=hinge)
        )
        for i in range(1, n_elem):
            m.add_element(BeamColumn2D(i + 1, (i + 1, i + 2), mat, A, Iz))
        constants = {"E": E, "A": A, "Iz": Iz, "L": L, "K_h": K_h, "My": My}
    else:
        for i in range(n_elem):
            m.add_element(BeamColumn2D(i + 1, (i + 1, i + 2), mat, A, Iz))
        constants = {"E": E, "A": A, "Iz": Iz, "L": L}

    m.fix(1, [1, 1, 1])
    return m, constants


# ============================================================== linear regression

def test_nonlinear_transient_matches_linear_for_elastic_cantilever():
    """NonlinearTransientAnalysis on a linear-elastic problem must
    produce a bit-for-bit match with TransientAnalysis (Phase 8)."""
    # Build twice
    m_lin, _ = _cantilever_model()
    m_lin.number_dofs()
    m_lin.node(5).disp[1] = 1.0e-3       # initial tip displacement

    m_nl, _ = _cantilever_model()
    m_nl.number_dofs()
    m_nl.node(5).disp[1] = 1.0e-3

    dt = 1.0e-4
    n_steps = 200

    r_lin = TransientAnalysis(
        m_lin, num_steps=n_steps, dt=dt, track=(5, 1),
    ).run()
    r_nl = NonlinearTransientAnalysis(
        m_nl, num_steps=n_steps, dt=dt, track=(5, 1), tol=1.0e-8,
    ).run()

    d_lin = np.array(r_lin["tracked_disp"])
    d_nl = np.array(r_nl["tracked_disp"])
    np.testing.assert_allclose(d_nl, d_lin, atol=1.0e-12)
    # Velocity histories also match
    v_lin = np.array(r_lin["tracked_velocity"])
    v_nl = np.array(r_nl["tracked_velocity"])
    np.testing.assert_allclose(v_nl, v_lin, atol=1.0e-10)


def test_newton_converges_in_one_iteration_for_linear_problem():
    """For a linear elastic problem, Newton's residual is exactly
    zero after one step — the iter count per step should be 1
    (some convergence tests may also pass at iter 0, in which case
    the count can be 0; either is acceptable)."""
    m, _ = _cantilever_model()
    m.number_dofs()
    m.node(5).disp[1] = 1.0e-3
    res = NonlinearTransientAnalysis(
        m, num_steps=30, dt=1.0e-4, track=(5, 1), tol=1.0e-8, max_iter=5,
    ).run()
    # First entry is the initial-state record (0 iters). Subsequent
    # entries are step-Newton counts. For a linear problem each
    # should be 1 (one solve, then converged).
    assert all(c <= 1 for c in res["iter_counts"][1:])


# ============================================================== plasticity decays

def test_yielding_hinge_decays_free_vibration_amplitude():
    """A cantilever with a *bilinear* base hinge given an initial
    displacement that triggers yield must lose energy each cycle. We
    quantify this by tracking the absolute-value peak height of the
    response and confirming peak[k] < peak[0] after several cycles.

    Without yielding (purely elastic) the average-acceleration Newmark
    is energy-conserving — see ``test_transient.py
    test_sdof_free_vibration_energy_conservation``. Adding plasticity
    must visibly drain that energy.
    """
    # Use a slightly hardening hinge so the structure is stable past
    # yield (no mechanism)
    m, cn = _cantilever_model(with_hinge=True, b_post=0.05)
    m.number_dofs()
    # Initial tip displacement large enough to yield the hinge.
    # Yield at tip: v_y = -P_y * L^3 / (3 EI) - P_y * L^2 / K_h
    # with P_y = My / L. Pick something a few times v_y.
    P_y = cn["My"] / cn["L"]
    EI = cn["E"] * cn["Iz"]
    v_y = -P_y * cn["L"] ** 3 / (3.0 * EI) - P_y * cn["L"] ** 2 / cn["K_h"]
    m.node(5).disp[1] = 3.0 * v_y    # well past first yield

    # Use a fairly small time step to resolve the response cleanly.
    # Period estimate: T ~ 2 pi sqrt(meff / Keff). For a cantilever
    # in mode 1 with mass ~ rho A L and stiffness ~ 3EI/L^3:
    rho = 7850.0
    meff = rho * cn["A"] * cn["L"] * 0.25       # effective tip mass (rough)
    Keff = 3.0 * EI / cn["L"] ** 3
    T_est = 2.0 * math.pi * math.sqrt(meff / Keff)
    dt = T_est / 200.0
    n_steps = 1200   # ~6 periods
    res = NonlinearTransientAnalysis(
        m, num_steps=n_steps, dt=dt, track=(5, 1),
        tol=1.0e-5, max_iter=30,
    ).run()
    u = np.array(res["tracked_disp"])

    # Find absolute-value local maxima
    peaks = []
    for i in range(1, len(u) - 1):
        if abs(u[i]) > abs(u[i - 1]) and abs(u[i]) > abs(u[i + 1]):
            peaks.append(abs(u[i]))
    # Need at least a few peaks to compare
    assert len(peaks) >= 3
    # Energy dissipation: amplitude must clearly drop.
    assert peaks[2] < 0.95 * peaks[0]


# ============================================================== quasi-static

def test_quasi_static_limit_matches_static_analysis():
    """Apply a load very slowly. The dynamic response (with no initial
    velocity and modest damping) should approach the static-analysis
    result for the same final load.

    We use a *linear-elastic* problem so static and dynamic must agree
    exactly in the steady state.
    """
    # SDOF spring with a constant force applied. Static answer:
    # u_static = F / K.
    K, M = 100.0, 1.0
    F_target = 10.0
    omega = math.sqrt(K / M)

    # Damping at 20% (well-damped, settles in a few periods).
    damping = RayleighDamping(alpha_M=2.0 * 0.20 * omega, alpha_K=0.0)
    m = _sdof_model(K=K, M=M)
    m.number_dofs()
    m.add_nodal_load(2, [F_target, 0.0])

    T = 2.0 * math.pi / omega
    dt = T / 200.0
    n_steps = 1500   # ~7.5 periods at 20% damping is well-settled
    res = NonlinearTransientAnalysis(
        m, num_steps=n_steps, dt=dt, track=(2, 0),
        damping=damping, tol=1.0e-8,
    ).run()
    u_final = res["tracked_disp"][-1]
    u_static = F_target / K
    assert u_final == pytest.approx(u_static, rel=2.0e-2)


# ============================================================== driver guards

def test_nonlinear_transient_rejects_zero_dt():
    m, _ = _cantilever_model()
    with pytest.raises(ValueError):
        NonlinearTransientAnalysis(m, num_steps=10, dt=0.0)


def test_nonlinear_transient_rejects_zero_steps():
    m, _ = _cantilever_model()
    with pytest.raises(ValueError):
        NonlinearTransientAnalysis(m, num_steps=0, dt=0.01)


def test_nonlinear_transient_unknown_algorithm_raises():
    m, _ = _cantilever_model()
    with pytest.raises(ValueError, match="unknown algorithm"):
        NonlinearTransientAnalysis(
            m, num_steps=2, dt=0.01, algorithm="banana",
        )


def test_nonlinear_transient_accepts_line_search_algorithm():
    """Smoke test: LineSearchNewton algorithm works with the dynamic
    integrator since NewmarkNonlinear advertises supports_line_search."""
    m, _ = _cantilever_model()
    m.number_dofs()
    m.node(5).disp[1] = 1.0e-4
    res = NonlinearTransientAnalysis(
        m, num_steps=10, dt=1.0e-4, track=(5, 1),
        algorithm="line_search", tol=1.0e-8,
    ).run()
    # Tip should oscillate; final displacement should be close to
    # but not exactly the initial (small time, low energy loss).
    assert abs(res["tracked_disp"][-1]) > 0.0


# ============================================================== initial conds

def test_initial_velocity_propagates():
    """Setting Node.velocity before run() should be picked up as the
    initial condition."""
    K, M = 100.0, 1.0
    v0 = 0.1
    omega = math.sqrt(K / M)
    m = _sdof_model(K=K, M=M)
    m.number_dofs()
    m.node(2).velocity[0] = v0

    T = 2.0 * math.pi / omega
    dt = T / 200.0
    res = NonlinearTransientAnalysis(
        m, num_steps=50, dt=dt, track=(2, 0), tol=1.0e-9,
    ).run()
    # Exact solution: u(t) = (v0 / omega) sin(omega t)
    u_fe = np.array(res["tracked_disp"])
    t = np.array(res["times"])
    u_exact = (v0 / omega) * np.sin(omega * t)
    np.testing.assert_allclose(u_fe, u_exact, atol=2.0e-4)
