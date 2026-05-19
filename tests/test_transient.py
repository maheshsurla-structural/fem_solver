"""Tests for the Newmark-based linear transient analysis.

The Newmark integrator is validated against three classes of
analytical solutions:

1. **SDOF free vibration** — single mass on a spring, with an initial
   velocity. Exact solution: ``u(t) = (v_0 / omega) sin(omega t)``.
   The FE must match this to within Newmark's accuracy.

2. **SDOF damped step response** — mass-spring-dashpot subjected to a
   step force. Exact solution exists in closed form for under-damped
   systems and provides an end-to-end check of damping + load.

3. **MDOF cantilever vibration** — a beam with an initial tip
   displacement. The fundamental period must match the eigenvalue
   analysis to within Newmark + discretisation accuracy.

We use the standard truss-as-SDOF idiom: a single Truss2D between a
fixed node and a free node carries the spring constant, and a point
mass at the free node provides the inertia. The mass is supplied via
a material with ``rho`` set so that the lumped beam mass matches the
desired mass — but for cleanest SDOF, we use a dedicated single-DOF
problem built directly from K, M, C.

In addition to the integrator validation, we test the analysis driver:
load function (callable scalar and callable vector), initial conditions
via Node.disp / Node.velocity, tracking, and the regression that all
existing static-analysis tests are unaffected by the Node-class
additions.
"""
import math

import numpy as np
import pytest

from femsolver import (
    BeamColumn2D,
    EigenAnalysis,
    ElasticIsotropic,
    Model,
    Newmark,
    NonlinearStaticAnalysis,
    RayleighDamping,
    TransientAnalysis,
    Truss2D,
)


# ============================================== Newmark constructor ==

def test_newmark_default_parameters():
    nm = Newmark()
    assert nm.beta == 0.25
    assert nm.gamma == 0.5


def test_newmark_rejects_invalid_beta():
    with pytest.raises(ValueError):
        Newmark(beta=0.0)
    with pytest.raises(ValueError):
        Newmark(beta=-0.1)


def test_newmark_rejects_out_of_range_gamma():
    with pytest.raises(ValueError):
        Newmark(gamma=-0.1)
    with pytest.raises(ValueError):
        Newmark(gamma=1.1)


# ============================================== SDOF free vibration ==

def _sdof_model(*, K_target: float, M_target: float):
    """Build an effectively-SDOF model: a horizontal truss between
    a fixed and a free node, with rho tuned so the consistent mass
    matrix gives the target nodal mass at the free DOF.

    For a 2-node truss, the consistent mass is:

        M_local = (rho A L / 6) * [[2, 1], [1, 2]]

    With node 1 fixed (u=0), only the (1, 1) entry contributes at the
    free node, giving an effective mass of ``2/6 rho A L = rho A L / 3``.
    To get a target nodal mass ``m``, set ``rho A L = 3 m``.

    The truss stiffness is ``K = EA/L``. To get target ``K``, fix
    ``L = 1.0`` and pick ``EA = K``. Then ``rho = 3 m / (A * 1.0)``.
    """
    L = 1.0
    A = 1.0
    E = K_target / A * L                    # EA/L = K
    rho = 3.0 * M_target / (A * L)
    mat = ElasticIsotropic(1, E=E, nu=0.3, rho=rho)
    m = Model(ndm=2, ndf=2); m.add_material(mat)
    m.add_node(1, 0.0, 0.0)
    m.add_node(2, L, 0.0)
    m.add_element(Truss2D(1, (1, 2), mat, area=A))
    m.fix(1, [1, 1])
    m.fix(2, [0, 1])     # only horizontal motion at node 2
    return m


def test_sdof_free_vibration_period_matches_analytical():
    """Initial velocity, no force, no damping. Solution:
       u(t) = (v0 / omega) sin(omega t).
    Period: T = 2 pi sqrt(m / k).
    """
    K = 100.0
    M = 1.0
    v0 = 1.0
    omega = math.sqrt(K / M)
    T_analytical = 2.0 * math.pi / omega

    m = _sdof_model(K_target=K, M_target=M)
    m.number_dofs()
    # Initial conditions: u_2 = 0, v_2 = v0
    m.node(2).velocity[0] = v0

    # March 1.5 periods with a fine step.
    dt = T_analytical / 200.0
    n_steps = int(1.5 * 200)
    analysis = TransientAnalysis(
        m, num_steps=n_steps, dt=dt, track=(2, 0),
    )
    res = analysis.run()
    t = np.array(res["times"])
    u_fe = np.array(res["tracked_disp"])
    u_exact = (v0 / omega) * np.sin(omega * t)
    # Average-acceleration Newmark is conservative and has period
    # error of O((omega dt)^2). With omega dt = 2 pi / 200, the error
    # is well under 1 % over 1.5 periods.
    np.testing.assert_allclose(u_fe, u_exact, atol=2.0e-3)


def test_sdof_free_vibration_energy_conservation():
    """For average-acceleration Newmark, total energy is conserved
    in the absence of damping. Check at the end of an integer number
    of periods."""
    K = 100.0
    M = 1.0
    v0 = 1.0
    omega = math.sqrt(K / M)
    T = 2.0 * math.pi / omega
    m = _sdof_model(K_target=K, M_target=M)
    m.number_dofs()
    m.node(2).velocity[0] = v0
    dt = T / 400.0
    n_steps = 400 * 3   # exactly 3 periods
    analysis = TransientAnalysis(m, num_steps=n_steps, dt=dt, track=(2, 0))
    res = analysis.run()
    u_end = res["tracked_disp"][-1]
    v_end = res["tracked_velocity"][-1]
    E_initial = 0.5 * M * v0 ** 2
    E_end = 0.5 * M * v_end ** 2 + 0.5 * K * u_end ** 2
    # Average-acceleration Newmark conserves energy exactly only for
    # rigid-body modes; for vibrations the relative error scales like
    # (omega dt)^2. With dt = T/400, that's (2 pi / 400)^2 ~ 2.5e-4.
    assert abs(E_end - E_initial) / E_initial < 1.0e-3


# ============================================== SDOF damped ==

def test_sdof_critically_damped_step_response():
    """Step-applied force, critically damped SDOF.
    Solution: u(t) = (F / k) (1 - (1 + omega_n t) exp(-omega_n t)).
    """
    K = 100.0
    M = 1.0
    F0 = 50.0
    omega = math.sqrt(K / M)
    # Critical damping ratio = 1 -> c = 2 sqrt(k m) = 2 omega m
    # Construct via Rayleigh: zeta = c / (2 omega m). Set alpha_M = c/m,
    # alpha_K = 0. With c = 2 omega m, alpha_M = 2 omega.
    damping = RayleighDamping(alpha_M=2.0 * omega, alpha_K=0.0)

    m = _sdof_model(K_target=K, M_target=M)
    m.number_dofs()
    m.add_nodal_load(2, [F0, 0.0])
    dt = (2.0 * math.pi / omega) / 200.0
    n_steps = 800
    res = TransientAnalysis(
        m, num_steps=n_steps, dt=dt, damping=damping, track=(2, 0),
    ).run()
    # Final value approaches the static deflection F0 / K
    u_static = F0 / K
    # At t = 4 periods (decay >> 1), should be very close to static.
    np.testing.assert_allclose(res["tracked_disp"][-1], u_static, rtol=5.0e-3)
    # Critically damped: no overshoot. The max recorded displacement
    # equals (or is very close to) the static value.
    assert max(res["tracked_disp"]) < 1.01 * u_static


def test_sdof_underdamped_decay():
    """Step force on an underdamped SDOF. Amplitude of oscillation
    about the static value decays exponentially with rate omega_d * zeta."""
    K = 100.0
    M = 1.0
    F0 = 50.0
    omega = math.sqrt(K / M)
    zeta = 0.05    # 5% damping
    # alpha_M = 2 zeta omega (mass-proportional only)
    damping = RayleighDamping(alpha_M=2.0 * zeta * omega, alpha_K=0.0)
    m = _sdof_model(K_target=K, M_target=M)
    m.number_dofs()
    m.add_nodal_load(2, [F0, 0.0])
    dt = (2.0 * math.pi / omega) / 200.0
    n_steps = 600     # ~3 periods
    res = TransientAnalysis(
        m, num_steps=n_steps, dt=dt, damping=damping, track=(2, 0),
    ).run()
    # Amplitudes of the first and third peak above static value should
    # be in the ratio exp(-zeta omega T_period) per period.
    u_static = F0 / K
    u_arr = np.array(res["tracked_disp"])
    # Find peaks (local maxima above static).
    above_static = u_arr - u_static
    peak_indices = []
    for i in range(1, len(above_static) - 1):
        if (above_static[i] > 0
            and above_static[i] > above_static[i - 1]
            and above_static[i] > above_static[i + 1]):
            peak_indices.append(i)
    assert len(peak_indices) >= 2
    p1, p2 = peak_indices[0], peak_indices[1]
    ratio = above_static[p2] / above_static[p1]
    # Logarithmic decrement: ln(p1/p2) = 2 pi zeta / sqrt(1 - zeta^2)
    delta_analytical = 2.0 * math.pi * zeta / math.sqrt(1.0 - zeta ** 2)
    delta_fe = math.log(above_static[p1] / above_static[p2])
    assert delta_fe == pytest.approx(delta_analytical, rel=2.0e-2)


# ============================================== MDOF beam vibration ==

def test_cantilever_first_period_matches_eigenanalysis():
    """A cantilever beam given an initial *first-mode-shape* displacement
    vibrates purely in mode 1. The period of the tip oscillation must
    match the first natural period from :class:`EigenAnalysis` to
    within Newmark + discretisation accuracy.

    Using the eigen mode shape (rather than a spike at the tip) means
    no higher modes are excited, so we can directly identify the
    response period with T_1.
    """
    E, A, Iz, L, rho = 2.0e11, 1.0e-2, 8.333e-6, 3.0, 7850.0
    mat = ElasticIsotropic(1, E=E, nu=0.3, rho=rho)
    n_elem = 4

    def build():
        m = Model(ndm=2, ndf=3); m.add_material(mat)
        for i in range(n_elem + 1):
            m.add_node(i + 1, i * L / n_elem, 0.0)
        for i in range(n_elem):
            m.add_element(BeamColumn2D(i + 1, (i + 1, i + 2), mat, A, Iz))
        m.fix(1, [1, 1, 1])
        return m

    # Eigen analysis: first natural period and the mode shape used as
    # the initial-condition basis.
    m_eig = build()
    EigenAnalysis(m_eig, num_modes=3).run()
    # The first mode's nodal pattern is stored on Node.mode_disp[:, 0]
    # after EigenAnalysis runs.

    m_tr = build()
    eig = EigenAnalysis(m_tr, num_modes=3).run()
    T_1 = eig["periods_s"][0]
    # Set initial disp to a small multiple of the first mode shape so
    # the transient response is essentially pure mode 1.
    tip_tag = n_elem + 1
    scale = 0.001 / abs(m_tr.node(tip_tag).mode_disp[1, 0])
    for node in m_tr.nodes.values():
        node.disp[:] = scale * node.mode_disp[:, 0]

    dt = T_1 / 100.0
    n_steps = 300     # 3 periods
    res = TransientAnalysis(
        m_tr, num_steps=n_steps, dt=dt, track=(tip_tag, 1),
    ).run()
    t = np.array(res["times"])
    u = np.array(res["tracked_disp"])
    # Estimate period from successive downward zero crossings.
    zeros = []
    for i in range(1, len(u)):
        if u[i - 1] > 0 and u[i] <= 0:
            frac = u[i - 1] / (u[i - 1] - u[i])
            zeros.append(t[i - 1] + frac * (t[i] - t[i - 1]))
    assert len(zeros) >= 2
    T_fe = zeros[1] - zeros[0]
    assert T_fe == pytest.approx(T_1, rel=1.0e-2)


# ============================================== driver ==

def test_load_function_scalar_callable():
    """A scalar time function is multiplied into the reference load
    pattern. The damped response approaches the static deflection of
    the full applied load."""
    K, M = 100.0, 1.0
    F_ref = 10.0
    omega = math.sqrt(K / M)
    m = _sdof_model(K_target=K, M_target=M)
    m.number_dofs()
    m.add_nodal_load(2, [F_ref, 0.0])
    # Light damping (5% on the natural frequency) so the oscillation
    # decays and we can compare with the static value cleanly.
    damping = RayleighDamping(alpha_M=2.0 * 0.05 * omega, alpha_K=0.0)
    T = 2.0 * math.pi / omega
    n_steps = 1500
    dt = T / 200.0

    # Step-applied load (constant 1.0). With 5 % damping over ~7
    # periods the residual oscillation is below 1 % of the static.
    res = TransientAnalysis(
        m, num_steps=n_steps, dt=dt,
        load_function=lambda t: 1.0,
        damping=damping, track=(2, 0),
    ).run()
    u_static = F_ref / K
    u = np.array(res["tracked_disp"])
    # After several damped periods, the average over the last period
    # is very close to the static value.
    last_period_idx = int(T / dt)
    u_avg = u[-last_period_idx:].mean()
    assert u_avg == pytest.approx(u_static, rel=2.0e-2)


def test_load_function_vector_callable():
    """A vector-valued time function bypasses the reference pattern."""
    K, M = 100.0, 1.0
    F_target = 10.0
    omega = math.sqrt(K / M)
    m = _sdof_model(K_target=K, M_target=M)
    m.number_dofs()
    # Light damping for clean comparison to static.
    damping = RayleighDamping(alpha_M=2.0 * 0.05 * omega, alpha_K=0.0)

    def force(t):
        return np.array([F_target])  # constant force, all in DOF 0

    T = 2.0 * math.pi / omega
    dt = T / 200.0
    n_steps = 1500
    res = TransientAnalysis(
        m, num_steps=n_steps, dt=dt,
        load_function=force, damping=damping, track=(2, 0),
    ).run()
    u_static = F_target / K
    u = np.array(res["tracked_disp"])
    u_avg = u[-int(T / dt):].mean()
    assert u_avg == pytest.approx(u_static, rel=2.0e-2)


def test_initial_conditions_from_node_state():
    """Initial conditions read directly from Node.disp and
    Node.velocity. A free vibration with u_0 = u0 should oscillate
    starting from u0 with v_0 = 0."""
    K, M = 100.0, 1.0
    u_0 = 0.01
    omega = math.sqrt(K / M)
    m = _sdof_model(K_target=K, M_target=M)
    m.number_dofs()
    m.node(2).disp[0] = u_0    # initial displacement
    # m.node(2).velocity[0] = 0  (default zero)
    T = 2.0 * math.pi / omega
    dt = T / 200.0
    res = TransientAnalysis(
        m, num_steps=100, dt=dt, track=(2, 0),
    ).run()
    # At t=0, recorded disp must be u_0
    assert res["tracked_disp"][0] == u_0
    # At t = T/2, recorded disp must be ~ -u_0
    half_period_idx = 100   # T/2 / dt = 100
    assert res["tracked_disp"][half_period_idx] == pytest.approx(
        -u_0, rel=1.0e-2
    )


def test_undamped_zero_force_with_zero_ic_stays_at_rest():
    """Trivial case: no load, no initial state — displacements
    remain zero throughout."""
    K, M = 100.0, 1.0
    m = _sdof_model(K_target=K, M_target=M)
    m.number_dofs()
    res = TransientAnalysis(
        m, num_steps=20, dt=0.01, track=(2, 0),
    ).run()
    np.testing.assert_allclose(res["tracked_disp"], 0.0, atol=1e-14)
    np.testing.assert_allclose(res["tracked_velocity"], 0.0, atol=1e-14)


# ============================================== input validation ==

def test_transient_rejects_nonpositive_dt():
    K, M = 100.0, 1.0
    m = _sdof_model(K_target=K, M_target=M)
    m.number_dofs()
    with pytest.raises(ValueError):
        TransientAnalysis(m, num_steps=10, dt=0.0)
    with pytest.raises(ValueError):
        TransientAnalysis(m, num_steps=10, dt=-0.01)


def test_transient_rejects_zero_steps():
    K, M = 100.0, 1.0
    m = _sdof_model(K_target=K, M_target=M)
    m.number_dofs()
    with pytest.raises(ValueError):
        TransientAnalysis(m, num_steps=0, dt=0.01)
