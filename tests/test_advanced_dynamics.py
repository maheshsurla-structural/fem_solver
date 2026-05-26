"""Tests for Phase 17 advanced dynamics: HHT-alpha, generalized-alpha,
central difference, and multi-support excitation.
"""
import math

import numpy as np
import pytest

from femsolver import (
    BeamColumn2D,
    CentralDifference,
    ElasticIsotropic,
    GeneralizedAlpha,
    HHTAlpha,
    Model,
    Newmark,
    TransientAnalysis,
    Truss2D,
    ground_motion_force,
    multi_support_ground_motion_force,
)


# ====================================================== helpers

def _build_sdof(K: float = 100.0, M_mass: float = 1.0):
    """SDOF horizontal truss with tip mass M, axial stiffness K."""
    L, A = 1.0, 1.0
    E = K * L / A
    rho = 3.0 * M_mass / (A * L)
    mat = ElasticIsotropic(1, E=E, nu=0.3, rho=rho)
    m = Model(ndm=2, ndf=2); m.add_material(mat)
    m.add_node(1, 0, 0); m.add_node(2, L, 0)
    m.add_element(Truss2D(1, (1, 2), mat, area=A))
    m.fix(1, [1, 1])
    m.fix(2, [0, 1])
    return m


# ====================================================== HHT-alpha

def test_hht_alpha_construction_validates_range():
    with pytest.raises(ValueError, match="alpha"):
        HHTAlpha(alpha=-0.5)     # below -1/3
    with pytest.raises(ValueError, match="alpha"):
        HHTAlpha(alpha=0.1)       # above 0


def test_hht_alpha_zero_matches_newmark_average_acceleration():
    """HHT-alpha with alpha = 0 reduces exactly to Newmark beta=1/4,
    gamma=1/2 (average acceleration)."""
    omega = 10.0
    T = 2.0 * math.pi / omega
    u0 = 0.01
    dt = T / 100
    n_steps = 200

    m_nm = _build_sdof(K=omega ** 2)
    m_nm.number_dofs()
    m_nm.node(2).disp[0] = u0
    res_nm = TransientAnalysis(
        m_nm, num_steps=n_steps, dt=dt,
        integrator=Newmark(), track=(2, 0),
    ).run()

    m_hht = _build_sdof(K=omega ** 2)
    m_hht.number_dofs()
    m_hht.node(2).disp[0] = u0
    res_hht = TransientAnalysis(
        m_hht, num_steps=n_steps, dt=dt,
        integrator=HHTAlpha(alpha=0.0), track=(2, 0),
    ).run()

    u_nm = np.array(res_nm["tracked_disp"])
    u_hht = np.array(res_hht["tracked_disp"])
    assert np.max(np.abs(u_nm - u_hht)) < 1e-12


def test_hht_alpha_damps_high_frequency_response():
    """At large omega*dt, HHT with alpha < 0 should attenuate the
    response significantly compared to alpha = 0."""
    omega = 10.0
    T = 2.0 * math.pi / omega
    u0 = 0.01
    # Coarse step puts the dynamics into the HF regime
    dt = T / 2
    n_steps = 30

    m_hht0 = _build_sdof(K=omega ** 2)
    m_hht0.number_dofs()
    m_hht0.node(2).disp[0] = u0
    res_no = TransientAnalysis(
        m_hht0, num_steps=n_steps, dt=dt,
        integrator=HHTAlpha(alpha=0.0), track=(2, 0),
    ).run()

    m_hht3 = _build_sdof(K=omega ** 2)
    m_hht3.number_dofs()
    m_hht3.node(2).disp[0] = u0
    res_dmp = TransientAnalysis(
        m_hht3, num_steps=n_steps, dt=dt,
        integrator=HHTAlpha(alpha=-0.3), track=(2, 0),
    ).run()

    u_no = np.array(res_no["tracked_disp"])
    u_dmp = np.array(res_dmp["tracked_disp"])
    # Damped magnitude at the end should be substantially smaller.
    assert abs(u_dmp[-1]) < 0.05 * abs(u_no[-1])


# ====================================================== generalized-alpha

def test_genalpha_construction_validates_range():
    with pytest.raises(ValueError, match="rho_inf"):
        GeneralizedAlpha(rho_inf=-0.1)
    with pytest.raises(ValueError, match="rho_inf"):
        GeneralizedAlpha(rho_inf=1.1)


def test_genalpha_rho1_preserves_low_freq_response():
    """rho_inf = 1.0 gives no algorithmic damping at low frequencies."""
    omega = 10.0
    T = 2.0 * math.pi / omega
    u0 = 0.01
    dt = T / 100
    n_steps = 200

    m = _build_sdof(K=omega ** 2)
    m.number_dofs()
    m.node(2).disp[0] = u0
    res = TransientAnalysis(
        m, num_steps=n_steps, dt=dt,
        integrator=GeneralizedAlpha(rho_inf=1.0), track=(2, 0),
    ).run()
    u = np.array(res["tracked_disp"])
    # Energy preserved (no algorithmic damping)
    assert np.max(np.abs(u)) == pytest.approx(u0, rel=1e-3)


def test_genalpha_rho0_annihilates_high_freq():
    """rho_inf = 0.0 annihilates HF modes in one step (the strongest
    high-frequency dissipation any 2nd-order scheme provides)."""
    omega = 10.0
    T = 2.0 * math.pi / omega
    u0 = 0.01
    # HF regime
    dt = T / 2
    n_steps = 40

    m = _build_sdof(K=omega ** 2)
    m.number_dofs()
    m.node(2).disp[0] = u0
    res = TransientAnalysis(
        m, num_steps=n_steps, dt=dt,
        integrator=GeneralizedAlpha(rho_inf=0.0), track=(2, 0),
    ).run()
    u = np.array(res["tracked_disp"])
    # After enough steps, the response should be effectively zero.
    assert abs(u[-1]) < 1e-10


def test_genalpha_damping_increases_as_rho_decreases():
    omega = 10.0
    T = 2.0 * math.pi / omega
    u0 = 0.01
    dt = T / 2
    n_steps = 30

    finals = {}
    for rho in (1.0, 0.7, 0.3, 0.0):
        m = _build_sdof(K=omega ** 2)
        m.number_dofs(); m.node(2).disp[0] = u0
        res = TransientAnalysis(
            m, num_steps=n_steps, dt=dt,
            integrator=GeneralizedAlpha(rho_inf=rho), track=(2, 0),
        ).run()
        finals[rho] = abs(res["tracked_disp"][-1])
    # Monotonically decreasing damping
    assert finals[1.0] >= finals[0.7] >= finals[0.3] >= finals[0.0]


# ====================================================== central difference

def test_cd_stable_below_courant_limit():
    """CD with dt < 2/omega should give a bounded response."""
    omega = 10.0
    T = 2.0 * math.pi / omega
    u0 = 0.01
    dt = T / 40
    n_steps = 200
    m = _build_sdof(K=omega ** 2)
    m.number_dofs()
    m.node(2).disp[0] = u0
    res = TransientAnalysis(
        m, num_steps=n_steps, dt=dt,
        integrator=CentralDifference(), track=(2, 0),
    ).run()
    u = np.array(res["tracked_disp"])
    # Response should stay bounded near u0
    assert np.max(np.abs(u)) < 2.0 * u0


def test_cd_unstable_above_courant_limit():
    """CD with dt > 2/omega blows up exponentially."""
    omega = 10.0
    u0 = 0.01
    dt_crit = 2.0 / omega
    dt = 1.5 * dt_crit
    n_steps = 30
    m = _build_sdof(K=omega ** 2)
    m.number_dofs()
    m.node(2).disp[0] = u0
    res = TransientAnalysis(
        m, num_steps=n_steps, dt=dt,
        integrator=CentralDifference(), track=(2, 0),
    ).run()
    u = np.array(res["tracked_disp"])
    # Should blow up (final magnitude >> initial)
    assert abs(u[-1]) > 100 * u0


# ====================================================== multi-support

def _build_2_column_frame():
    """Tiny 2-column frame: nodes 1 (base left), 2 (base right),
    3 (roof). Each column is a beam carrying mass. Roof is free."""
    E = 2.0e10; A = 1.0e-2; Iz = 1.0e-4; L = 3.0
    mat = ElasticIsotropic(1, E=E, nu=0.3, rho=7850.0)
    m = Model(ndm=2, ndf=3); m.add_material(mat)
    m.add_node(1, 0.0, 0.0)
    m.add_node(2, 5.0, 0.0)
    m.add_node(3, 0.0, L)
    m.add_node(4, 5.0, L)
    m.add_element(BeamColumn2D(1, (1, 3), mat, A, Iz))
    m.add_element(BeamColumn2D(2, (2, 4), mat, A, Iz))
    # Tie 3-4 with a beam (rigid roof girder)
    m.add_element(BeamColumn2D(3, (3, 4), mat, A * 10, Iz * 10))
    m.fix(1, [1, 1, 1])
    m.fix(2, [1, 1, 1])
    m.number_dofs()
    return m


def test_multi_support_single_spec_matches_single_support():
    """One spec with nodes=None reduces to a single-support
    ground_motion_force."""
    m = _build_2_column_frame()

    def ag(t):
        return math.sin(t)

    load_single = ground_motion_force(m, direction="x", accel_function=ag)
    load_multi = multi_support_ground_motion_force(
        m, supports=[{"direction": "x", "accel_function": ag, "nodes": None}]
    )
    # Both should give the same force vector at any t.
    for t in (0.0, 0.5, 1.7, 3.2):
        f_s = load_single(t)
        f_m = load_multi(t)
        assert np.allclose(f_s, f_m)


def test_multi_support_two_specs_sum_correctly():
    """Two specs whose nodes cover the entire model give the same total
    force as one spec with nodes=None, provided both spec functions
    return the same value."""
    m = _build_2_column_frame()
    # Roof nodes 3 and 4 are the only free DOFs in x.

    def ag(t):
        return 1.5 * math.sin(t)

    # Single-spec version
    load_one = multi_support_ground_motion_force(
        m, supports=[{"direction": "x", "accel_function": ag, "nodes": None}]
    )
    # Two-spec version: split node 3 from node 4 (each spec has the
    # same accel function -> same total).
    load_two = multi_support_ground_motion_force(
        m, supports=[
            {"direction": "x", "accel_function": ag, "nodes": [3]},
            {"direction": "x", "accel_function": ag, "nodes": [4]},
        ]
    )
    for t in (0.0, 0.5, 1.7, 3.2):
        f1 = load_one(t)
        f2 = load_two(t)
        assert np.allclose(f1, f2)


def test_multi_support_different_motions_give_different_forces():
    """Specs with *different* accel histories must produce different
    contributions to the total force vector."""
    m = _build_2_column_frame()

    def ag1(t):
        return math.sin(t)

    def ag2(t):
        return math.cos(t)

    # Split: node 3 follows ag1, node 4 follows ag2
    load = multi_support_ground_motion_force(
        m, supports=[
            {"direction": "x", "accel_function": ag1, "nodes": [3]},
            {"direction": "x", "accel_function": ag2, "nodes": [4]},
        ]
    )
    # At t = 0: ag1=0, ag2=1 -> only node 4 contributes
    f_t0 = load(0.0)
    # At t = pi/2: ag1=1, ag2=0 -> only node 3 contributes
    f_tpi2 = load(math.pi / 2)
    # The two force vectors should differ (have different supports active)
    assert not np.allclose(f_t0, f_tpi2)


def test_multi_support_requires_accel_function():
    m = _build_2_column_frame()
    with pytest.raises(ValueError, match="accel_function"):
        multi_support_ground_motion_force(
            m, supports=[{"direction": "x", "nodes": [3]}]
        )


def test_multi_support_rejects_empty_supports_list():
    m = _build_2_column_frame()
    with pytest.raises(ValueError, match="support"):
        multi_support_ground_motion_force(m, supports=[])
