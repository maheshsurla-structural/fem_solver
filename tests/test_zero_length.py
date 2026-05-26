"""Tests for the Phase 18 connector / isolator infrastructure:

* ``ZeroLengthElement`` with various uniaxial materials
* ``UniaxialGap`` material (compression / tension)
* ``lead_rubber_bearing`` and ``friction_pendulum`` macros
"""
import math

import numpy as np
import pytest

from femsolver import (
    BeamColumn2D,
    ElasticIsotropic,
    Model,
    NonlinearStaticAnalysis,
    UniaxialBilinear,
    UniaxialElastic,
    UniaxialGap,
    ZeroLengthElement,
    friction_pendulum,
    lead_rubber_bearing,
)
from femsolver.analysis.linear_static import LinearStaticAnalysis


# ====================================================== ZeroLengthElement

def test_zero_length_construction_validates_materials():
    """ZeroLengthElement needs at least one material direction."""
    mat = ElasticIsotropic(1, E=2e11, nu=0.3)
    m = Model(ndm=2, ndf=2); m.add_material(mat)
    m.add_node(1, 0, 0); m.add_node(2, 0, 0)
    with pytest.raises(ValueError, match="at least one"):
        ZeroLengthElement(1, (1, 2), materials={}, dofs_per_node=2)


def test_zero_length_rejects_dof_out_of_range():
    spring = UniaxialElastic(E=1e4)
    mat = ElasticIsotropic(1, E=2e11, nu=0.3)
    m = Model(ndm=2, ndf=2); m.add_material(mat)
    m.add_node(1, 0, 0); m.add_node(2, 0, 0)
    with pytest.raises(ValueError, match="dofs_per_node"):
        ZeroLengthElement(1, (1, 2),
                            materials={5: spring}, dofs_per_node=2)


def test_zero_length_rejects_negative_dof():
    spring = UniaxialElastic(E=1e4)
    with pytest.raises(ValueError, match="DOF index"):
        ZeroLengthElement(1, (1, 2),
                            materials={-1: spring}, dofs_per_node=2)


def test_zero_length_elastic_spring_displacement():
    """A single ZeroLength spring K reproduces u = P/K under static load."""
    K = 1.0e4
    mat = ElasticIsotropic(1, E=2e11, nu=0.3)
    m = Model(ndm=2, ndf=2); m.add_material(mat)
    m.add_node(1, 0, 0); m.add_node(2, 0, 0)
    spring = UniaxialElastic(E=K)
    m.add_element(ZeroLengthElement(
        1, (1, 2), materials={0: spring}, dofs_per_node=2
    ))
    m.fix(1, [1, 1])
    m.fix(2, [0, 1])     # y locked
    P = 1500.0
    m.add_nodal_load(2, [P, 0])
    LinearStaticAnalysis(m).run()
    assert m.node(2).disp[0] == pytest.approx(P / K, rel=1e-12)


def test_zero_length_two_dof_independent_springs():
    """Two springs in orthogonal DOFs don't couple."""
    Kx, Ky = 1.0e4, 2.0e4
    mat = ElasticIsotropic(1, E=2e11, nu=0.3)
    m = Model(ndm=2, ndf=2); m.add_material(mat)
    m.add_node(1, 0, 0); m.add_node(2, 0, 0)
    m.add_element(ZeroLengthElement(
        1, (1, 2),
        materials={0: UniaxialElastic(E=Kx),
                    1: UniaxialElastic(E=Ky)},
        dofs_per_node=2,
    ))
    m.fix(1, [1, 1])
    Px, Py = 1000.0, 2000.0
    m.add_nodal_load(2, [Px, Py])
    LinearStaticAnalysis(m).run()
    assert m.node(2).disp[0] == pytest.approx(Px / Kx, rel=1e-12)
    assert m.node(2).disp[1] == pytest.approx(Py / Ky, rel=1e-12)


def test_zero_length_K_global_is_symmetric():
    K = 1.0e4
    mat = ElasticIsotropic(1, E=2e11, nu=0.3)
    m = Model(ndm=2, ndf=2); m.add_material(mat)
    m.add_node(1, 0, 0); m.add_node(2, 0, 0)
    elem = ZeroLengthElement(
        1, (1, 2),
        materials={0: UniaxialElastic(E=K), 1: UniaxialElastic(E=2 * K)},
        dofs_per_node=2,
    )
    m.add_element(elem)
    m.number_dofs()
    K_g = elem.K_global()
    assert K_g.shape == (4, 4)
    assert np.allclose(K_g, K_g.T)


def test_zero_length_internal_force_balances_applied():
    """For an elastic spring at equilibrium, f_int on node j equals
    the applied load (opposite sign on node i, matching standard
    finite-element ``F_ext = F_int`` convention)."""
    K = 5.0e4
    mat = ElasticIsotropic(1, E=2e11, nu=0.3)
    m = Model(ndm=2, ndf=2); m.add_material(mat)
    m.add_node(1, 0, 0); m.add_node(2, 0, 0)
    elem = ZeroLengthElement(
        1, (1, 2),
        materials={0: UniaxialElastic(E=K)},
        dofs_per_node=2,
    )
    m.add_element(elem)
    m.fix(1, [1, 1])
    m.fix(2, [0, 1])
    P = 1000.0
    m.add_nodal_load(2, [P, 0])
    LinearStaticAnalysis(m).run()
    f = elem.f_int_global()
    # Standard F_ext = F_int convention: f_int[j] = +P, f_int[i] = -P.
    assert f[2] == pytest.approx(+P, rel=1e-10)
    assert f[0] == pytest.approx(-P, rel=1e-10)


# ====================================================== UniaxialGap

def test_gap_construction_validates_params():
    with pytest.raises(ValueError, match="E"):
        UniaxialGap(E=0.0)
    with pytest.raises(ValueError, match="gap"):
        UniaxialGap(E=1e4, gap=-0.1)
    with pytest.raises(ValueError, match="kind"):
        UniaxialGap(E=1e4, kind="foo")


def test_gap_compression_only_no_force_in_tension():
    """Strictly-positive eps (tension) gives zero force and zero
    tangent (the gap is fully open)."""
    g = UniaxialGap(E=1e5, gap=0.0, kind="compression")
    for eps in (0.001, 0.01, 0.1):
        s, Et = g.get_response(eps)
        assert s == 0.0
        assert Et == 0.0


def test_gap_compression_only_force_when_closed():
    g = UniaxialGap(E=1e5, gap=0.0, kind="compression")
    s, Et = g.get_response(-0.002)
    assert s == pytest.approx(-200.0, rel=1e-12)
    assert Et == pytest.approx(1e5, rel=1e-12)


def test_gap_initial_separation():
    """With gap > 0, the spring stays inactive until the relative
    displacement exceeds the gap."""
    g = UniaxialGap(E=1e5, gap=0.001, kind="compression")
    # eps just below the threshold: gap still open
    s, _ = g.get_response(-0.0005)
    assert s == 0.0
    # eps past the threshold: gap closed by 1 mm extra
    s, _ = g.get_response(-0.002)
    assert s == pytest.approx(1e5 * (-0.002 - (-0.001)), rel=1e-12)
    assert s == pytest.approx(-100.0, rel=1e-12)


def test_gap_tension_only():
    g = UniaxialGap(E=1e5, gap=0.0, kind="tension")
    assert g.get_response(-0.001) == (0.0, 0.0)
    s, _ = g.get_response(0.002)
    assert s == pytest.approx(200.0, rel=1e-12)


def test_gap_in_zero_length_compression_only_uplift_simulation():
    """A compression-only spring under a node lets it 'lift off' when
    a tensile load is applied -- u becomes large (no resistance)."""
    K_spring = 1.0e6
    mat = ElasticIsotropic(1, E=2e11, nu=0.3)
    m = Model(ndm=2, ndf=2); m.add_material(mat)
    m.add_node(1, 0, 0); m.add_node(2, 0, 0)
    m.add_element(ZeroLengthElement(
        1, (1, 2),
        materials={0: UniaxialGap(E=K_spring, kind="compression")},
        dofs_per_node=2,
    ))
    m.fix(1, [1, 1])
    m.fix(2, [0, 1])
    # COMPRESSIVE load (P negative) -- spring engages.
    m.add_nodal_load(2, [-100.0, 0])
    # Run nonlinear static (linear-static would fail with zero diagonal).
    NonlinearStaticAnalysis(
        m, num_steps=1, dlambda=1.0, tol=1e-6, max_iter=10,
    ).run()
    # eps = u_j - u_i = u_2 - 0 < 0 -> compression closes gap
    # F = K * eps -> -100 = K * eps -> eps = -100/K -> u_2 = -100/K
    assert m.node(2).disp[0] == pytest.approx(-100.0 / K_spring, rel=1e-9)


# ====================================================== Lead-rubber bearing

def test_lrb_construction_validates_params():
    with pytest.raises(ValueError, match="K1"):
        lead_rubber_bearing(1, (1, 2), K1=0.0, K2=1e5, Q=10e3,
                              dofs_per_node=2)
    with pytest.raises(ValueError, match="K2"):
        lead_rubber_bearing(1, (1, 2), K1=1e6, K2=2e6, Q=10e3,
                              dofs_per_node=2)
    with pytest.raises(ValueError, match="Q"):
        lead_rubber_bearing(1, (1, 2), K1=1e6, K2=1e5, Q=-10.0,
                              dofs_per_node=2)


def test_lrb_initial_stiffness_matches_K1():
    """Below yield, the lateral spring has stiffness K1."""
    K1 = 1.0e7; K2 = 5.0e5; Q = 20.0e3
    mat = ElasticIsotropic(1, E=2e11, nu=0.3)
    m = Model(ndm=2, ndf=2); m.add_material(mat)
    m.add_node(1, 0, 0); m.add_node(2, 0, 0)
    lrb = lead_rubber_bearing(1, (1, 2), K1=K1, K2=K2, Q=Q,
                                dofs_per_node=2)
    m.add_element(lrb)
    m.fix(1, [1, 1])
    m.fix(2, [0, 1])
    P = 5.0e3       # below F_y = K1 * d_y = K1 * Q/(K1-K2) ≈ 21 kN
    m.add_nodal_load(2, [P, 0])
    LinearStaticAnalysis(m).run()
    # Linear analysis uses K1
    assert m.node(2).disp[0] == pytest.approx(P / K1, rel=1e-10)


def test_lrb_yields_under_lateral_load():
    """Under a load beyond F_y, nonlinear analysis shows post-yield
    behaviour."""
    K1 = 1.0e7; K2 = 5.0e5; Q = 20.0e3
    d_y = Q / (K1 - K2)
    F_y = K1 * d_y
    mat = ElasticIsotropic(1, E=2e11, nu=0.3)
    m = Model(ndm=2, ndf=2); m.add_material(mat)
    m.add_node(1, 0, 0); m.add_node(2, 0, 0)
    m.add_element(lead_rubber_bearing(
        1, (1, 2), K1=K1, K2=K2, Q=Q, dofs_per_node=2,
    ))
    m.fix(1, [1, 1])
    m.fix(2, [0, 1])
    # Apply load 2x F_y and run nonlinear pushover
    P = 2.0 * F_y
    m.add_nodal_load(2, [P, 0])
    NonlinearStaticAnalysis(
        m, num_steps=10, dlambda=1.0 / 10, tol=1e-6, max_iter=30,
    ).run()
    u = m.node(2).disp[0]
    # Expected: u = d_y + (P - F_y) / K2
    expected_u = d_y + (P - F_y) / K2
    assert u == pytest.approx(expected_u, rel=1e-2)


# ====================================================== Friction pendulum

def test_friction_pendulum_construction_validates_params():
    with pytest.raises(ValueError, match="mu"):
        friction_pendulum(1, (1, 2), mu=0.0, R=2.0, W=1e6,
                            dofs_per_node=2)
    with pytest.raises(ValueError, match="R"):
        friction_pendulum(1, (1, 2), mu=0.05, R=0.0, W=1e6,
                            dofs_per_node=2)
    with pytest.raises(ValueError, match="W"):
        friction_pendulum(1, (1, 2), mu=0.05, R=2.0, W=-1.0,
                            dofs_per_node=2)


def test_friction_pendulum_yields_at_friction_force():
    """The friction force is mu * W; lateral force at first slip
    equals mu * W."""
    mu = 0.05
    R = 2.0
    W = 1.0e6     # 1000 kN vertical load
    F_f = mu * W   # = 50 kN
    K2 = W / R     # = 5e5 N/m (pendulum restoring stiffness)
    mat = ElasticIsotropic(1, E=2e11, nu=0.3)
    m = Model(ndm=2, ndf=2); m.add_material(mat)
    m.add_node(1, 0, 0); m.add_node(2, 0, 0)
    m.add_element(friction_pendulum(
        1, (1, 2), mu=mu, R=R, W=W, dofs_per_node=2,
    ))
    m.fix(1, [1, 1])
    m.fix(2, [0, 1])
    # Apply load 2 * F_f -> structure has slipped
    P = 2.0 * F_f
    m.add_nodal_load(2, [P, 0])
    NonlinearStaticAnalysis(
        m, num_steps=20, dlambda=1.0 / 20, tol=1e-6, max_iter=30,
    ).run()
    u = m.node(2).disp[0]
    # With K_initial_factor=100 (default), K1 = 100 * K2 = 5e7. So
    # d_y_slip = F_f / (K1 - K2) ≈ F_f / K1
    K1 = 100.0 * K2
    d_y_slip = F_f / (K1 - K2)
    F_y_slip = K1 * d_y_slip
    expected_u = d_y_slip + (P - F_y_slip) / K2
    assert u == pytest.approx(expected_u, rel=2e-2)
