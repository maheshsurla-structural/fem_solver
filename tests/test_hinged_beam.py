"""Tests for HingedBeamColumn2D — concentrated-plasticity beam element.

The element is validated in three regimes:

1. **Rigid-spring limit**: with the hinge stiffness driven well above
   the elastic-beam rotational stiffness, the condensed K and the
   resulting displacements must converge to plain :class:`BeamColumn2D`.
2. **Elastic regime**: with finite linear hinges, the cantilever tip
   deflection has a known closed form
   ``v_tip = -P L^3 / 3 EI - P L^2 / K_h`` (for a single end-hinge).
3. **Plastic pushover**: with an EPP hinge at the fixed end of a
   cantilever, the force-displacement curve must be bilinear with yield
   at ``P_y = M_y / L`` and a perfectly-flat plateau afterwards.
"""
import numpy as np
import pytest

from femsolver import (
    BeamColumn2D,
    BilinearMomentRotationSpring,
    ElasticIsotropic,
    HingedBeamColumn2D,
    LinearStaticAnalysis,
    Model,
    NonlinearStaticAnalysis,
)


# ============================================================ rigid-spring

@pytest.mark.parametrize("hinge_pos", ["i", "j", "both"])
def test_rigid_hinges_match_plain_beam(hinge_pos):
    """A hinge with K0 >> 4EI/L behaves like no hinge at all — the
    condensed K_local must agree with BeamColumn2D's K_local."""
    E, A, Iz, L = 2.0e11, 1.0e-2, 8.333e-6, 3.0
    EI_per_L = 4.0 * E * Iz / L
    K_rigid = 1.0e8 * EI_per_L     # 8 orders of magnitude above the beam term
    M_huge = 1.0e30                 # never yields

    mat = ElasticIsotropic(1, E=E, nu=0.3)
    m = Model(ndm=2, ndf=3); m.add_material(mat)
    m.add_node(1, 0.0, 0.0); m.add_node(2, L, 0.0)
    plain = BeamColumn2D(1, (1, 2), mat, A, Iz)
    m.add_element(plain)

    m2 = Model(ndm=2, ndf=3); m2.add_material(mat)
    m2.add_node(1, 0.0, 0.0); m2.add_node(2, L, 0.0)
    if hinge_pos == "i":
        h_i = BilinearMomentRotationSpring(K0=K_rigid, My=M_huge); h_j = None
    elif hinge_pos == "j":
        h_i = None; h_j = BilinearMomentRotationSpring(K0=K_rigid, My=M_huge)
    else:
        h_i = BilinearMomentRotationSpring(K0=K_rigid, My=M_huge)
        h_j = BilinearMomentRotationSpring(K0=K_rigid, My=M_huge)
    hinged = HingedBeamColumn2D(1, (1, 2), mat, A, Iz, hinge_i=h_i, hinge_j=h_j)
    m2.add_element(hinged)

    np.testing.assert_allclose(plain.K_local(), hinged.K_local(),
                               rtol=1e-7, atol=1e-3)


def test_rejects_no_hinges():
    mat = ElasticIsotropic(1, E=2.0e11, nu=0.3)
    with pytest.raises(ValueError):
        HingedBeamColumn2D(1, (1, 2), mat, 1.0e-2, 8.333e-6)


def test_rejects_both_section_and_legacy():
    from femsolver import ElasticSection2D
    mat = ElasticIsotropic(1, E=2.0e11, nu=0.3)
    sec = ElasticSection2D(E=2.0e11, A=1.0e-2, Iz=8.333e-6)
    h = BilinearMomentRotationSpring(K0=1e6, My=1e3)
    with pytest.raises(ValueError):
        HingedBeamColumn2D(1, (1, 2), mat, area=1e-2, Iz=1e-6, section=sec, hinge_i=h)


# ============================================================== elastic

def test_elastic_cantilever_with_one_finite_hinge():
    """Cantilever with linear-elastic spring at the fixed end::

        v_tip = -P L^3 / 3 EI  -  P L^2 / K_h

    The added flexibility from the spring is :math:`P L^2 / K_h`,
    coming from the moment ``M = P L`` at the spring producing a hinge
    rotation of ``P L / K_h`` which translates the tip by ``P L^2 / K_h``.
    """
    E, A, Iz, L, P = 2.0e11, 1.0e-2, 8.333e-6, 3.0, 1.0e3
    K_h = 4.0 * E * Iz / L         # comparable order to beam stiffness
    My = 1.0e30                     # stays elastic
    h_i = BilinearMomentRotationSpring(K0=K_h, My=My)
    mat = ElasticIsotropic(1, E=E, nu=0.3)
    m = Model(ndm=2, ndf=3); m.add_material(mat)
    m.add_node(1, 0.0, 0.0); m.add_node(2, L, 0.0)
    elem = HingedBeamColumn2D(1, (1, 2), mat, A, Iz, hinge_i=h_i)
    m.add_element(elem)
    m.fix(1, [1, 1, 1])
    m.add_nodal_load(2, [0.0, -P, 0.0])
    LinearStaticAnalysis(m).run()
    v_tip = m.node(2).disp[1]
    expected = -P * L ** 3 / (3.0 * E * Iz) - P * L ** 2 / K_h
    assert v_tip == pytest.approx(expected, rel=1e-10)


def test_elastic_axial_unaffected_by_hinges():
    """Axial behaviour is decoupled from rotational hinges; the axial
    flexibility of the element is just L/EA."""
    E, A, Iz, L, P = 2.0e11, 1.0e-2, 8.333e-6, 3.0, 1.0e4
    h = BilinearMomentRotationSpring(K0=1e3, My=1.0)   # very weak hinge
    mat = ElasticIsotropic(1, E=E, nu=0.3)
    m = Model(ndm=2, ndf=3); m.add_material(mat)
    m.add_node(1, 0.0, 0.0); m.add_node(2, L, 0.0)
    elem = HingedBeamColumn2D(1, (1, 2), mat, A, Iz,
                              hinge_i=h,
                              hinge_j=BilinearMomentRotationSpring(K0=1e3, My=1.0))
    m.add_element(elem)
    m.fix(1, [1, 1, 1])
    m.fix(2, [0, 1, 1])
    m.add_nodal_load(2, [P, 0.0, 0.0])
    LinearStaticAnalysis(m).run()
    assert m.node(2).disp[0] == pytest.approx(P * L / (E * A), rel=1e-12)


# ============================================ plastic pushover

def _build_cantilever_with_hinge_at_base(b: float = 0.0):
    """Cantilever with a hinge at the fixed end. Tip-load DOF = (2, 1).

    Note: with EPP (``b = 0``) the structure becomes a kinematic
    mechanism the moment the hinge yields — a cantilever with a free
    rotational connection at the support has zero stiffness in pure
    rotation. Load control can therefore only converge for P < P_y.
    Past-yield pushovers use ``b > 0`` so the hinge retains some
    rotational stiffness and the structure remains stable.
    """
    E, A, Iz, L = 2.0e11, 1.0e-2, 8.333e-6, 3.0
    # Spring stiffness comparable to the beam's rotational stiffness 4EI/L,
    # so that the post-yield softening at b > 0 actually shows up in the
    # tip force-displacement curve. With K_h >> 4EI/L the beam flexibility
    # dominates and the elastic vs. post-yield slopes look nearly identical.
    K_h = 4.0 * E * Iz / L
    My = 5.0e3                          # yield moment (N.m)
    h = BilinearMomentRotationSpring(K0=K_h, My=My, b=b)
    mat = ElasticIsotropic(1, E=E, nu=0.3)
    m = Model(ndm=2, ndf=3); m.add_material(mat)
    m.add_node(1, 0.0, 0.0); m.add_node(2, L, 0.0)
    elem = HingedBeamColumn2D(1, (1, 2), mat, A, Iz, hinge_i=h)
    m.add_element(elem)
    m.fix(1, [1, 1, 1])
    return m, elem, L, My, K_h, E, Iz


def test_epp_just_below_yield_stays_elastic():
    """Load to 90% of P_y with an EPP hinge: hinge stays elastic, no
    plastic rotation accumulates."""
    m, elem, L, My, K_h, E, Iz = _build_cantilever_with_hinge_at_base(b=0.0)
    P_yield = My / L
    P_max = 0.9 * P_yield               # safely below yield
    num_steps = 10
    dlambda = 1.0 / num_steps
    m.add_nodal_load(2, [0.0, -P_max, 0.0])
    NonlinearStaticAnalysis(
        m, num_steps=num_steps, dlambda=dlambda, tol=1e-8, max_iter=30,
    ).run()
    assert elem.hinge_i.theta_p_committed == 0.0
    # Hinge moment should be just below My
    assert abs(elem.hinge_i.M_trial) < My
    assert abs(elem.hinge_i.M_trial) > 0.85 * My


def test_pushover_yield_with_bilinear_hinge():
    """Bilinear (b > 0) hinge at base: load past P_y, verify the hinge
    yielded and accumulated plastic rotation."""
    m, elem, L, My, K_h, E, Iz = _build_cantilever_with_hinge_at_base(b=0.05)
    P_yield = My / L
    P_max = 1.5 * P_yield
    num_steps = 30
    dlambda = 1.0 / num_steps
    m.add_nodal_load(2, [0.0, -P_max, 0.0])
    NonlinearStaticAnalysis(
        m, num_steps=num_steps, dlambda=dlambda, tol=1e-8, max_iter=30,
    ).run()
    assert elem.hinge_i.theta_p_committed > 0.0


def test_pushover_force_displacement_is_bilinear():
    """Track tip displacement vs lambda. The post-yield slope must be
    much softer than the elastic slope but not zero (b > 0 case).
    """
    m, elem, L, My, K_h, E, Iz = _build_cantilever_with_hinge_at_base(b=0.05)
    P_yield = My / L
    P_max = 2.0 * P_yield
    num_steps = 40
    dlambda = 1.0 / num_steps
    m.add_nodal_load(2, [0.0, -P_max, 0.0])
    res = NonlinearStaticAnalysis(
        m, num_steps=num_steps, dlambda=dlambda,
        track=(2, 1), tol=1e-8, max_iter=30,
    ).run()
    lambdas = np.array(res["lambdas"])
    disps = np.array(res["tracked"])
    forces = lambdas * P_max
    below_yield = forces < 0.4 * P_yield
    above_yield = forces > 1.4 * P_yield
    assert below_yield.any() and above_yield.any()
    k_below = (forces[below_yield][-1] - forces[below_yield][0]) / abs(
        disps[below_yield][-1] - disps[below_yield][0]
    )
    k_above = (forces[above_yield][-1] - forces[above_yield][0]) / abs(
        disps[above_yield][-1] - disps[above_yield][0]
    )
    assert k_above < 0.5 * k_below
    assert k_above > 0.0


def test_epp_load_past_yield_diverges():
    """EPP hinge at the base of a cantilever forms a mechanism the
    moment it yields. Trying to push past ``P_y`` under load control
    must fail to converge — there is no equilibrium configuration.
    """
    m, elem, L, My, K_h, E, Iz = _build_cantilever_with_hinge_at_base(b=0.0)
    P_yield = My / L
    P_max = 1.5 * P_yield
    num_steps = 30
    dlambda = 1.0 / num_steps
    m.add_nodal_load(2, [0.0, -P_max, 0.0])
    with pytest.raises((RuntimeError, Exception)):
        NonlinearStaticAnalysis(
            m, num_steps=num_steps, dlambda=dlambda, tol=1e-8, max_iter=30,
        ).run()


# ============================================ bilinear (b > 0) pushover

def test_bilinear_pushover_post_yield_slope():
    """With ``b = 0.1`` the spring's post-yield tangent is ``0.1 * K0``.
    The structural post-yield stiffness is therefore non-zero, and the
    pushover curve continues to rise with a clear bilinear shape."""
    E, A, Iz, L = 2.0e11, 1.0e-2, 8.333e-6, 3.0
    K_h = 1.0e3 * 4.0 * E * Iz / L
    My = 5.0e3
    b = 0.1
    h = BilinearMomentRotationSpring(K0=K_h, My=My, b=b)
    mat = ElasticIsotropic(1, E=E, nu=0.3)
    m = Model(ndm=2, ndf=3); m.add_material(mat)
    m.add_node(1, 0.0, 0.0); m.add_node(2, L, 0.0)
    elem = HingedBeamColumn2D(1, (1, 2), mat, A, Iz, hinge_i=h)
    m.add_element(elem)
    m.fix(1, [1, 1, 1])
    P_yield = My / L
    P_max = 2.0 * P_yield
    num_steps = 40
    dlambda = 1.0 / num_steps
    m.add_nodal_load(2, [0.0, -P_max, 0.0])
    res = NonlinearStaticAnalysis(
        m, num_steps=num_steps, dlambda=dlambda,
        track=(2, 1), tol=1e-8, max_iter=30,
    ).run()
    lambdas = np.array(res["lambdas"])
    disps = np.array(res["tracked"])
    forces = lambdas * P_max
    below = forces < 0.5 * P_yield
    above = forces > 1.5 * P_yield
    k_below = (forces[below][-1] - forces[below][0]) / abs(disps[below][-1] - disps[below][0])
    k_above = (forces[above][-1] - forces[above][0]) / abs(disps[above][-1] - disps[above][0])
    # post-yield is softer than elastic but NOT vanishing
    assert k_above < k_below
    assert k_above > 0.02 * k_below


# =========================================================== state revert

def test_revert_undoes_uncommitted_hinge_flow():
    """After running an analysis to convergence, calling revert_state
    on the element should roll the spring's trial state back to the
    last commit. This is the contract relied on when a Newton step
    fails. Uses ``b = 0.05`` so the post-yield analysis is stable.
    """
    m, elem, L, My, K_h, E, Iz = _build_cantilever_with_hinge_at_base(b=0.05)
    P_max = 1.3 * (My / L)
    m.add_nodal_load(2, [0.0, -P_max, 0.0])
    NonlinearStaticAnalysis(
        m, num_steps=20, dlambda=1.0 / 20, tol=1e-8, max_iter=30,
    ).run()
    tp_committed = elem.hinge_i.theta_p_committed
    assert tp_committed > 0.0     # sanity: did yield
    # mutate trial state, then revert
    elem.hinge_i.theta_p_trial = tp_committed + 1.0
    elem.revert_state()
    assert elem.hinge_i.theta_p_trial == tp_committed
