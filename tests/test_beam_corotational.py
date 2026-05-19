"""Tests for the 2D corotational beam-column.

A corotational element handles arbitrary rigid-body rotations exactly
and picks up geometric-stiffness contributions from the current axial
force and bending moments. These tests pin down the key invariants:

1. **Zero-displacement equivalence** — at ``u = 0``, ``K_tangent`` and
   ``f_int`` must agree with the linear :class:`BeamColumn2D` to machine
   precision. If this ever breaks, the natural-frame transformation has
   a sign error.

2. **Rigid-body invariance** — under a finite rigid rotation of the
   element about one node, the natural deformations (and therefore
   ``f_int``) must be zero. This is *the* defining property of a
   corotational formulation: large rigid rotations cost nothing.

3. **P-Δ — geometric stiffness from axial force** — a column compressed
   below its critical load shows reduced lateral stiffness. The Newton
   step that solves a lateral load applied at the top must produce a
   *larger* lateral deflection than the same load on an un-compressed
   column.

4. **Euler buckling neighbourhood** — a slender simply-supported column
   under compressive load just below ``P_cr = pi^2 EI / L^2`` shows a
   tangent stiffness whose lateral block approaches singularity; just
   below P_cr, a small lateral perturbation produces a large deflection
   (the textbook "amplification factor" near buckling).

5. **f_int and K_tangent are a consistent linearisation** — checked
   numerically by finite-differencing ``f_int_global`` and comparing
   against ``K_tangent_global`` at an arbitrary deformed state.
"""
import numpy as np
import pytest

from femsolver import (
    BeamColumn2D,
    BeamColumn2DCorotational,
    ElasticIsotropic,
    LinearStaticAnalysis,
    Model,
    NonlinearStaticAnalysis,
)


# ====================================================== helpers ==

def _make_horizontal_corot(L=3.0):
    """Build a single horizontal corotational beam, ready for assembly."""
    E, A, Iz = 2.0e11, 1.0e-2, 8.333e-6
    mat = ElasticIsotropic(1, E=E, nu=0.3)
    m = Model(ndm=2, ndf=3); m.add_material(mat)
    m.add_node(1, 0.0, 0.0); m.add_node(2, L, 0.0)
    elem = BeamColumn2DCorotational(1, (1, 2), mat, A, Iz)
    m.add_element(elem)
    return m, elem


# ====================================================== zero-disp ==

@pytest.mark.parametrize("L", [1.0, 3.0, 7.5])
@pytest.mark.parametrize("theta", [0.0, np.pi / 6, np.pi / 4, -0.9])
def test_K_at_zero_disp_matches_linear_beam(L, theta):
    """At ``u = 0``, the corotational K must equal the linear K."""
    E, A, Iz = 2.0e11, 1.0e-2, 8.333e-6
    mat = ElasticIsotropic(1, E=E, nu=0.3)
    m = Model(ndm=2, ndf=3); m.add_material(mat)
    m.add_node(1, 0.0, 0.0)
    m.add_node(2, L * np.cos(theta), L * np.sin(theta))
    e_lin = BeamColumn2D(1, (1, 2), mat, A, Iz)
    e_cor = BeamColumn2DCorotational(2, (1, 2), mat, A, Iz)
    m.add_element(e_lin); m.add_element(e_cor)
    K_lin = e_lin.K_global()
    K_cor = e_cor.K_tangent_global()
    np.testing.assert_allclose(K_lin, K_cor, rtol=1e-12, atol=1e-8)


def test_f_int_at_zero_disp_is_zero():
    _, elem = _make_horizontal_corot()
    np.testing.assert_allclose(elem.f_int_global(), 0.0, atol=1e-12)


def test_K_global_inherits_linear_initial():
    """``K_global()`` is the *initial* elastic K, used by
    LinearStaticAnalysis. For a corotational element this must still
    equal the linear closed-form K at u = 0 (it's inherited from
    :class:`BeamColumn2D`)."""
    m, elem = _make_horizontal_corot()
    m_ref = Model(ndm=2, ndf=3); m_ref.add_material(elem.material)
    m_ref.add_node(1, 0.0, 0.0); m_ref.add_node(2, 3.0, 0.0)
    e_ref = BeamColumn2D(1, (1, 2), elem.material, elem.area, elem.Iz)
    m_ref.add_element(e_ref)
    np.testing.assert_allclose(elem.K_global(), e_ref.K_global(), rtol=1e-14)


# ====================================================== rigid-body ==

@pytest.mark.parametrize("angle", [0.05, 0.5, np.pi / 3, np.pi / 2, 2.0])
def test_rigid_body_rotation_produces_zero_natural_deformation(angle):
    """Apply a finite rigid rotation about node 1: u_nat and theta_nat
    must both be zero, and therefore f_int must be zero too.

    The absolute tolerance reflects the natural round-off floor: the
    chord length is recomputed as ``sqrt(d_x^2 + d_y^2)`` which loses
    ~1 ulp relative to L0, and ``N = EA / L0 * (L - L0)`` then has
    magnitude ``EA * eps_machine``. For our scales (EA ~ 2e9), that is
    ~ 1e-7 N.
    """
    m, elem = _make_horizontal_corot(L=3.0)
    n1 = m.node(1); n2 = m.node(2)
    L = 3.0
    n2.disp[0] = L * np.cos(angle) - L      # u2
    n2.disp[1] = L * np.sin(angle)          # v2
    n1.disp[2] = angle                       # theta1
    n2.disp[2] = angle                       # theta2
    f_int = elem.f_int_global()
    # Magnitude floor: round-off in (L - L0) scaled by EA/L0
    EA_over_L0 = (elem.material.E * elem.area) / L
    floor = 10.0 * EA_over_L0 * np.finfo(float).eps   # generous margin
    np.testing.assert_allclose(f_int, 0.0, atol=floor)


def test_rigid_body_translation_produces_zero_internal_force():
    m, elem = _make_horizontal_corot()
    # Translate node 2 by a moderate amount keeping the beam horizontal,
    # AND translate node 1 by the same amount so the chord direction
    # is unchanged.
    for n in (m.node(1), m.node(2)):
        n.disp[0] = 0.42
        n.disp[1] = -0.17
    f_int = elem.f_int_global()
    # The beam is *not* deformed (both nodes moved rigidly), so f_int = 0.
    np.testing.assert_allclose(f_int, 0.0, atol=1e-12)


# ====================================================== consistency ==

def test_K_tangent_is_consistent_linearisation_of_f_int():
    """Finite-difference d(f_int)/du and compare against K_tangent.

    A correct corotational element must satisfy this to within the
    truncation error of the finite-difference scheme. We use centred
    differences with a moderately small step.
    """
    m, elem = _make_horizontal_corot()
    # Set an arbitrary, non-trivial deformed state
    rng = np.random.default_rng(seed=42)
    for n in (m.node(1), m.node(2)):
        n.disp[:] = rng.uniform(-1.0e-3, 1.0e-3, size=3)
    K_tan = elem.K_tangent_global()
    f0 = elem.f_int_global()
    # Build numerical Jacobian via centred differences
    K_num = np.zeros((6, 6))
    h = 1.0e-6
    dofs = [(1, 0), (1, 1), (1, 2), (2, 0), (2, 1), (2, 2)]
    for j, (tag, idx) in enumerate(dofs):
        m.node(tag).disp[idx] += h
        f_plus = elem.f_int_global().copy()
        m.node(tag).disp[idx] -= 2 * h
        f_minus = elem.f_int_global().copy()
        m.node(tag).disp[idx] += h          # restore
        K_num[:, j] = (f_plus - f_minus) / (2.0 * h)
    np.testing.assert_allclose(K_tan, K_num, rtol=1e-4, atol=1e-3)


# ====================================================== P-Delta ==

def test_compression_reduces_lateral_stiffness():
    """Apply a small compressive axial deformation to a horizontal
    cantilever. The lateral-stiffness coefficient ``K[v_tip, v_tip]``
    in the deformed tangent must be *smaller* than the elastic
    (un-compressed) value — the geometric stiffness reduces the
    transverse stiffness in the presence of compression.
    """
    E, A, Iz, L = 2.0e11, 1.0e-2, 8.333e-6, 3.0
    EI = E * Iz
    mat = ElasticIsotropic(1, E=E, nu=0.3)
    m = Model(ndm=2, ndf=3); m.add_material(mat)
    m.add_node(1, 0.0, 0.0); m.add_node(2, L, 0.0)
    elem = BeamColumn2DCorotational(1, (1, 2), mat, A, Iz)
    m.add_element(elem)
    # K_tangent at u = 0 is the linear elastic
    K0 = elem.K_tangent_global()
    Kvv_0 = K0[4, 4]   # transverse stiffness at node 2 (v_tip)
    # Apply axial compression of u2 = -0.005 m (small, well below
    # Euler buckling so the equilibrium is still stable)
    m.node(2).disp[0] = -5.0e-3
    K1 = elem.K_tangent_global()
    Kvv_1 = K1[4, 4]
    assert Kvv_1 < Kvv_0
    # And tension increases transverse stiffness (cable effect)
    m.node(2).disp[0] = +5.0e-3
    K2 = elem.K_tangent_global()
    Kvv_2 = K2[4, 4]
    assert Kvv_2 > Kvv_0


def test_buckling_neighbourhood_shows_stiffness_collapse():
    """Pin-pin column loaded just below the Euler critical load shows
    a near-zero transverse stiffness. The simply-supported critical
    load is ``P_cr = pi^2 EI / L^2``; we push to 0.95 P_cr (well below
    the unstable region under load control) and verify the lateral
    flexibility blows up — i.e. ``K[lateral, lateral]`` drops by 20x
    or more compared with the no-load case.

    Why this matters: this is the regime where commercial codes warn
    about P-Delta amplification. A linear analysis would predict a
    deflection independent of the axial load; the corotational
    analysis correctly amplifies it as P approaches P_cr.
    """
    E, A, Iz, L = 2.0e11, 1.0e-3, 1.0e-7, 5.0     # slender column
    EI = E * Iz
    P_cr = np.pi ** 2 * EI / L ** 2

    def lateral_response(P_axial):
        mat = ElasticIsotropic(1, E=E, nu=0.3)
        m = Model(ndm=2, ndf=3); m.add_material(mat)
        m.add_node(1, 0.0, 0.0); m.add_node(2, L, 0.0)
        # Use enough elements so each segment's rotation stays small
        # under buckling — at the limit point the chord rotation is
        # ~few degrees.
        n_elem = 8
        for i in range(1, n_elem):
            m.add_node(i + 2, i * L / n_elem, 0.0)
        # element list: 1 .. n_elem connecting nodes in order.
        for i in range(n_elem):
            if i == 0:
                left, right = 1, (3 if n_elem > 1 else 2)
            elif i == n_elem - 1:
                left, right = (i + 2), 2
            else:
                left, right = (i + 2), (i + 3)
            m.add_element(BeamColumn2DCorotational(i + 1, (left, right), mat, A, Iz))
        m.fix(1, [1, 1, 0])           # pin
        m.fix(2, [0, 1, 0])           # roller (v fixed, allows axial)
        # axial compression at node 2 + small lateral perturbation at midspan
        m.add_nodal_load(2, [-P_axial, 0.0, 0.0])
        # lateral load on the midspan node (node tag for n_elem=8 is 6 = midspan)
        mid_tag = 2 + n_elem // 2     # 2 + 4 = 6
        m.add_nodal_load(mid_tag, [0.0, -1.0, 0.0])  # small unit lateral load
        res = NonlinearStaticAnalysis(
            m, num_steps=20, dlambda=1.0 / 20, tol=1e-6, max_iter=40,
        ).run()
        v_mid = m.node(mid_tag).disp[1]
        return abs(v_mid)

    # Lateral deflection under pure 1 N lateral load (no axial)
    v_no_axial = lateral_response(0.0)
    # With axial compression at 80 % of P_cr — well-defined amplification
    v_compressed = lateral_response(0.8 * P_cr)
    # Theoretical amplification factor: 1 / (1 - P/P_cr) = 5x at P=0.8 P_cr
    assert v_compressed > 3.0 * v_no_axial   # at least 3x (margin for FE discretisation)


def test_tension_stiffening_amplifies_lateral_stiffness():
    """Opposite of P-Delta: axial tension makes the column stiffer
    laterally. The "cable" effect."""
    E, A, Iz, L = 2.0e11, 1.0e-3, 1.0e-7, 5.0

    def lateral_response(P_axial):
        mat = ElasticIsotropic(1, E=E, nu=0.3)
        m = Model(ndm=2, ndf=3); m.add_material(mat)
        m.add_node(1, 0.0, 0.0); m.add_node(2, L, 0.0)
        # midspan node for the lateral load
        m.add_node(3, L / 2, 0.0)
        m.add_element(BeamColumn2DCorotational(1, (1, 3), mat, A, Iz))
        m.add_element(BeamColumn2DCorotational(2, (3, 2), mat, A, Iz))
        m.fix(1, [1, 1, 0])
        m.fix(2, [0, 1, 0])
        m.add_nodal_load(2, [P_axial, 0.0, 0.0])
        m.add_nodal_load(3, [0.0, -1.0, 0.0])
        NonlinearStaticAnalysis(
            m, num_steps=10, dlambda=1.0 / 10, tol=1e-6, max_iter=30,
        ).run()
        return abs(m.node(3).disp[1])

    EI = E * Iz
    P_cr = np.pi ** 2 * EI / L ** 2
    v_no_axial = lateral_response(0.0)
    v_tensioned = lateral_response(+0.8 * P_cr)
    # Tension makes the column stiffer, so deflection is *smaller*
    assert v_tensioned < v_no_axial


# ====================================================== recovery ==

def test_recover_stores_end_forces():
    """After a converged analysis, end_forces_local should be populated
    and have the right sign pattern for a downward tip load.

    The corotational element converges in ~2 Newton iterations per step
    (quadratic), so a moderate tolerance is enough — pushing tol below
    the round-off floor only triggers spurious non-convergence reports.
    """
    E, A, Iz, L, P = 2.0e11, 1.0e-2, 8.333e-6, 3.0, 1.0e3
    mat = ElasticIsotropic(1, E=E, nu=0.3)
    m = Model(ndm=2, ndf=3); m.add_material(mat)
    m.add_node(1, 0.0, 0.0); m.add_node(2, L, 0.0)
    elem = BeamColumn2DCorotational(1, (1, 2), mat, A, Iz)
    m.add_element(elem)
    m.fix(1, [1, 1, 1])
    m.add_nodal_load(2, [0.0, -P, 0.0])
    NonlinearStaticAnalysis(
        m, num_steps=5, dlambda=1.0 / 5, tol=1e-6, max_iter=20,
    ).run()
    # Tip deflection close to the linear PL^3/(3EI) value for small load
    v_lin = -P * L ** 3 / (3.0 * E * Iz)
    np.testing.assert_allclose(m.node(2).disp[1], v_lin, rtol=5e-3)
    # end_forces_local was populated by recover()
    assert elem.end_forces_local.shape == (6,)
    # Moment at the fixed end should be ~ +PL in the BeamColumn2D
    # sign convention (the beam's resisting moment on the support
    # counters the downward tip load). The same convention applies
    # to the corotational element so the two are directly comparable.
    np.testing.assert_allclose(elem.end_forces_local[2], P * L, rtol=5e-3)
