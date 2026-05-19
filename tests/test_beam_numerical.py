"""Tests for the numerically-integrated beam-column stiffness path.

The closed-form ``K_local`` is the analytical integral of
:math:`B^T k_s B` along the element. Replacing that integral with
Gauss-Lobatto quadrature must reproduce the closed form to machine
precision when:

* the section is elastic (so ``k_s`` is constant in ``x``), and
* the rule has enough points to integrate the integrand exactly.

For Euler-Bernoulli bending the integrand is quadratic in ``xi``
(the second derivatives of the Hermite cubic basis are linear in ``xi``).
Gauss-Lobatto with ``n`` points is exact for polynomials of degree
``2n - 3``, so ``n_int >= 3`` is the minimum that reproduces the
closed-form ``K``. ``n_int = 2`` is exact for degree 1 only and is
**deliberately** included as a negative test so we know the integrator
is actually being exercised.
"""
import numpy as np
import pytest

from femsolver import (
    BeamColumn2D,
    BeamColumn3D,
    ElasticIsotropic,
    LinearStaticAnalysis,
    Model,
)


# --------------------------------------------------------------- helpers ----

def _make_2d_beam(L=3.0, theta=0.0):
    """Build a single BeamColumn2D oriented at angle ``theta`` from +x."""
    E, A, Iz = 2.0e11, 1.0e-2, 8.333e-6
    m = Model(ndm=2, ndf=3)
    m.add_node(1, 0.0, 0.0)
    m.add_node(2, L * np.cos(theta), L * np.sin(theta))
    mat = ElasticIsotropic(1, E=E, nu=0.3)
    m.add_material(mat)
    elem = BeamColumn2D(1, (1, 2), mat, A, Iz)
    m.add_element(elem)
    return m, elem


def _make_3d_beam(L=3.0):
    E, nu = 2.0e11, 0.3
    A, Iy, Iz, J = 1.0e-2, 5.0e-6, 8.333e-6, 1.4e-5
    m = Model(ndm=3, ndf=6)
    m.add_node(1, 0.0, 0.0, 0.0)
    m.add_node(2, L, 0.0, 0.0)
    mat = ElasticIsotropic(1, E=E, nu=nu)
    m.add_material(mat)
    elem = BeamColumn3D(1, (1, 2), mat, A, Iy, Iz, J)
    m.add_element(elem)
    return m, elem


# ============================================================ B-matrix tests

def test_b_matrix_2d_axial_constant_in_xi():
    """Axial row of B is constant in xi (linear shape function)."""
    _, elem = _make_2d_beam()
    L = 3.0
    for xi in (-1.0, -0.3, 0.0, 0.7, 1.0):
        B = elem._strain_disp_matrix(xi, L)
        assert B[0, 0] == pytest.approx(-1.0 / L)
        assert B[0, 3] == pytest.approx(1.0 / L)
        # only axial DOFs (0, 3) appear in row 0
        assert B[0, 1] == B[0, 2] == B[0, 4] == B[0, 5] == 0.0


def test_b_matrix_2d_bending_at_endpoints():
    """Bending B-row at xi=-1 (x=0) and xi=+1 (x=L) reproduces the textbook
    second-derivative values of the Hermite cubic basis."""
    _, elem = _make_2d_beam()
    L = 3.0
    B_left = elem._strain_disp_matrix(-1.0, L)
    np.testing.assert_allclose(
        B_left[1, [1, 2, 4, 5]],
        np.array([-6.0 / L**2, -4.0 / L, 6.0 / L**2, -2.0 / L]),
        rtol=1e-14,
    )
    B_right = elem._strain_disp_matrix(1.0, L)
    np.testing.assert_allclose(
        B_right[1, [1, 2, 4, 5]],
        np.array([6.0 / L**2, 2.0 / L, -6.0 / L**2, 4.0 / L]),
        rtol=1e-14,
    )


# ================================================ 2D K equivalence ==========

@pytest.mark.parametrize("theta", [0.0, np.pi / 6, np.pi / 4, -0.7])
@pytest.mark.parametrize("n_int", [3, 4, 5, 6, 8])
def test_K2d_numerical_matches_closed_form(theta, n_int):
    """Numerical = closed form to machine precision for n_int >= 3."""
    _, elem = _make_2d_beam(theta=theta)
    K_cf = elem._K_local_closed_form()
    elem.n_int = n_int
    elem.use_numerical_integration = True
    K_num = elem.K_local()
    np.testing.assert_allclose(K_cf, K_num, rtol=1e-12, atol=1e-9)


def test_K2d_numerical_is_symmetric():
    """A small but non-trivial extra check: numerical K must be symmetric."""
    _, elem = _make_2d_beam()
    elem.use_numerical_integration = True
    K = elem.K_local()
    np.testing.assert_allclose(K, K.T, rtol=1e-12, atol=1e-9)


def test_K2d_numerical_n2_is_inexact():
    """Sanity: with only 2 Gauss-Lobatto points the bending block CANNOT
    integrate the quadratic-in-xi integrand exactly. If this test fails
    we are not really exercising the numerical path."""
    _, elem = _make_2d_beam()
    K_cf = elem._K_local_closed_form()
    elem.n_int = 2
    elem.use_numerical_integration = True
    K_num = elem.K_local()
    # axial part (DOFs 0, 3) is exact even with n=2 because the integrand is
    # constant in xi
    np.testing.assert_allclose(K_cf[0, 0], K_num[0, 0], rtol=1e-12)
    # bending diagonal must NOT match
    assert not np.isclose(K_cf[1, 1], K_num[1, 1], rtol=1e-3)


def test_K2d_global_via_numerical_path():
    """K_global must use whichever K_local path is active."""
    _, elem = _make_2d_beam(theta=np.pi / 5)
    K_cf = elem.K_global()
    elem.use_numerical_integration = True
    K_num = elem.K_global()
    np.testing.assert_allclose(K_cf, K_num, rtol=1e-12, atol=1e-9)


# =================================== 2D end-to-end analysis equivalence =====

def test_beam2d_cantilever_full_analysis_matches():
    """Full LinearStaticAnalysis with numerical-integration path must give
    the same displacements as the closed-form path to machine precision."""
    E, A, Iz, L, P = 2.0e11, 1.0e-2, 8.333e-6, 3.0, 1.0e3

    def build(use_num: bool):
        m = Model(ndm=2, ndf=3)
        m.add_node(1, 0.0, 0.0)
        m.add_node(2, L, 0.0)
        mat = ElasticIsotropic(1, E=E, nu=0.3)
        m.add_material(mat)
        elem = BeamColumn2D(1, (1, 2), mat, A, Iz)
        elem.use_numerical_integration = use_num
        m.add_element(elem)
        m.fix(1, [1, 1, 1])
        m.add_nodal_load(2, [0.0, -P, 0.0])
        LinearStaticAnalysis(m).run()
        return m

    m_cf = build(False)
    m_num = build(True)
    np.testing.assert_allclose(
        m_cf.node(2).disp, m_num.node(2).disp, rtol=1e-12, atol=1e-14
    )
    # also matches the analytical solution
    w_expected = -P * L**3 / (3.0 * E * Iz)
    np.testing.assert_allclose(m_num.node(2).disp[1], w_expected, rtol=1e-10)


def test_beam2d_udl_full_analysis_matches():
    """Distributed load goes through f_eq_local (closed form), but the
    stiffness K can come from either path. Verify the displacements still
    match analytical solution under numerical K."""
    E, A, Iz, L, w = 2.0e11, 1.0e-2, 8.333e-6, 3.0, 1.0e3
    m = Model(ndm=2, ndf=3)
    m.add_node(1, 0.0, 0.0)
    m.add_node(2, L, 0.0)
    mat = ElasticIsotropic(1, E=E, nu=0.3)
    m.add_material(mat)
    elem = BeamColumn2D(1, (1, 2), mat, A, Iz)
    elem.use_numerical_integration = True
    m.add_element(elem)
    elem.add_uniform_load(-w)
    m.fix(1, [1, 1, 1])
    LinearStaticAnalysis(m).run()
    w_expected = -w * L**4 / (8.0 * E * Iz)
    np.testing.assert_allclose(m.node(2).disp[1], w_expected, rtol=1e-10)


# ============================================================ 3D B-matrix ==

def test_b_matrix_3d_torsion_constant_in_xi():
    _, elem = _make_3d_beam()
    L = 3.0
    for xi in (-1.0, 0.0, 0.5, 1.0):
        B = elem._strain_disp_matrix(xi, L)
        assert B[3, 3] == pytest.approx(-1.0 / L)
        assert B[3, 9] == pytest.approx(1.0 / L)
        # row 3 has nothing else
        cols_with_value = [j for j in range(12) if B[3, j] != 0.0]
        assert cols_with_value == [3, 9]


def test_b_matrix_3d_bending_y_sign_flip():
    """Bending-about-y rotational columns flip sign relative to bending-z.

    Reason: with local axes (ex, ey, ez) and y = z x x, dw/dx = -theta_y
    (whereas dv/dx = +theta_z). The B-matrix encodes this directly.
    """
    _, elem = _make_3d_beam()
    L = 3.0
    for xi in (-0.5, 0.2, 0.9):
        B = elem._strain_disp_matrix(xi, L)
        # |B[1, 5]| == |B[2, 4]| but signs are opposite
        assert B[1, 5] == pytest.approx(-B[2, 4])
        assert B[1, 11] == pytest.approx(-B[2, 10])
        # translational columns have the same sign
        assert B[1, 1] == pytest.approx(B[2, 2])
        assert B[1, 7] == pytest.approx(B[2, 8])


# ================================================ 3D K equivalence ==========

@pytest.mark.parametrize("n_int", [3, 4, 5, 6])
def test_K3d_numerical_matches_closed_form(n_int):
    _, elem = _make_3d_beam()
    K_cf = elem._K_local_closed_form()
    elem.n_int = n_int
    elem.use_numerical_integration = True
    K_num = elem.K_local()
    np.testing.assert_allclose(K_cf, K_num, rtol=1e-12, atol=1e-8)


def test_K3d_numerical_is_symmetric():
    _, elem = _make_3d_beam()
    elem.use_numerical_integration = True
    K = elem.K_local()
    np.testing.assert_allclose(K, K.T, rtol=1e-12, atol=1e-8)


def test_beam3d_cantilever_full_analysis_matches():
    """Cantilever in 3D: numerical path must produce identical displacements."""
    E, nu = 2.0e11, 0.3
    A, Iy, Iz, J = 1.0e-2, 8.333e-6, 8.333e-6, 1.4e-5
    L, P = 3.0, 1.0e3

    def build(use_num: bool):
        m = Model(ndm=3, ndf=6)
        m.add_node(1, 0.0, 0.0, 0.0)
        m.add_node(2, L, 0.0, 0.0)
        mat = ElasticIsotropic(1, E=E, nu=nu)
        m.add_material(mat)
        elem = BeamColumn3D(1, (1, 2), mat, A, Iy, Iz, J)
        elem.use_numerical_integration = use_num
        m.add_element(elem)
        m.fix(1, [1, 1, 1, 1, 1, 1])
        m.add_nodal_load(2, [0, P, 0, 0, 0, 0])
        LinearStaticAnalysis(m).run()
        return m

    m_cf = build(False)
    m_num = build(True)
    np.testing.assert_allclose(
        m_cf.node(2).disp, m_num.node(2).disp, rtol=1e-12, atol=1e-14
    )


def test_beam3d_torsion_via_numerical_path():
    """Torsion goes through the numerically-integrated K row 3 (gamma)."""
    E, nu = 2.0e11, 0.3
    A, Iy, Iz, J = 1.0e-2, 8.333e-6, 8.333e-6, 1.4e-5
    L, T_load = 3.0, 1.0e2
    m = Model(ndm=3, ndf=6)
    m.add_node(1, 0.0, 0.0, 0.0)
    m.add_node(2, L, 0.0, 0.0)
    mat = ElasticIsotropic(1, E=E, nu=nu)
    G = E / (2.0 * (1 + nu))
    m.add_material(mat)
    elem = BeamColumn3D(1, (1, 2), mat, A, Iy, Iz, J)
    elem.use_numerical_integration = True
    m.add_element(elem)
    m.fix(1, [1, 1, 1, 1, 1, 1])
    m.add_nodal_load(2, [0, 0, 0, T_load, 0, 0])
    LinearStaticAnalysis(m).run()
    phi_expected = T_load * L / (G * J)
    np.testing.assert_allclose(m.node(2).disp[3], phi_expected, rtol=1e-10)


# =========================================== default behavior preserved ====

def test_default_path_is_closed_form():
    """A freshly constructed beam must default to the closed-form path so
    nothing in the rest of the codebase changes behavior."""
    _, e2 = _make_2d_beam()
    assert e2.use_numerical_integration is False
    _, e3 = _make_3d_beam()
    assert e3.use_numerical_integration is False
