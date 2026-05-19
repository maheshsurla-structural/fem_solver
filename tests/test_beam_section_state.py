"""Tests for the per-section recovery output on beam-column elements.

After ``recover()`` runs on a beam, three new arrays are populated:

* ``section_locations[i]`` — local x-coordinate of integration point i
  (always Gauss-Lobatto, so first point is 0 and last is L).
* ``section_strains[i]`` — section generalized strain at point i.
* ``section_forces[i]`` — section generalized force at point i.

The forces along the element must agree with the analytical moment /
axial / torsion / bending diagrams of classical beam theory. These tests
pin those agreements down so that downstream output (and the state
determination loops introduced in Phases 4-5) build on a verified base.
"""
import numpy as np
import pytest

from femsolver import (
    BeamColumn2D,
    BeamColumn3D,
    ElasticIsotropic,
    ElasticSection2D,
    LinearStaticAnalysis,
    Model,
)


# =============================================================== 2D outputs

def test_section_arrays_have_correct_shape_2d():
    """A linear-static run must populate the per-section output arrays."""
    E, A, Iz, L, P = 2.0e11, 1.0e-2, 8.333e-6, 3.0, 1.0e3
    m = Model(ndm=2, ndf=3)
    m.add_node(1, 0.0, 0.0)
    m.add_node(2, L, 0.0)
    mat = ElasticIsotropic(1, E=E, nu=0.3)
    m.add_material(mat)
    elem = BeamColumn2D(1, (1, 2), mat, A, Iz)
    m.add_element(elem)
    m.fix(1, [1, 1, 1])
    m.add_nodal_load(2, [0.0, -P, 0.0])
    LinearStaticAnalysis(m).run()
    n = elem.n_int
    assert elem.section_locations.shape == (n,)
    assert elem.section_strains.shape == (n, 2)
    assert elem.section_forces.shape == (n, 2)


def test_section_endpoints_lie_at_x_zero_and_L():
    """Gauss-Lobatto rule includes endpoints, so the first and last
    section locations must coincide with x=0 and x=L exactly."""
    L = 3.0
    m = Model(ndm=2, ndf=3)
    m.add_node(1, 0.0, 0.0)
    m.add_node(2, L, 0.0)
    mat = ElasticIsotropic(1, E=2.0e11, nu=0.3)
    m.add_material(mat)
    elem = BeamColumn2D(1, (1, 2), mat, 1.0e-2, 8.333e-6)
    m.add_element(elem)
    m.fix(1, [1, 1, 1])
    m.add_nodal_load(2, [0.0, -1.0, 0.0])
    LinearStaticAnalysis(m).run()
    assert elem.section_locations[0] == pytest.approx(0.0, abs=1e-14)
    assert elem.section_locations[-1] == pytest.approx(L, abs=1e-14)


def test_section_moment_diagram_cantilever_tip_load():
    """Cantilever with tip vertical load P: M(x) = -P (L - x).

    Sign is negative because the moment at the fixed end resists the load
    (with our ``kappa = d^2 v / dx^2`` convention and a tip load downward,
    deflection is negative and curvature is negative at the support).
    """
    E, A, Iz, L, P = 2.0e11, 1.0e-2, 8.333e-6, 3.0, 1.0e3
    m = Model(ndm=2, ndf=3)
    m.add_node(1, 0.0, 0.0)
    m.add_node(2, L, 0.0)
    mat = ElasticIsotropic(1, E=E, nu=0.3)
    m.add_material(mat)
    elem = BeamColumn2D(1, (1, 2), mat, A, Iz)
    m.add_element(elem)
    m.fix(1, [1, 1, 1])
    m.add_nodal_load(2, [0.0, -P, 0.0])
    LinearStaticAnalysis(m).run()
    # M_internal(x) is *linear* in x. With our integrator (cubic Hermite
    # shape functions, exact for cubic transverse displacement), the
    # recovered moments must match -P (L - x) at every IP.
    expected_M = -P * (L - elem.section_locations)
    np.testing.assert_allclose(elem.section_forces[:, 1], expected_M,
                               rtol=1e-10, atol=1e-9)


def test_section_moment_at_fixed_end_matches_reaction():
    """The fixed-end section moment must equal the reaction moment at
    the support. Cross-checks recover() and the global reaction vector.
    """
    E, A, Iz, L, P = 2.0e11, 1.0e-2, 8.333e-6, 3.0, 1.0e3
    m = Model(ndm=2, ndf=3)
    m.add_node(1, 0.0, 0.0)
    m.add_node(2, L, 0.0)
    mat = ElasticIsotropic(1, E=E, nu=0.3)
    m.add_material(mat)
    elem = BeamColumn2D(1, (1, 2), mat, A, Iz)
    m.add_element(elem)
    m.fix(1, [1, 1, 1])
    m.add_nodal_load(2, [0.0, -P, 0.0])
    LinearStaticAnalysis(m).run()
    # |reaction moment at node 1| == |internal moment at x=0|
    assert abs(elem.section_forces[0, 1]) == pytest.approx(
        abs(m.node(1).reaction[2]), rel=1e-10
    )


def test_section_axial_force_constant_along_element():
    """Pure axial load: N(x) is constant along the element."""
    E, A, Iz, L, P = 2.0e11, 1.0e-2, 8.333e-6, 3.0, 1.0e4
    m = Model(ndm=2, ndf=3)
    m.add_node(1, 0.0, 0.0)
    m.add_node(2, L, 0.0)
    mat = ElasticIsotropic(1, E=E, nu=0.3)
    m.add_material(mat)
    elem = BeamColumn2D(1, (1, 2), mat, A, Iz)
    m.add_element(elem)
    m.fix(1, [1, 1, 1])
    m.fix(2, [0, 1, 1])
    m.add_nodal_load(2, [P, 0.0, 0.0])
    LinearStaticAnalysis(m).run()
    # Axial force = P at every integration point
    np.testing.assert_allclose(elem.section_forces[:, 0], P, rtol=1e-10)


def test_section_strain_equals_B_times_u_local():
    """Section strain must equal B(xi) @ u_local at each integration point.

    This is the *defining* relation; if it ever breaks, every downstream
    state-determination loop will be broken too.
    """
    E, A, Iz, L, P = 2.0e11, 1.0e-2, 8.333e-6, 3.0, 1.0e3
    m = Model(ndm=2, ndf=3)
    m.add_node(1, 0.0, 0.0)
    m.add_node(2, L, 0.0)
    mat = ElasticIsotropic(1, E=E, nu=0.3)
    m.add_material(mat)
    elem = BeamColumn2D(1, (1, 2), mat, A, Iz)
    m.add_element(elem)
    m.fix(1, [1, 1, 1])
    m.add_nodal_load(2, [0.0, -P, 0.0])
    LinearStaticAnalysis(m).run()
    # Re-derive u_local manually
    T = elem.transform_matrix()
    u_l = T @ elem.gather_u()
    from femsolver.numerics import gauss_lobatto_1d
    xi_pts, _ = gauss_lobatto_1d(elem.n_int)
    for i, xi in enumerate(xi_pts):
        B = elem._strain_disp_matrix(xi, L)
        np.testing.assert_allclose(elem.section_strains[i], B @ u_l, rtol=1e-12)


def test_section_force_equals_ks_times_strain_for_elastic():
    """For an elastic section, s = ks @ e at every IP."""
    sec = ElasticSection2D(E=2.0e11, A=1.0e-2, Iz=8.333e-6)
    L, P = 3.0, 1.0e3
    m = Model(ndm=2, ndf=3)
    m.add_node(1, 0.0, 0.0)
    m.add_node(2, L, 0.0)
    mat = ElasticIsotropic(1, E=2.0e11, nu=0.3)
    m.add_material(mat)
    elem = BeamColumn2D(1, (1, 2), mat, section=sec)
    m.add_element(elem)
    m.fix(1, [1, 1, 1])
    m.add_nodal_load(2, [0.0, -P, 0.0])
    LinearStaticAnalysis(m).run()
    for i in range(elem.n_int):
        e_i = elem.section_strains[i]
        s_i = elem.section_forces[i]
        s_expected, _ = sec.get_response(e_i)
        np.testing.assert_allclose(s_i, s_expected, rtol=1e-14)


def test_section_moment_under_udl_matches_cubic_hermite_fem():
    """Cantilever under downward UDL — moment diagram from cubic Hermite.

    The exact analytical moment at the fixed end is ``-w L^2 / 2``, but
    a single cubic-Hermite displacement-based element does NOT recover
    this value. With UDL the exact ``v(x)`` is quartic; the cubic Hermite
    interpolation only matches the exact at the nodes (Galerkin's
    nodal-exactness property), and the recovered ``M(x) = EI v''(x)`` is
    the second derivative of the *interpolant*, not the exact field.

    Substituting the (exact) tip values v_tip = -wL^4/(8EI) and
    theta_tip = -wL^3/(6EI) into the second derivative of the Hermite
    cubic at x=0 gives::

        M(0)_FEM = EI * v''(0) = (6/L^2) v_tip + (-2/L) theta_tip
                 = -5 w L^2 / 12         (NOT -w L^2 / 2)

    The discrepancy disappears with mesh refinement — and is one of the
    motivations for force-based formulations (which carry an exact
    linear/quadratic moment diagram regardless of mesh density). For now
    we lock in the cubic-Hermite value so any future change to the
    integration kernel is caught immediately.
    """
    E, A, Iz, L, w = 2.0e11, 1.0e-2, 8.333e-6, 3.0, 1.0e3
    m = Model(ndm=2, ndf=3)
    m.add_node(1, 0.0, 0.0)
    m.add_node(2, L, 0.0)
    mat = ElasticIsotropic(1, E=E, nu=0.3)
    m.add_material(mat)
    elem = BeamColumn2D(1, (1, 2), mat, A, Iz)
    m.add_element(elem)
    elem.add_uniform_load(-w)
    m.fix(1, [1, 1, 1])
    LinearStaticAnalysis(m).run()
    M_fixed_cubic_hermite = -5.0 * w * L ** 2 / 12.0
    assert elem.section_forces[0, 1] == pytest.approx(
        M_fixed_cubic_hermite, rel=1e-10
    )


def test_section_moment_under_udl_converges_with_mesh_refinement():
    """The cubic-Hermite under-estimate at the fixed end shrinks as the
    mesh is refined. Two elements should already be much closer to the
    exact ``-w L^2 / 2`` than one.
    """
    E, A, Iz, L, w = 2.0e11, 1.0e-2, 8.333e-6, 3.0, 1.0e3

    def fixed_end_moment(n_elem: int) -> float:
        m = Model(ndm=2, ndf=3)
        for i in range(n_elem + 1):
            m.add_node(i + 1, i * L / n_elem, 0.0)
        mat = ElasticIsotropic(1, E=E, nu=0.3)
        m.add_material(mat)
        elements = []
        for i in range(n_elem):
            e = BeamColumn2D(i + 1, (i + 1, i + 2), mat, A, Iz)
            e.add_uniform_load(-w)
            m.add_element(e)
            elements.append(e)
        m.fix(1, [1, 1, 1])
        LinearStaticAnalysis(m).run()
        return elements[0].section_forces[0, 1]

    M_exact = -w * L ** 2 / 2.0
    err_1 = abs(fixed_end_moment(1) - M_exact)
    err_2 = abs(fixed_end_moment(2) - M_exact)
    err_4 = abs(fixed_end_moment(4) - M_exact)
    assert err_2 < err_1
    assert err_4 < err_2


# =============================================================== 3D outputs

def test_section_arrays_have_correct_shape_3d():
    E, nu = 2.0e11, 0.3
    A, Iy, Iz, J = 1.0e-2, 5.0e-6, 8.333e-6, 1.4e-5
    L, P = 3.0, 1.0e3
    m = Model(ndm=3, ndf=6)
    m.add_node(1, 0.0, 0.0, 0.0)
    m.add_node(2, L, 0.0, 0.0)
    mat = ElasticIsotropic(1, E=E, nu=nu)
    m.add_material(mat)
    elem = BeamColumn3D(1, (1, 2), mat, A, Iy, Iz, J)
    m.add_element(elem)
    m.fix(1, [1, 1, 1, 1, 1, 1])
    m.add_nodal_load(2, [0, P, 0, 0, 0, 0])
    LinearStaticAnalysis(m).run()
    n = elem.n_int
    assert elem.section_locations.shape == (n,)
    assert elem.section_strains.shape == (n, 4)
    assert elem.section_forces.shape == (n, 4)


def test_section_torque_constant_along_3d_beam():
    """Pure torsion: T(x) = T_applied at every IP, gamma = T / (G J)."""
    E, nu = 2.0e11, 0.3
    A, Iy, Iz, J = 1.0e-2, 8.333e-6, 8.333e-6, 1.4e-5
    L, T_load = 3.0, 1.0e2
    m = Model(ndm=3, ndf=6)
    m.add_node(1, 0.0, 0.0, 0.0)
    m.add_node(2, L, 0.0, 0.0)
    mat = ElasticIsotropic(1, E=E, nu=nu)
    G = mat.G
    m.add_material(mat)
    elem = BeamColumn3D(1, (1, 2), mat, A, Iy, Iz, J)
    m.add_element(elem)
    m.fix(1, [1, 1, 1, 1, 1, 1])
    m.add_nodal_load(2, [0, 0, 0, T_load, 0, 0])
    LinearStaticAnalysis(m).run()
    np.testing.assert_allclose(elem.section_forces[:, 3], T_load, rtol=1e-10)
    np.testing.assert_allclose(
        elem.section_strains[:, 3], T_load / (G * J), rtol=1e-10
    )


def test_section_3d_cantilever_bending_about_z():
    """3D cantilever, tip load +Py: bending about z. Mz(x) = +P (L - x).

    With ``kappa_z = d^2 v / dx^2`` and an *upward* tip load, deflection
    and curvature are positive, so Mz = EIz * kappa_z is positive along
    the element (opposite sign from the 2D test, which uses a downward
    load).
    """
    E, nu = 2.0e11, 0.3
    A, Iy, Iz, J = 1.0e-2, 8.333e-6, 8.333e-6, 1.4e-5
    L, P = 3.0, 1.0e3
    m = Model(ndm=3, ndf=6)
    m.add_node(1, 0.0, 0.0, 0.0)
    m.add_node(2, L, 0.0, 0.0)
    mat = ElasticIsotropic(1, E=E, nu=nu)
    m.add_material(mat)
    elem = BeamColumn3D(1, (1, 2), mat, A, Iy, Iz, J)
    m.add_element(elem)
    m.fix(1, [1, 1, 1, 1, 1, 1])
    m.add_nodal_load(2, [0, P, 0, 0, 0, 0])
    LinearStaticAnalysis(m).run()
    expected_Mz = P * (L - elem.section_locations)
    np.testing.assert_allclose(elem.section_forces[:, 1], expected_Mz,
                               rtol=1e-10, atol=1e-9)
    # bending about y is zero
    np.testing.assert_allclose(elem.section_forces[:, 2], 0.0, atol=1e-6)


def test_section_3d_cantilever_bending_about_y():
    """Sign-flip check: tip load Pz produces non-zero My only.

    With kappa_y derived from -d^2 w / dx^2 (encoded in B-row 2) and
    section response My = EIy * kappa_y, the sign of My along the element
    matches Mz from the analogous y-direction load case (same magnitude
    pattern, no sign flip in the *moment diagram* itself).
    """
    E, nu = 2.0e11, 0.3
    A, Iy, Iz, J = 1.0e-2, 5.0e-6, 8.333e-6, 1.4e-5
    L, P = 3.0, 1.0e3
    m = Model(ndm=3, ndf=6)
    m.add_node(1, 0.0, 0.0, 0.0)
    m.add_node(2, L, 0.0, 0.0)
    mat = ElasticIsotropic(1, E=E, nu=nu)
    m.add_material(mat)
    elem = BeamColumn3D(1, (1, 2), mat, A, Iy, Iz, J)
    m.add_element(elem)
    m.fix(1, [1, 1, 1, 1, 1, 1])
    m.add_nodal_load(2, [0, 0, P, 0, 0, 0])
    LinearStaticAnalysis(m).run()
    # Magnitude at the fixed end should equal P * L
    assert abs(elem.section_forces[0, 2]) == pytest.approx(P * L, rel=1e-10)
    # Mz must be zero
    np.testing.assert_allclose(elem.section_forces[:, 1], 0.0, atol=1e-6)


# ============================================================ commit/revert

class _StatefulSection:
    """A tiny stand-in section that records lifecycle calls."""
    n_resultants = 2

    def __init__(self):
        self._ks = np.diag([1.0e9, 1.0e6])
        self.commit_calls = 0
        self.revert_calls = 0

    # element constructor reads these attributes off the section
    A = 1.0
    Iz = 1.0

    def get_response(self, e):
        return self._ks @ e, self._ks.copy()

    def commit_state(self):
        self.commit_calls += 1

    def revert_state(self):
        self.revert_calls += 1


def test_beam_commit_state_forwards_to_section():
    sec = _StatefulSection()
    mat = ElasticIsotropic(1, E=2.0e11, nu=0.3)
    m = Model(ndm=2, ndf=3)
    m.add_node(1, 0.0, 0.0)
    m.add_node(2, 1.0, 0.0)
    m.add_material(mat)
    elem = BeamColumn2D(1, (1, 2), mat, section=sec)
    m.add_element(elem)
    elem.commit_state()
    elem.commit_state()
    assert sec.commit_calls == 2


def test_beam_revert_state_forwards_to_section():
    sec = _StatefulSection()
    mat = ElasticIsotropic(1, E=2.0e11, nu=0.3)
    m = Model(ndm=2, ndf=3)
    m.add_node(1, 0.0, 0.0)
    m.add_node(2, 1.0, 0.0)
    m.add_material(mat)
    elem = BeamColumn2D(1, (1, 2), mat, section=sec)
    m.add_element(elem)
    elem.revert_state()
    assert sec.revert_calls == 1
