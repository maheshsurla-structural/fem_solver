"""Beam-column element validation against analytical solutions."""
import numpy as np
import pytest

from femsolver import (
    Model,
    ElasticIsotropic,
    BeamColumn2D,
    BeamColumn3D,
    LinearStaticAnalysis,
)


# ------------------------------------------------------------------ 2D beam --

def test_beam2d_cantilever_tip_load():
    """Cantilever, tip vertical load P. w_tip = PL^3/(3EI), theta = PL^2/(2EI)."""
    E, A, Iz, L, P = 2.0e11, 1.0e-2, 8.333e-6, 3.0, 1.0e3  # 100x100 mm rect
    m = Model(ndm=2, ndf=3)
    m.add_node(1, 0.0, 0.0)
    m.add_node(2, L, 0.0)
    mat = ElasticIsotropic(1, E=E, nu=0.3)
    m.add_material(mat)
    m.add_element(BeamColumn2D(1, (1, 2), mat, A, Iz))
    m.fix(1, [1, 1, 1])
    m.add_nodal_load(2, [0.0, -P, 0.0])
    LinearStaticAnalysis(m).run()
    w_expected = -P * L ** 3 / (3.0 * E * Iz)
    th_expected = -P * L ** 2 / (2.0 * E * Iz)
    np.testing.assert_allclose(m.node(2).disp[1], w_expected, rtol=1e-10)
    np.testing.assert_allclose(m.node(2).disp[2], th_expected, rtol=1e-10)


def test_beam2d_cantilever_udl():
    """Cantilever, downward UDL w. w_tip = -wL^4/(8EI), theta = -wL^3/(6EI)."""
    E, A, Iz, L, w = 2.0e11, 1.0e-2, 8.333e-6, 3.0, 1.0e3
    m = Model(ndm=2, ndf=3)
    m.add_node(1, 0.0, 0.0)
    m.add_node(2, L, 0.0)
    mat = ElasticIsotropic(1, E=E, nu=0.3)
    m.add_material(mat)
    elem = BeamColumn2D(1, (1, 2), mat, A, Iz)
    m.add_element(elem)
    elem.add_uniform_load(-w)  # downward
    m.fix(1, [1, 1, 1])
    LinearStaticAnalysis(m).run()
    w_expected = -w * L ** 4 / (8.0 * E * Iz)
    th_expected = -w * L ** 3 / (6.0 * E * Iz)
    np.testing.assert_allclose(m.node(2).disp[1], w_expected, rtol=1e-10)
    np.testing.assert_allclose(m.node(2).disp[2], th_expected, rtol=1e-10)


def test_beam2d_simply_supported_center_load():
    """Simply supported, point load P at midspan. w_mid = PL^3/(48EI).
    Use 2 elements so we have a midspan node."""
    E, A, Iz, L, P = 2.0e11, 1.0e-2, 8.333e-6, 4.0, 1.0e3
    m = Model(ndm=2, ndf=3)
    m.add_node(1, 0.0, 0.0)
    m.add_node(2, L / 2.0, 0.0)
    m.add_node(3, L, 0.0)
    mat = ElasticIsotropic(1, E=E, nu=0.3)
    m.add_material(mat)
    m.add_element(BeamColumn2D(1, (1, 2), mat, A, Iz))
    m.add_element(BeamColumn2D(2, (2, 3), mat, A, Iz))
    m.fix(1, [1, 1, 0])  # pin
    m.fix(3, [0, 1, 0])  # roller
    m.add_nodal_load(2, [0.0, -P, 0.0])
    LinearStaticAnalysis(m).run()
    w_expected = -P * L ** 3 / (48.0 * E * Iz)
    np.testing.assert_allclose(m.node(2).disp[1], w_expected, rtol=1e-10)
    # equilibrium: each support reaction in y = P/2
    np.testing.assert_allclose(m.node(1).reaction[1], P / 2.0, rtol=1e-10)
    np.testing.assert_allclose(m.node(3).reaction[1], P / 2.0, rtol=1e-10)


def test_beam2d_simply_supported_udl():
    """Simply supported, UDL w. w_mid = 5wL^4/(384EI), end shear = wL/2."""
    E, A, Iz, L, w = 2.0e11, 1.0e-2, 8.333e-6, 4.0, 1.0e3
    m = Model(ndm=2, ndf=3)
    m.add_node(1, 0.0, 0.0)
    m.add_node(2, L / 2.0, 0.0)
    m.add_node(3, L, 0.0)
    mat = ElasticIsotropic(1, E=E, nu=0.3)
    m.add_material(mat)
    e1 = BeamColumn2D(1, (1, 2), mat, A, Iz)
    e2 = BeamColumn2D(2, (2, 3), mat, A, Iz)
    m.add_element(e1)
    m.add_element(e2)
    e1.add_uniform_load(-w)
    e2.add_uniform_load(-w)
    m.fix(1, [1, 1, 0])
    m.fix(3, [0, 1, 0])
    LinearStaticAnalysis(m).run()
    w_expected = -5.0 * w * L ** 4 / (384.0 * E * Iz)
    np.testing.assert_allclose(m.node(2).disp[1], w_expected, rtol=1e-10)
    # Each support takes wL/2 of the total downward UDL (acting downward, so
    # reaction is upward = +w*L/2)
    np.testing.assert_allclose(m.node(1).reaction[1], w * L / 2.0, rtol=1e-10)
    np.testing.assert_allclose(m.node(3).reaction[1], w * L / 2.0, rtol=1e-10)


def test_beam2d_fixed_fixed_udl():
    """Fixed-fixed beam under UDL. End moments = wL^2/12, midspan = wL^2/24,
    midspan deflection = wL^4/(384EI)."""
    E, A, Iz, L, w = 2.0e11, 1.0e-2, 8.333e-6, 4.0, 1.0e3
    m = Model(ndm=2, ndf=3)
    m.add_node(1, 0.0, 0.0)
    m.add_node(2, L / 2.0, 0.0)
    m.add_node(3, L, 0.0)
    mat = ElasticIsotropic(1, E=E, nu=0.3)
    m.add_material(mat)
    e1 = BeamColumn2D(1, (1, 2), mat, A, Iz)
    e2 = BeamColumn2D(2, (2, 3), mat, A, Iz)
    m.add_element(e1); m.add_element(e2)
    e1.add_uniform_load(-w)
    e2.add_uniform_load(-w)
    m.fix(1, [1, 1, 1])
    m.fix(3, [1, 1, 1])
    LinearStaticAnalysis(m).run()
    w_expected = -w * L ** 4 / (384.0 * E * Iz)
    np.testing.assert_allclose(m.node(2).disp[1], w_expected, rtol=1e-10)
    # End reactions: shear = wL/2 upward each, moment magnitude = wL^2/12
    np.testing.assert_allclose(m.node(1).reaction[1], w * L / 2.0, rtol=1e-10)
    np.testing.assert_allclose(abs(m.node(1).reaction[2]), w * L * L / 12.0, rtol=1e-10)


def test_beam2d_axial_load():
    """Axial load through a beam element should give same result as truss.
    u = PL/(EA). Tests that the EA/L block is correct."""
    E, A, Iz, L, P = 2.0e11, 1.0e-2, 8.333e-6, 3.0, 1.0e4
    m = Model(ndm=2, ndf=3)
    m.add_node(1, 0.0, 0.0)
    m.add_node(2, L, 0.0)
    mat = ElasticIsotropic(1, E=E, nu=0.3)
    m.add_material(mat)
    m.add_element(BeamColumn2D(1, (1, 2), mat, A, Iz))
    m.fix(1, [1, 1, 1])
    m.fix(2, [0, 1, 1])
    m.add_nodal_load(2, [P, 0.0, 0.0])
    LinearStaticAnalysis(m).run()
    np.testing.assert_allclose(m.node(2).disp[0], P * L / (E * A), rtol=1e-12)


# ------------------------------------------------------------------ 3D beam --

def test_beam3d_cantilever_load_in_y():
    """3D cantilever along x, tip load Py. Bend about z (Iz controls)."""
    E, nu = 2.0e11, 0.3
    A, Iy, Iz, J = 1.0e-2, 8.333e-6, 8.333e-6, 1.4e-5
    L, P = 3.0, 1.0e3
    m = Model(ndm=3, ndf=6)
    m.add_node(1, 0.0, 0.0, 0.0)
    m.add_node(2, L, 0.0, 0.0)
    mat = ElasticIsotropic(1, E=E, nu=nu)
    m.add_material(mat)
    m.add_element(BeamColumn3D(1, (1, 2), mat, A, Iy, Iz, J))
    m.fix(1, [1, 1, 1, 1, 1, 1])
    m.add_nodal_load(2, [0, P, 0, 0, 0, 0])  # Py at tip
    LinearStaticAnalysis(m).run()
    w_expected = P * L ** 3 / (3.0 * E * Iz)
    np.testing.assert_allclose(m.node(2).disp[1], w_expected, rtol=1e-10)


def test_beam3d_cantilever_load_in_z():
    """3D cantilever along x, tip load Pz. Bend about y (Iy controls)."""
    E, nu = 2.0e11, 0.3
    A, Iy, Iz, J = 1.0e-2, 5.0e-6, 8.333e-6, 1.4e-5  # Iy != Iz to confirm
    L, P = 3.0, 1.0e3
    m = Model(ndm=3, ndf=6)
    m.add_node(1, 0.0, 0.0, 0.0)
    m.add_node(2, L, 0.0, 0.0)
    mat = ElasticIsotropic(1, E=E, nu=nu)
    m.add_material(mat)
    m.add_element(BeamColumn3D(1, (1, 2), mat, A, Iy, Iz, J))
    m.fix(1, [1, 1, 1, 1, 1, 1])
    m.add_nodal_load(2, [0, 0, P, 0, 0, 0])  # Pz at tip
    LinearStaticAnalysis(m).run()
    w_expected = P * L ** 3 / (3.0 * E * Iy)
    np.testing.assert_allclose(m.node(2).disp[2], w_expected, rtol=1e-10)


def test_beam3d_torsion():
    """3D cantilever with applied torque. phi = TL/(GJ)."""
    E, nu = 2.0e11, 0.3
    A, Iy, Iz, J = 1.0e-2, 8.333e-6, 8.333e-6, 1.4e-5
    L, T_load = 3.0, 1.0e2
    m = Model(ndm=3, ndf=6)
    m.add_node(1, 0.0, 0.0, 0.0)
    m.add_node(2, L, 0.0, 0.0)
    mat = ElasticIsotropic(1, E=E, nu=nu)
    G = E / (2.0 * (1 + nu))
    m.add_material(mat)
    m.add_element(BeamColumn3D(1, (1, 2), mat, A, Iy, Iz, J))
    m.fix(1, [1, 1, 1, 1, 1, 1])
    m.add_nodal_load(2, [0, 0, 0, T_load, 0, 0])  # torque about x
    LinearStaticAnalysis(m).run()
    phi_expected = T_load * L / (G * J)
    np.testing.assert_allclose(m.node(2).disp[3], phi_expected, rtol=1e-10)


def test_beam3d_axial():
    """3D axial: u = PL/(EA)."""
    E, nu = 2.0e11, 0.3
    A, Iy, Iz, J = 1.0e-2, 8.333e-6, 8.333e-6, 1.4e-5
    L, P = 3.0, 1.0e4
    m = Model(ndm=3, ndf=6)
    m.add_node(1, 0.0, 0.0, 0.0)
    m.add_node(2, L, 0.0, 0.0)
    mat = ElasticIsotropic(1, E=E, nu=nu)
    m.add_material(mat)
    m.add_element(BeamColumn3D(1, (1, 2), mat, A, Iy, Iz, J))
    m.fix(1, [1, 1, 1, 1, 1, 1])
    m.fix(2, [0, 1, 1, 1, 1, 1])
    m.add_nodal_load(2, [P, 0, 0, 0, 0, 0])
    LinearStaticAnalysis(m).run()
    np.testing.assert_allclose(m.node(2).disp[0], P * L / (E * A), rtol=1e-12)


def test_beam3d_oriented_along_y():
    """Beam oriented along global y — verify orientation transform.
    Cantilever along y, load in z direction at tip. w_tip = PL^3/(3 E Iz_local).
    With default vecxz=(0,0,1) the local z is along global z and local y
    is along global -x (since y_local = z_local x x_local = (0,0,1) x (0,1,0) = (-1,0,0))."""
    E, nu = 2.0e11, 0.3
    A, Iy, Iz, J = 1.0e-2, 5.0e-6, 8.333e-6, 1.4e-5
    L, P = 3.0, 1.0e3
    m = Model(ndm=3, ndf=6)
    m.add_node(1, 0.0, 0.0, 0.0)
    m.add_node(2, 0.0, L, 0.0)
    mat = ElasticIsotropic(1, E=E, nu=nu)
    m.add_material(mat)
    # default vecxz = (0,0,1) since not vertical
    m.add_element(BeamColumn3D(1, (1, 2), mat, A, Iy, Iz, J))
    m.fix(1, [1, 1, 1, 1, 1, 1])
    # Load along global +z = local +z direction. Bending happens about local y.
    # Iy controls. Tip displacement in z = P L^3 / (3 E Iy).
    m.add_nodal_load(2, [0, 0, P, 0, 0, 0])
    LinearStaticAnalysis(m).run()
    w_expected = P * L ** 3 / (3.0 * E * Iy)
    np.testing.assert_allclose(m.node(2).disp[2], w_expected, rtol=1e-10)


# ------------------------------------------------------- mixed-element model

def test_truss_beam_combined_2d():
    """A 2D beam with a tie (truss) bracing — both element families coexist.
    Beam-column ndf=3 model. Truss is a Truss2D (dofs_per_node=2): only the
    translational DOFs at its nodes get its stiffness contribution. Rotational
    DOFs at truss-only nodes must therefore have stiffness from a beam, or
    be constrained.

    Geometry: cantilever beam horizontal from (0,0) to (L,0), tied diagonally
    by a truss from (L,0) up to a fixed node at (L, h)."""
    from femsolver import Truss2D
    E, A_b, Iz, L = 2.0e11, 1.0e-2, 8.333e-6, 3.0
    A_t = 1.0e-4
    h = 1.5
    P = 1.0e3
    m = Model(ndm=2, ndf=3)
    m.add_node(1, 0.0, 0.0)
    m.add_node(2, L, 0.0)
    m.add_node(3, L, h)  # fixed
    mat = ElasticIsotropic(1, E=E, nu=0.3)
    m.add_material(mat)
    m.add_element(BeamColumn2D(1, (1, 2), mat, A_b, Iz))
    m.add_element(Truss2D(2, (2, 3), mat, A_t))
    m.fix(1, [1, 1, 1])
    m.fix(3, [1, 1, 1])
    m.add_nodal_load(2, [0.0, -P, 0.0])
    LinearStaticAnalysis(m).run()
    # tip should deflect less than free cantilever PL^3/(3EI)
    w_free = -P * L ** 3 / (3.0 * E * Iz)
    assert abs(m.node(2).disp[1]) < abs(w_free)
    assert m.node(2).disp[1] < 0  # still goes down
