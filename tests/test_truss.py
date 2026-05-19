"""Truss element validation against analytical solutions."""
import numpy as np
import pytest

from femsolver import Model, ElasticIsotropic, Truss2D, Truss3D, LinearStaticAnalysis


# ---------------------------------------------------------------------- 2D --

def test_truss2d_axial_bar():
    """Single horizontal bar under axial tension. u_tip = PL/(EA)."""
    E, A, L, P = 2.0e11, 1.0e-4, 2.0, 1.0e4
    m = Model(ndm=2, ndf=2)
    m.add_node(1, 0.0, 0.0)
    m.add_node(2, L, 0.0)
    mat = ElasticIsotropic(1, E=E, nu=0.3)
    m.add_material(mat)
    m.add_element(Truss2D(1, (1, 2), mat, A))
    m.fix(1, [1, 1])
    m.fix(2, [0, 1])
    m.add_nodal_load(2, [P, 0.0])
    LinearStaticAnalysis(m).run()
    u_expected = P * L / (E * A)
    np.testing.assert_allclose(m.node(2).disp[0], u_expected, rtol=1e-12)
    np.testing.assert_allclose(m.element(1).axial_force, P, rtol=1e-12)
    np.testing.assert_allclose(m.element(1).axial_stress, P / A, rtol=1e-12)
    # equilibrium: reaction at node 1 should be -P in x
    np.testing.assert_allclose(m.node(1).reaction[0], -P, rtol=1e-10)


def test_truss2d_inclined_bar():
    """Inclined bar at 45 deg under axial load along bar axis."""
    E, A = 2.0e11, 1.0e-4
    L = np.sqrt(2.0)  # bar from (0,0) to (1,1)
    P = 1.0e4
    m = Model(ndm=2, ndf=2)
    m.add_node(1, 0.0, 0.0)
    m.add_node(2, 1.0, 1.0)
    mat = ElasticIsotropic(1, E=E, nu=0.3)
    m.add_material(mat)
    m.add_element(Truss2D(1, (1, 2), mat, A))
    m.fix(1, [1, 1])
    # apply load along bar axis: P/sqrt(2) in x and y
    f = P / np.sqrt(2.0)
    m.add_nodal_load(2, [f, f])
    # Need to constrain rigid body modes: also fix node 2 in transverse direction.
    # Trick: rotate constraint by adding a stiff diagonal spring? Easier: use
    # an additional bar perpendicular to fix the transverse mode.
    # For simplicity, test the displacement projection along the bar instead:
    # remove transverse rigid mode by adding a second element.
    # Here we just constrain the perpendicular DOF via a second node and bar.
    # Alternate: leave both DOFs free at node 2, supply a load only along bar.
    # The truss axial-only formulation has zero stiffness perpendicular to the
    # bar, so we'd be singular. Add a constraint: fix node 2 v_perp.
    # Simplest: add a vertical stub bar at node 2 that goes to a pinned node.
    m.add_node(3, 1.0, 0.0)
    m.fix(3, [1, 1])
    m.add_element(Truss2D(2, (2, 3), mat, A))
    LinearStaticAnalysis(m).run()
    # axial force on member 1 should equal applied load magnitude P
    np.testing.assert_allclose(m.element(1).axial_force, P, rtol=1e-8)


def test_truss2d_three_bar():
    """3-bar pin-jointed truss, classic textbook problem.

    Geometry: equilateral triangle with apex at top.
    Bottom chord nodes 1 (pinned) at (0,0), 2 (roller) at (L,0).
    Apex node 3 at (L/2, L*sqrt(3)/2). Vertical load P at node 3.

    By symmetry: F1 (1->3) = F2 (2->3) (compression), F3 (1->2) (tension).
    From equilibrium at node 3: 2 * F1 * sin(60) = P, so F1 = P/(2 sin 60) = P/sqrt(3).
    """
    E, A, L_chord, P = 2.0e11, 1.0e-4, 2.0, 1.0e4
    m = Model(ndm=2, ndf=2)
    m.add_node(1, 0.0, 0.0)
    m.add_node(2, L_chord, 0.0)
    m.add_node(3, L_chord / 2.0, L_chord * np.sqrt(3.0) / 2.0)
    mat = ElasticIsotropic(1, E=E, nu=0.3)
    m.add_material(mat)
    m.add_element(Truss2D(1, (1, 3), mat, A))
    m.add_element(Truss2D(2, (2, 3), mat, A))
    m.add_element(Truss2D(3, (1, 2), mat, A))
    m.fix(1, [1, 1])
    m.fix(2, [0, 1])
    m.add_nodal_load(3, [0.0, -P])
    LinearStaticAnalysis(m).run()
    # element 1 (1->3): axial force is in the direction from node 1 to node 3
    # the magnitude (with our sign) should be -P/sqrt(3) (compression because
    # tip is being pushed down), i.e., elongation is negative.
    expected = -P / np.sqrt(3.0)
    np.testing.assert_allclose(m.element(1).axial_force, expected, rtol=1e-10)
    np.testing.assert_allclose(m.element(2).axial_force, expected, rtol=1e-10)
    # equilibrium check on supports
    Rx1 = m.node(1).reaction[0]
    Ry1 = m.node(1).reaction[1]
    Ry2 = m.node(2).reaction[1]
    np.testing.assert_allclose(Rx1, 0.0, atol=1e-6)
    np.testing.assert_allclose(Ry1 + Ry2, P, rtol=1e-10)
    np.testing.assert_allclose(Ry1, P / 2.0, rtol=1e-10)


# ---------------------------------------------------------------------- 3D --

def test_truss3d_axial_bar_along_x():
    E, A, L, P = 2.0e11, 1.0e-4, 2.0, 1.0e4
    m = Model(ndm=3, ndf=3)
    m.add_node(1, 0.0, 0.0, 0.0)
    m.add_node(2, L, 0.0, 0.0)
    mat = ElasticIsotropic(1, E=E, nu=0.3)
    m.add_material(mat)
    m.add_element(Truss3D(1, (1, 2), mat, A))
    m.fix(1, [1, 1, 1])
    m.fix(2, [0, 1, 1])
    m.add_nodal_load(2, [P, 0, 0])
    LinearStaticAnalysis(m).run()
    np.testing.assert_allclose(m.node(2).disp[0], P * L / (E * A), rtol=1e-12)
    np.testing.assert_allclose(m.element(1).axial_force, P, rtol=1e-12)


def test_truss3d_skewed_bar():
    """3D bar from origin to (1,1,1). Axial force = P given correct projection."""
    E, A, P = 2.0e11, 1.0e-4, 1.0e4
    m = Model(ndm=3, ndf=3)
    m.add_node(1, 0.0, 0.0, 0.0)
    m.add_node(2, 1.0, 1.0, 1.0)
    mat = ElasticIsotropic(1, E=E, nu=0.3)
    m.add_material(mat)
    m.add_element(Truss3D(1, (1, 2), mat, A))
    m.fix(1, [1, 1, 1])
    # apply load along bar: P * (1/sqrt(3)) in each component
    f = P / np.sqrt(3.0)
    m.add_nodal_load(2, [f, f, f])
    # need to constrain rigid-body modes perpendicular to bar — add extra bars
    m.add_node(3, 1.0, 0.0, 0.0)
    m.add_node(4, 1.0, 1.0, 0.0)
    m.fix(3, [1, 1, 1])
    m.fix(4, [1, 1, 1])
    m.add_element(Truss3D(2, (2, 3), mat, A))
    m.add_element(Truss3D(3, (2, 4), mat, A))
    LinearStaticAnalysis(m).run()
    np.testing.assert_allclose(m.element(1).axial_force, P, rtol=1e-6)


def test_truss2d_zero_length_raises():
    m = Model(ndm=2, ndf=2)
    m.add_node(1, 0.0, 0.0)
    m.add_node(2, 0.0, 0.0)
    mat = ElasticIsotropic(1, E=1e9, nu=0.3)
    m.add_material(mat)
    m.add_element(Truss2D(1, (1, 2), mat, 1e-4))
    m.fix(1, [1, 1])
    with pytest.raises(ValueError, match="zero length"):
        LinearStaticAnalysis(m).run()
