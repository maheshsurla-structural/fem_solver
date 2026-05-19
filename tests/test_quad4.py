"""Quad4 element validation: patch test, pure tension, beam-bending."""
import numpy as np
import pytest

from femsolver import Model, ElasticIsotropic, Quad4, LinearStaticAnalysis


def test_quad4_patch_test_constant_strain():
    """Patch test: a single Q4 with prescribed corner displacements that
    correspond to a constant strain field should reproduce that strain
    exactly at every Gauss point.

    Here we apply u_x(x, y) = a * x (pure x-tension, exx = a) and
    verify that gp_strain = [a, 0, 0] within machine precision and
    sigma_xx = E/(1-nu^2) * a (plane stress)."""
    E, nu = 2.0e11, 0.3
    a = 1.0e-3  # uniform strain
    m = Model(ndm=2, ndf=2)
    m.add_node(1, 0.0, 0.0)
    m.add_node(2, 1.0, 0.0)
    m.add_node(3, 1.0, 1.0)
    m.add_node(4, 0.0, 1.0)
    mat = ElasticIsotropic(1, E=E, nu=nu)
    m.add_material(mat)
    m.add_element(Quad4(1, (1, 2, 3, 4), mat, thickness=0.1))
    # Prescribe displacements via fixities + assigning disp directly through
    # known boundary motion: enforce u_x = a*x at the two right nodes,
    # u_x = 0 at the left, u_y = 0 along the bottom.
    m.fix(1, [1, 1])
    m.fix(4, [1, 0])
    m.fix(2, [0, 1])  # roller in x
    # apply load that produces pure uniaxial stress sigma_xx = E * a (with
    # the y-direction free, eps_yy = -nu*a comes from Poisson contraction).
    sigma_xx = E * a
    F_per_node = sigma_xx * 0.1 * 0.5  # half tributary length on each right node
    m.add_nodal_load(2, [F_per_node, 0.0])
    m.add_nodal_load(3, [F_per_node, 0.0])
    LinearStaticAnalysis(m).run()
    # check displacements at corners 2 and 3
    np.testing.assert_allclose(m.node(2).disp[0], a * 1.0, rtol=1e-10)
    np.testing.assert_allclose(m.node(3).disp[0], a * 1.0, rtol=1e-10)
    # check strain and stress at all Gauss points
    elem = m.element(1)
    for eps in elem.gp_strain:
        np.testing.assert_allclose(eps, [a, -nu * a, 0.0], atol=1e-10, rtol=1e-10)
    for sig in elem.gp_stress:
        np.testing.assert_allclose(
            sig, [sigma_xx, 0.0, 0.0], atol=1e-3, rtol=1e-10
        )


def test_quad4_jacobian_sign_flip_raises():
    """Reversing node order (clockwise) gives negative jacobian, must raise."""
    E, nu = 2.0e11, 0.3
    m = Model(ndm=2, ndf=2)
    m.add_node(1, 0.0, 0.0)
    m.add_node(2, 0.0, 1.0)  # clockwise from (0,0)
    m.add_node(3, 1.0, 1.0)
    m.add_node(4, 1.0, 0.0)
    mat = ElasticIsotropic(1, E=E, nu=nu)
    m.add_material(mat)
    m.add_element(Quad4(1, (1, 2, 3, 4), mat, thickness=0.1))
    m.fix(1, [1, 1])
    m.fix(2, [1, 0])
    m.fix(4, [0, 1])
    m.add_nodal_load(3, [1.0, 0.0])
    with pytest.raises(ValueError, match="non-positive Jacobian"):
        LinearStaticAnalysis(m).run()


def test_quad4_uniform_tension_mesh():
    """Refined mesh under uniform tension. Strain should be uniform at every
    Gauss point of every element."""
    E, nu = 2.0e11, 0.3
    Lx, Ly, t = 2.0, 1.0, 0.1
    a = 1.0e-3  # target strain
    nx, ny = 4, 2  # 4x2 quads
    m = Model(ndm=2, ndf=2)
    mat = ElasticIsotropic(1, E=E, nu=nu)
    m.add_material(mat)
    # nodes
    tag = 1
    node_grid = {}
    for j in range(ny + 1):
        for i in range(nx + 1):
            x = Lx * i / nx
            y = Ly * j / ny
            m.add_node(tag, x, y)
            node_grid[(i, j)] = tag
            tag += 1
    # elements
    etag = 1
    for j in range(ny):
        for i in range(nx):
            n1 = node_grid[(i, j)]
            n2 = node_grid[(i + 1, j)]
            n3 = node_grid[(i + 1, j + 1)]
            n4 = node_grid[(i, j + 1)]
            m.add_element(Quad4(etag, (n1, n2, n3, n4), mat, thickness=t))
            etag += 1
    # BC: left edge fixed in x; bottom-left corner fixed in y
    for j in range(ny + 1):
        m.fix(node_grid[(0, j)], [1, 0])
    m.fix(node_grid[(0, 0)], [1, 1])
    # load on right edge: pure uniaxial tension sigma_xx = E * a (the y
    # direction is unconstrained, so the element Poisson-contracts naturally
    # to give eps_yy = -nu * a).
    sigma_xx = E * a
    edge_force_per_y = sigma_xx * t
    for j in range(ny + 1):
        # tributary length: full element except at corners (half)
        if j == 0 or j == ny:
            f = edge_force_per_y * (Ly / ny) / 2.0
        else:
            f = edge_force_per_y * (Ly / ny)
        m.add_nodal_load(node_grid[(nx, j)], [f, 0.0])
    LinearStaticAnalysis(m).run()
    # check uniform tension on every gauss point
    for e in m.elements.values():
        for eps in e.gp_strain:
            np.testing.assert_allclose(eps[0], a, rtol=1e-8)
            np.testing.assert_allclose(eps[1], -nu * a, rtol=1e-8)
            np.testing.assert_allclose(eps[2], 0.0, atol=1e-12)
