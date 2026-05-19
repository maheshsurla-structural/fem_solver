"""Sanity checks on Model, Node, materials, DOF numbering."""
import numpy as np
import pytest

from femsolver import Model, ElasticIsotropic, Truss2D


def test_model_construction():
    m = Model(ndm=3, ndf=6)
    assert m.ndm == 3 and m.ndf == 6
    assert len(m.nodes) == 0 and len(m.elements) == 0


def test_invalid_ndm_raises():
    with pytest.raises(ValueError):
        Model(ndm=4, ndf=2)


def test_node_coord_count_validation():
    m = Model(ndm=2, ndf=2)
    with pytest.raises(ValueError):
        m.add_node(1, 0.0, 0.0, 0.0)


def test_duplicate_node_tag_raises():
    m = Model(ndm=2, ndf=2)
    m.add_node(1, 0.0, 0.0)
    with pytest.raises(ValueError, match="duplicate"):
        m.add_node(1, 1.0, 0.0)


def test_duplicate_element_tag_raises():
    m = Model(ndm=2, ndf=2)
    m.add_node(1, 0.0, 0.0)
    m.add_node(2, 1.0, 0.0)
    mat = ElasticIsotropic(1, E=1e9, nu=0.3)
    m.add_material(mat)
    m.add_element(Truss2D(1, (1, 2), mat, 1e-4))
    with pytest.raises(ValueError, match="duplicate"):
        m.add_element(Truss2D(1, (1, 2), mat, 1e-4))


def test_element_unknown_node_raises():
    m = Model(ndm=2, ndf=2)
    m.add_node(1, 0.0, 0.0)
    mat = ElasticIsotropic(1, E=1e9, nu=0.3)
    m.add_material(mat)
    with pytest.raises(ValueError, match="unknown node"):
        m.add_element(Truss2D(1, (1, 99), mat, 1e-4))


def test_dof_numbering():
    m = Model(ndm=2, ndf=2)
    m.add_node(1, 0.0, 0.0)
    m.add_node(2, 1.0, 0.0)
    m.add_node(3, 1.0, 1.0)
    m.fix(1, [1, 1])
    m.fix(2, [0, 1])
    n_eq = m.number_dofs()
    # 6 total - 3 fixed = 3 free
    assert n_eq == 3
    assert (m.node(1).eqn == np.array([-1, -1])).all()
    assert m.node(2).eqn[0] >= 0
    assert m.node(2).eqn[1] == -1
    assert (m.node(3).eqn >= 0).all()


def test_material_validation():
    with pytest.raises(ValueError):
        ElasticIsotropic(1, E=-1.0, nu=0.3)
    with pytest.raises(ValueError):
        ElasticIsotropic(1, E=1e9, nu=0.7)


def test_elastic_isotropic_D_matrices():
    mat = ElasticIsotropic(1, E=2.0e11, nu=0.3)
    Dps = mat.D_plane_stress()
    assert Dps.shape == (3, 3)
    np.testing.assert_allclose(Dps, Dps.T)  # symmetric
    Dpe = mat.D_plane_strain()
    np.testing.assert_allclose(Dpe, Dpe.T)
    D3 = mat.D_3d()
    assert D3.shape == (6, 6)
    np.testing.assert_allclose(D3, D3.T)
    # plane stress E11 = E/(1-nu^2)
    np.testing.assert_allclose(Dps[0, 0], 2.0e11 / (1 - 0.09))
    # G consistency
    G = 2.0e11 / (2 * (1 + 0.3))
    np.testing.assert_allclose(mat.G, G)
    np.testing.assert_allclose(D3[3, 3], G)


def test_kg_symmetric_truss():
    """Element stiffness must be symmetric."""
    mat = ElasticIsotropic(1, E=2.0e11, nu=0.3)
    m = Model(ndm=2, ndf=2)
    m.add_node(1, 0.0, 0.0)
    m.add_node(2, 1.5, 2.5)
    m.add_material(mat)
    e = Truss2D(1, (1, 2), mat, 1e-4)
    m.add_element(e)
    K = e.K_global()
    np.testing.assert_allclose(K, K.T, atol=1e-10)
    # rank should be 1 (only axial mode)
    assert np.linalg.matrix_rank(K, tol=1e-6) == 1
