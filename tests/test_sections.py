"""Tests for the section abstraction.

These exercise the constitutive interface (forces and tangent stiffness)
and the constructor seam on :class:`BeamColumn2D` / :class:`BeamColumn3D`,
which now accept either the legacy ``(area, ...)`` parameters or a
``section=`` keyword. The downstream stiffness / displacement results must
be identical to the closed-form path so existing analyses are unaffected.
"""
import numpy as np
import pytest

from femsolver import (
    BeamColumn2D,
    BeamColumn3D,
    ElasticIsotropic,
    ElasticSection2D,
    ElasticSection3D,
    LinearStaticAnalysis,
    Model,
)


# ----------------------------------------------------------- ElasticSection2D

def test_elastic_section_2d_response_is_linear():
    sec = ElasticSection2D(E=2.0e11, A=1.0e-2, Iz=8.333e-6)
    e = np.array([1.0e-4, 2.0e-3])  # axial strain, curvature
    s, ks = sec.get_response(e)
    # forces: N = EA * eps, Mz = EIz * kappa
    np.testing.assert_allclose(s[0], sec.EA * e[0], rtol=1e-14)
    np.testing.assert_allclose(s[1], sec.EIz * e[1], rtol=1e-14)
    # tangent: diagonal, constant
    np.testing.assert_allclose(ks, np.diag([sec.EA, sec.EIz]), rtol=1e-14)


def test_elastic_section_2d_tangent_independent_of_strain():
    sec = ElasticSection2D(E=2.0e11, A=1.0e-2, Iz=8.333e-6)
    _, ks0 = sec.get_response(np.zeros(2))
    _, ks1 = sec.get_response(np.array([1.0, -1.0]))
    np.testing.assert_allclose(ks0, ks1, rtol=1e-14)


def test_elastic_section_2d_tangent_is_a_copy():
    """Mutating the returned tangent must not corrupt the section."""
    sec = ElasticSection2D(E=2.0e11, A=1.0e-2, Iz=8.333e-6)
    _, ks = sec.get_response(np.zeros(2))
    ks[0, 0] = 0.0
    _, ks2 = sec.get_response(np.zeros(2))
    assert ks2[0, 0] == sec.EA


def test_elastic_section_2d_rejects_nonpositive():
    with pytest.raises(ValueError):
        ElasticSection2D(E=0.0, A=1.0, Iz=1.0)
    with pytest.raises(ValueError):
        ElasticSection2D(E=1.0, A=-1.0, Iz=1.0)
    with pytest.raises(ValueError):
        ElasticSection2D(E=1.0, A=1.0, Iz=0.0)


# ----------------------------------------------------------- ElasticSection3D

def test_elastic_section_3d_response_is_linear():
    sec = ElasticSection3D(E=2.0e11, G=8.0e10, A=1.0e-2,
                           Iy=5.0e-6, Iz=8.333e-6, J=1.4e-5)
    e = np.array([1.0e-4, 2.0e-3, 3.0e-3, 1.0e-3])
    s, ks = sec.get_response(e)
    np.testing.assert_allclose(s[0], sec.EA * e[0], rtol=1e-14)
    np.testing.assert_allclose(s[1], sec.EIz * e[1], rtol=1e-14)
    np.testing.assert_allclose(s[2], sec.EIy * e[2], rtol=1e-14)
    np.testing.assert_allclose(s[3], sec.GJ * e[3], rtol=1e-14)
    np.testing.assert_allclose(
        ks, np.diag([sec.EA, sec.EIz, sec.EIy, sec.GJ]), rtol=1e-14
    )


def test_elastic_section_3d_rejects_nonpositive():
    kwargs = dict(E=1.0, G=1.0, A=1.0, Iy=1.0, Iz=1.0, J=1.0)
    for key in kwargs:
        bad = dict(kwargs)
        bad[key] = 0.0
        with pytest.raises(ValueError):
            ElasticSection3D(**bad)


# --------------------------------------------- Beam constructor: section kwarg

def _build_2d_cantilever(*, use_section: bool):
    """Cantilever: tip load P, w_tip = PL^3/(3EI)."""
    E, A, Iz, L, P = 2.0e11, 1.0e-2, 8.333e-6, 3.0, 1.0e3
    m = Model(ndm=2, ndf=3)
    m.add_node(1, 0.0, 0.0)
    m.add_node(2, L, 0.0)
    mat = ElasticIsotropic(1, E=E, nu=0.3)
    m.add_material(mat)
    if use_section:
        sec = ElasticSection2D(E=E, A=A, Iz=Iz)
        m.add_element(BeamColumn2D(1, (1, 2), mat, section=sec))
    else:
        m.add_element(BeamColumn2D(1, (1, 2), mat, A, Iz))
    m.fix(1, [1, 1, 1])
    m.add_nodal_load(2, [0.0, -P, 0.0])
    LinearStaticAnalysis(m).run()
    return m, P, L, E, Iz


def test_beam2d_section_kwarg_matches_legacy():
    """Constructing with a section= must yield identical displacements."""
    m_legacy, *_ = _build_2d_cantilever(use_section=False)
    m_section, *_ = _build_2d_cantilever(use_section=True)
    np.testing.assert_allclose(
        m_legacy.node(2).disp, m_section.node(2).disp, rtol=1e-14, atol=0.0
    )


def test_beam2d_section_attached_to_element():
    """The element must hold a reference to its section after construction."""
    mat = ElasticIsotropic(1, E=2.0e11, nu=0.3)
    sec = ElasticSection2D(E=2.0e11, A=1.0e-2, Iz=8.333e-6)
    m = Model(ndm=2, ndf=3)
    m.add_node(1, 0.0, 0.0)
    m.add_node(2, 1.0, 0.0)
    m.add_material(mat)
    elem = BeamColumn2D(1, (1, 2), mat, section=sec)
    m.add_element(elem)
    assert elem.section is sec
    assert elem.area == sec.A
    assert elem.Iz == sec.Iz


def test_beam2d_legacy_constructor_builds_section():
    """Legacy (area, Iz) construction must still attach an ElasticSection2D."""
    mat = ElasticIsotropic(1, E=2.0e11, nu=0.3)
    m = Model(ndm=2, ndf=3)
    m.add_node(1, 0.0, 0.0)
    m.add_node(2, 1.0, 0.0)
    m.add_material(mat)
    elem = BeamColumn2D(1, (1, 2), mat, 1.0e-2, 8.333e-6)
    m.add_element(elem)
    assert isinstance(elem.section, ElasticSection2D)
    assert elem.section.EA == mat.E * 1.0e-2
    assert elem.section.EIz == mat.E * 8.333e-6


def test_beam2d_rejects_both_section_and_legacy():
    mat = ElasticIsotropic(1, E=2.0e11, nu=0.3)
    sec = ElasticSection2D(E=2.0e11, A=1.0e-2, Iz=8.333e-6)
    with pytest.raises(ValueError):
        BeamColumn2D(1, (1, 2), mat, area=1.0e-2, Iz=8.333e-6, section=sec)


def test_beam2d_rejects_missing_arguments():
    mat = ElasticIsotropic(1, E=2.0e11, nu=0.3)
    with pytest.raises(ValueError):
        BeamColumn2D(1, (1, 2), mat)


# ------------------------------------------------------ 3D beam section kwarg

def _build_3d_cantilever(*, use_section: bool):
    E, nu = 2.0e11, 0.3
    A, Iy, Iz, J = 1.0e-2, 8.333e-6, 8.333e-6, 1.4e-5
    L, P = 3.0, 1.0e3
    m = Model(ndm=3, ndf=6)
    m.add_node(1, 0.0, 0.0, 0.0)
    m.add_node(2, L, 0.0, 0.0)
    mat = ElasticIsotropic(1, E=E, nu=nu)
    m.add_material(mat)
    if use_section:
        sec = ElasticSection3D(E=E, G=mat.G, A=A, Iy=Iy, Iz=Iz, J=J)
        m.add_element(BeamColumn3D(1, (1, 2), mat, section=sec))
    else:
        m.add_element(BeamColumn3D(1, (1, 2), mat, A, Iy, Iz, J))
    m.fix(1, [1, 1, 1, 1, 1, 1])
    m.add_nodal_load(2, [0, P, 0, 0, 0, 0])
    LinearStaticAnalysis(m).run()
    return m


def test_beam3d_section_kwarg_matches_legacy():
    m_legacy = _build_3d_cantilever(use_section=False)
    m_section = _build_3d_cantilever(use_section=True)
    np.testing.assert_allclose(
        m_legacy.node(2).disp, m_section.node(2).disp, rtol=1e-14, atol=0.0
    )


def test_beam3d_legacy_constructor_builds_section():
    mat = ElasticIsotropic(1, E=2.0e11, nu=0.3)
    m = Model(ndm=3, ndf=6)
    m.add_node(1, 0.0, 0.0, 0.0)
    m.add_node(2, 1.0, 0.0, 0.0)
    m.add_material(mat)
    elem = BeamColumn3D(1, (1, 2), mat, 1.0e-2, 5.0e-6, 8.333e-6, 1.4e-5)
    m.add_element(elem)
    assert isinstance(elem.section, ElasticSection3D)
    assert elem.section.EA == mat.E * 1.0e-2
    assert elem.section.GJ == mat.G * 1.4e-5


def test_beam3d_rejects_both_section_and_legacy():
    mat = ElasticIsotropic(1, E=2.0e11, nu=0.3)
    sec = ElasticSection3D(E=2.0e11, G=mat.G, A=1.0e-2,
                           Iy=5.0e-6, Iz=8.333e-6, J=1.4e-5)
    with pytest.raises(ValueError):
        BeamColumn3D(1, (1, 2), mat, area=1.0e-2, Iy=5.0e-6, Iz=8.333e-6, J=1.4e-5,
                     section=sec)


def test_beam3d_rejects_missing_arguments():
    mat = ElasticIsotropic(1, E=2.0e11, nu=0.3)
    with pytest.raises(ValueError):
        BeamColumn3D(1, (1, 2), mat)
