"""Tests for the 3-D chord-corotational beam-column.

The element handles arbitrary chord rotations (no upper-bound angle
on the chord direction) while keeping nodal rotations linearised.
This is the same formulation that powers SAP2000 / ETABS "P-Delta"
analyses for slender 3-D frames.

Tests:

1. **Zero-displacement equivalence** — at ``u = 0`` the tangent and
   internal force match :class:`BeamColumn3D` to machine precision.

2. **Elastic cantilever in any transverse direction** — tip
   displacement under ``Py`` and ``Pz`` separately matches the
   analytical ``PL^3 / 3 EI``.

3. **Rigid-body translation** produces zero internal force.

4. **P-Delta amplification in 3-D** — under combined axial
   compression + lateral perturbation, the lateral deflection
   amplifies by ``1 / (1 - P/P_cr)`` as P approaches the cantilever's
   Euler critical load.

5. **Compression reduces lateral stiffness, tension increases it** —
   the cable-stiffening signature, applied to any transverse
   direction.
"""
import math

import numpy as np
import pytest

from femsolver import (
    BeamColumn3D,
    BeamColumn3DCorotational,
    ElasticIsotropic,
    Model,
    NonlinearStaticAnalysis,
)


# ====================================================== helpers

def _build_horizontal_3d_corot(*, L=3.0, A=1.0e-2, Iy=8.333e-6,
                                Iz=8.333e-6, J=1.4e-5):
    E = 2.0e11
    mat = ElasticIsotropic(1, E=E, nu=0.3)
    m = Model(ndm=3, ndf=6); m.add_material(mat)
    m.add_node(1, 0.0, 0.0, 0.0); m.add_node(2, L, 0.0, 0.0)
    elem = BeamColumn3DCorotational(1, (1, 2), mat, A, Iy, Iz, J)
    m.add_element(elem)
    return m, elem, dict(E=E, A=A, Iy=Iy, Iz=Iz, J=J, L=L)


# ====================================================== zero-disp eq

def test_K_at_zero_disp_matches_linear_3d_beam():
    """3D corotational and linear BeamColumn3D agree at u=0 to
    machine precision."""
    E, A, Iy, Iz, J, L = 2.0e11, 1.0e-2, 8.333e-6, 8.333e-6, 1.4e-5, 3.0
    mat = ElasticIsotropic(1, E=E, nu=0.3)
    m_db = Model(ndm=3, ndf=6); m_db.add_material(mat)
    m_db.add_node(1, 0,0,0); m_db.add_node(2, L,0,0)
    e_db = BeamColumn3D(1, (1, 2), mat, A, Iy, Iz, J)
    m_db.add_element(e_db)
    m_co = Model(ndm=3, ndf=6); m_co.add_material(mat)
    m_co.add_node(1, 0,0,0); m_co.add_node(2, L,0,0)
    e_co = BeamColumn3DCorotational(1, (1, 2), mat, A, Iy, Iz, J)
    m_co.add_element(e_co)
    K_db = e_db.K_global()
    K_co = e_co.K_tangent_global()
    np.testing.assert_allclose(K_co, K_db, rtol=1e-10, atol=1e-8)


def test_f_int_at_zero_disp_is_zero():
    _, elem, _ = _build_horizontal_3d_corot()
    np.testing.assert_allclose(elem.f_int_global(), 0.0, atol=1e-9)


@pytest.mark.parametrize("direction", ["y", "z"])
def test_elastic_cantilever_matches_PL3_over_3EI(direction):
    """Cantilever under transverse load in y or z direction:
    tip displacement = PL^3 / (3 E I) with appropriate I."""
    P = 1.0e3
    m, elem, cn = _build_horizontal_3d_corot()
    m.fix(1, [1,1,1,1,1,1])
    if direction == "y":
        m.add_nodal_load(2, [0, -P, 0, 0, 0, 0])
        I_used = cn["Iz"]
    else:
        m.add_nodal_load(2, [0, 0, -P, 0, 0, 0])
        I_used = cn["Iy"]
    NonlinearStaticAnalysis(m, num_steps=1, dlambda=1.0, tol=1e-6).run()
    v_expected = -P * cn["L"] ** 3 / (3.0 * cn["E"] * I_used)
    idx = 1 if direction == "y" else 2
    assert m.node(2).disp[idx] == pytest.approx(v_expected, rel=1e-3)


# ====================================================== rigid translation

def test_rigid_translation_produces_zero_internal_force():
    """Translate both nodes by the same vector — chord direction
    unchanged, no deformation, f_int = 0."""
    m, elem, _ = _build_horizontal_3d_corot()
    m.number_dofs()
    delta = np.array([0.1, 0.05, -0.03])
    for tag in (1, 2):
        m.node(tag).disp[0:3] = delta
    f = elem.f_int_global()
    np.testing.assert_allclose(f, 0.0, atol=1e-8)


# ====================================================== P-Delta in 3D

def _pinpin_3d_column(*, n_elem: int = 8, axis: str = "y"):
    """Slender pin-pin column oriented along x, with lateral load
    applied in the chosen ``axis`` direction (y or z). Returns
    (model, midspan_tag, constants_dict).
    """
    E = 2.0e11
    A = 1.0e-3
    I = 1.0e-7
    L = 5.0
    J = 1.0e-7

    mat = ElasticIsotropic(1, E=E, nu=0.3)
    m = Model(ndm=3, ndf=6); m.add_material(mat)
    for i in range(n_elem + 1):
        m.add_node(i + 1, i * L / n_elem, 0.0, 0.0)
    for i in range(n_elem):
        m.add_element(BeamColumn3DCorotational(
            i + 1, (i + 1, i + 2), mat, A, I, I, J,
        ))
    # Pin-pin: support 1 fixes u, v, w + rotational about x (no twist
    # without other axis fixity it would be free); support 2 is a
    # roller in x.
    m.fix(1, [1, 1, 1, 1, 0, 0])
    m.fix(n_elem + 1, [0, 1, 1, 1, 0, 0])
    midspan_tag = 2 + n_elem // 2 - 1   # interior node nearest midspan
    return m, midspan_tag, dict(E=E, I=I, L=L, A=A, J=J)


def test_compression_reduces_lateral_stiffness():
    """Apply small axial compression and check that the lateral
    component of K_tangent at one of the free nodes decreases — the
    P-Delta softening signature.
    """
    m, elem, cn = _build_horizontal_3d_corot()
    m.number_dofs()
    K0 = elem.K_tangent_global()
    Kvv_0 = K0[7, 7]    # transverse stiffness in y at node 2
    Kww_0 = K0[8, 8]    # transverse stiffness in z at node 2
    # Apply axial compression (u_x at node 2 < 0)
    m.node(2).disp[0] = -2.0e-3
    K_compressed = elem.K_tangent_global()
    assert K_compressed[7, 7] < Kvv_0
    assert K_compressed[8, 8] < Kww_0

    # And tension increases transverse stiffness
    m.node(2).disp[0] = +2.0e-3
    K_tensed = elem.K_tangent_global()
    assert K_tensed[7, 7] > Kvv_0
    assert K_tensed[8, 8] > Kww_0


def test_p_delta_amplification_in_3d():
    """Pin-pin column with axial compression + small lateral load.
    Midspan deflection grows by the textbook ``1 / (1 - P/P_cr)``
    factor as ``P -> P_cr``."""
    E = 2.0e11; I = 1.0e-7; A = 1.0e-3; L = 5.0
    P_cr = math.pi ** 2 * E * I / L ** 2
    P_lateral = 1.0
    mat = ElasticIsotropic(1, E=E, nu=0.3)

    def midspan_deflection(P_axial: float, axis: str = "y") -> float:
        m = Model(ndm=3, ndf=6); m.add_material(mat)
        n_elem = 8
        for i in range(n_elem + 1):
            m.add_node(i + 1, i * L / n_elem, 0.0, 0.0)
        for i in range(n_elem):
            m.add_element(BeamColumn3DCorotational(
                i + 1, (i + 1, i + 2), mat, A, I, I, 1e-7,
            ))
        # Pin-pin: end 1 fully fixed except for rotation about
        # transverse axes; end 2 is a roller in x (axial free).
        m.fix(1, [1, 1, 1, 1, 0, 0])
        m.fix(n_elem + 1, [0, 1, 1, 1, 0, 0])
        # Axial compression at end 2
        m.add_nodal_load(n_elem + 1, [-P_axial, 0, 0, 0, 0, 0])
        # Lateral perturbation at midspan
        mid_tag = 1 + n_elem // 2
        if axis == "y":
            m.add_nodal_load(mid_tag, [0, -P_lateral, 0, 0, 0, 0])
            idx = 1
        else:
            m.add_nodal_load(mid_tag, [0, 0, -P_lateral, 0, 0, 0])
            idx = 2
        NonlinearStaticAnalysis(
            m, num_steps=10, dlambda=1.0 / 10, tol=1e-6, max_iter=30,
        ).run()
        return abs(m.node(mid_tag).disp[idx])

    # No axial — pure transverse response (baseline)
    v0_y = midspan_deflection(0.0, "y")
    v0_z = midspan_deflection(0.0, "z")
    # 80 % of Euler: theoretical amplification factor 1/(1-0.8) = 5x
    v80_y = midspan_deflection(0.8 * P_cr, "y")
    v80_z = midspan_deflection(0.8 * P_cr, "z")
    # Both transverse directions show ~5x amplification (with FE
    # discretisation error allowance)
    assert v80_y > 3.0 * v0_y, f"y-amplification too small: {v80_y / v0_y}"
    assert v80_z > 3.0 * v0_z, f"z-amplification too small: {v80_z / v0_z}"
