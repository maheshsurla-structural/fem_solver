"""Tests for 3-D solid elements -- Hex8 and Tet4 (Phase 15).

The following properties pin down each element's correctness:

1. **Construction & validation** -- K is symmetric, of correct shape.
2. **Rigid-body modes** -- translation in any direction gives zero
   internal force.
3. **Patch test (prescribed-displacement form)** -- a uniform-strain
   state imposed at nodes is reproduced exactly at every Gauss point
   / element. This is the rigorous patch test for irregular meshes.
4. **Uniaxial tension closed form** -- a single Hex8 under uniaxial
   tension gives u = P L / (E A) exactly.
5. **Hex8 cantilever convergence** -- a 3-D cantilever block under
   tip load converges to Euler beam theory under mesh refinement
   (full integration is well-known to converge slowly for bending;
   the test only requires monotone convergence to within 20% at
   N_x = 16, N_y = N_z = 1).
6. **Tet4 same patch test** on a 6-tet diagonal-fan decomposition
   of a unit cube.
"""
import numpy as np
import pytest

from femsolver import ElasticIsotropic, Hex8, Model, Tet4
from femsolver.analysis.linear_static import LinearStaticAnalysis


# ====================================================== unit-cube helpers

_CUBE_CORNERS = [
    (0, 0, 0), (1, 0, 0), (1, 1, 0), (0, 1, 0),
    (0, 0, 1), (1, 0, 1), (1, 1, 1), (0, 1, 1),
]

_TET_FAN = [
    (1, 2, 3, 7), (1, 3, 4, 7), (1, 4, 8, 7),
    (1, 8, 5, 7), (1, 5, 6, 7), (1, 6, 2, 7),
]


def _unit_cube_hex(E: float = 1.0e10, nu: float = 0.3):
    mat = ElasticIsotropic(1, E=E, nu=nu)
    m = Model(ndm=3, ndf=3); m.add_material(mat)
    for i, (x, y, z) in enumerate(_CUBE_CORNERS):
        m.add_node(i + 1, x, y, z)
    e = Hex8(1, tuple(range(1, 9)), mat)
    m.add_element(e)
    return m, e


def _unit_cube_tet_fan(E: float = 1.0e10, nu: float = 0.3):
    mat = ElasticIsotropic(1, E=E, nu=nu)
    m = Model(ndm=3, ndf=3); m.add_material(mat)
    for i, (x, y, z) in enumerate(_CUBE_CORNERS):
        m.add_node(i + 1, x, y, z)
    elements = []
    for tag, ts in enumerate(_TET_FAN, start=1):
        e = Tet4(tag, ts, mat)
        m.add_element(e); elements.append(e)
    return m, elements


# ====================================================== Hex8

def test_hex8_K_is_24x24_and_symmetric():
    _, e = _unit_cube_hex()
    K = e.K_global()
    assert K.shape == (24, 24)
    assert np.allclose(K, K.T, atol=1e-8 * np.max(np.abs(K)))


def test_hex8_zero_force_under_rigid_translation():
    m, e = _unit_cube_hex()
    m.number_dofs()
    for tag in range(1, 9):
        m.node(tag).disp[:] = [1.0, 0.5, -0.3]
    f_int = e.f_int_global()
    EA = e.material.E * 1.0
    assert np.max(np.abs(f_int)) < 1.0e-9 * EA


def test_hex8_uniaxial_tension_closed_form():
    """A unit cube pulled in x with force P at the +x face should
    give u_x = P * L / (E * A) at every +x face node."""
    E, nu = 1e10, 0.0     # nu=0 for clean closed form
    m, _ = _unit_cube_hex(E=E, nu=nu)
    # Fix left face (x=0): u=0 plus minimal constraints to remove
    # rigid-body modes.
    m.fix(1, [1, 1, 1])
    m.fix(4, [1, 1, 0])
    m.fix(5, [1, 0, 1])
    m.fix(8, [1, 0, 0])
    P = 1000.0
    for tag in (2, 3, 6, 7):
        m.add_nodal_load(tag, [P / 4, 0, 0])
    LinearStaticAnalysis(m).run()
    for tag in (2, 3, 6, 7):
        assert m.node(tag).disp[0] == pytest.approx(
            P * 1.0 / (E * 1.0), rel=1e-12
        )


def test_hex8_prescribed_displacement_patch_test():
    """Impose u = (alpha*x, beta*y, gamma*z) on every node; every GP
    should recover the analytical uniform strain state."""
    m, e = _unit_cube_hex(E=1e10, nu=0.3)
    m.number_dofs()
    alpha, beta, gamma = 1e-3, 2e-4, -5e-4
    for tag in range(1, 9):
        x, y, z = m.node(tag).coords
        m.node(tag).disp[:] = [alpha * x, beta * y, gamma * z]
    e.recover()
    for eps in e.gp_strain:
        assert eps[0] == pytest.approx(alpha, abs=1e-12)
        assert eps[1] == pytest.approx(beta, abs=1e-12)
        assert eps[2] == pytest.approx(gamma, abs=1e-12)
        for i in (3, 4, 5):
            assert abs(eps[i]) < 1e-12


def test_hex8_cantilever_converges_to_beam_theory():
    """Cantilever block (1 x 0.1 x 0.1) under tip load should converge
    to beam theory under x-refinement. Hex8 with full integration is
    notoriously stiff in bending; we require *monotone* improvement
    and a 16-element ratio above 0.8."""
    E, nu = 2e11, 0.0
    L, b, h = 1.0, 0.1, 0.1
    P = 1.0
    I = b * h ** 3 / 12.0
    w_beam = P * L ** 3 / (3.0 * E * I)

    def build(N_x: int):
        mat = ElasticIsotropic(1, E=E, nu=nu)
        m = Model(ndm=3, ndf=3); m.add_material(mat)
        nx, ny, nz = N_x + 1, 2, 2
        for k in range(nz):
            for j in range(ny):
                for i in range(nx):
                    tag = k * nx * ny + j * nx + i + 1
                    m.add_node(tag, i * L / N_x, j * b, k * h)
        etag = 1
        def n(i, j, k): return k * nx * ny + j * nx + i + 1
        for k in range(nz - 1):
            for j in range(ny - 1):
                for i in range(N_x):
                    m.add_element(Hex8(etag, (
                        n(i, j, k), n(i+1, j, k), n(i+1, j+1, k), n(i, j+1, k),
                        n(i, j, k+1), n(i+1, j, k+1), n(i+1, j+1, k+1), n(i, j+1, k+1),
                    ), mat)); etag += 1
        for k in range(nz):
            for j in range(ny):
                m.fix(n(0, j, k), [1, 1, 1])
        tip = [n(N_x, j, k) for k in range(nz) for j in range(ny)]
        for nt in tip:
            m.add_nodal_load(nt, [0, 0, -P / len(tip)])
        LinearStaticAnalysis(m).run()
        return -np.mean([m.node(nt).disp[2] for nt in tip])

    w4 = build(4); w8 = build(8); w16 = build(16)
    # Monotone improvement
    assert w4 < w8 < w16
    # 16-element ratio above 80% of beam theory (Hex8 has well-known
    # bending shear-locking; better convergence requires B-bar or
    # incompatible modes — future Phase 15.x).
    assert w16 / w_beam > 0.80, f"got ratio {w16 / w_beam:.4f}"


# ====================================================== Tet4

def test_tet4_K_is_12x12_and_symmetric():
    mat = ElasticIsotropic(1, E=1e10, nu=0.3)
    m = Model(ndm=3, ndf=3); m.add_material(mat)
    m.add_node(1, 0, 0, 0)
    m.add_node(2, 1, 0, 0)
    m.add_node(3, 0, 1, 0)
    m.add_node(4, 0, 0, 1)
    e = Tet4(1, (1, 2, 3, 4), mat)
    m.add_element(e)
    K = e.K_global()
    assert K.shape == (12, 12)
    assert np.allclose(K, K.T, atol=1e-8 * np.max(np.abs(K)))


def test_tet4_rejects_negative_volume():
    mat = ElasticIsotropic(1, E=1e10, nu=0.3)
    m = Model(ndm=3, ndf=3); m.add_material(mat)
    m.add_node(1, 0, 0, 0)
    m.add_node(2, 1, 0, 0)
    m.add_node(3, 0, 1, 0)
    m.add_node(4, 0, 0, 1)
    # Wrong orientation: swap nodes 2 and 3 -> negative volume
    e = Tet4(1, (1, 3, 2, 4), mat)
    m.add_element(e)
    with pytest.raises(ValueError, match="signed volume"):
        e.K_global()


def test_tet4_zero_force_under_rigid_translation():
    mat = ElasticIsotropic(1, E=1e10, nu=0.3)
    m = Model(ndm=3, ndf=3); m.add_material(mat)
    m.add_node(1, 0, 0, 0)
    m.add_node(2, 1, 0, 0)
    m.add_node(3, 0, 1, 0)
    m.add_node(4, 0, 0, 1)
    e = Tet4(1, (1, 2, 3, 4), mat)
    m.add_element(e)
    m.number_dofs()
    for tag in (1, 2, 3, 4):
        m.node(tag).disp[:] = [1.0, -0.5, 0.2]
    f_int = e.f_int_global()
    EA = mat.E * 1.0
    assert np.max(np.abs(f_int)) < 1.0e-9 * EA


def test_tet4_prescribed_displacement_patch_test():
    """A uniform-strain state imposed at the 8 cube corners must be
    reproduced exactly at every tet of a 6-tet fan decomposition."""
    m, elements = _unit_cube_tet_fan(E=1e10, nu=0.3)
    m.number_dofs()
    alpha, beta, gamma = 1e-3, 2e-4, -5e-4
    for tag in range(1, 9):
        x, y, z = m.node(tag).coords
        m.node(tag).disp[:] = [alpha * x, beta * y, gamma * z]
    for e in elements:
        e.recover()
        assert e.strain[0] == pytest.approx(alpha, abs=1e-12)
        assert e.strain[1] == pytest.approx(beta, abs=1e-12)
        assert e.strain[2] == pytest.approx(gamma, abs=1e-12)
        for k in (3, 4, 5):
            assert abs(e.strain[k]) < 1e-12


def test_tet4_uniaxial_tension_uniform_stress():
    """Pull a 6-tet cube in x with the *consistent* nodal forces for a
    uniform face traction. All tets should compute the same sigma_xx."""
    E, nu = 1e10, 0.0
    m, elements = _unit_cube_tet_fan(E=E, nu=nu)
    m.fix(1, [1, 1, 1])
    m.fix(4, [1, 1, 0])
    m.fix(5, [1, 0, 1])
    m.fix(8, [1, 0, 0])
    # Consistent face-traction nodal forces for the 6-tet fan: corners 2
    # and 7 are touched by 2 triangles, 3 and 6 by 1. Each triangle has
    # area 1/2, traction P/A = P, so each triangle contributes P/2
    # equivalent nodal force, distributed P/6 per node.
    P = 1000.0
    m.add_nodal_load(2, [P / 3, 0, 0])    # corner 2: 2 triangles
    m.add_nodal_load(7, [P / 3, 0, 0])    # corner 7: 2 triangles
    m.add_nodal_load(3, [P / 6, 0, 0])    # corner 3: 1 triangle
    m.add_nodal_load(6, [P / 6, 0, 0])    # corner 6: 1 triangle
    LinearStaticAnalysis(m).run()
    for e in elements:
        e.recover()
        # All tets should have the same uniaxial-tension stress
        assert e.stress[0] == pytest.approx(P, rel=1e-10)
        assert abs(e.stress[1]) < 1e-6
        assert abs(e.stress[2]) < 1e-6
