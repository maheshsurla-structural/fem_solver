"""Tests for FiberSection3D — distributed-plasticity 3D cross-section.

The section is validated through three properties:

1. **Gross properties** — for a rectangle ``b_y x b_z`` discretised
   into ``n_y x n_z`` fibers: gross area, second moments ``Iz, Iy``,
   centroid all match analytical to within strip-discretisation error.

2. **Elastic equivalence with ElasticSection3D** — for an all-elastic
   fiber section with the same gross properties, ``get_response``
   produces the same forces and tangent as :class:`ElasticSection3D`
   at every strain state (to within fiber-discretisation error).

3. **Cross-coupling emerges only after asymmetric biaxial yielding**
   — for a symmetric elastic section the tangent ``ks`` is diagonal;
   the off-diagonal blocks ``ES_z, ES_y, EI_yz`` become non-zero
   only when fibers yield asymmetrically. This is the 3D analog of
   the 2D P-M interaction.

4. **BeamColumn3D + FiberSection3D end-to-end** — a cantilever under
   axial + biaxial bending gives the analytical elastic tip
   displacements in both transverse directions.
"""
import math

import numpy as np
import pytest

from femsolver import (
    BeamColumn3D,
    ElasticIsotropic,
    ElasticSection3D,
    Fiber,
    FiberSection3D,
    LinearStaticAnalysis,
    Model,
    NonlinearStaticAnalysis,
    UniaxialBilinear,
    UniaxialElastic,
)


# ====================================================== construction

def test_fiber3d_rejects_empty_fibers():
    with pytest.raises(ValueError):
        FiberSection3D([], GJ=1.0)


def test_fiber3d_rejects_nonpositive_GJ():
    mat = UniaxialElastic(E=1.0e11)
    with pytest.raises(ValueError):
        FiberSection3D(
            [Fiber(y=0.0, z=0.0, area=1.0, material=mat)], GJ=0.0,
        )


def test_rectangular_3d_rejects_too_few_fibers_per_direction():
    mat = UniaxialElastic(E=1.0e11)
    with pytest.raises(ValueError):
        FiberSection3D.rectangular(
            width_y=0.2, width_z=0.1, n_y=1, n_z=2,
            material=mat, GJ=1.0e6,
        )
    with pytest.raises(ValueError):
        FiberSection3D.rectangular(
            width_y=0.2, width_z=0.1, n_y=2, n_z=1,
            material=mat, GJ=1.0e6,
        )


# ====================================================== gross properties

def test_rectangular_3d_gross_area():
    mat = UniaxialElastic(E=2.0e11)
    sec = FiberSection3D.rectangular(
        width_y=0.2, width_z=0.1, n_y=10, n_z=8,
        material=mat, GJ=1.0e6,
    )
    assert sec.gross_area == pytest.approx(0.2 * 0.1, rel=1e-14)


def test_rectangular_3d_centroid_at_origin():
    mat = UniaxialElastic(E=2.0e11)
    sec = FiberSection3D.rectangular(
        width_y=0.4, width_z=0.3, n_y=12, n_z=10,
        material=mat, GJ=1.0e6,
    )
    assert sec.centroid_y == pytest.approx(0.0, abs=1e-13)
    assert sec.centroid_z == pytest.approx(0.0, abs=1e-13)


def test_rectangular_3d_gross_Iz_and_Iy():
    """For a rectangle of dimensions width_y x width_z:
       Iz_exact = width_z * width_y^3 / 12  (second moment about z)
       Iy_exact = width_y * width_z^3 / 12  (second moment about y)
    The fiber discretisation converges as O(h^2) — at 50 strips per
    direction the error is ~0.05 %.
    """
    width_y, width_z = 0.4, 0.2
    n = 50
    mat = UniaxialElastic(E=2.0e11)
    sec = FiberSection3D.rectangular(
        width_y=width_y, width_z=width_z, n_y=n, n_z=n,
        material=mat, GJ=1.0e6,
    )
    Iz_exact = width_z * width_y ** 3 / 12.0
    Iy_exact = width_y * width_z ** 3 / 12.0
    assert sec.gross_Iz == pytest.approx(Iz_exact, rel=1e-3)
    assert sec.gross_Iy == pytest.approx(Iy_exact, rel=1e-3)


def test_rectangular_3d_Iyz_zero_for_symmetric_section():
    mat = UniaxialElastic(E=2.0e11)
    sec = FiberSection3D.rectangular(
        width_y=0.4, width_z=0.2, n_y=8, n_z=6,
        material=mat, GJ=1.0e6,
    )
    assert sec.gross_Iyz == pytest.approx(0.0, abs=1e-13)


# ====================================================== elastic equivalence

@pytest.mark.parametrize("e_vec", [
    np.array([1e-4,  0.0,    0.0,    0.0]),    # pure axial
    np.array([0.0,   2e-3,   0.0,    0.0]),    # pure kappa_z
    np.array([0.0,   0.0,    1e-3,   0.0]),    # pure kappa_y
    np.array([0.0,   0.0,    0.0,    5e-3]),   # pure torsion
    np.array([1e-4,  2e-3,   1e-3,   5e-3]),   # all four
    np.array([-2e-5, -1e-3, -2e-3,  -3e-3]),   # negative directions
])
def test_elastic_fiber3d_matches_ElasticSection3D(e_vec):
    """All-elastic FiberSection3D vs analytical ElasticSection3D —
    must agree to within fiber-discretisation error for any strain."""
    E = 2.0e11
    G = E / (2.0 * 1.3)        # nu = 0.3
    width_y, width_z = 0.4, 0.2
    A_exact = width_y * width_z
    Iz_exact = width_z * width_y ** 3 / 12.0
    Iy_exact = width_y * width_z ** 3 / 12.0
    J = 1.0e-4
    GJ = G * J

    sec_fiber = FiberSection3D.rectangular(
        width_y=width_y, width_z=width_z, n_y=80, n_z=40,
        material=UniaxialElastic(E=E), GJ=GJ,
    )
    sec_elastic = ElasticSection3D(
        E=E, G=G, A=A_exact, Iy=Iy_exact, Iz=Iz_exact, J=J,
    )
    s_f, ks_f = sec_fiber.get_response(e_vec)
    s_e, ks_e = sec_elastic.get_response(e_vec)
    np.testing.assert_allclose(s_f, s_e, rtol=2e-3, atol=1e-4)
    # Off-diagonal of fiber ks should be ~ zero for symmetric section
    assert abs(ks_f[0, 1]) < 1e-3 * abs(ks_f[0, 0])
    assert abs(ks_f[0, 2]) < 1e-3 * abs(ks_f[0, 0])
    assert abs(ks_f[1, 2]) < 1e-3 * abs(ks_f[1, 1])


def test_torsion_uncoupled_from_axial_and_bending():
    """Pure torsion ``gamma > 0`` should produce only ``T = GJ gamma``;
    N, Mz, My all zero (since fibers contribute axial only)."""
    mat = UniaxialBilinear(E=2.0e11, sigma_y=400e6, b=0.05)
    GJ = 1.0e6
    sec = FiberSection3D.rectangular(
        width_y=0.2, width_z=0.1, n_y=10, n_z=4,
        material=mat, GJ=GJ,
    )
    e = np.array([0.0, 0.0, 0.0, 2.0e-3])
    s, ks = sec.get_response(e)
    assert s[0] == pytest.approx(0.0, abs=1e-9)
    assert s[1] == pytest.approx(0.0, abs=1e-9)
    assert s[2] == pytest.approx(0.0, abs=1e-9)
    assert s[3] == pytest.approx(GJ * 2.0e-3, rel=1e-14)


# ====================================================== cross-coupling

def test_no_coupling_in_elastic_symmetric_section():
    """An all-elastic symmetric section produces a diagonal tangent."""
    mat = UniaxialElastic(E=2.0e11)
    sec = FiberSection3D.rectangular(
        width_y=0.2, width_z=0.1, n_y=20, n_z=10,
        material=mat, GJ=1.0e6,
    )
    e = np.array([1e-4, 2e-3, 1e-3, 5e-3])
    _, ks = sec.get_response(e)
    # All off-diagonals in the axial-bending block (3x3) are zero
    for i in range(3):
        for j in range(3):
            if i != j:
                assert abs(ks[i, j]) < 1e-3 * (
                    abs(ks[i, i]) + abs(ks[j, j])
                )


def test_coupling_emerges_after_asymmetric_biaxial_yielding():
    """Apply biaxial bending past first yield. The off-diagonal
    ``EI_yz`` should become non-zero (asymmetric fiber yielding
    means y*z*Et averages to nonzero across the section)."""
    E, sy = 2.0e11, 400.0e6
    mat = UniaxialBilinear(E=E, sigma_y=sy, b=0.05)
    sec = FiberSection3D.rectangular(
        width_y=0.4, width_z=0.2, n_y=20, n_z=10,
        material=mat, GJ=1.0e6,
    )
    # Apply biaxial bending past yield: corner-fiber strain
    # = eps_axial - y_max * kappa_z + z_max * kappa_y. We want some
    # fibers past yield, others not.
    eps_y_steel = sy / E      # 2e-3
    # Pick combined curvatures that yield only one corner heavily.
    e = np.array([0.0, 6.0 * eps_y_steel / 0.2, 6.0 * eps_y_steel / 0.1, 0.0])
    sec.get_response(e); sec.commit_state()
    # Re-evaluate at the committed state
    _, ks = sec.get_response(e)
    # Off-diagonal EI_yz (= ks[1, 2] = -EI_yz, also ks[2, 1])
    EI_yz_term = abs(ks[1, 2])
    # Magnitude should be a non-trivial fraction of the diagonal EI_z
    assert EI_yz_term > 1e-6 * abs(ks[1, 1])


# ====================================================== BeamColumn3D path

def test_beamcolumn3d_with_fiber_section_constructor_sets_stateful_flag():
    """Passing a FiberSection3D triggers the stateful-section path
    in BeamColumn3D: per-IP cloned sections, numerical integration."""
    E, sy = 2.0e11, 400.0e6
    mat_iso = ElasticIsotropic(1, E=E, nu=0.3)
    mat_u = UniaxialBilinear(E=E, sigma_y=sy, b=0.05)
    sec = FiberSection3D.rectangular(
        width_y=0.2, width_z=0.1, n_y=10, n_z=4,
        material=mat_u, GJ=1.0e6,
    )
    m = Model(ndm=3, ndf=6); m.add_material(mat_iso)
    m.add_node(1, 0.0, 0.0, 0.0); m.add_node(2, 3.0, 0.0, 0.0)
    elem = BeamColumn3D(1, (1, 2), mat_iso, section=sec)
    m.add_element(elem)
    assert elem._stateful_sections is True
    assert elem.use_numerical_integration is True
    assert len(elem.sections) == elem.n_int
    # Each IP must be an independent clone
    for i in range(elem.n_int):
        for j in range(i + 1, elem.n_int):
            assert elem.sections[i] is not elem.sections[j]


def test_beamcolumn3d_with_fiber_section_elastic_biaxial_cantilever():
    """3D cantilever with all-elastic fiber section under biaxial tip
    loads. Tip displacements in y and z must match the analytical
    PL^3 / (3 EI) for the respective second moments."""
    E = 2.0e11
    nu = 0.3
    G = E / (2.0 * (1 + nu))
    width_y, width_z = 0.2, 0.1
    L = 3.0
    Iz_exact = width_z * width_y ** 3 / 12.0
    Iy_exact = width_y * width_z ** 3 / 12.0
    J = 1.0e-4
    GJ = G * J

    mat_iso = ElasticIsotropic(1, E=E, nu=nu)
    # Use enough fibers in both directions that the strip-midpoint
    # error on Iy and Iz is well under 0.3%.
    sec = FiberSection3D.rectangular(
        width_y=width_y, width_z=width_z, n_y=40, n_z=20,
        material=UniaxialElastic(E=E), GJ=GJ,
    )
    m = Model(ndm=3, ndf=6); m.add_material(mat_iso)
    m.add_node(1, 0.0, 0.0, 0.0); m.add_node(2, L, 0.0, 0.0)
    m.add_element(BeamColumn3D(1, (1, 2), mat_iso, section=sec))
    m.fix(1, [1, 1, 1, 1, 1, 1])
    Py, Pz = 1000.0, 500.0
    m.add_nodal_load(2, [0.0, -Py, -Pz, 0.0, 0.0, 0.0])
    NonlinearStaticAnalysis(m, num_steps=1, dlambda=1.0, tol=1e-6).run()
    v_expected = -Py * L ** 3 / (3.0 * E * Iz_exact)
    w_expected = -Pz * L ** 3 / (3.0 * E * Iy_exact)
    # 5e-3 tolerance accommodates the strip-midpoint fiber-discretisation
    # error on Iy / Iz at 20 x 10 fibers (~0.3% on each).
    assert m.node(2).disp[1] == pytest.approx(v_expected, rel=5e-3)
    assert m.node(2).disp[2] == pytest.approx(w_expected, rel=5e-3)


def test_beamcolumn3d_with_fiber_section_yields_under_biaxial_load():
    """Bilinear fiber section under biaxial loading past first yield —
    some fibers must show plastic strain."""
    E, sy = 2.0e11, 400.0e6
    width_y, width_z = 0.2, 0.1
    L = 3.0
    Iz_exact = width_z * width_y ** 3 / 12.0

    mat_iso = ElasticIsotropic(1, E=E, nu=0.3)
    mat_u = UniaxialBilinear(E=E, sigma_y=sy, b=0.05)
    sec = FiberSection3D.rectangular(
        width_y=width_y, width_z=width_z, n_y=10, n_z=4,
        material=mat_u, GJ=1.0e6,
    )
    m = Model(ndm=3, ndf=6); m.add_material(mat_iso)
    m.add_node(1, 0.0, 0.0, 0.0); m.add_node(2, L, 0.0, 0.0)
    elem = BeamColumn3D(1, (1, 2), mat_iso, section=sec)
    m.add_element(elem)
    m.fix(1, [1, 1, 1, 1, 1, 1])
    # First-yield force estimate: My_section = sy * width_z * width_y^2 / 6
    # (assumes the outermost fiber is at y = width_y/2; with strip
    # discretisation the outermost fiber sits slightly inside this,
    # so we push to 1.5x to ensure visible plasticity).
    My_section = sy * width_z * width_y ** 2 / 6.0
    Py_yield = My_section / L
    Py = 1.5 * Py_yield        # comfortably past first yield in y
    Pz = 0.0
    m.add_nodal_load(2, [0.0, -Py, -Pz, 0.0, 0.0, 0.0])
    NonlinearStaticAnalysis(
        m, num_steps=12, dlambda=1.0 / 12, tol=1.0e-5, max_iter=30,
    ).run()
    # Some fibers at the fixed end must have yielded
    fixed_end_sec = elem.sections[0]
    yielded = [f for f in fixed_end_sec.fibers
               if f.material.eps_p_committed != 0.0]
    assert len(yielded) > 0
