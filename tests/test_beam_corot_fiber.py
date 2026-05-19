"""Tests for the corotational beam-column with a fiber section.

Phase 6.5 combines Phase 5 (distributed plasticity via fiber sections)
with Phase 6 (corotational kinematics — large rotations and P-Delta).
The element is the canonical nonlinear column model used in
performance-based seismic design: progressive yielding through the
depth of the section, modified by the axial-load-induced P-Delta
amplification of lateral deflections.

The tests pin down four properties:

1. **Elastic-section corotational unchanged** — passing a
   :class:`ElasticSection2D` must produce identical results to the
   legacy ``(area, Iz)`` constructor. This is the strongest regression
   against the Phase 6 elastic-only path.

2. **Fiber-section corotational matches the displacement-based fiber
   beam at small deformation** — at infinitesimal load, P-Delta
   effects vanish and a fiber-section corotational should agree with
   :class:`BeamColumn2D`-with-fiber to within fiber-discretisation
   error.

3. **Rigid-body invariance with a fiber section** — the corotational
   guarantee that rigid rotations cost nothing must still hold when
   the constitutive layer is nonlinear.

4. **Combined P-Delta + plasticity** — under axial compression *plus*
   a lateral perturbation, the column shows BOTH amplification of the
   lateral deflection (from P-Delta) AND the gradual stiffness drop
   from progressive yielding. At higher axial loads the column yields
   earlier (smaller lateral capacity), an effect neither pure
   geometric nor pure material nonlinearity can capture alone.
"""
import numpy as np
import pytest

from femsolver import (
    BeamColumn2D,
    BeamColumn2DCorotational,
    ElasticIsotropic,
    ElasticSection2D,
    FiberSection2D,
    LinearStaticAnalysis,
    Model,
    NonlinearStaticAnalysis,
    UniaxialBilinear,
    UniaxialElastic,
)


# ============================================== elastic-section regression

def _make_horizontal_corot_with_section(section, L=3.0):
    mat = ElasticIsotropic(1, E=2.0e11, nu=0.3)
    m = Model(ndm=2, ndf=3); m.add_material(mat)
    m.add_node(1, 0.0, 0.0); m.add_node(2, L, 0.0)
    elem = BeamColumn2DCorotational(1, (1, 2), mat, section=section)
    m.add_element(elem)
    return m, elem


def test_elastic_section_corot_matches_legacy_constructor_at_zero():
    """Corotational with section=ElasticSection2D must give identical
    K and f_int to the legacy (area, Iz) corotational at any state."""
    E, A, Iz, L = 2.0e11, 1.0e-2, 8.333e-6, 3.0
    mat = ElasticIsotropic(1, E=E, nu=0.3)
    # Two beams with the same elastic properties, one via section= and
    # one via legacy constructor
    m1 = Model(ndm=2, ndf=3); m1.add_material(mat)
    m1.add_node(1, 0.0, 0.0); m1.add_node(2, L, 0.0)
    e1 = BeamColumn2DCorotational(
        1, (1, 2), mat, section=ElasticSection2D(E=E, A=A, Iz=Iz)
    )
    m1.add_element(e1)

    m2 = Model(ndm=2, ndf=3); m2.add_material(mat)
    m2.add_node(1, 0.0, 0.0); m2.add_node(2, L, 0.0)
    e2 = BeamColumn2DCorotational(1, (1, 2), mat, A, Iz)
    m2.add_element(e2)

    np.testing.assert_allclose(
        e1.K_tangent_global(), e2.K_tangent_global(), rtol=1e-12, atol=1e-8
    )
    np.testing.assert_allclose(
        e1.f_int_global(), e2.f_int_global(), atol=1e-8
    )


@pytest.mark.parametrize("rotation", [0.0, 0.1, 0.5, 1.0])
def test_elastic_section_corot_matches_legacy_under_deformation(rotation):
    """At an arbitrary deformed state, the two construction paths
    must still produce the same K_tangent."""
    E, A, Iz, L = 2.0e11, 1.0e-2, 8.333e-6, 3.0
    mat = ElasticIsotropic(1, E=E, nu=0.3)
    # Build twice
    m1, e1 = _make_horizontal_corot_with_section(
        ElasticSection2D(E=E, A=A, Iz=Iz), L=L
    )
    m2 = Model(ndm=2, ndf=3); m2.add_material(mat)
    m2.add_node(1, 0.0, 0.0); m2.add_node(2, L, 0.0)
    e2 = BeamColumn2DCorotational(1, (1, 2), mat, A, Iz)
    m2.add_element(e2)
    # Apply the same arbitrary displacement state to both
    for m in (m1, m2):
        m.node(1).disp[:] = 0.0
        m.node(2).disp[0] = L * np.cos(rotation) - L
        m.node(2).disp[1] = L * np.sin(rotation)
        m.node(2).disp[2] = rotation
        m.node(1).disp[2] = 0.1 * rotation     # add some θ1 mismatch
    np.testing.assert_allclose(
        e1.K_tangent_global(), e2.K_tangent_global(), rtol=1e-12, atol=1e-6
    )
    np.testing.assert_allclose(
        e1.f_int_global(), e2.f_int_global(), atol=1e-6
    )


# ============================================== fiber-section construction

def test_fiber_section_corot_constructor_sets_stateful_flag():
    """Passing a FiberSection2D must trigger per-IP state on the
    corotational element, just like it does on BeamColumn2D."""
    E, sigma_y = 2.0e11, 400.0e6
    sec = FiberSection2D.rectangular(
        width=0.1, height=0.2, n_fibers=20,
        material=UniaxialBilinear(E=E, sigma_y=sigma_y, b=0.05),
    )
    mat = ElasticIsotropic(1, E=E, nu=0.3)
    m = Model(ndm=2, ndf=3); m.add_material(mat)
    m.add_node(1, 0.0, 0.0); m.add_node(2, 3.0, 0.0)
    elem = BeamColumn2DCorotational(1, (1, 2), mat, section=sec)
    m.add_element(elem)
    assert elem._stateful_sections is True
    assert len(elem.sections) == elem.n_int
    # Each IP has its own clone
    for i in range(elem.n_int):
        for j in range(i + 1, elem.n_int):
            assert elem.sections[i] is not elem.sections[j]


# ====================================== fiber elastic matches displacement

def test_fiber_corot_K_at_zero_matches_displacement_fiber_beam():
    """At u = 0 the corotational K_tangent must match the
    displacement-based fiber beam's K (which itself matches a
    closed-form ElasticSection2D to within fiber-discretisation error).
    """
    E, b, h, L = 2.0e11, 0.1, 0.2, 3.0
    sec = FiberSection2D.rectangular(
        width=b, height=h, n_fibers=40,
        material=UniaxialElastic(E=E),
    )
    mat = ElasticIsotropic(1, E=E, nu=0.3)
    # Corotational with fiber section
    m1 = Model(ndm=2, ndf=3); m1.add_material(mat)
    m1.add_node(1, 0.0, 0.0); m1.add_node(2, L, 0.0)
    e_cor = BeamColumn2DCorotational(1, (1, 2), mat, section=sec.clone())
    m1.add_element(e_cor)
    # Displacement-based with fiber section
    m2 = Model(ndm=2, ndf=3); m2.add_material(mat)
    m2.add_node(1, 0.0, 0.0); m2.add_node(2, L, 0.0)
    e_db = BeamColumn2D(1, (1, 2), mat, section=sec.clone())
    m2.add_element(e_db)
    np.testing.assert_allclose(
        e_cor.K_tangent_global(), e_db.K_tangent_global(),
        rtol=1e-10, atol=1e-3,
    )


# ============================================== rigid-body invariance

def test_fiber_corot_rigid_rotation_produces_no_section_strain():
    """Under a finite rigid rotation, the natural deformations are
    zero, so every fiber sees zero strain and the section stays
    elastic. The internal force must therefore be zero (well, within
    round-off)."""
    E, sigma_y, L = 2.0e11, 400.0e6, 3.0
    sec = FiberSection2D.rectangular(
        width=0.1, height=0.2, n_fibers=20,
        material=UniaxialBilinear(E=E, sigma_y=sigma_y, b=0.05),
    )
    mat = ElasticIsotropic(1, E=E, nu=0.3)
    m, elem = _make_horizontal_corot_with_section(sec, L=L)
    angle = 0.7
    m.node(2).disp[0] = L * np.cos(angle) - L
    m.node(2).disp[1] = L * np.sin(angle)
    m.node(1).disp[2] = angle
    m.node(2).disp[2] = angle
    f_int = elem.f_int_global()
    # Round-off floor: EA * eps / L0
    EA_over_L0 = E * sec.gross_area / L
    floor = 10.0 * EA_over_L0 * np.finfo(float).eps
    np.testing.assert_allclose(f_int, 0.0, atol=floor)
    # None of the fibers should have yielded under a rigid rotation
    for s in elem.sections:
        for f in s.fibers:
            assert f.material.eps_p_trial == 0.0


# ====================================== combined P-Delta + plasticity

def _build_fiber_cantilever(b_post: float = 0.05):
    """Cantilever steel column with a bilinear fiber section."""
    E, sigma_y = 2.0e11, 400.0e6
    b, h, L = 0.1, 0.2, 3.0
    sec = FiberSection2D.rectangular(
        width=b, height=h, n_fibers=20,
        material=UniaxialBilinear(E=E, sigma_y=sigma_y, b=b_post),
    )
    mat = ElasticIsotropic(1, E=E, nu=0.3)
    m = Model(ndm=2, ndf=3); m.add_material(mat)
    m.add_node(1, 0.0, 0.0); m.add_node(2, L, 0.0)
    elem = BeamColumn2DCorotational(1, (1, 2), mat, section=sec)
    m.add_element(elem)
    m.fix(1, [1, 1, 1])
    Mp = sigma_y * b * h ** 2 / 4.0     # plastic moment
    My = sigma_y * b * h ** 2 / 6.0     # first-yield moment
    return m, elem, L, My, Mp, E, sec.gross_Iz


def test_fiber_corot_pushover_yields_progressively():
    """Moderate axial preload + lateral pushover just past first yield.

    With the corotational + fiber combination, the fixed-end IP must
    show plasticity (some fibers yielded). We stay below the limit
    point so load control converges. Keeping below ``Mp / L`` and
    moderate axial preload avoids the geometric-instability region.
    """
    m, elem, L, My, Mp, E, Iz = _build_fiber_cantilever(b_post=0.1)
    EI = E * Iz
    P_yield = My / L
    # Lateral load just past first fiber yield, low axial preload
    P_lateral = 1.05 * P_yield
    P_cantilever_cr = np.pi ** 2 * EI / (4.0 * L ** 2)
    P_axial = 0.1 * P_cantilever_cr
    m.add_nodal_load(2, [-P_axial, -P_lateral, 0.0])
    NonlinearStaticAnalysis(
        m, num_steps=20, dlambda=1.0 / 20, tol=1e-5, max_iter=40,
    ).run()
    # At least one fiber at the fixed end must have yielded
    n_yielded = sum(
        1 for f in elem.sections[0].fibers if f.material.eps_p_committed != 0.0
    )
    assert n_yielded > 0


def test_axial_compression_amplifies_lateral_response():
    """Same lateral load applied with and without axial compression.

    The corotational + fiber combination must show: tip deflection is
    LARGER under axial compression than without. The P-Delta
    amplification is the only mechanism that can produce this — a
    pure fiber section would give an axial-independent lateral
    response.

    We keep the lateral load below first yield so the column stays
    elastic at the section level — the test isolates the geometric
    contribution.
    """
    def lateral_deflection(P_axial_ratio: float) -> float:
        m, elem, L, My, Mp, E, Iz = _build_fiber_cantilever(b_post=0.1)
        EI = E * Iz
        P_cantilever_cr = np.pi ** 2 * EI / (4.0 * L ** 2)
        P_axial = P_axial_ratio * P_cantilever_cr
        P_lateral = 0.5 * My / L      # safely below first yield
        m.add_nodal_load(2, [-P_axial, -P_lateral, 0.0])
        NonlinearStaticAnalysis(
            m, num_steps=10, dlambda=1.0 / 10, tol=1e-5, max_iter=30,
        ).run()
        return abs(m.node(2).disp[1])

    v_no_axial = lateral_deflection(0.0)
    v_with_axial = lateral_deflection(0.5)   # 50 % of cantilever buckling
    # Lateral deflection must be at least 30 % larger under significant
    # axial compression — the P-Delta amplification.
    assert v_with_axial > 1.3 * v_no_axial


def test_axial_compression_lowers_first_yield_lateral_load():
    """M-P interaction signature: axial compression brings the column
    to first-fiber yield at a smaller lateral load than the
    no-compression case.

    Mechanism: axial compression already loads the cross-section's
    +y fibers (where flexure-compression coincides under negative
    bending). The extra compressive strain from bending therefore
    crosses the yield threshold sooner. P-Delta amplification of the
    lateral deflection further accelerates this.
    """
    def fixed_end_max_strain(P_axial_ratio: float, P_lateral_ratio: float) -> float:
        """Run the analysis and return the maximum |eps| over the
        fixed-end section's fibers."""
        m, elem, L, My, Mp, E, Iz = _build_fiber_cantilever(b_post=0.1)
        EI = E * Iz
        P_cantilever_cr = np.pi ** 2 * EI / (4.0 * L ** 2)
        P_axial = P_axial_ratio * P_cantilever_cr
        P_lateral = P_lateral_ratio * My / L
        m.add_nodal_load(2, [-P_axial, -P_lateral, 0.0])
        NonlinearStaticAnalysis(
            m, num_steps=10, dlambda=1.0 / 10, tol=1e-5, max_iter=30,
        ).run()
        # The extreme-fiber strain at the fixed end is what triggers
        # first yield. Read it off the section state.
        eps_max = max(
            abs(f.material.eps_p_committed)
            + abs(f.material.sigma_trial / f.material.E)
            for f in elem.sections[0].fibers
        )
        return eps_max

    # Same lateral ratio in both cases; axial compression adds extra
    # compressive strain through the depth.
    P_lat_ratio = 0.95   # 95 % of "first-yield lateral force"
    eps_no_axial = fixed_end_max_strain(0.0, P_lat_ratio)
    eps_with_axial = fixed_end_max_strain(0.3, P_lat_ratio)
    # With axial compression, the extreme-fiber strain is larger — the
    # column is closer to (or past) first yield under the same lateral
    # load. The exact ratio depends on geometry but a clear ordering
    # is expected.
    assert eps_with_axial > eps_no_axial
