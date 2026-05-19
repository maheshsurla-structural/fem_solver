"""Tests for BeamColumn2D with a FiberSection2D — distributed plasticity
at the element level.

These tests check the *composition* of the pieces: the element's
per-IP section list, the consistent tangent ``K_tangent_global``, the
state-aware ``f_int_global``, the ``commit_state`` / ``revert_state``
forwarding, and that a full pushover analysis converges and produces
the expected bilinear force-displacement curve.

The elastic regime is also pinned down: a fiber-section beam under
linear-static analysis must reproduce the closed-form cantilever
deflection to within fiber-discretisation error.
"""
import numpy as np
import pytest

from femsolver import (
    BeamColumn2D,
    ElasticIsotropic,
    FiberSection2D,
    LinearStaticAnalysis,
    Model,
    NonlinearStaticAnalysis,
    UniaxialBilinear,
    UniaxialElastic,
)


# =================================================== construction ==

def test_elastic_fiber_beam_constructor_sets_stateful_flag():
    """Passing a FiberSection2D must trigger the stateful per-IP path."""
    mat = ElasticIsotropic(1, E=2.0e11, nu=0.3)
    sec = FiberSection2D.rectangular(
        width=0.1, height=0.2, n_fibers=20, material=UniaxialElastic(E=2.0e11)
    )
    m = Model(ndm=2, ndf=3); m.add_material(mat)
    m.add_node(1, 0.0, 0.0); m.add_node(2, 3.0, 0.0)
    elem = BeamColumn2D(1, (1, 2), mat, section=sec)
    m.add_element(elem)
    assert elem._stateful_sections is True
    assert elem.use_numerical_integration is True
    assert len(elem.sections) == elem.n_int
    # each IP has an independent fiber section (different objects)
    for i in range(elem.n_int):
        for j in range(i + 1, elem.n_int):
            assert elem.sections[i] is not elem.sections[j]


def test_elastic_fiber_beam_area_Iz_match_gross():
    mat = ElasticIsotropic(1, E=2.0e11, nu=0.3)
    b, h = 0.1, 0.2
    sec = FiberSection2D.rectangular(
        width=b, height=h, n_fibers=20, material=UniaxialElastic(E=2.0e11)
    )
    m = Model(ndm=2, ndf=3); m.add_material(mat)
    m.add_node(1, 0.0, 0.0); m.add_node(2, 3.0, 0.0)
    elem = BeamColumn2D(1, (1, 2), mat, section=sec)
    m.add_element(elem)
    assert elem.area == pytest.approx(b * h, rel=1e-14)
    # gross Iz of the fiber section is what got stored; matches b h^3 / 12
    # to within the strip-midpoint discretisation error (~0.3 % at 20 strips,
    # 1/n^2 convergence)
    assert elem.Iz == pytest.approx(b * h ** 3 / 12.0, rel=5e-3)


# =================================================== elastic regime ==

def test_elastic_fiber_beam_cantilever_matches_analytical():
    """Cantilever, tip vertical load P, all-elastic fibers.
    Tip deflection must match ``-PL^3/(3 EI)`` to within
    fiber-discretisation error (~0.1 % at 40 fibers)."""
    E, b, h, L, P = 2.0e11, 0.1, 0.2, 3.0, 1.0e4
    Iz_exact = b * h ** 3 / 12.0
    mat = ElasticIsotropic(1, E=E, nu=0.3)
    sec = FiberSection2D.rectangular(
        width=b, height=h, n_fibers=40, material=UniaxialElastic(E=E)
    )
    m = Model(ndm=2, ndf=3); m.add_material(mat)
    m.add_node(1, 0.0, 0.0); m.add_node(2, L, 0.0)
    m.add_element(BeamColumn2D(1, (1, 2), mat, section=sec))
    m.fix(1, [1, 1, 1])
    m.add_nodal_load(2, [0.0, -P, 0.0])
    LinearStaticAnalysis(m).run()
    v_expected = -P * L ** 3 / (3.0 * E * Iz_exact)
    assert m.node(2).disp[1] == pytest.approx(v_expected, rel=2e-3)


def test_elastic_fiber_beam_axial_matches_PL_over_EA():
    """Axial-only loading: a fiber-section beam must give ``u = P L / (E A)``."""
    E, b, h, L, P = 2.0e11, 0.1, 0.2, 3.0, 1.0e4
    mat = ElasticIsotropic(1, E=E, nu=0.3)
    sec = FiberSection2D.rectangular(
        width=b, height=h, n_fibers=20, material=UniaxialElastic(E=E)
    )
    m = Model(ndm=2, ndf=3); m.add_material(mat)
    m.add_node(1, 0.0, 0.0); m.add_node(2, L, 0.0)
    m.add_element(BeamColumn2D(1, (1, 2), mat, section=sec))
    m.fix(1, [1, 1, 1]); m.fix(2, [0, 1, 1])
    m.add_nodal_load(2, [P, 0.0, 0.0])
    LinearStaticAnalysis(m).run()
    assert m.node(2).disp[0] == pytest.approx(P * L / (E * b * h), rel=1e-12)


def test_elastic_fiber_K_matches_closed_form_elastic_section():
    """At zero strain, K_tangent_global of a fiber-section beam must
    match that of an equivalent ElasticSection2D beam (gross properties)
    to within fiber-discretisation error.
    """
    E, b, h, L = 2.0e11, 0.1, 0.2, 3.0
    mat = ElasticIsotropic(1, E=E, nu=0.3)
    # Fiber elastic
    sec = FiberSection2D.rectangular(
        width=b, height=h, n_fibers=80, material=UniaxialElastic(E=E)
    )
    m = Model(ndm=2, ndf=3); m.add_material(mat)
    m.add_node(1, 0.0, 0.0); m.add_node(2, L, 0.0)
    e_fiber = BeamColumn2D(1, (1, 2), mat, section=sec)
    m.add_element(e_fiber)
    # Equivalent elastic beam
    m2 = Model(ndm=2, ndf=3); m2.add_material(mat)
    m2.add_node(1, 0.0, 0.0); m2.add_node(2, L, 0.0)
    e_elast = BeamColumn2D(1, (1, 2), mat, b * h, b * h ** 3 / 12.0)
    m2.add_element(e_elast)
    K_fiber = e_fiber.K_tangent_global()
    K_elast = e_elast.K_global()
    np.testing.assert_allclose(K_fiber, K_elast, rtol=2e-3, atol=1e-3)


# =================================================== pushover ==

def _build_cantilever_with_fiber_section(*, b_post=0.05):
    """Cantilever column with a rectangular bilinear-fiber section."""
    E, sigma_y = 2.0e11, 400.0e6
    b, h, L = 0.1, 0.2, 3.0
    n_fibers = 20
    mat = ElasticIsotropic(1, E=E, nu=0.3)
    uniaxial = UniaxialBilinear(E=E, sigma_y=sigma_y, b=b_post)
    sec = FiberSection2D.rectangular(
        width=b, height=h, n_fibers=n_fibers, material=uniaxial
    )
    model = Model(ndm=2, ndf=3); model.add_material(mat)
    model.add_node(1, 0.0, 0.0); model.add_node(2, L, 0.0)
    elem = BeamColumn2D(1, (1, 2), mat, section=sec)
    model.add_element(elem)
    model.fix(1, [1, 1, 1])
    Mp_rect = sigma_y * b * h ** 2 / 4.0     # plastic moment
    My_rect = sigma_y * b * h ** 2 / 6.0     # first-yield moment
    P_yield = My_rect / L                    # tip force at first fiber yield
    return model, elem, P_yield, Mp_rect, My_rect, L


def test_fiber_pushover_stays_elastic_below_yield():
    """Load to 80% of first-yield force — no fiber should yield, the
    section state at every IP must be elastic."""
    model, elem, P_yield, _, _, _ = _build_cantilever_with_fiber_section()
    P_target = 0.8 * P_yield
    model.add_nodal_load(2, [0.0, -P_target, 0.0])
    NonlinearStaticAnalysis(
        model, num_steps=8, dlambda=1.0 / 8, tol=1e-6, max_iter=30,
    ).run()
    # No fiber should have plasticised
    for sec in elem.sections:
        for f in sec.fibers:
            assert f.material.eps_p_committed == 0.0


def test_fiber_pushover_yields_first_at_extreme_fibers():
    """Past first-yield, only the outermost fibers should have plastic
    strain — the middle of the section stays elastic. This is the
    spread-of-plasticity story that distinguishes fiber sections from
    lumped hinges."""
    model, elem, P_yield, Mp, _, _ = _build_cantilever_with_fiber_section(
        b_post=0.05
    )
    P_target = 1.2 * P_yield   # above first yield, well below Mp/L
    model.add_nodal_load(2, [0.0, -P_target, 0.0])
    NonlinearStaticAnalysis(
        model, num_steps=15, dlambda=1.0 / 15, tol=1e-6, max_iter=30,
    ).run()
    # IP 0 (the fixed end, where moment is largest) should show plastic
    # flow on the extreme fibers but not on the centroidal fibers.
    fixed_end_sec = elem.sections[0]
    fibers_sorted = sorted(fixed_end_sec.fibers, key=lambda f: abs(f.y))
    inner_fiber = fibers_sorted[0]
    outer_fiber = fibers_sorted[-1]
    assert outer_fiber.material.eps_p_committed != 0.0
    assert inner_fiber.material.eps_p_committed == 0.0


def test_fiber_pushover_force_displacement_is_bilinear():
    """The structural force-deflection curve must show a clear knee
    between deep-elastic and deep-post-plastic regimes.

    Unlike a lumped-plastic hinge (which has an abrupt yield knee), a
    fiber section yields *gradually* — first the extreme fibers, then
    progressively inwards. The structural stiffness therefore softens
    smoothly between ``My / L`` (first fiber yield) and ``Mp / L`` (full
    section plastification). To see a clear bilinear character we
    contrast a deep-elastic regime (forces < 0.5 P_yield) against a
    deep-post-plastic regime (forces > 1.1 Mp / L), where the response
    is dominated by the hardening modulus.
    """
    model, elem, P_yield, Mp, _, L = _build_cantilever_with_fiber_section(
        b_post=0.05
    )
    P_target = 1.4 * Mp / L
    n_steps = 70
    model.add_nodal_load(2, [0.0, -P_target, 0.0])
    res = NonlinearStaticAnalysis(
        model, num_steps=n_steps, dlambda=1.0 / n_steps,
        track=(2, 1), tol=1e-6, max_iter=40,
    ).run()
    lambdas = np.array(res["lambdas"])
    disps = np.array(res["tracked"])
    forces = lambdas * P_target
    below = forces < 0.5 * P_yield        # deep elastic
    above = forces > 1.1 * Mp / L          # deep post-plastic plateau
    assert below.any() and above.any()
    k_below = (forces[below][-1] - forces[below][0]) / abs(
        disps[below][-1] - disps[below][0]
    )
    k_above = (forces[above][-1] - forces[above][0]) / abs(
        disps[above][-1] - disps[above][0]
    )
    # Post-plateau stiffness must be much softer than the elastic one
    # (driven by the hardening modulus b * E, not the elastic E).
    assert k_above < 0.2 * k_below
    assert k_above > 0.0    # but non-zero — hardening keeps us stable


# =================================================== revert ==

def test_revert_undoes_uncommitted_fiber_plastic_flow():
    """Across all per-IP sections and all fibers within, ``revert_state``
    on the element rolls trial back to committed."""
    model, elem, P_yield, *_ = _build_cantilever_with_fiber_section()
    model.add_nodal_load(2, [0.0, -1.5 * P_yield, 0.0])
    NonlinearStaticAnalysis(
        model, num_steps=15, dlambda=1.0 / 15, tol=1e-6, max_iter=30,
    ).run()
    # Capture committed state, then mutate trial across all fibers/IPs
    committed = []
    for sec in elem.sections:
        for f in sec.fibers:
            committed.append(f.material.eps_p_committed)
            f.material.eps_p_trial = f.material.eps_p_committed + 1.0
    elem.revert_state()
    k = 0
    for sec in elem.sections:
        for f in sec.fibers:
            assert f.material.eps_p_trial == committed[k]
            k += 1
