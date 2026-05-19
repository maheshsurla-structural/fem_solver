"""Tests for FiberSection2D — distributed-plasticity cross-section.

The fiber section is validated in three regimes:

1. **Gross-section equivalence** — with all fibers carrying the same
   elastic material, the section's ``EA`` and ``EIz`` (read off the
   tangent at zero strain) must equal the analytical gross-section
   values for a rectangle: ``E A`` and ``E b h^3 / 12``.

2. **Section response matches an :class:`ElasticSection2D`** when the
   constituent material is :class:`UniaxialElastic`. This is the
   strongest possible regression check on the kinematic assumption
   ``eps_f = eps_axial - y * kappa_z`` and the integration formulae
   ``N = sum sigma A``, ``Mz = -sum y sigma A``.

3. **Plastic-moment under pure bending** — for a rectangle with EPP
   fibers, ``Mp = sigma_y * b * h^2 / 4``. The fiber section, in the
   limit of large curvature with finely-discretised fibers, must
   approach this analytical value.

4. **P-M interaction / coupling** — once one side yields and the other
   does not, the section tangent stiffness picks up a non-zero
   off-diagonal coupling ``-ES``. We check that this coupling appears
   *only* after asymmetric yielding, not before.
"""
import numpy as np
import pytest

from femsolver import (
    ElasticSection2D,
    Fiber,
    FiberSection2D,
    UniaxialBilinear,
    UniaxialElastic,
)


# ====================================================== construction ==

def test_fiber_rejects_empty_list():
    with pytest.raises(ValueError):
        FiberSection2D([])


def test_fiber_rejects_nonpositive_area():
    mat = UniaxialElastic(E=2.0e11)
    with pytest.raises(ValueError):
        FiberSection2D([Fiber(y=0.0, z=0.0, area=0.0, material=mat)])


def test_rectangular_factory_rejects_too_few_fibers():
    with pytest.raises(ValueError):
        FiberSection2D.rectangular(
            width=0.1, height=0.2, n_fibers=1, material=UniaxialElastic(E=1e11)
        )


def test_rectangular_factory_rejects_bad_dimensions():
    mat = UniaxialElastic(E=1.0e11)
    with pytest.raises(ValueError):
        FiberSection2D.rectangular(width=0.0, height=0.2, n_fibers=5, material=mat)
    with pytest.raises(ValueError):
        FiberSection2D.rectangular(width=0.1, height=-0.2, n_fibers=5, material=mat)


# ============================================== gross-section sanity ==

def test_rectangular_gross_area_matches_b_times_h():
    b, h, n = 0.3, 0.5, 40
    sec = FiberSection2D.rectangular(
        width=b, height=h, n_fibers=n, material=UniaxialElastic(E=2.0e11)
    )
    assert sec.gross_area == pytest.approx(b * h, rel=1e-14)


def test_rectangular_gross_Iz_matches_bh3_over_12():
    b, h, n = 0.3, 0.5, 200
    sec = FiberSection2D.rectangular(
        width=b, height=h, n_fibers=n, material=UniaxialElastic(E=2.0e11)
    )
    # The strip-based discretisation under-estimates Iz by a few percent
    # at low n (sum of (h_strip/n)^2 area terms is a midpoint sum). With
    # 200 strips the error is well under 0.1 %.
    Iz_exact = b * h ** 3 / 12.0
    assert sec.gross_Iz == pytest.approx(Iz_exact, rel=2e-4)


def test_rectangular_centroid_at_origin():
    sec = FiberSection2D.rectangular(
        width=0.3, height=0.5, n_fibers=20, material=UniaxialElastic(E=1e11)
    )
    assert sec.centroid_y == pytest.approx(0.0, abs=1e-14)


def test_rectangular_centroid_offset():
    """Passing ``centroid_y`` should shift the fiber centroid by that amount."""
    sec = FiberSection2D.rectangular(
        width=0.3, height=0.5, n_fibers=20,
        material=UniaxialElastic(E=1e11), centroid_y=0.25,
    )
    assert sec.centroid_y == pytest.approx(0.25, abs=1e-14)


# ===================================== matches ElasticSection2D ===========

def test_elastic_fiber_section_matches_ElasticSection2D_at_zero_strain():
    """At zero strain the tangent of an all-elastic fiber section is the
    gross-section diag(EA, EI). With many fibers this must converge to
    :class:`ElasticSection2D`'s constant tangent.
    """
    E, b, h, n = 2.0e11, 0.3, 0.5, 200
    fiber_sec = FiberSection2D.rectangular(
        width=b, height=h, n_fibers=n, material=UniaxialElastic(E=E)
    )
    elastic_sec = ElasticSection2D(E=E, A=b * h, Iz=b * h ** 3 / 12.0)
    s_f, ks_f = fiber_sec.get_response(np.zeros(2))
    s_e, ks_e = elastic_sec.get_response(np.zeros(2))
    np.testing.assert_allclose(s_f, s_e, atol=1e-9)
    # Iz from discretisation differs by ~1e-4 from continuous; ks
    # diagonal terms agree to that tolerance.
    np.testing.assert_allclose(ks_f, ks_e, rtol=2e-4)


@pytest.mark.parametrize("eps_axial,kappa", [
    (1.0e-4, 0.0),    # pure axial
    (0.0, 2.0e-3),    # pure bending
    (1.0e-4, 2.0e-3), # combined
    (-5.0e-5, -1.0e-3),
])
def test_elastic_fiber_section_response_matches_analytical(eps_axial, kappa):
    """For an all-elastic fiber section: N = EA eps, Mz = EI kappa,
    no axial-bending coupling because the centroid is at y = 0."""
    E, b, h, n = 2.0e11, 0.3, 0.5, 200
    EA_exact = E * b * h
    EI_exact = E * b * h ** 3 / 12.0
    fiber_sec = FiberSection2D.rectangular(
        width=b, height=h, n_fibers=n, material=UniaxialElastic(E=E)
    )
    s, ks = fiber_sec.get_response(np.array([eps_axial, kappa]))
    # forces
    assert s[0] == pytest.approx(EA_exact * eps_axial, rel=1e-10, abs=1e-6)
    assert s[1] == pytest.approx(EI_exact * kappa, rel=1e-3, abs=1e-6)
    # tangent: diagonal, no coupling
    assert ks[0, 1] == pytest.approx(0.0, abs=1e-3)
    assert ks[1, 0] == pytest.approx(0.0, abs=1e-3)


# ===================================== plastic-moment limit ================

def test_plastic_moment_rectangle_under_pure_bending():
    """For a rectangle with EPP fibers, ``Mp = sigma_y * b * h^2 / 4``.

    We drive the curvature large (10x yield curvature) so essentially
    all fibers have yielded; the fiber section's recovered moment must
    approach the analytical Mp.
    """
    E = 2.0e11
    sigma_y = 400.0e6
    b, h = 0.3, 0.5
    n = 200             # many fibers — discretisation error well under 1 %
    mat = UniaxialBilinear(E=E, sigma_y=sigma_y, b=0.0)
    sec = FiberSection2D.rectangular(
        width=b, height=h, n_fibers=n, material=mat
    )
    # Yield curvature ≈ 2 sigma_y / (E h). Drive to ~10x.
    kappa_yield = 2.0 * sigma_y / (E * h)
    kappa = 10.0 * kappa_yield
    s, _ = sec.get_response(np.array([0.0, kappa]))
    Mp_exact = sigma_y * b * h ** 2 / 4.0
    # 1% tolerance because the strip discretisation under-counts the
    # outer fibers slightly.
    assert s[1] == pytest.approx(Mp_exact, rel=1e-2)


def test_plastic_moment_is_1p5_times_yield_moment_for_rectangle():
    """Shape factor of a rectangle is exactly 1.5: ``Mp / My = 1.5``.

    This is the cleanest single-number identity that distinguishes a
    fiber section (which knows section shape) from a lumped-plastic
    hinge (which has a user-specified M_y but no shape factor).
    """
    E, sigma_y, b, h = 2.0e11, 400.0e6, 0.3, 0.5
    n = 400
    mat_yield = UniaxialBilinear(E=E, sigma_y=sigma_y, b=0.0)
    sec_yield = FiberSection2D.rectangular(
        width=b, height=h, n_fibers=n, material=mat_yield
    )
    # Curvature at *first* yield: outermost fiber strain = sigma_y/E.
    kappa_first_yield = (sigma_y / E) / (h / 2.0)
    s_y, _ = sec_yield.get_response(
        np.array([0.0, kappa_first_yield * 0.999])  # just below yield
    )
    My = s_y[1]
    # Reset for the plastic-moment leg
    mat_plastic = UniaxialBilinear(E=E, sigma_y=sigma_y, b=0.0)
    sec_plastic = FiberSection2D.rectangular(
        width=b, height=h, n_fibers=n, material=mat_plastic
    )
    s_p, _ = sec_plastic.get_response(np.array([0.0, 20.0 * kappa_first_yield]))
    Mp = s_p[1]
    assert Mp / My == pytest.approx(1.5, rel=2e-2)


# ===================================== off-diagonal coupling ===============

def test_no_coupling_when_section_is_symmetric_and_elastic():
    """Symmetric elastic section: ks is diagonal (no axial-bending coupling)."""
    sec = FiberSection2D.rectangular(
        width=0.3, height=0.5, n_fibers=40, material=UniaxialElastic(E=2.0e11)
    )
    _, ks = sec.get_response(np.array([1.0e-4, 1.0e-3]))
    assert ks[0, 1] == pytest.approx(0.0, abs=1e-3)


def test_coupling_emerges_after_asymmetric_yielding():
    """Apply combined axial + bending so the +y side yields before the
    -y side. The section tangent should pick up a non-zero off-diagonal
    ``-ES``, which is what causes P-M interaction post-yield.
    """
    E, sigma_y, b, h = 2.0e11, 400.0e6, 0.3, 0.5
    mat = UniaxialBilinear(E=E, sigma_y=sigma_y, b=0.0)
    sec = FiberSection2D.rectangular(
        width=b, height=h, n_fibers=80, material=mat
    )
    # Strain state: +ve axial + +ve curvature → +y fibers yield first
    eps_y = sigma_y / E
    e = np.array([0.6 * eps_y, 4.0 * eps_y / (h / 2.0)])
    _, ks = sec.get_response(e)
    # Off-diagonal must be non-zero. Compare magnitudes to spot bias.
    assert abs(ks[0, 1]) > 0.0
    assert ks[0, 1] == pytest.approx(ks[1, 0], rel=1e-12)  # symmetric


# ===================================== lifecycle ===========================

def test_commit_revert_forwards_to_every_fiber_material():
    """Section ``commit_state`` must reach every fiber's material."""
    E, sigma_y = 2.0e11, 400.0e6
    mat = UniaxialBilinear(E=E, sigma_y=sigma_y, b=0.0)
    sec = FiberSection2D.rectangular(
        width=0.3, height=0.5, n_fibers=10, material=mat
    )
    # Drive past yield, then commit and revert
    sec.get_response(np.array([0.0, 5.0 * sigma_y / E / 0.25]))
    pre_commit_trial = [f.material.eps_p_trial for f in sec.fibers]
    sec.commit_state()
    # After commit, every fiber's committed state should equal what was trial
    for f, tp in zip(sec.fibers, pre_commit_trial):
        assert f.material.eps_p_committed == tp


def test_revert_resets_every_fiber_trial_state():
    E, sigma_y = 2.0e11, 400.0e6
    sec = FiberSection2D.rectangular(
        width=0.3, height=0.5, n_fibers=10,
        material=UniaxialBilinear(E=E, sigma_y=sigma_y, b=0.0),
    )
    sec.get_response(np.array([0.0, 5.0 * sigma_y / E / 0.25]))
    sec.revert_state()
    for f in sec.fibers:
        assert f.material.eps_p_trial == 0.0


def test_clone_produces_independent_state():
    """Each clone owns independent fiber states — required for per-IP usage."""
    E, sigma_y = 2.0e11, 400.0e6
    sec = FiberSection2D.rectangular(
        width=0.3, height=0.5, n_fibers=10,
        material=UniaxialBilinear(E=E, sigma_y=sigma_y, b=0.0),
    )
    sec2 = sec.clone()
    assert sec2 is not sec
    assert sec2.fibers[0].material is not sec.fibers[0].material
    # mutate sec
    sec.get_response(np.array([0.0, 5.0 * sigma_y / E / 0.25])); sec.commit_state()
    # sec2 must be untouched
    for f in sec2.fibers:
        assert f.material.eps_p_committed == 0.0
