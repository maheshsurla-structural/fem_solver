"""Tests for concrete uniaxial materials (Phase 16.0).

The Kent-Park-Scott and Mander concrete models are pinned down by
seven properties:

1. **Construction guard rails** — invalid fpc, eps_c0, fpcu, eps_cu.
2. **Tension is zero** for any positive strain (cracked concrete).
3. **Parabolic ascent** matches the analytical formula on the ascending
   branch (initial modulus = 2 fpc / eps_c0, zero tangent at peak).
4. **Linear descent** correctly interpolates from peak to crushing.
5. **Karsan-Jirsa unloading** lands at zero stress at the predicted
   plastic-strain offset.
6. **Commit / revert lifecycle** preserves and rolls back the most-
   compressive-strain history.
7. **Mander Popovics envelope** smoothly ascends to fpc at eps_c0 and
   descends past it; initial tangent equals the user-supplied Ec.
"""
import math

import numpy as np
import pytest

from femsolver import ConcreteKentPark, ConcreteMander


# ====================================================== construction

def test_concrete_kent_park_rejects_invalid_fpc():
    with pytest.raises(ValueError, match="fpc"):
        ConcreteKentPark(fpc=0.0, eps_c0=0.002, fpcu=0.0, eps_cu=0.003)
    with pytest.raises(ValueError, match="fpc"):
        ConcreteKentPark(fpc=-30e6, eps_c0=0.002, fpcu=0.0, eps_cu=0.003)


def test_concrete_kent_park_rejects_invalid_eps_c0():
    with pytest.raises(ValueError, match="eps_c0"):
        ConcreteKentPark(fpc=30e6, eps_c0=0.0, fpcu=0.0, eps_cu=0.003)


def test_concrete_kent_park_rejects_fpcu_above_fpc():
    with pytest.raises(ValueError, match="fpcu"):
        ConcreteKentPark(fpc=30e6, eps_c0=0.002, fpcu=40e6, eps_cu=0.003)


def test_concrete_kent_park_rejects_eps_cu_below_eps_c0():
    with pytest.raises(ValueError, match="eps_cu"):
        ConcreteKentPark(fpc=30e6, eps_c0=0.002, fpcu=0.0, eps_cu=0.001)


def test_concrete_mander_rejects_invalid_Ec():
    """Ec must exceed the secant fpc / eps_c0 for the Popovics shape
    parameter r to be finite and positive."""
    with pytest.raises(ValueError, match="Ec"):
        # Ec = 1e9 < secant = fpc/eps_c0 = 30e6/0.002 = 15e9
        ConcreteMander(fpc=30e6, eps_c0=0.002, Ec=1.0e9)


# ====================================================== envelope shape

def test_kent_park_tension_returns_zero():
    mat = ConcreteKentPark(fpc=30e6, eps_c0=0.002, fpcu=6e6, eps_cu=0.0035)
    for eps in (1e-5, 1e-4, 1e-3, 1e-2):
        sigma, Et = mat.get_response(eps)
        assert sigma == 0.0
        assert Et == 0.0


def test_kent_park_parabolic_ascent_matches_formula():
    """sigma = fpc * (2 r - r^2), r = eps/eps_c0."""
    fpc = 30e6; eps_c0 = 0.002
    mat = ConcreteKentPark(fpc=fpc, eps_c0=eps_c0, fpcu=6e6, eps_cu=0.0035)
    for ratio in (0.25, 0.5, 0.75, 0.9, 1.0):
        eps = -ratio * eps_c0
        sigma, Et = mat.get_response(eps)
        sigma_expected = -fpc * (2.0 * ratio - ratio * ratio)
        assert sigma == pytest.approx(sigma_expected, rel=1e-12)


def test_kent_park_initial_modulus_is_2_fpc_over_eps_c0():
    fpc = 30e6; eps_c0 = 0.002
    mat = ConcreteKentPark(fpc=fpc, eps_c0=eps_c0, fpcu=6e6, eps_cu=0.0035)
    # Tangent at eps -> 0^- (infinitesimal compressive strain)
    sigma, Et = mat.get_response(-1.0e-8)
    assert Et == pytest.approx(2.0 * fpc / eps_c0, rel=1e-3)


def test_kent_park_tangent_is_zero_at_peak():
    mat = ConcreteKentPark(fpc=30e6, eps_c0=0.002, fpcu=6e6, eps_cu=0.0035)
    _, Et = mat.get_response(-0.002)
    assert abs(Et) < 1.0


def test_kent_park_peak_stress_equals_fpc_at_eps_c0():
    fpc = 30e6
    mat = ConcreteKentPark(fpc=fpc, eps_c0=0.002, fpcu=6e6, eps_cu=0.0035)
    sigma, _ = mat.get_response(-0.002)
    assert sigma == pytest.approx(-fpc, rel=1e-12)


def test_kent_park_linear_descent_interpolates_correctly():
    fpc = 30e6; eps_c0 = 0.002; fpcu = 6e6; eps_cu = 0.0035
    mat = ConcreteKentPark(fpc=fpc, eps_c0=eps_c0, fpcu=fpcu, eps_cu=eps_cu)
    # Midpoint of the descending branch
    eps_mid = -0.5 * (eps_c0 + eps_cu)
    sigma, Et = mat.get_response(eps_mid)
    sigma_expected = -0.5 * (fpc + fpcu)
    assert sigma == pytest.approx(sigma_expected, rel=1e-12)
    # In signed coords the slope (Et = d sigma / d eps) is negative:
    # d(sigma)/d(eps) = (sigma_cu - sigma_peak) / (eps_cu - eps_peak)
    #                 = (-fpcu - (-fpc)) / (-eps_cu - (-eps_c0))
    #                 = (fpc - fpcu) / (eps_c0 - eps_cu)
    # Both numerator and denominator are positive over negative -> negative
    slope_expected = (fpc - fpcu) / (eps_c0 - eps_cu)    # negative number
    assert Et == pytest.approx(slope_expected, rel=1e-12)


def test_kent_park_residual_plateau_past_eps_cu():
    fpc = 30e6; fpcu = 6e6
    mat = ConcreteKentPark(fpc=fpc, eps_c0=0.002, fpcu=fpcu, eps_cu=0.0035)
    for eps in (-0.004, -0.005, -0.01):
        sigma, Et = mat.get_response(eps)
        assert sigma == pytest.approx(-fpcu, rel=1e-12)
        assert Et == pytest.approx(0.0, abs=1e-3)


# ====================================================== cyclic

def test_kent_park_unloading_to_eps_p_gives_zero_stress():
    """Karsan-Jirsa: on unloading from eps_min, stress hits zero at
    eps_p = eps_c0 * (0.145 r^2 + 0.13 r),  r = eps_min/eps_c0."""
    fpc = 30e6; eps_c0 = 0.002
    mat = ConcreteKentPark(fpc=fpc, eps_c0=eps_c0, fpcu=6e6, eps_cu=0.0035)
    # Load to eps = -0.003 (deep in descending branch) and commit
    mat.get_response(-0.003)
    mat.commit_state()
    # Karsan-Jirsa formula
    r = 1.5
    eps_p = -eps_c0 * (0.145 * r * r + 0.13 * r)
    sigma, _ = mat.get_response(eps_p)
    assert abs(sigma) < 1.0


def test_kent_park_reload_rejoins_envelope():
    """After unloading and reloading deeper, the stress should land
    back on the monotonic envelope."""
    mat = ConcreteKentPark(fpc=30e6, eps_c0=0.002, fpcu=6e6, eps_cu=0.0035)
    # Load to -0.0025 (just past peak), commit
    mat.get_response(-0.0025)
    mat.commit_state()
    # Unload to tension, commit
    mat.get_response(0.0005)
    mat.commit_state()
    # Reload to -0.003 (further than committed eps_min) -> should be
    # on the envelope at -0.003
    sigma, _ = mat.get_response(-0.003)
    # Envelope value at -0.003 (linear descent from -30 MPa at -0.002
    # to -6 MPa at -0.0035): slope = (-6 - -30)/(0.0035-0.002)*1e6 =
    # +16e9 in compression-magnitude form; at -0.003, sigma = -30e6
    # + (-6e6 - -30e6) * (0.003-0.002)/(0.0035-0.002) = -30e6 + 16e6 = -14e6
    assert sigma == pytest.approx(-14.0e6, rel=1e-6)


def test_kent_park_commit_revert_lifecycle():
    """Reverting must roll back eps_min and sigma to the last commit."""
    mat = ConcreteKentPark(fpc=30e6, eps_c0=0.002, fpcu=6e6, eps_cu=0.0035)
    # First load and commit
    mat.get_response(-0.001)
    mat.commit_state()
    eps_min_after_commit = mat.eps_min_committed
    # Trial deeper without committing
    mat.get_response(-0.003)
    # Revert
    mat.revert_state()
    assert mat.eps_min_trial == eps_min_after_commit


# ====================================================== Mander

def test_mander_peak_is_exactly_fpc_at_eps_c0():
    fpc = 45e6
    mat = ConcreteMander(fpc=fpc, eps_c0=0.005)
    sigma, _ = mat.get_response(-0.005)
    assert sigma == pytest.approx(-fpc, rel=1e-12)


def test_mander_zero_stress_at_origin():
    mat = ConcreteMander(fpc=45e6, eps_c0=0.005)
    sigma, Et = mat.get_response(0.0)
    assert sigma == 0.0
    # Tangent at origin should be the user-supplied (or computed) Ec
    sigma2, Et2 = mat.get_response(-1.0e-8)
    assert Et2 == pytest.approx(mat.E0, rel=1.0e-3)


def test_mander_tension_returns_zero():
    mat = ConcreteMander(fpc=45e6, eps_c0=0.005)
    for eps in (1e-5, 1e-4, 1e-3):
        sigma, Et = mat.get_response(eps)
        assert sigma == 0.0
        assert Et == 0.0


def test_mander_envelope_is_smooth():
    """Verify the Popovics envelope is C^1 by comparing finite-difference
    tangents to the analytical tangent at several interior points."""
    fpc = 45e6; eps_c0 = 0.005
    mat = ConcreteMander(fpc=fpc, eps_c0=eps_c0)
    h = 1.0e-7
    # Skip the peak (Et = 0 exactly there, so a relative tolerance is
    # meaningless). Test points on the ascending and descending branches.
    for eps_mag in (0.001, 0.003, 0.008, 0.012):
        eps = -eps_mag
        sigma_minus, _ = mat.get_response(eps - h)
        sigma_plus, _ = mat.get_response(eps + h)
        slope_fd = (sigma_plus - sigma_minus) / (2.0 * h)
        mat2 = ConcreteMander(fpc=fpc, eps_c0=eps_c0)
        _, Et = mat2.get_response(eps)
        assert slope_fd == pytest.approx(Et, rel=2.0e-2), \
            f"at eps={eps}: FD={slope_fd:.3e}, analytic={Et:.3e}"


def test_mander_descent_past_peak_below_fpc():
    """Past eps_cc the stress should fall below fpc, monotonically."""
    fpc = 45e6
    mat = ConcreteMander(fpc=fpc, eps_c0=0.005)
    s_peak, _ = mat.get_response(-0.005)
    s_past, _ = mat.get_response(-0.010)
    assert abs(s_past) < abs(s_peak)


def test_mander_initial_modulus_matches_aci_default():
    """Default Ec uses 4700 sqrt(fpc[MPa]) MPa. For fpc=30 MPa this is
    25.7 GPa."""
    mat = ConcreteMander(fpc=30e6, eps_c0=0.002)
    aci = 4700.0 * math.sqrt(30.0) * 1.0e6
    assert mat.E0 == pytest.approx(aci, rel=1e-12)


def test_mander_commit_revert_lifecycle():
    mat = ConcreteMander(fpc=45e6, eps_c0=0.005)
    mat.get_response(-0.003)
    mat.commit_state()
    eps_min = mat.eps_min_committed
    mat.get_response(-0.008)
    mat.revert_state()
    assert mat.eps_min_trial == eps_min


# ====================================================== integration

def test_mander_damage_factor_zero_unchanged():
    """damage_factor = 0 reproduces the original undegraded Mander."""
    mat_no_dmg = ConcreteMander(fpc=45e6, eps_c0=0.005, damage_factor=0.0)
    mat_default = ConcreteMander(fpc=45e6, eps_c0=0.005)
    for eps_mag in (0.001, 0.005, 0.01, 0.02):
        s1, _ = mat_no_dmg.get_response(-eps_mag); mat_no_dmg.commit_state()
        s2, _ = mat_default.get_response(-eps_mag); mat_default.commit_state()
        assert s1 == pytest.approx(s2, rel=1e-12)


def test_mander_damage_reduces_envelope_peak_strength():
    """Pushing past the peak repeatedly should reduce sigma at any
    fixed strain past eps_c0."""
    fpc = 45e6; eps_c0 = 0.005
    mat = ConcreteMander(fpc=fpc, eps_c0=eps_c0, damage_factor=2.0)
    # Push to eps = -3*eps_c0 with several intermediate reversals to
    # accumulate damage.
    for eps in np.linspace(0.0, -2.0 * eps_c0, 30):
        mat.get_response(eps); mat.commit_state()
    for eps in np.linspace(-2.0 * eps_c0, 0.0, 30):
        mat.get_response(eps); mat.commit_state()
    for eps in np.linspace(0.0, -3.0 * eps_c0, 50):
        sigma_damaged, _ = mat.get_response(eps); mat.commit_state()
    # Compare to undamaged at the same final strain
    mat_undamaged = ConcreteMander(fpc=fpc, eps_c0=eps_c0, damage_factor=0.0)
    for eps in np.linspace(0.0, -3.0 * eps_c0, 50):
        sigma_undamaged, _ = mat_undamaged.get_response(eps)
        mat_undamaged.commit_state()
    assert abs(sigma_damaged) < abs(sigma_undamaged)


def test_mander_damage_alpha_accumulates():
    """alpha increases monotonically as we push past eps_c0."""
    mat = ConcreteMander(fpc=45e6, eps_c0=0.005, damage_factor=0.5)
    alphas = []
    for eps in np.linspace(0.0, -0.02, 40):
        mat.get_response(eps); mat.commit_state()
        alphas.append(mat.alpha_committed)
    # alpha grows once we go past eps_c0
    assert all(alphas[i + 1] >= alphas[i] for i in range(len(alphas) - 1))
    assert mat.alpha_committed > 0.0


def test_mander_damage_rejects_negative_damage_factor():
    with pytest.raises(ValueError, match="damage_factor"):
        ConcreteMander(fpc=45e6, eps_c0=0.005, damage_factor=-0.1)


def test_mander_damage_clamped_at_min_strength():
    mat = ConcreteMander(fpc=45e6, eps_c0=0.005, damage_factor=10.0,
                          min_strength_ratio=0.3)
    # Pile up large excursion
    for eps in np.linspace(0.0, -0.05, 100):
        mat.get_response(eps); mat.commit_state()
    ratio = mat._strength_ratio(mat.alpha_committed)
    assert ratio >= 0.3 - 1e-12
    assert ratio <= 1.0


def test_concrete_works_in_fiber_section():
    """Smoke test: ConcreteKentPark can be plugged into a FiberSection2D
    and used by a BeamColumn2D in a static analysis without crashing."""
    from femsolver import (
        BeamColumn2D,
        ElasticIsotropic,
        FiberSection2D,
        Fiber,
        Model,
    )
    from femsolver.analysis.linear_static import LinearStaticAnalysis

    mat_conc = ConcreteKentPark(fpc=30e6, eps_c0=0.002,
                                 fpcu=6e6, eps_cu=0.0035)
    mat_iso = ElasticIsotropic(1, E=30e9, nu=0.2)
    # 6 fibers across a 200x200 mm section
    fibers = []
    h = 0.20
    n = 6
    for i in range(n):
        y = -h / 2 + (i + 0.5) * h / n
        fibers.append(Fiber(y=y, z=0.0, area=0.20 * h / n, material=mat_conc))
    sec = FiberSection2D(fibers)
    m = Model(ndm=2, ndf=3); m.add_material(mat_iso)
    m.add_node(1, 0.0, 0.0)
    m.add_node(2, 1.0, 0.0)
    m.add_element(BeamColumn2D(1, (1, 2), mat_iso, section=sec))
    m.fix(1, [1, 1, 1])
    # Apply a small axial load (compression); the concrete fibers'
    # tangents at very-small strain are E0 = 2 fpc / eps_c0 = 3e10.
    m.add_nodal_load(2, [-1.0e3, 0.0, 0.0])
    LinearStaticAnalysis(m).run()
    # End-node x displacement (compression -> negative)
    assert m.node(2).disp[0] < 0.0
