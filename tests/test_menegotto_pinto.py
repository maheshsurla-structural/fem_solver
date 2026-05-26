"""Tests for UniaxialMenegottoPinto cyclic-steel material (Phase 16.3).

The Menegotto-Pinto model is pinned down by seven properties:

1. **Construction guard rails** — invalid E, sigma_y, b, R.
2. **Elastic-range linearity** — for ``|eps| << eps_y``, sigma = E*eps.
3. **Asymptotic yield approach** — for ``|eps| >> eps_y``, sigma
   approaches the hardening asymptote ``b * E * eps + sign * (1-b) * sigma_y``.
4. **Initial tangent equals E** — at eps = 0 the tangent equals the
   elastic modulus.
5. **Cyclic symmetry** — loading symmetrically in tension and
   compression gives sigma values of equal magnitude.
6. **Bauschinger curvature** — on reload after a full reversal, the
   stress reaches a value *between* zero and the previous-peak
   stress at zero strain (the textbook curved reload signature).
7. **Large R approaches bilinear** — with R = 50, the MP curve sits
   within a few percent of the UniaxialBilinear envelope.
"""
import numpy as np
import pytest

from femsolver import UniaxialBilinear, UniaxialMenegottoPinto


# ====================================================== construction

def test_mp_rejects_invalid_E():
    with pytest.raises(ValueError, match="E"):
        UniaxialMenegottoPinto(E=0.0, sigma_y=400e6)


def test_mp_rejects_invalid_sigma_y():
    with pytest.raises(ValueError, match="sigma_y"):
        UniaxialMenegottoPinto(E=2e11, sigma_y=0.0)
    with pytest.raises(ValueError, match="sigma_y"):
        UniaxialMenegottoPinto(E=2e11, sigma_y=-1.0)


def test_mp_rejects_invalid_b():
    with pytest.raises(ValueError, match="b"):
        UniaxialMenegottoPinto(E=2e11, sigma_y=400e6, b=1.0)
    with pytest.raises(ValueError, match="b"):
        UniaxialMenegottoPinto(E=2e11, sigma_y=400e6, b=-0.01)


def test_mp_rejects_invalid_R():
    with pytest.raises(ValueError, match="R"):
        UniaxialMenegottoPinto(E=2e11, sigma_y=400e6, R=0.5)


# ====================================================== monotonic

def test_mp_initial_tangent_equals_E():
    E = 2.0e11
    mat = UniaxialMenegottoPinto(E=E, sigma_y=400e6, b=0.01, R=15.0)
    sigma, Et = mat.get_response(0.0)
    assert sigma == pytest.approx(0.0, abs=1e-6)
    assert Et == pytest.approx(E, rel=1e-12)


def test_mp_elastic_range_linear():
    """For strains well below eps_y, sigma = E * eps."""
    E = 2.0e11
    sigma_y = 400e6
    mat = UniaxialMenegottoPinto(E=E, sigma_y=sigma_y, b=0.01, R=15.0)
    eps_y = sigma_y / E
    # eps = 1/10 of yield: well into the elastic range
    eps_small = 0.1 * eps_y
    sigma, _ = mat.get_response(eps_small)
    # MP gives slightly below E*eps because of the Bauschinger function
    # influence, but the deviation should be small (< 5%) in the
    # elastic range.
    sigma_elastic = E * eps_small
    assert sigma == pytest.approx(sigma_elastic, rel=5e-2)


def test_mp_large_strain_approaches_hardening_asymptote():
    """For eps >> eps_y the stress approaches b*E*eps + (1-b)*sigma_y."""
    E = 2.0e11; sigma_y = 400e6; b = 0.01
    mat = UniaxialMenegottoPinto(E=E, sigma_y=sigma_y, b=b, R=15.0)
    # Sweep monotonically (committing at each step) to a large strain
    for eps in np.linspace(0.0, 0.05, 200):
        mat.get_response(eps); mat.commit_state()
    sigma_final = mat.sigma_committed
    sigma_asymp = b * E * 0.05 + (1.0 - b) * sigma_y
    assert sigma_final == pytest.approx(sigma_asymp, rel=1e-3)


def test_mp_with_large_R_matches_bilinear():
    """As R grows the MP curve approaches the bilinear (sharp-yield)
    envelope. At R = 50, residual error should be small."""
    E = 2.0e11; sigma_y = 400e6; b = 0.01
    mat_mp = UniaxialMenegottoPinto(E=E, sigma_y=sigma_y, b=b, R=50.0)
    mat_bl = UniaxialBilinear(E=E, sigma_y=sigma_y, b=b)
    eps_y = sigma_y / E
    # At eps = 3 * eps_y (well into yield), both should agree
    eps = 3.0 * eps_y
    s_mp, _ = mat_mp.get_response(eps)
    s_bl, _ = mat_bl.get_response(eps)
    assert s_mp == pytest.approx(s_bl, rel=2e-2)


# ====================================================== cyclic

def test_mp_symmetric_cycle_gives_symmetric_stress():
    """Loading +eps_max then -eps_max (committed monotonically) gives
    sigma of equal magnitude."""
    E = 2.0e11; sigma_y = 400e6
    mat = UniaxialMenegottoPinto(E=E, sigma_y=sigma_y, b=0.01, R=15.0)
    eps_max = 0.01
    for eps in np.linspace(0.0, eps_max, 50):
        mat.get_response(eps); mat.commit_state()
    sigma_plus = mat.sigma_committed
    for eps in np.linspace(eps_max, -eps_max, 100):
        mat.get_response(eps); mat.commit_state()
    sigma_minus = mat.sigma_committed
    assert sigma_minus == pytest.approx(-sigma_plus, rel=1.0e-3)


def test_mp_bauschinger_curvature_visible_on_reload():
    """After full reversal from +eps_max to -eps_max, the stress at
    eps = 0 during the reload should fall in (0, sigma_max), reflecting
    the Bauschinger-effect smooth approach to the new asymptote."""
    E = 2.0e11; sigma_y = 400e6; b = 0.01
    mat = UniaxialMenegottoPinto(E=E, sigma_y=sigma_y, b=b, R=15.0)
    eps_max = 0.01
    # Load to +eps_max, then -eps_max, then back to +eps_max
    for eps in np.linspace(0.0, eps_max, 50):
        mat.get_response(eps); mat.commit_state()
    sigma_peak_plus = mat.sigma_committed
    for eps in np.linspace(eps_max, -eps_max, 100):
        mat.get_response(eps); mat.commit_state()
    # Now reload: at eps = 0 expect stress between 0 and sigma_peak_plus
    for eps in np.linspace(-eps_max, 0.0, 100):
        mat.get_response(eps); mat.commit_state()
    sigma_at_zero = mat.sigma_committed
    # MP with R=15 should give a meaningful nonzero stress at zero
    # strain during reload (curved reload trajectory).
    assert sigma_at_zero > 0.0, \
        f"reload at eps=0 gave sigma={sigma_at_zero:.3e} (should be positive)"
    assert sigma_at_zero < sigma_peak_plus, \
        f"reload at eps=0 gave sigma={sigma_at_zero:.3e}, " \
        f"sigma_peak={sigma_peak_plus:.3e}"


def test_mp_commit_revert_lifecycle():
    """Reverting must roll back trial state to the last commit."""
    mat = UniaxialMenegottoPinto(E=2e11, sigma_y=400e6, b=0.01, R=15.0)
    # Load and commit
    mat.get_response(0.005); mat.commit_state()
    eps_committed = mat.eps_committed
    sigma_committed = mat.sigma_committed
    direction_committed = mat.direction_committed
    # Trial in opposite direction (would trigger a reversal)
    mat.get_response(-0.005)
    # Revert
    mat.revert_state()
    assert mat.eps_trial == eps_committed
    assert mat.sigma_trial == sigma_committed
    assert mat.direction_trial == direction_committed


# ====================================================== integration

def test_giuffre_mp_a1_zero_matches_constant_R_mp():
    """With a1 = 0 the Giuffre evolution is disabled; behavior matches
    the constant-R formulation."""
    E, sigma_y = 2e11, 400e6
    mat_const = UniaxialMenegottoPinto(E=E, sigma_y=sigma_y, b=0.01, R=15.0)
    mat_evolv = UniaxialMenegottoPinto(E=E, sigma_y=sigma_y, b=0.01,
                                          R0=15.0, a1=0.0)
    for eps in (0.001, 0.003, 0.01, -0.005, 0.005, -0.01):
        s1, _ = mat_const.get_response(eps); mat_const.commit_state()
        s2, _ = mat_evolv.get_response(eps); mat_evolv.commit_state()
        assert s1 == pytest.approx(s2, rel=1e-12)


def test_giuffre_mp_R_decreases_with_cyclic_excursion():
    """With a1 > 0, R should decrease after large strain swings."""
    mat = UniaxialMenegottoPinto(E=2e11, sigma_y=400e6, b=0.01,
                                    R0=20.0, a1=18.5, a2=0.15)
    eps_y = 400e6 / 2e11
    R_initial = mat.R
    # Cycle: load +5*eps_y, then -5*eps_y -> large reversal
    for eps in np.linspace(0.0, 5 * eps_y, 50):
        mat.get_response(eps); mat.commit_state()
    for eps in np.linspace(5 * eps_y, -5 * eps_y, 100):
        mat.get_response(eps); mat.commit_state()
    # After the big reversal the new branch's R should be smaller
    R_after = mat.R_committed
    assert R_after < R_initial, f"R: initial={R_initial}, after={R_after}"


def test_giuffre_mp_larger_swing_gives_smaller_R():
    """Bigger strain swing -> smaller R (more Bauschinger curvature)."""
    def R_after_swing(swing_factor):
        m = UniaxialMenegottoPinto(E=2e11, sigma_y=400e6, b=0.01,
                                      R0=20.0, a1=18.5, a2=0.15)
        eps_y = 400e6 / 2e11
        peak = swing_factor * eps_y
        for eps in np.linspace(0.0, peak, 50):
            m.get_response(eps); m.commit_state()
        for eps in np.linspace(peak, -peak, 50):
            m.get_response(eps); m.commit_state()
        return m.R_committed

    R_small = R_after_swing(2.0)
    R_big = R_after_swing(10.0)
    assert R_big < R_small, f"swing 2: R={R_small}, swing 10: R={R_big}"


def test_giuffre_mp_R_stays_above_minimum():
    """R should not drop below the safety clamp (1.5) under any swing."""
    mat = UniaxialMenegottoPinto(E=2e11, sigma_y=400e6, b=0.01,
                                    R0=20.0, a1=18.5, a2=0.15)
    eps_y = 400e6 / 2e11
    huge = 100 * eps_y
    for eps in np.linspace(0.0, huge, 100):
        mat.get_response(eps); mat.commit_state()
    for eps in np.linspace(huge, -huge, 100):
        mat.get_response(eps); mat.commit_state()
    assert mat.R_committed >= 1.5


def test_mp_works_in_fiber_section():
    """Smoke test: UniaxialMenegottoPinto can drive a fiber-section
    cantilever in a static analysis."""
    from femsolver import (
        BeamColumn2D,
        ElasticIsotropic,
        Fiber,
        FiberSection2D,
        Model,
    )
    from femsolver.analysis.linear_static import LinearStaticAnalysis

    mat_steel = UniaxialMenegottoPinto(E=2.0e11, sigma_y=400e6,
                                          b=0.01, R=15.0)
    mat_iso = ElasticIsotropic(1, E=2.0e11, nu=0.3)
    fibers = []
    h = 0.10
    n = 4
    for i in range(n):
        y = -h / 2 + (i + 0.5) * h / n
        fibers.append(Fiber(y=y, z=0.0, area=0.05 * h / n,
                             material=mat_steel.clone()))
    sec = FiberSection2D(fibers)
    m = Model(ndm=2, ndf=3); m.add_material(mat_iso)
    m.add_node(1, 0.0, 0.0); m.add_node(2, 1.0, 0.0)
    m.add_element(BeamColumn2D(1, (1, 2), mat_iso, section=sec))
    m.fix(1, [1, 1, 1])
    m.add_nodal_load(2, [-1.0e3, 0.0, 0.0])
    LinearStaticAnalysis(m).run()
    # End-node x displacement should be small and negative (compression)
    assert m.node(2).disp[0] < 0.0
    assert m.node(2).disp[0] > -1e-3
