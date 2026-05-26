"""Tests for UniaxialHysteretic (Phase 16.5).

The pinched-hysteretic model is pinned down by these properties:

1. **Construction guard rails** — invalid E, sigma_y, b, pinch_x, pinch_y.
2. **No-pinch case matches UniaxialBilinear** — with pinch_x = pinch_y = 1,
   the material reproduces the bilinear envelope and direct cyclic
   reload (no pinch point intervenes).
3. **Pure elastic at first loading** — for any pinch values, the
   first loading from origin gives sigma = E * eps for |eps| <= eps_y.
4. **Yield envelope** — pushing past yield in either direction lands
   on the bilinear envelope: sigma_y + b*E*(eps - eps_y).
5. **Pinch point is reached on reload trajectory** — after reversal
   from a yielded state, the trajectory passes through
   (pinch_x * eps_target, pinch_y * sigma_target).
6. **Symmetric cyclic response** — symmetric loading produces
   symmetric stresses.
7. **Commit/revert lifecycle** preserves state correctly.
"""
import numpy as np
import pytest

from femsolver import UniaxialBilinear, UniaxialHysteretic


# ====================================================== construction

def test_hysteretic_rejects_invalid_E():
    with pytest.raises(ValueError, match="E"):
        UniaxialHysteretic(E=0.0, sigma_y=400e6)


def test_hysteretic_rejects_invalid_sigma_y():
    with pytest.raises(ValueError, match="sigma_y"):
        UniaxialHysteretic(E=2e11, sigma_y=-1.0)


def test_hysteretic_rejects_invalid_b():
    with pytest.raises(ValueError, match="b"):
        UniaxialHysteretic(E=2e11, sigma_y=400e6, b=1.0)


def test_hysteretic_rejects_invalid_pinch():
    with pytest.raises(ValueError, match="pinch_x"):
        UniaxialHysteretic(E=2e11, sigma_y=400e6, pinch_x=0.0)
    with pytest.raises(ValueError, match="pinch_x"):
        UniaxialHysteretic(E=2e11, sigma_y=400e6, pinch_x=1.5)
    with pytest.raises(ValueError, match="pinch_y"):
        UniaxialHysteretic(E=2e11, sigma_y=400e6, pinch_y=1.5)


# ====================================================== no-pinch limit

def test_hysteretic_no_pinch_matches_bilinear_monotonic():
    """With pinch_x = pinch_y = 1, the monotonic envelope matches
    UniaxialBilinear exactly."""
    E, sigma_y, b = 2e11, 400e6, 0.01
    mat_h = UniaxialHysteretic(E=E, sigma_y=sigma_y, b=b,
                                 pinch_x=1.0, pinch_y=1.0)
    mat_b = UniaxialBilinear(E=E, sigma_y=sigma_y, b=b)
    eps_y = sigma_y / E
    for eps in (0.5 * eps_y, eps_y, 1.5 * eps_y, 3 * eps_y):
        s_h, _ = mat_h.get_response(eps); mat_h.commit_state()
        s_b, _ = mat_b.get_response(eps); mat_b.commit_state()
        assert s_h == pytest.approx(s_b, rel=1e-12)


# ====================================================== first-loading elastic

def test_hysteretic_first_loading_is_purely_elastic():
    """For any pinch values, the first loading from origin within the
    elastic range follows sigma = E * eps exactly."""
    E, sigma_y = 2e11, 400e6
    for pinch_x, pinch_y in ((0.5, 0.25), (0.3, 0.1), (0.9, 0.6)):
        mat = UniaxialHysteretic(E=E, sigma_y=sigma_y, b=0.01,
                                   pinch_x=pinch_x, pinch_y=pinch_y)
        eps_y = sigma_y / E
        for ratio in (0.25, 0.5, 0.75, 0.99):
            eps = ratio * eps_y
            mat = UniaxialHysteretic(E=E, sigma_y=sigma_y, b=0.01,
                                       pinch_x=pinch_x, pinch_y=pinch_y)
            sigma, _ = mat.get_response(eps)
            assert sigma == pytest.approx(E * eps, rel=1e-12), (
                f"pinch=({pinch_x},{pinch_y}), eps={eps}: got {sigma}, "
                f"expected {E*eps}"
            )


# ====================================================== envelope

def test_hysteretic_envelope_above_yield():
    """Pushing past yield lands on the bilinear envelope."""
    E, sigma_y, b = 2e11, 400e6, 0.01
    mat = UniaxialHysteretic(E=E, sigma_y=sigma_y, b=b,
                               pinch_x=0.5, pinch_y=0.25)
    eps_y = sigma_y / E
    for eps_mag in (1.5 * eps_y, 3 * eps_y, 5 * eps_y):
        mat = UniaxialHysteretic(E=E, sigma_y=sigma_y, b=b,
                                   pinch_x=0.5, pinch_y=0.25)
        sigma, _ = mat.get_response(eps_mag)
        expected = sigma_y + b * E * (eps_mag - eps_y)
        assert sigma == pytest.approx(expected, rel=1e-12)


# ====================================================== pinching

def test_hysteretic_reload_passes_through_pinch_point():
    """After loading to a positive yield peak, reversing to a negative
    yield peak, then reloading toward positive: the trajectory passes
    through (pinch_x * eps_pos_peak, pinch_y * sigma_pos_peak)."""
    E, sigma_y, b = 2e11, 400e6, 0.01
    pinch_x, pinch_y = 0.5, 0.3
    mat = UniaxialHysteretic(E=E, sigma_y=sigma_y, b=b,
                               pinch_x=pinch_x, pinch_y=pinch_y)
    eps_y = sigma_y / E
    eps_peak = 2.0 * eps_y
    # Push to +eps_peak (envelope)
    for eps in np.linspace(0.0, eps_peak, 30):
        mat.get_response(eps); mat.commit_state()
    sigma_peak_pos = mat.sigma_committed
    # Reverse to -eps_peak (back through 0 and onto negative envelope)
    for eps in np.linspace(eps_peak, -eps_peak, 60):
        mat.get_response(eps); mat.commit_state()
    # Now reload from (-eps_peak, -sigma_peak_pos) toward (+eps_peak, +sigma_peak_pos)
    # The pinch point is at (pinch_x * eps_peak, pinch_y * sigma_peak_pos).
    eps_pinch = pinch_x * eps_peak
    sigma_pinch = pinch_y * sigma_peak_pos
    sigma_at_pinch, _ = mat.get_response(eps_pinch)
    mat.commit_state()
    assert sigma_at_pinch == pytest.approx(sigma_pinch, rel=1e-10)


# ====================================================== cyclic symmetry

def test_hysteretic_symmetric_cycle():
    """Loading to +eps_max then -eps_max gives sigma magnitudes equal
    (envelope is symmetric)."""
    E, sigma_y = 2e11, 400e6
    mat = UniaxialHysteretic(E=E, sigma_y=sigma_y, b=0.01,
                               pinch_x=0.5, pinch_y=0.3)
    eps_y = sigma_y / E
    eps_max = 3.0 * eps_y
    # Load to +eps_max
    for eps in np.linspace(0.0, eps_max, 30):
        mat.get_response(eps); mat.commit_state()
    sigma_pos = mat.sigma_committed
    # Load to -eps_max
    for eps in np.linspace(eps_max, -eps_max, 60):
        mat.get_response(eps); mat.commit_state()
    sigma_neg = mat.sigma_committed
    assert sigma_neg == pytest.approx(-sigma_pos, rel=1e-10)


# ====================================================== lifecycle

def test_hysteretic_damage_factor_zero_unchanged():
    """damage_factor = 0 reproduces the original undegraded model."""
    E, sigma_y = 2e11, 400e6
    mat_no_dmg = UniaxialHysteretic(E=E, sigma_y=sigma_y, b=0.01,
                                       pinch_x=0.5, pinch_y=0.3,
                                       damage_factor=0.0)
    eps_y = sigma_y / E
    # Sweep a full cycle and confirm peaks are at full strength
    for eps in np.linspace(0.0, 3 * eps_y, 30):
        mat_no_dmg.get_response(eps); mat_no_dmg.commit_state()
    sigma_peak1 = mat_no_dmg.sigma_committed
    for eps in np.linspace(3 * eps_y, -3 * eps_y, 60):
        mat_no_dmg.get_response(eps); mat_no_dmg.commit_state()
    for eps in np.linspace(-3 * eps_y, 3 * eps_y, 60):
        mat_no_dmg.get_response(eps); mat_no_dmg.commit_state()
    sigma_peak2 = mat_no_dmg.sigma_committed
    # Without damage, the peak stress is the same on reload
    assert sigma_peak2 == pytest.approx(sigma_peak1, rel=1e-10)


def test_hysteretic_damage_reduces_envelope_strength():
    """Under repeated cycling, the yield strength on the envelope
    drops as damage accumulates."""
    E, sigma_y = 2e11, 400e6
    mat = UniaxialHysteretic(E=E, sigma_y=sigma_y, b=0.01,
                                pinch_x=0.5, pinch_y=0.3,
                                damage_factor=0.5)   # significant degradation
    eps_y = sigma_y / E
    # First peak (1 excursion to +3 eps_y)
    for eps in np.linspace(0.0, 3 * eps_y, 30):
        mat.get_response(eps); mat.commit_state()
    sigma_first_peak = mat.sigma_committed
    # Cycle several times: -3..+3 eps_y
    for _ in range(5):
        for eps in np.linspace(mat.eps_committed, -3 * eps_y, 50):
            mat.get_response(eps); mat.commit_state()
        for eps in np.linspace(-3 * eps_y, 3 * eps_y, 50):
            mat.get_response(eps); mat.commit_state()
    sigma_last_peak = mat.sigma_committed
    # After cycling, the reload at +3 eps_y is weaker than the first
    assert abs(sigma_last_peak) < 0.95 * abs(sigma_first_peak)


def test_hysteretic_damage_alpha_accumulates():
    """alpha state grows monotonically under envelope excursions."""
    E, sigma_y = 2e11, 400e6
    mat = UniaxialHysteretic(E=E, sigma_y=sigma_y, b=0.01,
                                pinch_x=0.5, pinch_y=0.3,
                                damage_factor=0.1)
    eps_y = sigma_y / E
    alphas = []
    for eps in np.linspace(0.0, 5 * eps_y, 50):
        mat.get_response(eps); mat.commit_state()
        alphas.append(mat.alpha_committed)
    # alpha increases monotonically once we push past yield
    assert all(alphas[i+1] >= alphas[i] for i in range(len(alphas)-1))
    # Final alpha is positive (excursion = 5 eps_y - 1 eps_y = 4 eps_y)
    assert mat.alpha_committed > 0.0


def test_hysteretic_damage_clamped_at_min_strength():
    """Strength ratio never drops below min_strength_ratio."""
    E, sigma_y = 2e11, 400e6
    mat = UniaxialHysteretic(E=E, sigma_y=sigma_y, b=0.01,
                                damage_factor=10.0,    # aggressive
                                min_strength_ratio=0.2)
    # Pile up huge plastic excursion
    eps_y = sigma_y / E
    for eps in np.linspace(0.0, 100 * eps_y, 100):
        mat.get_response(eps); mat.commit_state()
    ratio = mat._strength_ratio(mat.alpha_committed)
    assert ratio >= 0.2 - 1e-12
    assert ratio <= 1.0


def test_hysteretic_commit_revert():
    mat = UniaxialHysteretic(E=2e11, sigma_y=400e6, b=0.01,
                               pinch_x=0.5, pinch_y=0.3)
    # Push past yield
    mat.get_response(0.005); mat.commit_state()
    eps_max_pos_after_commit = mat.eps_max_pos_committed
    sigma_after_commit = mat.sigma_committed
    # Trial revert
    mat.get_response(-0.001)
    mat.revert_state()
    assert mat.eps_max_pos_trial == eps_max_pos_after_commit
    assert mat.sigma_trial == sigma_after_commit
