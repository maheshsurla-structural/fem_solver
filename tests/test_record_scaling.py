"""Phase 26 tests -- ASCE 7 amplitude scaling + Baker-Jayaram CMS.
"""
from __future__ import annotations

import math

import numpy as np
import pytest

from femsolver import (
    SuiteScalingResult,
    amplitude_scale_factor,
    baker_jayaram_correlation,
    compute_epsilon,
    compute_sdof_response_spectrum,
    conditional_mean_spectrum,
    conditional_spectrum_variance,
    period_range_mask,
    record_response_spectrum,
    scale_record_suite,
)


# ============================================================ SDOF spectrum

def test_sdof_spectrum_resonant_amplification():
    """Sinusoid at T = T_n should give ~ 1/(2 zeta) amplification."""
    T_drive = 1.0
    dt = 0.005
    duration = 50.0   # long enough to reach steady-state
    t = np.arange(0.0, duration, dt)
    omega = 2.0 * math.pi / T_drive
    PGA = 1.0
    ag = PGA * np.sin(omega * t)
    zeta = 0.05
    periods = np.array([T_drive])
    Sa = compute_sdof_response_spectrum(ag, dt, periods, zeta=zeta)
    expected = PGA / (2.0 * zeta)
    # 5-10% tolerance, since we're approaching but not at perfect steady state
    assert Sa[0] == pytest.approx(expected, rel=0.10)


def test_sdof_spectrum_zero_period_limit():
    """At T -> 0, Sa -> PGA (rigid oscillator follows ground)."""
    dt = 0.005
    t = np.arange(0.0, 10.0, dt)
    PGA = 2.0
    ag = PGA * np.sin(2.0 * math.pi * 1.5 * t)
    Sa = compute_sdof_response_spectrum(
        ag, dt, np.array([0.01]), zeta=0.05,
    )
    assert Sa[0] == pytest.approx(PGA, rel=0.10)


def test_sdof_spectrum_validates_inputs():
    with pytest.raises(ValueError, match="dt"):
        compute_sdof_response_spectrum(np.array([1, 2]), 0.0,
                                        np.array([1.0]))
    with pytest.raises(ValueError, match="positive"):
        compute_sdof_response_spectrum(np.array([1, 2]), 0.01,
                                        np.array([-1.0]))
    with pytest.raises(ValueError, match="at least 2"):
        compute_sdof_response_spectrum(np.array([1.0]), 0.01,
                                        np.array([1.0]))


def test_record_response_spectrum_wrapper():
    """Function-form wrapper produces same result as array form."""
    dt = 0.005
    t_end = 10.0
    periods = np.array([0.5, 1.0, 2.0])

    def ag(t):
        return math.sin(2.0 * math.pi * t)

    Sa = record_response_spectrum(ag, t_end=t_end, dt=dt, periods=periods)
    assert Sa.shape == (3,)
    assert np.all(Sa > 0.0)


# ============================================================ amplitude scaling

def test_amplitude_scale_factor_geometric_mean():
    """If record_Sa = c * target_Sa, scale factor should be 1/c."""
    target = np.array([0.5, 1.0, 0.8, 0.6])
    rec = 0.4 * target            # record is 40% of target
    sf = amplitude_scale_factor(rec, target)
    assert sf == pytest.approx(1.0 / 0.4, rel=1.0e-12)


def test_amplitude_scale_factor_min_geometric_mean_error():
    """For non-flat ratio, scale factor minimises log-space L2 error."""
    target = np.array([1.0, 2.0, 1.5, 0.9])
    rec = np.array([0.5, 0.8, 0.6, 0.4])
    sf = amplitude_scale_factor(rec, target)
    # After scaling, geometric mean of (scaled / target) should be 1
    ratio = sf * rec / target
    gm = np.exp(np.mean(np.log(ratio)))
    assert gm == pytest.approx(1.0, abs=1.0e-12)


def test_amplitude_scale_factor_rejects_invalid_shapes():
    with pytest.raises(ValueError, match="shape"):
        amplitude_scale_factor(np.array([1.0]), np.array([1.0, 2.0]))
    with pytest.raises(ValueError, match="positive"):
        amplitude_scale_factor(np.array([-1.0]), np.array([1.0]))


# ============================================================ suite scaling

def test_scale_record_suite_synthetic_target():
    """Five synthetic records, target is flat 1.0g across periods 0.5-2s."""
    periods = np.linspace(0.5, 2.0, 16)
    target = np.full_like(periods, 1.0)        # 1.0 g flat target
    # Five records, each a different flat level (under-scaled spectra)
    levels = [0.3, 0.5, 0.7, 0.9, 1.1]
    records = [np.full_like(periods, lv) for lv in levels]
    result = scale_record_suite(records, target)
    assert isinstance(result, SuiteScalingResult)
    # Each scale factor brings the record to 1.0
    for sf, lv in zip(result.scale_factors, levels):
        assert sf == pytest.approx(1.0 / lv, rel=1.0e-12)
    # Suite average after scaling = 1.0 at every period -> min_ratio = 1
    assert result.min_ratio == pytest.approx(1.0, rel=1.0e-12)
    assert result.passes_90pct
    assert result.n_records == 5


def test_scale_record_suite_period_range_mask():
    """Mask should restrict scaling decision to selected periods only."""
    periods = np.linspace(0.1, 5.0, 50)
    target = np.full_like(periods, 1.0)
    # Record that matches target only inside [0.5, 2.0], otherwise much lower
    rec = np.where((periods >= 0.5) & (periods <= 2.0),
                    1.0, 0.1)
    mask = (periods >= 0.5) & (periods <= 2.0)
    result = scale_record_suite([rec], target, period_range_mask=mask)
    # Inside the mask the record already matches target -> SF ~ 1
    assert result.scale_factors[0] == pytest.approx(1.0, rel=1.0e-12)
    assert result.passes_90pct


def test_period_range_mask_helper():
    periods = np.linspace(0.1, 3.0, 100)
    mask = period_range_mask(periods, T1=1.0, low_mult=0.2, high_mult=2.0)
    selected = periods[mask]
    assert selected.min() >= 0.2 - 1.0e-12
    assert selected.max() <= 2.0 + 1.0e-12


def test_period_range_mask_validates():
    with pytest.raises(ValueError, match="T1"):
        period_range_mask(np.array([1.0]), T1=-1.0)
    with pytest.raises(ValueError, match="low_mult"):
        period_range_mask(np.array([1.0]), T1=1.0, low_mult=2.0, high_mult=0.5)


# ============================================================ Baker-Jayaram

def test_baker_jayaram_symmetric():
    rho_ab = baker_jayaram_correlation(0.3, 1.2)
    rho_ba = baker_jayaram_correlation(1.2, 0.3)
    assert rho_ab == pytest.approx(rho_ba, rel=1.0e-12)


def test_baker_jayaram_self_correlation_is_one():
    for T in [0.05, 0.1, 0.3, 1.0, 5.0]:
        rho = baker_jayaram_correlation(T, T)
        assert rho == pytest.approx(1.0, abs=1.0e-12)


def test_baker_jayaram_decreases_with_separation():
    """Far-apart periods should have smaller correlation."""
    rho_close = baker_jayaram_correlation(1.0, 1.1)
    rho_far = baker_jayaram_correlation(0.1, 10.0)
    assert rho_close > rho_far
    assert 0.0 <= rho_far <= 0.2     # near-uncorrelated at decade separation


def test_baker_jayaram_rejects_invalid():
    with pytest.raises(ValueError, match="positive"):
        baker_jayaram_correlation(-1.0, 1.0)
    with pytest.raises(ValueError, match="positive"):
        baker_jayaram_correlation(1.0, 0.0)


# ============================================================ CMS

def test_compute_epsilon_basic():
    """epsilon = (ln target - ln median) / sigma."""
    eps = compute_epsilon(Sa_target=1.0, mu_Sa_at_Tstar=0.5,
                          sigma_lnSa_at_Tstar=0.6)
    expected = (math.log(1.0) - math.log(0.5)) / 0.6
    assert eps == pytest.approx(expected, rel=1.0e-12)


def test_compute_epsilon_validates():
    with pytest.raises(ValueError):
        compute_epsilon(Sa_target=-1.0, mu_Sa_at_Tstar=0.5,
                        sigma_lnSa_at_Tstar=0.6)
    with pytest.raises(ValueError):
        compute_epsilon(Sa_target=1.0, mu_Sa_at_Tstar=-0.5,
                        sigma_lnSa_at_Tstar=0.6)
    with pytest.raises(ValueError):
        compute_epsilon(Sa_target=1.0, mu_Sa_at_Tstar=0.5,
                        sigma_lnSa_at_Tstar=-0.6)


def test_conditional_mean_spectrum_anchored_at_T_star():
    """CMS at T_star = exp(mu + epsilon * sigma) since rho(T*,T*) = 1."""
    T_star = 1.0
    periods = np.array([0.5, 1.0, 2.0])
    mu_lnSa = np.array([math.log(0.3), math.log(0.5), math.log(0.4)])
    sigma_lnSa = np.array([0.5, 0.6, 0.55])
    eps = 2.0
    cms = conditional_mean_spectrum(T_star, eps, periods, mu_lnSa, sigma_lnSa)
    expected_at_Tstar = math.exp(math.log(0.5) + 2.0 * 0.6)
    assert cms[1] == pytest.approx(expected_at_Tstar, rel=1.0e-12)


def test_conditional_mean_spectrum_uniform_when_epsilon_zero():
    """epsilon = 0 -> CMS = exp(mu) (the median spectrum)."""
    T_star = 1.0
    periods = np.array([0.2, 1.0, 3.0])
    mu_lnSa = np.array([math.log(0.4), math.log(0.5), math.log(0.3)])
    sigma_lnSa = np.array([0.6, 0.6, 0.6])
    cms = conditional_mean_spectrum(T_star, 0.0, periods, mu_lnSa, sigma_lnSa)
    expected = np.exp(mu_lnSa)
    np.testing.assert_allclose(cms, expected, rtol=1.0e-12)


def test_conditional_spectrum_variance_zero_at_Tstar():
    """At T*, conditional sigma is 0 (we've conditioned on it exactly)."""
    T_star = 1.0
    periods = np.array([0.5, 1.0, 2.0])
    sigma_lnSa = np.array([0.5, 0.6, 0.55])
    sd_cond = conditional_spectrum_variance(T_star, periods, sigma_lnSa)
    assert sd_cond[1] == pytest.approx(0.0, abs=1.0e-7)
    # Off-conditioning, conditional sigma < marginal sigma
    assert sd_cond[0] < sigma_lnSa[0]
    assert sd_cond[2] < sigma_lnSa[2]
