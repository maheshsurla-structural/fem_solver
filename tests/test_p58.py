"""Phase 28 tests -- FEMA P-58 component damage + Monte-Carlo loss.
"""
from __future__ import annotations

import math

import numpy as np
import pytest

from femsolver import (
    ComponentDamageAssessment,
    ComponentFragility,
    ComponentGroup,
    DamageState,
)


# ============================================================ Damage state

def test_damage_state_prob_exceeds_at_median_is_half():
    """P(DS >= this | EDP = theta) = 0.5."""
    ds = DamageState("DS1", fragility_theta=0.010,
                     fragility_beta=0.4, cost_median=1000.0)
    assert ds.prob_exceeds(0.010) == pytest.approx(0.5, abs=1.0e-12)


def test_damage_state_prob_at_zero_is_zero():
    ds = DamageState("DS1", fragility_theta=0.010,
                     fragility_beta=0.4, cost_median=1000.0)
    assert ds.prob_exceeds(0.0) == 0.0
    assert ds.prob_exceeds(-0.001) == 0.0


def test_damage_state_prob_monotonic():
    """P(DS >= this) increases with EDP."""
    ds = DamageState("DS1", fragility_theta=0.010,
                     fragility_beta=0.4, cost_median=1000.0)
    for edp in np.linspace(0.001, 0.05, 50):
        pass
    edps = np.linspace(0.001, 0.05, 50)
    probs = np.array([ds.prob_exceeds(e) for e in edps])
    assert np.all(np.diff(probs) >= -1.0e-12)
    assert probs[0] < 0.1
    assert probs[-1] > 0.99


def test_damage_state_expected_cost_lognormal():
    """E[cost] = median * exp(beta^2/2)."""
    ds = DamageState("DS1", fragility_theta=0.010,
                     fragility_beta=0.4,
                     cost_median=1000.0, cost_beta=0.5)
    expected = 1000.0 * math.exp(0.5 * 0.5 ** 2)
    assert ds.expected_cost() == pytest.approx(expected, rel=1.0e-12)


def test_damage_state_sample_cost_mean_converges():
    """MC sample mean -> E[cost] for many samples."""
    ds = DamageState("DS1", fragility_theta=0.010,
                     fragility_beta=0.4,
                     cost_median=1000.0, cost_beta=0.3)
    rng = np.random.default_rng(42)
    samples = np.array([ds.sample_cost(rng) for _ in range(20_000)])
    expected = 1000.0 * math.exp(0.5 * 0.3 ** 2)
    assert samples.mean() == pytest.approx(expected, rel=0.03)


def test_damage_state_validates():
    with pytest.raises(ValueError, match="fragility_theta"):
        DamageState("DS1", -1.0, 0.4, 1000.0)
    with pytest.raises(ValueError, match="fragility_beta"):
        DamageState("DS1", 0.01, -0.4, 1000.0)
    with pytest.raises(ValueError, match="cost_median"):
        DamageState("DS1", 0.01, 0.4, -1000.0)


# ============================================================ ComponentFragility

def _three_state_drywall():
    """Standard 3-damage-state drywall partition."""
    ds1 = DamageState("DS1", fragility_theta=0.005,
                       fragility_beta=0.4, cost_median=200.0)
    ds2 = DamageState("DS2", fragility_theta=0.012,
                       fragility_beta=0.4, cost_median=800.0)
    ds3 = DamageState("DS3", fragility_theta=0.024,
                       fragility_beta=0.4, cost_median=2500.0)
    return ComponentFragility("Drywall", "PSD",
                              damage_states=[ds1, ds2, ds3])


def test_component_damage_state_probs_sum_to_one():
    comp = _three_state_drywall()
    for edp in [0.001, 0.005, 0.010, 0.020, 0.050, 0.100]:
        probs = comp.damage_state_probs(edp)
        assert probs.sum() == pytest.approx(1.0, abs=1.0e-10)


def test_component_damage_state_probs_non_negative():
    comp = _three_state_drywall()
    for edp in [0.001, 0.005, 0.010, 0.020, 0.050, 0.100]:
        probs = comp.damage_state_probs(edp)
        assert np.all(probs >= 0.0)


def test_component_at_theta_ds1_split():
    """At EDP = theta_DS1, P(DS = 0) = P(DS >= 1) = 0.5."""
    comp = _three_state_drywall()
    probs = comp.damage_state_probs(0.005)
    # P(DS = 0) = 1 - P(DS >= DS1) = 0.5
    assert probs[0] == pytest.approx(0.5, abs=1.0e-12)


def test_component_requires_increasing_thetas():
    """Damage-state thetas must be strictly increasing."""
    ds1 = DamageState("DS1", 0.020, 0.4, 100.0)
    ds2 = DamageState("DS2", 0.010, 0.4, 200.0)  # smaller theta than DS1!
    with pytest.raises(ValueError, match="increasing"):
        ComponentFragility("X", "PSD", damage_states=[ds1, ds2])


def test_component_requires_at_least_one_ds():
    with pytest.raises(ValueError, match="at least one"):
        ComponentFragility("X", "PSD", damage_states=[])


# ============================================================ ComponentGroup

def test_group_expected_loss_zero_at_zero_edp():
    """No EDP -> no damage -> zero expected loss."""
    comp = _three_state_drywall()
    group = ComponentGroup(component=comp, quantity=100.0,
                            edp_value=0.0001)
    assert group.expected_loss() < 1.0


def test_group_expected_loss_scales_with_quantity():
    comp = _three_state_drywall()
    g1 = ComponentGroup(component=comp, quantity=50.0, edp_value=0.015)
    g2 = ComponentGroup(component=comp, quantity=200.0, edp_value=0.015)
    assert g2.expected_loss() == pytest.approx(
        4.0 * g1.expected_loss(), rel=1.0e-12)


def test_group_expected_loss_increases_with_edp():
    comp = _three_state_drywall()
    losses = [
        ComponentGroup(comp, 100.0, edp).expected_loss()
        for edp in [0.001, 0.005, 0.010, 0.020, 0.040]
    ]
    assert all(losses[i + 1] > losses[i]
               for i in range(len(losses) - 1))


def test_group_sample_loss_zero_for_no_damage():
    """If we (somehow) get DS=0 every realisation, total loss is 0.

    With a very small EDP that's essentially what happens.
    """
    comp = _three_state_drywall()
    group = ComponentGroup(component=comp, quantity=100.0,
                           edp_value=1.0e-5)
    rng = np.random.default_rng(42)
    n = 1000
    losses = np.array([group.sample_loss(rng) for _ in range(n)])
    # With essentially-zero EDP, almost all realisations are zero
    assert (losses == 0.0).sum() > 0.99 * n


# ============================================================ Assessment

def test_assessment_expected_loss_is_sum():
    """Building expected loss = sum of group expected losses."""
    comp = _three_state_drywall()
    groups = [
        ComponentGroup(comp, 50.0, edp_value=0.005, location="F1"),
        ComponentGroup(comp, 80.0, edp_value=0.015, location="F2"),
        ComponentGroup(comp, 30.0, edp_value=0.025, location="F3"),
    ]
    assess = ComponentDamageAssessment(groups)
    total = assess.expected_loss()
    by_group = sum(g.expected_loss() for g in groups)
    assert total == pytest.approx(by_group, rel=1.0e-12)


def test_assessment_monte_carlo_mean_converges_to_expected():
    """MC mean over many realisations -> closed-form E[L]."""
    comp = _three_state_drywall()
    groups = [
        ComponentGroup(comp, 100.0, edp_value=0.010),
    ]
    assess = ComponentDamageAssessment(groups)
    result = assess.monte_carlo(n_realisations=5000, seed=123)
    assert result.mean_loss == pytest.approx(
        result.expected_loss, rel=0.05)


def test_assessment_loss_percentiles_ordering():
    """p84 > median; mean ~ E[L]."""
    comp = _three_state_drywall()
    group = ComponentGroup(comp, 100.0, edp_value=0.018)
    assess = ComponentDamageAssessment([group])
    result = assess.monte_carlo(n_realisations=4000, seed=42)
    assert result.p84_loss >= result.median_loss
    assert result.p95_loss >= result.p84_loss
    assert result.n_realisations == 4000


def test_assessment_requires_groups():
    with pytest.raises(ValueError, match="at least one"):
        ComponentDamageAssessment(groups=[])


def test_assessment_monte_carlo_validates_n():
    comp = _three_state_drywall()
    group = ComponentGroup(comp, 100.0, edp_value=0.01)
    assess = ComponentDamageAssessment([group])
    with pytest.raises(ValueError, match="n_realisations"):
        assess.monte_carlo(n_realisations=0)
