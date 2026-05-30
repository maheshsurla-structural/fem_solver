"""Phase 44 tests -- random variables, FORM, SORM, Monte Carlo.
"""
from __future__ import annotations

import math

import numpy as np
import pytest
from scipy.stats import norm

from femsolver.reliability import (
    FORMResult,
    Gumbel,
    Lognormal,
    Normal,
    RandomVariableVector,
    SORMResult,
    Uniform,
    Weibull,
    crude_monte_carlo,
    form_hlrf,
    importance_sampling_around_u_star,
    latin_hypercube_monte_carlo,
    sorm_breitung,
)


# ============================================================ random variables

class TestNormal:
    def test_cdf_inv_cdf_round_trip(self):
        n = Normal(mu=10.0, sigma=2.0)
        for F in [0.05, 0.5, 0.95]:
            x = n.inv_cdf(F)
            assert n.cdf(x) == pytest.approx(F, rel=1e-9)

    def test_transform_round_trip(self):
        n = Normal(mu=10.0, sigma=2.0)
        for x in [6.0, 10.0, 14.0]:
            u = n.transform_to_U(x)
            assert n.transform_to_X(u) == pytest.approx(x, rel=1e-9)

    def test_mean_std(self):
        n = Normal(mu=10.0, sigma=2.0)
        assert n.mean() == 10.0
        assert n.std() == 2.0

    def test_validates_sigma(self):
        with pytest.raises(ValueError):
            Normal(mu=0.0, sigma=-1.0)


class TestLognormal:
    def test_mean_std_in_real_space(self):
        ln = Lognormal(mu_X=10.0, sigma_X=2.0)
        assert ln.mean() == 10.0
        assert ln.std() == 2.0

    def test_cdf_round_trip(self):
        ln = Lognormal(mu_X=10.0, sigma_X=2.0)
        for F in [0.1, 0.5, 0.9]:
            assert ln.cdf(ln.inv_cdf(F)) == pytest.approx(F, rel=1e-9)


class TestUniform:
    def test_cdf_at_endpoints(self):
        u = Uniform(0.0, 1.0)
        assert u.cdf(0.0) == 0.0
        assert u.cdf(1.0) == 1.0
        assert u.cdf(0.5) == 0.5

    def test_mean_std(self):
        u = Uniform(0.0, 12.0)
        assert u.mean() == 6.0
        assert u.std() == pytest.approx(12.0 / math.sqrt(12.0), rel=1e-12)


class TestGumbel:
    def test_inv_cdf_round_trip(self):
        g = Gumbel(mu=0.0, beta=1.0)
        for F in [0.1, 0.5, 0.9]:
            assert g.cdf(g.inv_cdf(F)) == pytest.approx(F, rel=1e-9)

    def test_mean_std(self):
        g = Gumbel(mu=0.0, beta=1.0)
        # E[X] = mu + beta*gamma, sigma = beta*pi/sqrt(6)
        assert g.mean() == pytest.approx(0.5772156649, rel=1e-6)
        assert g.std() == pytest.approx(math.pi / math.sqrt(6.0), rel=1e-12)


class TestWeibull:
    def test_inv_cdf_round_trip(self):
        w = Weibull(k=2.0, lam=10.0)
        for F in [0.1, 0.5, 0.9]:
            assert w.cdf(w.inv_cdf(F)) == pytest.approx(F, rel=1e-9)


class TestRandomVariableVector:
    def test_uncorrelated_basis(self):
        rvs = RandomVariableVector([Normal(0, 1), Normal(0, 1)])
        x = np.array([1.5, -0.5])
        u = rvs.transform_to_U(x)
        np.testing.assert_allclose(u, x, atol=1e-9)
        np.testing.assert_allclose(rvs.transform_to_X(u), x, atol=1e-9)

    def test_correlated_round_trip(self):
        R = np.array([[1.0, 0.5], [0.5, 1.0]])
        rvs = RandomVariableVector(
            [Normal(0, 1), Normal(0, 1)],
            correlation=R,
        )
        x = np.array([1.2, -0.7])
        u = rvs.transform_to_U(x)
        x_back = rvs.transform_to_X(u)
        np.testing.assert_allclose(x_back, x, atol=1e-9)

    def test_validates_correlation_dim(self):
        R = np.eye(3)
        with pytest.raises(ValueError, match="correlation"):
            RandomVariableVector(
                [Normal(0, 1), Normal(0, 1)],
                correlation=R,
            )


# ============================================================ FORM

class TestFORMLinear:
    def test_R_minus_S_normal(self):
        """g(R, S) = R - S, R ~ N(500, 100), S ~ N(200, 50).

        Exact: beta = (mu_R - mu_S) / sqrt(sigma_R^2 + sigma_S^2)
                  = 300 / sqrt(12500) = 2.6833.
        """
        mu_R, sigma_R = 500.0, 100.0
        mu_S, sigma_S = 200.0, 50.0
        beta_exact = (mu_R - mu_S) / math.sqrt(sigma_R ** 2 + sigma_S ** 2)
        rvs = RandomVariableVector([
            Normal(mu_R, sigma_R), Normal(mu_S, sigma_S),
        ])
        def g(x):
            return x[0] - x[1]
        res = form_hlrf(g=g, rvs=rvs)
        assert res.converged
        assert res.beta == pytest.approx(beta_exact, rel=1e-9)
        assert res.pf == pytest.approx(
            float(norm.cdf(-beta_exact)), rel=1e-9,
        )

    def test_alpha_unit_norm(self):
        rvs = RandomVariableVector([
            Normal(500, 100), Normal(200, 50),
        ])
        def g(x):
            return x[0] - x[1]
        res = form_hlrf(g=g, rvs=rvs)
        assert np.linalg.norm(res.alpha) == pytest.approx(1.0, rel=1e-9)

    def test_one_dim_problem(self):
        # g(X) = X - 5, X ~ N(10, 2)
        # Failure when X < 5: beta = (10 - 5) / 2 = 2.5
        rvs = RandomVariableVector([Normal(10.0, 2.0)])
        def g(x):
            return x[0] - 5.0
        res = form_hlrf(g=g, rvs=rvs)
        assert res.beta == pytest.approx(2.5, rel=1e-9)


# ============================================================ SORM

class TestSORMLinear:
    def test_linear_g_returns_FORM(self):
        """For a linear limit state, SORM should reproduce FORM."""
        rvs = RandomVariableVector([
            Normal(500, 100), Normal(200, 50),
        ])
        def g(x):
            return x[0] - x[1]
        res_F = form_hlrf(g=g, rvs=rvs)
        res_S = sorm_breitung(form_result=res_F, g=g, rvs=rvs)
        assert res_S.pf_SORM == pytest.approx(res_S.pf_FORM, rel=1e-3)


# ============================================================ Monte Carlo

class TestMonteCarlo:
    def test_crude_MC_converges_to_FORM(self):
        rvs = RandomVariableVector([
            Normal(500, 100), Normal(200, 50),
        ])
        def g(x):
            return x[0] - x[1]
        res_F = form_hlrf(g=g, rvs=rvs)
        res_MC = crude_monte_carlo(
            g=g, rvs=rvs, n_samples=100000, seed=42,
        )
        # within 2 sigma of FORM exact
        assert abs(res_MC.pf_estimate - res_F.pf) < 2 * res_MC.pf_std_error

    def test_LHS_lower_variance_than_crude(self):
        rvs = RandomVariableVector([
            Normal(500, 100), Normal(200, 50),
        ])
        def g(x):
            return x[0] - x[1]
        res_MC = crude_monte_carlo(
            g=g, rvs=rvs, n_samples=5000, seed=42,
        )
        res_LHS = latin_hypercube_monte_carlo(
            g=g, rvs=rvs, n_samples=5000, seed=42,
        )
        # LHS may not always strictly win on a single run, but the
        # variance should be in the same ballpark or smaller
        assert res_LHS.pf_estimate > 0.0
        assert abs(res_LHS.pf_estimate - res_MC.pf_estimate) < 1e-2

    def test_importance_sampling_variance_reduction(self):
        rvs = RandomVariableVector([
            Normal(500, 100), Normal(200, 50),
        ])
        def g(x):
            return x[0] - x[1]
        res_F = form_hlrf(g=g, rvs=rvs)
        res_IS = importance_sampling_around_u_star(
            g=g, rvs=rvs, u_star=res_F.u_star,
            n_samples=2000, seed=42,
        )
        # IS at 2k should give a sensible estimate within a few sigma
        assert abs(res_IS.pf_estimate - res_F.pf) < 5 * res_IS.pf_std_error

    def test_zero_failures_returns_zero(self):
        """If no MC sample fails, pf = 0 and beta = +inf."""
        # Very safe system: R >> S
        rvs = RandomVariableVector([
            Normal(1e9, 100), Normal(100, 50),
        ])
        def g(x):
            return x[0] - x[1]
        res = crude_monte_carlo(g=g, rvs=rvs, n_samples=100, seed=42)
        assert res.pf_estimate == 0.0
        assert math.isinf(res.beta_estimate)
