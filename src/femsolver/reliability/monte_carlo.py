"""Monte-Carlo failure-probability estimators.

Three sampling strategies for estimating ``P_f = Pr(g(X) <= 0)``:

* :func:`crude_monte_carlo` -- draw ``n`` samples from the prior, count
  the fraction with ``g <= 0``. Unbiased but slow when ``P_f`` is
  small (need ``~ 1/P_f`` samples for one expected failure).
* :func:`latin_hypercube_monte_carlo` -- Latin-hypercube stratified
  draw of the same prior; substantially lower variance for
  moderate ``n`` because each marginal is covered evenly.
* :func:`importance_sampling_around_u_star` -- centres a sampling
  density at the FORM design point ``U*``; for typical structural
  reliability problems this reduces the variance by orders of
  magnitude.

All three return a :class:`MonteCarloResult` with the point estimate,
standard error, 95% CI, and the count of failures.
"""
from __future__ import annotations

from dataclasses import dataclass

import numpy as np
from scipy.stats import multivariate_normal, norm


@dataclass
class MonteCarloResult:
    """Outcome of a Monte-Carlo reliability evaluation.

    Attributes
    ----------
    pf_estimate : float
    pf_std_error : float
    pf_ci95_low, pf_ci95_high : float
    n_samples : int
    n_failures : int
    beta_estimate : float
        ``- Phi^{-1}(pf_estimate)`` -- equivalent reliability index.
    method : str
    """

    pf_estimate: float
    pf_std_error: float
    pf_ci95_low: float
    pf_ci95_high: float
    n_samples: int
    n_failures: int
    beta_estimate: float
    method: str


def _wrap(pf: float, se: float, n: int, n_fail: int, method: str) -> MonteCarloResult:
    pf = float(min(max(pf, 0.0), 1.0))
    z = 1.96
    lo = max(pf - z * se, 0.0)
    hi = min(pf + z * se, 1.0)
    if 1e-15 < pf < 1 - 1e-15:
        beta = float(-norm.ppf(pf))
    elif pf <= 1e-15:
        beta = float("inf")
    else:
        beta = float("-inf")
    return MonteCarloResult(
        pf_estimate=pf,
        pf_std_error=float(se),
        pf_ci95_low=float(lo),
        pf_ci95_high=float(hi),
        n_samples=int(n),
        n_failures=int(n_fail),
        beta_estimate=beta,
        method=method,
    )


# ============================================================ crude MC

def crude_monte_carlo(
    *,
    g, rvs, n_samples: int = 10000, seed: int | None = None,
) -> MonteCarloResult:
    """Crude (direct) Monte-Carlo failure-probability estimator.

    Parameters
    ----------
    g : callable
        ``g(X) -> float``. Failure when ``g <= 0``.
    rvs : RandomVariableVector
    n_samples : int
    seed : int, optional
    """
    rng = np.random.default_rng(seed)
    n = len(rvs)
    n_fail = 0
    for _ in range(n_samples):
        u = rng.standard_normal(n)
        x = rvs.transform_to_X(u)
        if g(x) <= 0.0:
            n_fail += 1
    pf = n_fail / n_samples
    se = np.sqrt(pf * (1.0 - pf) / n_samples) if n_samples > 0 else 0.0
    return _wrap(pf, se, n_samples, n_fail, "crude_MC")


# ============================================================ Latin hypercube

def latin_hypercube_monte_carlo(
    *,
    g, rvs, n_samples: int = 10000, seed: int | None = None,
) -> MonteCarloResult:
    """Latin-hypercube sampling estimator.

    Stratifies each marginal into ``n_samples`` equal-probability
    bins and draws one sample from each; the per-bin draws are then
    randomly permuted across dimensions. Variance is typically a
    fraction of the crude-MC estimator's at the same ``n``.
    """
    rng = np.random.default_rng(seed)
    n = len(rvs)
    # Build LHS in U-space directly (since the joint is standard normal
    # after Rosenblatt). For each dimension, generate stratified
    # standard-normal samples using Phi^{-1} of the strata midpoints
    # plus jitter, then permute across rows.
    midpoints = (np.arange(n_samples) + 0.5) / n_samples
    jitter = (rng.random((n_samples, n)) - 0.5) / n_samples
    # Per-dimension uniform draws in (0, 1)
    U_unif = np.tile(midpoints[:, None], (1, n)) + jitter
    # Permute each column independently
    for j in range(n):
        rng.shuffle(U_unif[:, j])
    U = norm.ppf(np.clip(U_unif, 1e-15, 1.0 - 1e-15))
    # Evaluate g
    n_fail = 0
    for i in range(n_samples):
        x = rvs.transform_to_X(U[i])
        if g(x) <= 0.0:
            n_fail += 1
    pf = n_fail / n_samples
    se = np.sqrt(pf * (1.0 - pf) / n_samples) if n_samples > 0 else 0.0
    return _wrap(pf, se, n_samples, n_fail, "LHS")


# ============================================================ importance sampling

def importance_sampling_around_u_star(
    *,
    g, rvs, u_star, n_samples: int = 2000, seed: int | None = None,
    sigma_scale: float = 1.0,
) -> MonteCarloResult:
    """Importance sampling with a Gaussian biasing density centred at
    the design point ``u_star``.

    Draws ``u ~ N(u_star, sigma_scale^2 I)`` (in U-space) and
    weighs each sample by the ratio of the original ``N(0, I)`` density
    to the biasing density:

        w_i = phi(u_i) / phi_bias(u_i)
            = exp(- ½ (||u_i||^2 - ||u_i - u_star||^2 / sigma_scale^2))
              · sigma_scale^n.

    For ``u_star`` correctly placed near the limit-state surface and
    ``sigma_scale ≈ 1``, the estimator achieves OOM lower variance
    than crude MC.

    Parameters
    ----------
    u_star : array
        Design point in U-space (from FORM).
    sigma_scale : float, default 1.0
        Standard-deviation scale of the biasing density. Increase
        to broaden the search; decrease for tighter focus.
    """
    rng = np.random.default_rng(seed)
    u_star = np.asarray(u_star, dtype=float).ravel()
    n = u_star.size
    samples = rng.standard_normal((n_samples, n)) * sigma_scale + u_star
    # Indicator
    ind = np.zeros(n_samples)
    for i in range(n_samples):
        x = rvs.transform_to_X(samples[i])
        if g(x) <= 0.0:
            ind[i] = 1.0
    # Weights: phi(u) / phi_bias(u)
    # log phi(u_i) = -½ ||u_i||² + const
    # log phi_bias(u_i) = -½ ||(u_i - u_star) / sigma||² - n log sigma + const
    log_w = (-0.5 * np.sum(samples ** 2, axis=1)
             + 0.5 * np.sum((samples - u_star) ** 2, axis=1)
             / (sigma_scale ** 2)
             + n * np.log(sigma_scale))
    w = np.exp(log_w)
    pf_est = float(np.mean(ind * w))
    var = float(np.var(ind * w, ddof=1)) / n_samples
    se = float(np.sqrt(max(var, 0.0)))
    n_fail = int(np.sum(ind))
    return _wrap(pf_est, se, n_samples, n_fail, "importance_sampling")
