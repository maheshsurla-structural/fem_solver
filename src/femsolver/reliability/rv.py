"""Random-variable definitions + Rosenblatt transformations.

A random variable is a distribution with a CDF / inverse CDF. The
**Rosenblatt transformation** maps each random variable to a
standard normal ``U`` via

    U_i = Phi^{-1}(F_i(X_i)),

so reliability methods (FORM, SORM, MC with importance sampling)
operate in the rotationally-symmetric standard-normal U-space.

The ``RandomVariable`` interface is a small ABC; concrete classes
include:

* :class:`Normal` -- ``N(mu, sigma)``.
* :class:`Lognormal` -- parameters in REAL space (``mu_X``, ``sigma_X``).
* :class:`Uniform` -- ``U(a, b)``.
* :class:`Gumbel` -- maximum (largest extreme value, EV-I).
* :class:`Weibull` -- two-parameter Weibull (shape ``k``, scale ``lam``).

Each implements ``cdf``, ``inv_cdf``, ``pdf``, ``mean``, ``std``,
``transform_to_U`` and ``transform_to_X``.

For correlated random variables, use :class:`RandomVariableVector`,
which holds a list of marginal RVs plus a correlation matrix and
implements the **Nataf transformation** (equivalent-normal approach
under the standard Gaussian copula assumption).
"""
from __future__ import annotations

import math
from abc import ABC, abstractmethod
from dataclasses import dataclass

import numpy as np
from scipy.stats import norm


# ============================================================ base

class RandomVariable(ABC):
    """Marginal random-variable interface."""

    @abstractmethod
    def cdf(self, x: float) -> float: ...

    @abstractmethod
    def inv_cdf(self, F: float) -> float: ...

    @abstractmethod
    def pdf(self, x: float) -> float: ...

    @abstractmethod
    def mean(self) -> float: ...

    @abstractmethod
    def std(self) -> float: ...

    def transform_to_U(self, x: float) -> float:
        """Standard-normal image: ``u = Phi^{-1}(F(x))``."""
        F = self.cdf(x)
        F_safe = min(max(F, 1.0e-15), 1.0 - 1.0e-15)
        return float(norm.ppf(F_safe))

    def transform_to_X(self, u: float) -> float:
        """Inverse Rosenblatt: ``x = F^{-1}(Phi(u))``."""
        F = float(norm.cdf(u))
        return float(self.inv_cdf(F))


# ============================================================ Normal

@dataclass
class Normal(RandomVariable):
    """Normal distribution ``N(mu, sigma)``."""

    mu: float
    sigma: float

    def __post_init__(self) -> None:
        if self.sigma <= 0.0:
            raise ValueError("sigma must be > 0")

    def cdf(self, x: float) -> float:
        return float(norm.cdf((x - self.mu) / self.sigma))

    def inv_cdf(self, F: float) -> float:
        if not (0.0 <= F <= 1.0):
            raise ValueError("F must be in [0, 1]")
        return float(self.mu + self.sigma * norm.ppf(F))

    def pdf(self, x: float) -> float:
        z = (x - self.mu) / self.sigma
        return float(np.exp(-0.5 * z * z) / (self.sigma * math.sqrt(2.0 * math.pi)))

    def mean(self) -> float:
        return float(self.mu)

    def std(self) -> float:
        return float(self.sigma)


# ============================================================ Lognormal

@dataclass
class Lognormal(RandomVariable):
    """Lognormal distribution.

    Parameters
    ----------
    mu_X : float
        Mean in real space.
    sigma_X : float
        Standard deviation in real space.

    Internally converts to log-normal parameters
    ``sigma_lnX^2 = ln(1 + (sigma_X / mu_X)^2)``,
    ``mu_lnX = ln(mu_X) - 0.5 sigma_lnX^2``.
    """

    mu_X: float
    sigma_X: float

    def __post_init__(self) -> None:
        if self.mu_X <= 0.0 or self.sigma_X <= 0.0:
            raise ValueError("Lognormal mu_X, sigma_X must be > 0")
        cov = self.sigma_X / self.mu_X
        self.sigma_lnX = math.sqrt(math.log(1.0 + cov * cov))
        self.mu_lnX = math.log(self.mu_X) - 0.5 * self.sigma_lnX ** 2

    def cdf(self, x: float) -> float:
        if x <= 0.0:
            return 0.0
        return float(norm.cdf((math.log(x) - self.mu_lnX) / self.sigma_lnX))

    def inv_cdf(self, F: float) -> float:
        F_safe = min(max(F, 1.0e-15), 1.0 - 1.0e-15)
        return float(math.exp(self.mu_lnX + self.sigma_lnX * norm.ppf(F_safe)))

    def pdf(self, x: float) -> float:
        if x <= 0.0:
            return 0.0
        z = (math.log(x) - self.mu_lnX) / self.sigma_lnX
        return float(math.exp(-0.5 * z * z)
                     / (x * self.sigma_lnX * math.sqrt(2.0 * math.pi)))

    def mean(self) -> float:
        return float(self.mu_X)

    def std(self) -> float:
        return float(self.sigma_X)


# ============================================================ Uniform

@dataclass
class Uniform(RandomVariable):
    """Uniform on ``[a, b]``."""

    a: float
    b: float

    def __post_init__(self) -> None:
        if self.b <= self.a:
            raise ValueError("Uniform requires b > a")

    def cdf(self, x: float) -> float:
        if x <= self.a:
            return 0.0
        if x >= self.b:
            return 1.0
        return float((x - self.a) / (self.b - self.a))

    def inv_cdf(self, F: float) -> float:
        F = min(max(F, 0.0), 1.0)
        return float(self.a + F * (self.b - self.a))

    def pdf(self, x: float) -> float:
        if self.a <= x <= self.b:
            return float(1.0 / (self.b - self.a))
        return 0.0

    def mean(self) -> float:
        return float(0.5 * (self.a + self.b))

    def std(self) -> float:
        return float((self.b - self.a) / math.sqrt(12.0))


# ============================================================ Gumbel (EV-I max)

@dataclass
class Gumbel(RandomVariable):
    """Gumbel distribution (largest extreme value, EV-I).

    ``F(x) = exp(-exp(-(x - mu) / beta))``.

    Parameters
    ----------
    mu : float
        Location parameter.
    beta : float
        Scale parameter (> 0).
    """

    mu: float
    beta: float

    def __post_init__(self) -> None:
        if self.beta <= 0.0:
            raise ValueError("Gumbel beta must be > 0")

    _EULER_GAMMA = 0.5772156649015329

    def cdf(self, x: float) -> float:
        return float(math.exp(-math.exp(-(x - self.mu) / self.beta)))

    def inv_cdf(self, F: float) -> float:
        F_safe = min(max(F, 1.0e-15), 1.0 - 1.0e-15)
        return float(self.mu - self.beta * math.log(-math.log(F_safe)))

    def pdf(self, x: float) -> float:
        z = (x - self.mu) / self.beta
        return float(math.exp(-z - math.exp(-z)) / self.beta)

    def mean(self) -> float:
        return float(self.mu + self.beta * self._EULER_GAMMA)

    def std(self) -> float:
        return float(self.beta * math.pi / math.sqrt(6.0))


# ============================================================ Weibull

@dataclass
class Weibull(RandomVariable):
    """Two-parameter Weibull distribution.

    ``F(x) = 1 - exp(-(x / lam)^k)``  for ``x >= 0``.
    """

    k: float
    lam: float

    def __post_init__(self) -> None:
        if self.k <= 0.0 or self.lam <= 0.0:
            raise ValueError("Weibull k and lam must be > 0")

    def cdf(self, x: float) -> float:
        if x <= 0.0:
            return 0.0
        return float(1.0 - math.exp(-(x / self.lam) ** self.k))

    def inv_cdf(self, F: float) -> float:
        F = min(max(F, 0.0), 1.0 - 1e-15)
        return float(self.lam * (-math.log(1.0 - F)) ** (1.0 / self.k))

    def pdf(self, x: float) -> float:
        if x <= 0.0:
            return 0.0
        z = x / self.lam
        return float((self.k / self.lam) * z ** (self.k - 1.0)
                     * math.exp(-z ** self.k))

    def mean(self) -> float:
        # mu = lam * Gamma(1 + 1/k)
        return float(self.lam * math.gamma(1.0 + 1.0 / self.k))

    def std(self) -> float:
        a = math.gamma(1.0 + 2.0 / self.k)
        b = math.gamma(1.0 + 1.0 / self.k)
        return float(self.lam * math.sqrt(a - b * b))


# ============================================================ multivariate

class RandomVariableVector:
    """A vector of marginal random variables, optionally correlated.

    Uses the **Nataf transformation** (Gaussian-copula approximation)
    when a correlation matrix is supplied: each marginal is first
    mapped to standard normal via its own Rosenblatt step, then the
    inverse Cholesky factor of the (Nataf-corrected) U-space
    correlation matrix is applied. For weak correlations the Nataf
    correction is small, so we keep the user-supplied correlation in
    X-space and use it directly in U-space as a serviceable
    approximation (Ditlevsen & Madsen 1996 §7.3 -- this is the form
    used in OpenTURNS' "Nataf-iterative" simplification).
    """

    def __init__(
        self,
        marginals: list,
        *,
        correlation: np.ndarray | None = None,
    ):
        self.marginals = list(marginals)
        n = len(self.marginals)
        if correlation is None:
            self.R = np.eye(n)
        else:
            R = np.asarray(correlation, dtype=float)
            if R.shape != (n, n):
                raise ValueError(
                    f"correlation must be ({n}, {n}); got {R.shape}"
                )
            if not np.allclose(R, R.T):
                raise ValueError("correlation matrix must be symmetric")
            self.R = R
        # Lower-Cholesky of the correlation
        try:
            self.L = np.linalg.cholesky(self.R)
        except np.linalg.LinAlgError as exc:
            raise ValueError(
                f"correlation matrix must be positive definite: {exc}"
            ) from exc

    def __len__(self) -> int:
        return len(self.marginals)

    def transform_to_U(self, x: np.ndarray) -> np.ndarray:
        """Map ``x`` (real values) to standard normal ``u`` per
        Rosenblatt + Cholesky decorrelation."""
        x = np.asarray(x, dtype=float).ravel()
        # First, individual Rosenblatt
        u_corr = np.array([
            self.marginals[i].transform_to_U(float(x[i]))
            for i in range(len(self))
        ])
        # Decorrelate: u = L^{-1} u_corr
        return np.linalg.solve(self.L, u_corr)

    def transform_to_X(self, u: np.ndarray) -> np.ndarray:
        """Inverse Rosenblatt: ``u`` (standard normal, decorrelated)
        -> ``x`` (real values)."""
        u = np.asarray(u, dtype=float).ravel()
        # Re-correlate: u_corr = L u
        u_corr = self.L @ u
        return np.array([
            self.marginals[i].transform_to_X(float(u_corr[i]))
            for i in range(len(self))
        ])

    def means(self) -> np.ndarray:
        return np.array([rv.mean() for rv in self.marginals])

    def stds(self) -> np.ndarray:
        return np.array([rv.std() for rv in self.marginals])
