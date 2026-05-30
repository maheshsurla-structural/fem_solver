"""First-Order Reliability Method (FORM) via the HLRF iteration.

For a limit-state function ``g(X)`` (with ``g <= 0`` defining failure),
FORM transforms the random vector ``X`` to standard normal ``U`` and
finds the **design point** ``U*`` -- the point on the limit-state
surface ``g(U) = 0`` closest to the origin. The **reliability index**
``beta = ||U*||`` is then the distance from the origin to the
linearised limit-state hyperplane, and the failure probability is
approximated as

    P_f ≈ Φ(-β).

The classic Hasofer-Lind-Rackwitz-Fiessler (HLRF) iteration:

    α = -∇g(U) / ||∇g(U)||
    U_new = (α · U + g(U) / ||∇g(U)||) · α

starts at the origin (or the mean point in X-space) and converges
quadratically to ``U*`` for well-behaved problems. The gradient
``∇g(U)`` is computed by chain rule from ``∇g(X)`` and the Jacobian
of the X→U transformation.

This module exposes :func:`form_hlrf`, which takes:

* a limit-state callable ``g(X) -> float``;
* an analytical gradient ``grad_g(X) -> array`` or finite-difference
  fallback;
* the :class:`~femsolver.reliability.rv.RandomVariableVector`.

References
----------
* Ditlevsen, O. & Madsen, H.O. (1996). *Structural Reliability
  Methods*. Wiley.
* Rackwitz, R. & Fiessler, B. (1978). "Structural reliability under
  combined random load sequences." *Computers & Structures*, 9(5).
"""
from __future__ import annotations

from dataclasses import dataclass
from typing import Callable

import numpy as np
from scipy.stats import norm


# ============================================================ result

@dataclass
class FORMResult:
    """Outcome of a FORM analysis.

    Attributes
    ----------
    beta : float
        Reliability index.
    pf : float
        Probability of failure ``P_f ≈ Phi(-beta)``.
    u_star : np.ndarray
        Design point in standard-normal U-space.
    x_star : np.ndarray
        Design point in real X-space.
    alpha : np.ndarray
        Direction-cosine vector (importance factors of the variables).
    n_iter : int
        Number of HLRF iterations to convergence.
    converged : bool
    g_at_design_point : float
        Should be ≈ 0 if converged.
    """

    beta: float
    pf: float
    u_star: np.ndarray
    x_star: np.ndarray
    alpha: np.ndarray
    n_iter: int
    converged: bool
    g_at_design_point: float


# ============================================================ HLRF

def _finite_difference_grad(g, X, h_rel=1.0e-6):
    """Central-difference gradient of g at X."""
    X = np.asarray(X, dtype=float).ravel()
    g_x = g(X)
    n = X.size
    grad = np.zeros(n)
    for i in range(n):
        h = max(abs(X[i]), 1.0) * h_rel
        Xp = X.copy(); Xp[i] += h
        Xm = X.copy(); Xm[i] -= h
        grad[i] = (g(Xp) - g(Xm)) / (2.0 * h)
    return grad


def _grad_g_in_U(g, grad_g, rvs, U, h_rel=1.0e-6):
    """Gradient of g(X(U)) in U-space.

    Uses the chain rule ``∇_U g = J^T ∇_X g`` where
    ``J_{ij} = ∂X_i / ∂U_j`` is approximated by finite differences
    (so it works for any RandomVariableVector, including ones with
    correlated Nataf transformations).
    """
    X = rvs.transform_to_X(U)
    if grad_g is not None:
        gX = np.asarray(grad_g(X), dtype=float).ravel()
    else:
        gX = _finite_difference_grad(g, X)
    n = U.size
    J = np.zeros((n, n))
    for j in range(n):
        h = 1.0e-6
        Up = U.copy(); Up[j] += h
        Um = U.copy(); Um[j] -= h
        Xp = rvs.transform_to_X(Up)
        Xm = rvs.transform_to_X(Um)
        J[:, j] = (Xp - Xm) / (2.0 * h)
    return J.T @ gX


def form_hlrf(
    *,
    g: Callable[[np.ndarray], float],
    rvs,
    grad_g: Callable[[np.ndarray], np.ndarray] | None = None,
    U0: np.ndarray | None = None,
    tol_g: float = 1.0e-6,
    tol_u: float = 1.0e-6,
    max_iter: int = 100,
    relaxation: float = 1.0,
) -> FORMResult:
    """Find the FORM design point via the HLRF iteration.

    Parameters
    ----------
    g : callable
        ``g(X) -> float``. Convention: ``g(X) <= 0`` defines failure.
    rvs : RandomVariableVector
        Joint distribution of ``X``.
    grad_g : callable, optional
        Analytical gradient ``∇g(X)`` in X-space. If omitted, central
        differences are used.
    U0 : array, optional
        Initial point in U-space. Defaults to the origin.
    tol_g : float, default 1e-6
        Tolerance on ``|g(X*)|`` (limit-state residual).
    tol_u : float, default 1e-6
        Tolerance on the U-space update norm.
    max_iter : int, default 100
    relaxation : float, default 1.0
        Damping factor on the HLRF update (try 0.5 if oscillating).
    """
    n = len(rvs)
    U = np.zeros(n) if U0 is None else np.asarray(U0, dtype=float).copy()
    converged = False
    for k in range(max_iter):
        X = rvs.transform_to_X(U)
        g_val = float(g(X))
        gradU = _grad_g_in_U(g, grad_g, rvs, U)
        norm_grad = float(np.linalg.norm(gradU))
        if norm_grad < 1.0e-30:
            break
        alpha = -gradU / norm_grad
        U_new = (alpha @ U + g_val / norm_grad) * alpha
        # Damping
        U_new = U + relaxation * (U_new - U)
        if (abs(g_val) < tol_g
                and np.linalg.norm(U_new - U) < tol_u):
            U = U_new
            converged = True
            break
        U = U_new
    X_star = rvs.transform_to_X(U)
    g_at = float(g(X_star))
    gradU = _grad_g_in_U(g, grad_g, rvs, U)
    norm_grad = float(np.linalg.norm(gradU))
    alpha = -gradU / norm_grad if norm_grad > 0 else np.zeros_like(gradU)
    beta = float(np.linalg.norm(U))
    # Sign: beta is positive iff origin is in the safe region (g > 0
    # at U=0); negative if mean point is already failed.
    if float(g(rvs.transform_to_X(np.zeros(n)))) < 0:
        beta = -beta
    pf = float(norm.cdf(-beta))
    return FORMResult(
        beta=beta, pf=pf,
        u_star=U, x_star=X_star, alpha=alpha,
        n_iter=k + 1, converged=converged,
        g_at_design_point=g_at,
    )
