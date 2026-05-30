"""Second-Order Reliability Method (SORM) -- Breitung's correction.

FORM linearises the limit-state surface ``g(U) = 0`` at the design
point and uses ``P_f ≈ Φ(-β)``. SORM refines this by fitting a
**quadratic** surface at the design point and applying Breitung's
1984 asymptotic formula::

    P_f ≈ Φ(-β) · ∏_{i=1}^{n-1} (1 - β · κ_i)^{-1/2}

where ``κ_i`` are the **principal curvatures** of the limit-state
surface at the design point (with respect to the tangent hyperplane
to the unit normal ``α``).

Curvatures are obtained from the Hessian of ``g(U)`` at ``U*``: project
the Hessian onto the (n-1)-dimensional tangent space (orthogonal to
``α``), normalise by ``||∇g||``, then take eigenvalues.

The improvement over FORM is usually modest for slowly-curving
surfaces and substantial for highly-curved ones. For the simplest
case of a linear limit state, the SORM correction vanishes and
``P_f^SORM = P_f^FORM``.
"""
from __future__ import annotations

from dataclasses import dataclass
from typing import Callable

import numpy as np
from scipy.stats import norm

from femsolver.reliability.form import FORMResult


# ============================================================ result

@dataclass
class SORMResult:
    """Outcome of a SORM analysis.

    Attributes
    ----------
    beta_FORM : float
        Reliability index from the underlying FORM.
    pf_FORM : float
        Φ(-β).
    pf_SORM : float
        Breitung-corrected failure probability.
    beta_SORM : float
        ``-Φ^{-1}(pf_SORM)`` -- equivalent reliability index.
    kappa : np.ndarray
        Principal curvatures at the design point.
    """

    beta_FORM: float
    pf_FORM: float
    pf_SORM: float
    beta_SORM: float
    kappa: np.ndarray


# ============================================================ Hessian in U

def _hessian_g_in_U(g, rvs, U, h: float = 1.0e-4) -> np.ndarray:
    """Finite-difference Hessian of ``g(X(U))`` at ``U``."""
    n = U.size
    H = np.zeros((n, n))

    def gU(Uvec):
        return float(g(rvs.transform_to_X(Uvec)))

    g0 = gU(U)
    # Diagonal: (g(U + h e_i) - 2 g + g(U - h e_i)) / h^2
    for i in range(n):
        Ui_p = U.copy(); Ui_p[i] += h
        Ui_m = U.copy(); Ui_m[i] -= h
        H[i, i] = (gU(Ui_p) - 2.0 * g0 + gU(Ui_m)) / (h * h)
    # Off-diagonal (symmetric)
    for i in range(n):
        for j in range(i + 1, n):
            U_pp = U.copy(); U_pp[i] += h; U_pp[j] += h
            U_pm = U.copy(); U_pm[i] += h; U_pm[j] -= h
            U_mp = U.copy(); U_mp[i] -= h; U_mp[j] += h
            U_mm = U.copy(); U_mm[i] -= h; U_mm[j] -= h
            H[i, j] = (gU(U_pp) - gU(U_pm) - gU(U_mp) + gU(U_mm)) \
                       / (4.0 * h * h)
            H[j, i] = H[i, j]
    return H


# ============================================================ SORM

def sorm_breitung(
    *,
    form_result: FORMResult,
    g: Callable[[np.ndarray], float],
    rvs,
    h_hessian: float = 1.0e-4,
) -> SORMResult:
    """Apply Breitung's curvature correction to a FORM result.

    Parameters
    ----------
    form_result : FORMResult
        From :func:`femsolver.reliability.form.form_hlrf`.
    g : callable
        ``g(X) -> float`` (same as the FORM run).
    rvs : RandomVariableVector
    h_hessian : float, default 1e-4
        Finite-difference step for the Hessian.
    """
    beta = form_result.beta
    pf_F = form_result.pf
    U_star = form_result.u_star
    alpha = form_result.alpha
    n = U_star.size
    if n < 2:
        # Single random variable: no curvature correction
        return SORMResult(
            beta_FORM=beta, pf_FORM=pf_F,
            pf_SORM=pf_F, beta_SORM=beta,
            kappa=np.zeros(0),
        )

    # Hessian of g in U-space at the design point
    H = _hessian_g_in_U(g, rvs, U_star, h=h_hessian)
    # Gradient norm at design point
    from femsolver.reliability.form import _grad_g_in_U
    gradU = _grad_g_in_U(g, None, rvs, U_star)
    norm_grad = float(np.linalg.norm(gradU))
    if norm_grad < 1.0e-30:
        return SORMResult(
            beta_FORM=beta, pf_FORM=pf_F,
            pf_SORM=pf_F, beta_SORM=beta,
            kappa=np.zeros(n - 1),
        )

    # Build an orthonormal basis for the tangent hyperplane (orthogonal
    # to alpha) -- a (n, n-1) matrix Q such that Q^T alpha = 0.
    # Householder reflection makes alpha the first basis vector of an
    # orthonormal frame.
    Q_full = np.eye(n) - 2.0 * np.outer(alpha, alpha)
    # Replace the column most aligned with alpha by alpha itself.
    # Simpler: SVD-based basis.
    # u, _, _ = np.linalg.svd(alpha.reshape(-1, 1))
    # Q = u[:, 1:]
    A = alpha.reshape(-1, 1)
    Q = np.eye(n) - A @ A.T          # projector orthogonal to alpha
    # Take any (n-1) orthonormal columns from Q via SVD
    U_svd, _, _ = np.linalg.svd(Q)
    T = U_svd[:, :n - 1]            # (n, n-1) tangent basis

    # Project Hessian onto tangent space and normalise
    A_curv = (T.T @ H @ T) / norm_grad
    # Symmetrise (numerical noise)
    A_curv = 0.5 * (A_curv + A_curv.T)
    kappa = np.sort(np.linalg.eigvalsh(A_curv))

    # Breitung formula
    abs_beta = abs(beta)
    prod = 1.0
    for k in kappa:
        term = 1.0 - abs_beta * k
        if term <= 0.0:
            # Curvature too large for asymptotic Breitung; fall back to
            # FORM with a warning sentinel
            return SORMResult(
                beta_FORM=beta, pf_FORM=pf_F,
                pf_SORM=pf_F, beta_SORM=beta,
                kappa=kappa,
            )
        prod *= 1.0 / np.sqrt(term)
    pf_S = float(pf_F * prod)
    # Equivalent reliability index
    beta_S = float(-norm.ppf(min(max(pf_S, 1.0e-15), 1.0 - 1.0e-15)))
    return SORMResult(
        beta_FORM=beta, pf_FORM=pf_F,
        pf_SORM=pf_S, beta_SORM=beta_S,
        kappa=kappa,
    )
