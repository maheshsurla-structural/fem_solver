"""J2 (von Mises) plasticity in 3-D with isotropic hardening.

The classical radial-return formulation for J2 plasticity:

    f(sigma, alpha) = ||s|| - sqrt(2/3) * sigma_y(alpha)

where ``s = dev(sigma)`` is the stress deviator and
``sigma_y(alpha) = sigma_y_0 + K_iso * alpha`` is the isotropic-
hardening yield stress with ``alpha`` the equivalent plastic strain
(``alpha = integral sqrt(2/3) * ||dot eps_p||``).

The trial-elastic predictor + radial-return corrector update is:

1. ``eps_e_trial = eps - eps_p_n``
2. ``sigma_trial = D_elastic * eps_e_trial``
3. ``s_trial = dev(sigma_trial)``, ``||s||_trial = sqrt(s:s)``
4. ``f_trial = ||s||_trial - sqrt(2/3) * sigma_y(alpha_n)``
5. If ``f_trial <= 0``: elastic step, no plastic update.
6. Else: plastic step,

       n = s_trial / ||s||_trial                    (unit deviator direction)
       d_lambda = f_trial / (2 G + (2/3) K_iso)
       sigma_n+1 = sigma_trial - 2 G d_lambda * n
       eps_p_n+1 = eps_p_n + d_lambda * n   (with x2 on shear components for
                                              engineering Voigt strain)
       alpha_n+1 = alpha_n + sqrt(2/3) d_lambda

The returned tangent is the **elastic** ``D``; Newton convergence
will then be linear (rather than quadratic). The consistent
*algorithmic* tangent that recovers quadratic Newton convergence is
a documented future refinement.

Voigt convention
----------------
Stress and strain are stored as 6-vectors in the OpenSees / standard-
FEM ordering ``(xx, yy, zz, xy, yz, zx)`` with **engineering** shear
strains (gamma = 2 * eps_tensor on shear components). The elastic D
matrix matches ``ElasticIsotropic.D_3d()``.
"""
from __future__ import annotations

import copy

import numpy as np


class J2Plasticity3D:
    """3-D J2 plasticity with isotropic hardening.

    Parameters
    ----------
    E : float
        Young's modulus.
    nu : float
        Poisson's ratio.
    sigma_y : float
        Initial yield stress (positive).
    K_iso : float, default 0.0
        Isotropic hardening modulus. ``K_iso = 0`` is perfectly
        plastic; positive values give a linearly-hardening yield
        surface ``sigma_y_n = sigma_y_0 + K_iso * alpha``.

    Notes
    -----
    State variables (committed + trial):

    * ``eps_p`` : Voigt 6-vector, plastic strain.
    * ``alpha`` : scalar, equivalent plastic strain.
    * ``sigma`` : Voigt 6-vector, last-computed stress (for output).

    The material exposes :meth:`get_response` returning ``(sigma_voigt,
    D_6x6)``. The returned D is the elastic constitutive matrix; the
    algorithmic / consistent tangent is a documented future refinement.
    """

    def __init__(self, E: float, nu: float, sigma_y: float,
                 K_iso: float = 0.0):
        if E <= 0.0:
            raise ValueError(f"E must be positive, got {E}")
        if not (-1.0 < nu < 0.5):
            raise ValueError(f"nu must be in (-1, 0.5), got {nu}")
        if sigma_y <= 0.0:
            raise ValueError(f"sigma_y must be positive, got {sigma_y}")
        if K_iso < 0.0:
            raise ValueError(f"K_iso must be non-negative, got {K_iso}")
        self.E = float(E)
        self.nu = float(nu)
        self.sigma_y_0 = float(sigma_y)
        self.K_iso = float(K_iso)
        # Derived elastic constants
        self.G = E / (2.0 * (1.0 + nu))
        self.K_bulk = E / (3.0 * (1.0 - 2.0 * nu))
        self._D_elastic = self._build_D_elastic()
        # ---- state ----
        self.eps_p_committed = np.zeros(6)
        self.alpha_committed = 0.0
        self.sigma_committed = np.zeros(6)
        self.eps_p_trial = np.zeros(6)
        self.alpha_trial = 0.0
        self.sigma_trial = np.zeros(6)

    def _build_D_elastic(self) -> np.ndarray:
        E, nu = self.E, self.nu
        lam = E * nu / ((1.0 + nu) * (1.0 - 2.0 * nu))
        mu = self.G
        D = np.zeros((6, 6))
        D[0, 0] = D[1, 1] = D[2, 2] = lam + 2.0 * mu
        D[0, 1] = D[0, 2] = D[1, 0] = D[1, 2] = D[2, 0] = D[2, 1] = lam
        D[3, 3] = D[4, 4] = D[5, 5] = mu
        return D

    def D_elastic(self) -> np.ndarray:
        return self._D_elastic.copy()

    @staticmethod
    def _algorithmic_tangent(G: float, K_bulk: float, H_iso: float,
                              n_voigt: np.ndarray, norm_s: float,
                              d_lambda: float) -> np.ndarray:
        """Consistent algorithmic tangent for J2 + isotropic hardening
        in Voigt form with engineering shear strain.

            C_alg = K * E1 + 2 mu * theta * D_dev
                    + 2 mu * (1 - theta - theta_n) * (n_voigt outer n_voigt)

        where ``theta = 1 - 2 mu d_lambda / ||s||`` and
        ``theta_n = 1 / (1 + H/(3 mu))``. Cross-check at the
        perfectly-plastic limit (H = 0, theta_n = 1): strain along the
        deviator direction n produces zero stress increment, as
        expected.
        """
        theta = 1.0 - 2.0 * G * d_lambda / norm_s
        theta_n = 1.0 / (1.0 + H_iso / (3.0 * G))
        # E1: 1-tensor-outer-1-tensor (3x3 block of ones)
        E1 = np.zeros((6, 6))
        E1[0:3, 0:3] = 1.0
        # I4 = diag(1, 1, 1, 1/2, 1/2, 1/2) -- 4th-order identity in
        # Voigt with engineering shears (factor 1/2 on shear diagonals
        # makes 2 mu I4 @ eps_voigt give the standard tensor stress
        # for both normals and shears).
        I4 = np.diag([1.0, 1.0, 1.0, 0.5, 0.5, 0.5])
        D_dev = I4 - (1.0 / 3.0) * E1
        NN = np.outer(n_voigt, n_voigt)
        return (K_bulk * E1
                + 2.0 * G * theta * D_dev
                + 2.0 * G * (1.0 - theta - theta_n) * NN)

    # ----------------------------------------------------- helpers
    @staticmethod
    def _deviator(sigma_voigt: np.ndarray) -> tuple[np.ndarray, float]:
        """Return ``(s_voigt, p)`` where ``s`` is the deviator and ``p``
        the (negative-)hydrostatic pressure."""
        p = (sigma_voigt[0] + sigma_voigt[1] + sigma_voigt[2]) / 3.0
        s = sigma_voigt.copy()
        s[0] -= p; s[1] -= p; s[2] -= p
        return s, p

    @staticmethod
    def _voigt_double_dot(s_voigt: np.ndarray) -> float:
        """s:s where ``s`` is stored in Voigt with shears unscaled."""
        return float(
            s_voigt[0] ** 2 + s_voigt[1] ** 2 + s_voigt[2] ** 2
            + 2.0 * (s_voigt[3] ** 2 + s_voigt[4] ** 2 + s_voigt[5] ** 2)
        )

    def yield_stress(self, alpha: float) -> float:
        return self.sigma_y_0 + self.K_iso * alpha

    def von_mises_stress(self, sigma_voigt: np.ndarray) -> float:
        s, _ = self._deviator(sigma_voigt)
        return float(np.sqrt(1.5 * self._voigt_double_dot(s)))

    # ----------------------------------------------------- response
    def get_response(self, eps_voigt) -> tuple[np.ndarray, np.ndarray]:
        """Radial-return mapping from total strain to stress.

        Returns ``(sigma, D)`` where ``D`` is the *elastic* tangent.
        The returned ``sigma`` is on the yield surface after plastic
        loading; the algorithmic-consistent tangent is a future
        refinement (Phase 16.7.x).
        """
        eps = np.asarray(eps_voigt, dtype=float).reshape(6)
        eps_e_trial = eps - self.eps_p_committed
        sigma_trial = self._D_elastic @ eps_e_trial
        s_trial, _ = self._deviator(sigma_trial)
        norm_s_sq = self._voigt_double_dot(s_trial)
        norm_s = float(np.sqrt(norm_s_sq))
        sqrt_2_3 = np.sqrt(2.0 / 3.0)
        sigma_y_n = self.yield_stress(self.alpha_committed)
        f_trial = norm_s - sqrt_2_3 * sigma_y_n
        if f_trial <= 0.0 or norm_s < 1.0e-30:
            self.sigma_trial = sigma_trial
            self.eps_p_trial = self.eps_p_committed.copy()
            self.alpha_trial = self.alpha_committed
            return sigma_trial.copy(), self._D_elastic.copy()
        # Plastic step: radial return
        n_voigt = s_trial / norm_s
        d_lambda = f_trial / (2.0 * self.G + (2.0 / 3.0) * self.K_iso)
        sigma_new = sigma_trial - 2.0 * self.G * d_lambda * n_voigt
        # Plastic strain update (engineering shear -> 2x on shear components)
        eps_p_new = self.eps_p_committed.copy()
        eps_p_new[0:3] += d_lambda * n_voigt[0:3]
        eps_p_new[3:6] += 2.0 * d_lambda * n_voigt[3:6]
        alpha_new = self.alpha_committed + sqrt_2_3 * d_lambda
        self.sigma_trial = sigma_new
        self.eps_p_trial = eps_p_new
        self.alpha_trial = alpha_new
        # Consistent algorithmic tangent (gives quadratic Newton)
        D_alg = self._algorithmic_tangent(
            self.G, self.K_bulk, self.K_iso,
            n_voigt, norm_s, d_lambda,
        )
        return sigma_new.copy(), D_alg

    # ----------------------------------------------------- state
    def commit_state(self) -> None:
        self.eps_p_committed = self.eps_p_trial.copy()
        self.alpha_committed = self.alpha_trial
        self.sigma_committed = self.sigma_trial.copy()

    def revert_state(self) -> None:
        self.eps_p_trial = self.eps_p_committed.copy()
        self.alpha_trial = self.alpha_committed
        self.sigma_trial = self.sigma_committed.copy()

    def clone(self) -> "J2Plasticity3D":
        """Deep copy with independent state."""
        return copy.deepcopy(self)

    def __repr__(self) -> str:
        return (
            f"J2Plasticity3D(E={self.E:g}, nu={self.nu:g}, "
            f"sigma_y={self.sigma_y_0:g}, K_iso={self.K_iso:g})"
        )
