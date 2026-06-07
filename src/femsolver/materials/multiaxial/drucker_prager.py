"""Drucker-Prager plasticity in 3-D for soils.

The Drucker-Prager (DP) yield function is

    f(sigma) = alpha * I1 + sqrt(J2) - k

where ``I1 = trace(sigma)`` and ``J2 = (1/2) s : s`` is the second
deviatoric invariant. The two parameters ``alpha`` and ``k`` are
related to the cohesion ``c`` and friction angle ``phi`` of a
Mohr-Coulomb material by an outer-cone (or inner-cone) projection:

    alpha = 2 sin(phi) / (sqrt(3) (3 - sin(phi)))
    k     = 6 c cos(phi) / (sqrt(3) (3 - sin(phi)))   # outer cone

(Inner cone uses ``(3 + sin(phi))`` in the denominator.) For
phi = 0 (purely cohesive soil) DP reduces to a J2-type model with
``k = 2 c / sqrt(3)``.

This implementation uses **associated** flow on the smooth cone face.
The apex region (where ``J2_n+1`` would become negative under the
return) is handled by projecting onto the cone tip ``I1 = k/alpha,
J2 = 0`` -- the standard cap that prevents non-physical states.

Sign convention
---------------
Standard solid-mechanics convention: tension positive, compression
negative for both stress and strain. Voigt order matches the rest
of the library: ``(xx, yy, zz, xy, yz, zx)``, with engineering shear
strain.
"""
from __future__ import annotations

import copy
import math

import numpy as np


class DruckerPrager3D:
    """3-D Drucker-Prager perfectly-plastic material.

    Parameters
    ----------
    E : float
        Young's modulus.
    nu : float
        Poisson's ratio.
    alpha : float
        DP friction-angle parameter (positive). For ``alpha = 0``,
        the model reduces to a J2-type cohesive material with no
        pressure dependence.
    k : float
        DP cohesion-strength parameter (positive).
    """

    def __init__(self, E: float, nu: float, alpha: float, k: float):
        if E <= 0.0:
            raise ValueError(f"E must be positive, got {E}")
        if not (-1.0 < nu < 0.5):
            raise ValueError(f"nu must be in (-1, 0.5), got {nu}")
        if alpha < 0.0:
            raise ValueError(f"alpha must be >= 0, got {alpha}")
        if k <= 0.0:
            raise ValueError(f"k must be positive, got {k}")
        self.E = float(E)
        self.nu = float(nu)
        self.alpha = float(alpha)
        self.k = float(k)
        # Derived elastic constants
        self.G = E / (2.0 * (1.0 + nu))
        self.K_bulk = E / (3.0 * (1.0 - 2.0 * nu))
        self._D_elastic = self._build_D_elastic()
        # ---- state ----
        self.eps_p_committed = np.zeros(6)
        self.sigma_committed = np.zeros(6)
        self.eps_p_trial = np.zeros(6)
        self.sigma_trial = np.zeros(6)

    @classmethod
    def from_mohr_coulomb(cls, E: float, nu: float, cohesion: float,
                          phi_deg: float, *,
                          outer_cone: bool = True) -> "DruckerPrager3D":
        """Build DP parameters by matching a Mohr-Coulomb material.

        Parameters
        ----------
        cohesion : float
            Mohr-Coulomb cohesion (positive).
        phi_deg : float
            Friction angle in degrees (0 <= phi < 90).
        outer_cone : bool, default True
            ``True`` -> outer-cone (circumscribed) match: aggressive,
            but matches axisymmetric MC compression triaxial. ``False``
            -> inner-cone (inscribed) match: conservative, matches MC
            extension triaxial.
        """
        if not (0.0 <= phi_deg < 90.0):
            raise ValueError(f"phi_deg must be in [0, 90), got {phi_deg}")
        if cohesion <= 0.0:
            raise ValueError(f"cohesion must be positive, got {cohesion}")
        phi = math.radians(phi_deg)
        sp = math.sin(phi); cp = math.cos(phi)
        if outer_cone:
            denom = math.sqrt(3.0) * (3.0 - sp)
        else:
            denom = math.sqrt(3.0) * (3.0 + sp)
        alpha = 2.0 * sp / denom
        k = 6.0 * cohesion * cp / denom
        return cls(E=E, nu=nu, alpha=alpha, k=k)

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

    def _continuum_tangent(self, s_voigt: np.ndarray,
                             sqrt_J2: float) -> np.ndarray:
        """Continuum elasto-plastic tangent on the smooth cone face.

            C_ep = D_e - (D_e r) (D_e r)^T / (r . D_e . r)

        with the associated flow direction ``r = alpha I + s / (2 sqrt(J2))``
        and the scalar ``r . D_e . r = 9 K alpha^2 + G``. The outer
        product ``(D_e r) (D_e r)^T`` works in Voigt with engineering
        shear because both factors are stress-like vectors.

        Returns the symmetric 6x6 elasto-plastic tangent.
        """
        G = self.G
        K = self.K_bulk
        alpha = self.alpha
        # D_e r in Voigt form: volumetric part 3K alpha I + deviatoric
        # part G s / sqrt(J2). For pure shear J2 = ||s||^2/2 so sqrt(J2)
        # may be small near origin -- caller has ensured sqrt_J2 > 0.
        a_voigt = np.zeros(6)
        a_voigt[0:3] = 3.0 * K * alpha
        a_voigt += (G / sqrt_J2) * s_voigt
        denom = 9.0 * K * alpha * alpha + G
        return self._D_elastic - np.outer(a_voigt, a_voigt) / denom

    # ----------------------------------------------------- helpers
    @staticmethod
    def _deviator(sigma: np.ndarray) -> tuple[np.ndarray, float]:
        p = (sigma[0] + sigma[1] + sigma[2]) / 3.0
        s = sigma.copy()
        s[0] -= p; s[1] -= p; s[2] -= p
        return s, p

    @staticmethod
    def _voigt_double_dot(s: np.ndarray) -> float:
        return float(
            s[0] ** 2 + s[1] ** 2 + s[2] ** 2
            + 2.0 * (s[3] ** 2 + s[4] ** 2 + s[5] ** 2)
        )

    def yield_function(self, sigma: np.ndarray) -> float:
        """``f = alpha I1 + sqrt(J2) - k``. ``f <= 0`` means inside the
        yield cone."""
        I1 = sigma[0] + sigma[1] + sigma[2]
        s, _ = self._deviator(sigma)
        J2 = 0.5 * self._voigt_double_dot(s)
        return self.alpha * I1 + math.sqrt(max(J2, 0.0)) - self.k

    # ----------------------------------------------------- response
    def get_response(self, eps_voigt) -> tuple[np.ndarray, np.ndarray]:
        """Drucker-Prager radial return.

        Returns ``(sigma, D_elastic)`` after radial-return mapping of
        the trial stress back to the smooth cone face. The apex case
        (``sqrt(J2_n+1) < 0``) projects onto the cone tip.
        """
        eps = np.asarray(eps_voigt, dtype=float).reshape(6)
        eps_e_trial = eps - self.eps_p_committed
        sigma_trial = self._D_elastic @ eps_e_trial
        I1_trial = sigma_trial[0] + sigma_trial[1] + sigma_trial[2]
        s_trial, _ = self._deviator(sigma_trial)
        norm_s_sq = self._voigt_double_dot(s_trial)
        sqrt_J2 = math.sqrt(0.5 * norm_s_sq)
        f_trial = self.alpha * I1_trial + sqrt_J2 - self.k

        if f_trial <= 0.0:
            self.sigma_trial = sigma_trial
            self.eps_p_trial = self.eps_p_committed.copy()
            return sigma_trial.copy(), self._D_elastic.copy()

        # Plastic step: smooth-cone return.
        G = self.G
        K = self.K_bulk
        alpha = self.alpha
        d_lambda = f_trial / (G + 9.0 * K * alpha * alpha)
        sqrt_J2_new = sqrt_J2 - G * d_lambda
        if sqrt_J2_new < 0.0:
            # Apex return: collapse deviator to zero and project I1 to k/alpha.
            d_lambda = sqrt_J2 / G        # consume deviator
            # Apex return: sigma = (k / (3 alpha)) * I_voigt
            sigma_new = np.zeros(6)
            apex_p = self.k / (3.0 * alpha) if alpha > 0.0 else 0.0
            sigma_new[0:3] = apex_p
            # eps_p update: tricky -- effectively all strain becomes plastic.
            # For simplicity, accumulate eps_p approximately using the
            # smooth-cone formula with the consumed d_lambda for the
            # deviator + the remaining volumetric demand:
            # We do a 2-stage update.
            # Stage 1: smooth-cone consumption (d_lambda from above).
            s_unit_factor = 1.0 / max(sqrt_J2, 1.0e-30)
            eps_p_inc = np.zeros(6)
            eps_p_inc[0:3] = d_lambda * (alpha + s_trial[0:3] / (2.0 * sqrt_J2))
            eps_p_inc[3:6] = d_lambda * s_trial[3:6] / sqrt_J2
            # Stage 2: extra volumetric consumption to land at apex
            # The extra volumetric demand: I1_target = k/alpha;
            # current I1 after smooth-cone = I1_trial - 9 K alpha d_lambda.
            I1_smooth = I1_trial - 9.0 * K * alpha * d_lambda
            apex_I1 = self.k / alpha
            extra_I1 = I1_smooth - apex_I1
            # The extra plastic volumetric strain to absorb is extra_I1/(3K)
            extra_e_v = extra_I1 / (3.0 * K)
            eps_p_inc[0:3] += extra_e_v / 3.0
            self.eps_p_trial = self.eps_p_committed + eps_p_inc
            self.sigma_trial = sigma_new
            return sigma_new.copy(), self._D_elastic.copy()

        # Standard smooth-cone return
        # sigma_new = sigma_trial - d_lambda * (3 K alpha I + G s_trial / sqrt_J2)
        sigma_new = sigma_trial.copy()
        # Volumetric component
        delta_p = -3.0 * K * alpha * d_lambda
        sigma_new[0:3] += delta_p
        # Deviatoric component
        ds_factor = -G * d_lambda / sqrt_J2
        sigma_new[0:3] += ds_factor * s_trial[0:3]
        sigma_new[3:6] += ds_factor * s_trial[3:6]

        # Plastic strain increment (associated flow)
        eps_p_inc = np.zeros(6)
        eps_p_inc[0:3] = d_lambda * (alpha + s_trial[0:3] / (2.0 * sqrt_J2))
        # Engineering shear: factor of 2 on shears for plastic strain
        eps_p_inc[3:6] = d_lambda * s_trial[3:6] / sqrt_J2
        self.eps_p_trial = self.eps_p_committed + eps_p_inc
        self.sigma_trial = sigma_new
        # Continuum elasto-plastic tangent on the smooth cone face.
        # (Algorithmic / consistent tangent that gives quadratic Newton
        # convergence requires extra terms accounting for the
        # volumetric-deviatoric coupling of finite-step return -- a
        # future refinement.)
        D_ep = self._continuum_tangent(s_trial, sqrt_J2)
        return sigma_new.copy(), D_ep

    # ----------------------------------------------------- state
    def commit_state(self) -> None:
        self.eps_p_committed = self.eps_p_trial.copy()
        self.sigma_committed = self.sigma_trial.copy()

    def revert_state(self) -> None:
        self.eps_p_trial = self.eps_p_committed.copy()
        self.sigma_trial = self.sigma_committed.copy()

    def clone(self) -> "DruckerPrager3D":
        return copy.deepcopy(self)

    def __repr__(self) -> str:
        return (
            f"DruckerPrager3D(E={self.E:g}, nu={self.nu:g}, "
            f"alpha={self.alpha:g}, k={self.k:g})"
        )
