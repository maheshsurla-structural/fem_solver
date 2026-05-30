"""Modified Cam-Clay (MCC) plasticity for clays.

The Modified Cam-Clay model (Roscoe-Burland 1968) is the workhorse
critical-state model for clay soils. The yield surface in the
``(p', q)`` plane (mean effective stress, deviatoric stress) is an
ellipse passing through the origin and a *preconsolidation pressure*
``p_c``::

    f(p, q, p_c) = q^2 + M^2 * p * (p - p_c) = 0

with ``M`` the slope of the critical-state line. The hardening law
is volumetric::

    dp_c = p_c * (eps_v^p) / (lambda - kappa)

where ``lambda`` is the slope of the virgin compression line in
``(e, ln p')`` space and ``kappa`` the slope of the swelling line.

Behaviour
---------
* **OC clay** (p < p_c / 2 on the dry side): dilatant, peak strength
  above critical state, post-peak softening to critical state.
* **NC clay** (p > p_c / 2 on the wet side): contractant, hardens
  with continued shearing until reaching the critical-state line.

The implementation here uses an *implicit* return mapping with a
single scalar Newton iteration on the plastic multiplier and a
volumetric hardening update of ``p_c``. Tension positive sign
convention is kept (so soil compression has ``p > 0`` here, even
though geotech texts often invert the sign).

Sign convention
---------------
Tension positive throughout. For this model, ``p`` is therefore the
**negative** of the soil-mechanics mean effective stress. The
yield surface ``q^2 + M^2 * p * (p - p_c)`` is interpreted with
``p_c < 0`` (compressive preconsolidation -- the yield surface lives
in the p < 0 half-plane).
"""
from __future__ import annotations

import copy
import math

import numpy as np


def _deviator(sigma: np.ndarray) -> tuple[np.ndarray, float]:
    """Return ``(s, p)`` with ``p = trace/3`` and ``s = sigma - p I``."""
    p = (sigma[0] + sigma[1] + sigma[2]) / 3.0
    s = sigma.copy()
    s[0] -= p; s[1] -= p; s[2] -= p
    return s, p


def _von_mises_q(s: np.ndarray) -> float:
    """``q = sqrt(3/2 * s:s)`` in Voigt order (eng. shear)."""
    s2 = s[0] ** 2 + s[1] ** 2 + s[2] ** 2 \
         + 2.0 * (s[3] ** 2 + s[4] ** 2 + s[5] ** 2)
    return math.sqrt(1.5 * s2)


class ModifiedCamClay3D:
    """3-D Modified Cam-Clay material with volumetric hardening.

    Parameters
    ----------
    E : float
        (Initial) Young's modulus. The bulk modulus scales with ``p'``
        in the rigorous MCC; here we keep a constant elastic ``E`` for
        simplicity. Set ``E`` to its value at the in-situ stress.
    nu : float
        Poisson's ratio.
    M : float
        Slope of the critical-state line in ``(p, q)`` space.
        Typical: 0.9 to 1.5 depending on soil.
    lambda_cc : float
        Slope of the *virgin compression line* in ``(e, ln p)``.
        Typical: 0.05 to 0.30.
    kappa_cc : float
        Slope of the *swelling line* in ``(e, ln p)``. Typical: 0.005
        to 0.05; must be strictly less than ``lambda_cc``.
    p_c0 : float
        Initial preconsolidation pressure (positive number; the model
        uses ``-p_c0`` internally given the tension-positive sign).
    """

    def __init__(
        self,
        E: float,
        nu: float,
        M: float,
        lambda_cc: float,
        kappa_cc: float,
        p_c0: float,
    ):
        if E <= 0.0:
            raise ValueError(f"E must be positive, got {E}")
        if not (-1.0 < nu < 0.5):
            raise ValueError(f"nu must be in (-1, 0.5), got {nu}")
        if M <= 0.0:
            raise ValueError(f"M must be positive, got {M}")
        if lambda_cc <= 0.0:
            raise ValueError(f"lambda_cc must be positive, got {lambda_cc}")
        if not (0.0 < kappa_cc < lambda_cc):
            raise ValueError(
                f"kappa_cc must satisfy 0 < kappa_cc < lambda_cc, "
                f"got kappa_cc={kappa_cc}, lambda_cc={lambda_cc}"
            )
        if p_c0 <= 0.0:
            raise ValueError(f"p_c0 must be positive, got {p_c0}")
        self.E = float(E)
        self.nu = float(nu)
        self.M = float(M)
        self.lambda_cc = float(lambda_cc)
        self.kappa_cc = float(kappa_cc)
        self._p_c0 = float(p_c0)
        # Elastic constants
        self.G = E / (2.0 * (1.0 + nu))
        self.K_bulk = E / (3.0 * (1.0 - 2.0 * nu))
        self._lambda_lame = E * nu / ((1.0 + nu) * (1.0 - 2.0 * nu))
        self._D_elastic = self._build_D_elastic()
        # state: internal "p_c" stored as a positive number (consol.
        # pressure magnitude). Tension-positive sign means yield occurs
        # for sigma with negative p (compressive).
        self.p_c = float(p_c0)
        self.p_c_committed = float(p_c0)
        self.eps_p_committed = np.zeros(6)
        self.eps_p_trial = np.zeros(6)
        self.sigma_committed = np.zeros(6)
        self.sigma_trial = np.zeros(6)

    def _build_D_elastic(self) -> np.ndarray:
        lam = self._lambda_lame
        mu = self.G
        D = np.zeros((6, 6))
        D[0, 0] = D[1, 1] = D[2, 2] = lam + 2.0 * mu
        D[0, 1] = D[0, 2] = D[1, 0] = D[1, 2] = D[2, 0] = D[2, 1] = lam
        D[3, 3] = D[4, 4] = D[5, 5] = mu
        return D

    def D_elastic(self) -> np.ndarray:
        return self._D_elastic.copy()

    def yield_function(self, sigma: np.ndarray) -> float:
        """``f = q^2 + M^2 * p_eff * (p_eff + p_c)`` where ``p_eff =
        -p_voigt`` (geotech compressive mean stress, always >= 0 on
        the yield surface)."""
        s, p_v = _deviator(np.asarray(sigma, dtype=float))
        p_eff = -p_v       # convert to compressive sign
        q = _von_mises_q(s)
        return q * q + self.M * self.M * p_eff * (p_eff - self.p_c)

    # ----------------------------------------------------- response
    def get_response(self, eps_voigt) -> tuple[np.ndarray, np.ndarray]:
        """MCC return mapping with volumetric hardening.

        We use a simple 1-D Newton solve on the plastic multiplier
        ``delta_lambda`` enforcing the consistency condition. The
        bulk and shear contributions are decoupled (additive split).
        """
        eps = np.asarray(eps_voigt, dtype=float).reshape(6)
        eps_e_trial = eps - self.eps_p_committed
        sigma_trial = self._D_elastic @ eps_e_trial
        s_trial, p_v_trial = _deviator(sigma_trial)
        p_eff_trial = -p_v_trial
        q_trial = _von_mises_q(s_trial)
        # Elastic check
        f_trial = (q_trial * q_trial
                    + self.M * self.M * p_eff_trial
                      * (p_eff_trial - self.p_c_committed))
        if f_trial <= 0.0:
            self.sigma_trial = sigma_trial
            self.eps_p_trial = self.eps_p_committed.copy()
            self.p_c = self.p_c_committed
            return sigma_trial.copy(), self._D_elastic.copy()

        # Return mapping. Variables:
        #   q  = q_trial / (1 + 6G * delta_lambda)
        #   p  = (p_eff_trial + K * delta_lambda * M^2 * p_c) /
        #        (1 + K * delta_lambda * M^2)
        # Hardening update:
        #   p_c = p_c_committed * exp[
        #              (M^2 * delta_lambda * (2p - p_c))
        #              / (lambda_cc - kappa_cc)
        #          ]
        # The consistency f(dl) = q^2 + M^2 p (p - p_c) = 0 is solved
        # with Newton on dl >= 0.
        K = self.K_bulk
        G = self.G
        M2 = self.M * self.M

        def state(dl: float):
            q_new = q_trial / (1.0 + 6.0 * G * dl)
            p_new = (p_eff_trial + K * dl * M2 * self.p_c_committed) \
                    / (1.0 + 2.0 * K * dl * M2)
            # Hardening (use semi-implicit update for stability)
            eps_v_p = 2.0 * dl * M2 * (p_new - 0.5 * self.p_c_committed)
            denom = self.lambda_cc - self.kappa_cc
            p_c_new = self.p_c_committed * math.exp(eps_v_p / denom)
            return q_new, p_new, p_c_new

        # Newton on dl. Start from elastic-predictor estimate.
        dl = 0.0
        max_iter = 40
        for it in range(max_iter):
            q_n, p_n, p_c_n = state(dl)
            f = q_n * q_n + M2 * p_n * (p_n - p_c_n)
            if abs(f) < 1.0e-8 * (q_trial * q_trial
                                    + M2 * abs(p_eff_trial) ** 2 + 1.0):
                break
            # numerical derivative for robustness
            dl_p = dl + 1e-8
            q_p, p_p, pc_p = state(dl_p)
            fp = q_p * q_p + M2 * p_p * (p_p - pc_p)
            df_ddl = (fp - f) / 1e-8
            if df_ddl == 0.0:
                break
            dl -= f / df_ddl
            if dl < 0.0:
                dl = 0.0
                break
        q_n, p_n, p_c_n = state(dl)
        # Reconstruct sigma_new
        s_new = s_trial * (q_n / max(q_trial, 1e-30))
        p_v_new = -p_n     # back to tension-positive
        sigma_new = s_new.copy()
        sigma_new[0:3] += p_v_new
        # Update plastic strain via D_e^-1
        eps_e_new = self._invert_D_e_voigt(sigma_new)
        eps_p_new = eps - eps_e_new
        self.sigma_trial = sigma_new
        self.eps_p_trial = eps_p_new
        self.p_c = p_c_n
        return sigma_new.copy(), self._D_elastic.copy()

    def _invert_D_e_voigt(self, sigma: np.ndarray) -> np.ndarray:
        E, nu = self.E, self.nu
        G = self.G
        eps = np.empty(6)
        eps[0] = (sigma[0] - nu * (sigma[1] + sigma[2])) / E
        eps[1] = (sigma[1] - nu * (sigma[0] + sigma[2])) / E
        eps[2] = (sigma[2] - nu * (sigma[0] + sigma[1])) / E
        eps[3] = sigma[3] / G
        eps[4] = sigma[4] / G
        eps[5] = sigma[5] / G
        return eps

    # ----------------------------------------------------- state
    def commit_state(self) -> None:
        self.eps_p_committed = self.eps_p_trial.copy()
        self.sigma_committed = self.sigma_trial.copy()
        self.p_c_committed = self.p_c

    def revert_state(self) -> None:
        self.eps_p_trial = self.eps_p_committed.copy()
        self.sigma_trial = self.sigma_committed.copy()
        self.p_c = self.p_c_committed

    def clone(self) -> "ModifiedCamClay3D":
        return copy.deepcopy(self)

    def __repr__(self) -> str:
        return (
            f"ModifiedCamClay3D(E={self.E:g}, nu={self.nu:g}, "
            f"M={self.M:g}, lambda={self.lambda_cc:g}, "
            f"kappa={self.kappa_cc:g}, p_c={self.p_c:g})"
        )
