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
        *,
        e_0: float = 0.5,
        stress_dependent: bool = True,
        p_min: float = 1.0e3,
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
        if e_0 <= 0.0:
            raise ValueError(f"e_0 must be positive, got {e_0}")
        if p_min <= 0.0:
            raise ValueError(f"p_min must be positive, got {p_min}")
        self.E = float(E)
        self.nu = float(nu)
        self.M = float(M)
        self.lambda_cc = float(lambda_cc)
        self.kappa_cc = float(kappa_cc)
        self._p_c0 = float(p_c0)
        self.stress_dependent = bool(stress_dependent)
        self.p_min = float(p_min)
        # Initial elastic constants (used when stress_dependent=False,
        # and as a starting reference for the K-update logic)
        self._G_initial = E / (2.0 * (1.0 + nu))
        self._K_initial = E / (3.0 * (1.0 - 2.0 * nu))
        # state: internal "p_c" stored as a positive number (consol.
        # pressure magnitude). Tension-positive sign means yield occurs
        # for sigma with negative p (compressive).
        self.p_c = float(p_c0)
        self.p_c_committed = float(p_c0)
        # Void ratio: tracked across plastic volumetric strain
        self.e_committed = float(e_0)
        self.e_trial = float(e_0)
        self.eps_p_committed = np.zeros(6)
        self.eps_p_trial = np.zeros(6)
        self.sigma_committed = np.zeros(6)
        self.sigma_trial = np.zeros(6)
        # Build initial D_e at p_eff = p_c0/2 (typical in-situ state)
        self._D_elastic = self._build_D_elastic_at(
            self._tangent_K(p_c0 / 2.0, e_0)
        )

    def _tangent_K(self, p_eff: float, e: float) -> float:
        """Tangent bulk modulus per critical-state theory::

            K' = (1 + e) * p_eff / kappa

        with a floor at ``p_eff = p_min`` so K stays finite near the
        apex / tension cut-off. When ``stress_dependent`` is False,
        return the initial K from the constructor.
        """
        if not self.stress_dependent:
            return self._K_initial
        p_safe = max(p_eff, self.p_min)
        return (1.0 + e) * p_safe / self.kappa_cc

    def _build_D_elastic_at(self, K: float) -> np.ndarray:
        """Build the (6,6) elastic stiffness with the supplied K and
        constant Poisson ratio. G follows from K via ``G = 3K(1 -
        2nu) / (2(1 + nu))``."""
        nu = self.nu
        G = 3.0 * K * (1.0 - 2.0 * nu) / (2.0 * (1.0 + nu))
        lam = K - 2.0 * G / 3.0
        D = np.zeros((6, 6))
        D[0, 0] = D[1, 1] = D[2, 2] = lam + 2.0 * G
        D[0, 1] = D[0, 2] = D[1, 0] = D[1, 2] = D[2, 0] = D[2, 1] = lam
        D[3, 3] = D[4, 4] = D[5, 5] = G
        return D

    @property
    def K_bulk(self) -> float:
        """Current tangent bulk modulus (stress-dependent if enabled)."""
        if not self.stress_dependent:
            return self._K_initial
        # Use committed mean effective stress
        if not np.any(self.sigma_committed):
            return self._tangent_K(self._p_c0 / 2.0, self.e_committed)
        s, p_v = _deviator(self.sigma_committed)
        return self._tangent_K(-p_v, self.e_committed)

    @property
    def G(self) -> float:
        K = self.K_bulk
        return 3.0 * K * (1.0 - 2.0 * self.nu) / (2.0 * (1.0 + self.nu))

    def D_elastic(self) -> np.ndarray:
        return self._build_D_elastic_at(self.K_bulk)

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
        # Stress-dependent elastic stiffness: build D_e using the
        # committed state's tangent K. This makes the elastic predictor
        # follow the swelling line in (e, ln p').
        D_e = self.D_elastic()
        sigma_trial = D_e @ eps_e_trial
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
            # Track void-ratio change from elastic volumetric strain.
            # In tension-positive convention: de = (1+e) * eps_v, so
            # compression (eps_v < 0) decreases e.
            eps_v_e = (eps_e_trial[0] + eps_e_trial[1] + eps_e_trial[2])
            self.e_trial = self.e_committed + (1.0 + self.e_committed) * eps_v_e
            return sigma_trial.copy(), D_e.copy()

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
        # Update plastic strain via D_e^-1 (using the current K, G)
        eps_e_new = self._invert_D_e_voigt(sigma_new, K=K, G=G)
        eps_p_new = eps - eps_e_new
        # Update void ratio from total volumetric strain
        # (tension-positive: compression decreases e)
        eps_v_total = eps[0] + eps[1] + eps[2]
        self.e_trial = self.e_committed + (1.0 + self.e_committed) * eps_v_total
        self.sigma_trial = sigma_new
        self.eps_p_trial = eps_p_new
        self.p_c = p_c_n
        return sigma_new.copy(), D_e.copy()

    def _invert_D_e_voigt(
        self, sigma: np.ndarray,
        K: float | None = None, G: float | None = None,
    ) -> np.ndarray:
        # Allow caller to pass the K, G that built the trial sigma so
        # that the inverse is consistent (stress-dependent stiffness).
        if K is None:
            K = self.K_bulk
        if G is None:
            G = self.G
        E = 9.0 * K * G / (3.0 * K + G)
        nu = (3.0 * K - 2.0 * G) / (2.0 * (3.0 * K + G))
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
        self.e_committed = self.e_trial

    def revert_state(self) -> None:
        self.eps_p_trial = self.eps_p_committed.copy()
        self.sigma_trial = self.sigma_committed.copy()
        self.p_c = self.p_c_committed
        self.e_trial = self.e_committed

    def clone(self) -> "ModifiedCamClay3D":
        return copy.deepcopy(self)

    def __repr__(self) -> str:
        return (
            f"ModifiedCamClay3D(E={self.E:g}, nu={self.nu:g}, "
            f"M={self.M:g}, lambda={self.lambda_cc:g}, "
            f"kappa={self.kappa_cc:g}, p_c={self.p_c:g})"
        )
