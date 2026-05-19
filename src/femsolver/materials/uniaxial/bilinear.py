"""Bilinear uniaxial plasticity with kinematic hardening.

This is the same return-mapping algorithm as
:class:`BilinearMomentRotationSpring`, only on (sigma, eps) instead of
(M, theta_h). Setting ``b = 0`` recovers elastic-perfectly-plastic
(EPP). With ``0 < b < 1`` the post-yield slope is ``b * E``.

Algorithm
---------
1. Elastic predictor:  sigma_trial = E * (eps - eps_p_committed)
2. Yield-function trial: f_trial = |sigma_trial - q_committed| - sigma_y
3. Elastic step (f_trial <= 0): no plastic flow.
4. Plastic step: standard return mapping,

       d_lambda = f_trial / (E + H_kin)
       d_eps_p  = sign(sigma_trial - q_committed) * d_lambda
       eps_p_trial = eps_p_committed + d_eps_p
       q_trial     = q_committed + H_kin * d_eps_p
       sigma       = E * (eps - eps_p_trial)
       Et          = b * E

The kinematic hardening modulus is
``H_kin = b * E / (1 - b)``, which gives a post-yield tangent of
``b * E`` exactly.
"""
from __future__ import annotations

from femsolver.materials.uniaxial.base import UniaxialMaterial


class UniaxialBilinear(UniaxialMaterial):
    """Bilinear uniaxial plasticity with kinematic hardening.

    Parameters
    ----------
    E : float
        Initial elastic modulus.
    sigma_y : float
        Yield stress magnitude (positive). Yield surface in stress space
        is symmetric about the back-stress ``q``.
    b : float, default 0.0
        Post-yield stiffness ratio in ``[0, 1)``. ``b = 0`` is EPP.
    """

    def __init__(self, E: float, sigma_y: float, b: float = 0.0):
        if E <= 0.0:
            raise ValueError(f"E must be positive, got {E}")
        if sigma_y <= 0.0:
            raise ValueError(f"sigma_y must be positive, got {sigma_y}")
        if not (0.0 <= b < 1.0):
            raise ValueError(f"b must be in [0, 1), got {b}")
        self.E = float(E)
        self.sigma_y = float(sigma_y)
        self.b = float(b)
        self.H_kin = (self.b * self.E / (1.0 - self.b)) if self.b > 0.0 else 0.0
        # state
        self.eps_p_committed: float = 0.0
        self.eps_p_trial: float = 0.0
        self.q_committed: float = 0.0
        self.q_trial: float = 0.0
        # most recent response — useful for output
        self.sigma_trial: float = 0.0
        self.Et: float = self.E

    def get_response(self, eps: float) -> tuple[float, float]:
        eps = float(eps)
        eps_e_trial = eps - self.eps_p_committed
        sigma_trial = self.E * eps_e_trial
        xi = sigma_trial - self.q_committed
        f_trial = abs(xi) - self.sigma_y
        if f_trial <= 0.0:
            self.eps_p_trial = self.eps_p_committed
            self.q_trial = self.q_committed
            self.sigma_trial = sigma_trial
            self.Et = self.E
            return sigma_trial, self.E
        # plastic step
        sign = 1.0 if xi >= 0.0 else -1.0
        d_lambda = f_trial / (self.E + self.H_kin)
        d_eps_p = d_lambda * sign
        self.eps_p_trial = self.eps_p_committed + d_eps_p
        self.q_trial = self.q_committed + self.H_kin * d_eps_p
        sigma = self.E * (eps - self.eps_p_trial)
        Et = (self.E * self.H_kin / (self.E + self.H_kin)
              if self.H_kin > 0.0 else 0.0)
        self.sigma_trial = sigma
        self.Et = Et
        return sigma, Et

    def commit_state(self) -> None:
        self.eps_p_committed = self.eps_p_trial
        self.q_committed = self.q_trial

    def revert_state(self) -> None:
        self.eps_p_trial = self.eps_p_committed
        self.q_trial = self.q_committed

    def __repr__(self) -> str:
        return (
            f"UniaxialBilinear(E={self.E:g}, sigma_y={self.sigma_y:g}, "
            f"b={self.b:g})"
        )
