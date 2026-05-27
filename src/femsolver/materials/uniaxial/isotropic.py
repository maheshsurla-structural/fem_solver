"""Bilinear uniaxial plasticity with isotropic hardening.

The same J2 return-mapping algorithm as :class:`UniaxialBilinear`, but
with the hardening modulus acting on the *yield-surface size* (σ_y
grows with cumulative plastic strain) instead of on the back-stress.

Yield function
--------------

    f(σ, p) = |σ| - σ_y(p)

with ``σ_y(p) = σ_y0 + H_iso · p`` (linear isotropic hardening),
where ``p = cumulative |ε_p|`` is the equivalent plastic strain.

Algorithm
---------

1. Elastic predictor:  σ_trial = E · (ε - ε_p_committed)
2. Trial yield:       f_trial = |σ_trial| - σ_y(p_committed)
3. Elastic step (f_trial ≤ 0): no plastic flow.
4. Plastic step: standard 1-D return mapping,

        Δλ        = f_trial / (E + H_iso)
        d_ε_p     = sign(σ_trial) · Δλ
        ε_p_trial = ε_p_committed + d_ε_p
        p_trial   = p_committed + Δλ
        σ         = E · (ε - ε_p_trial)
        Et        = E · H_iso / (E + H_iso) = b · E

In monotonic loading, isotropic and kinematic hardening are
indistinguishable. They diverge on **reversal**: the isotropic model
gives reverse yielding at ``-σ_y(p)`` (the expanded surface size),
while the kinematic model gives Bauschinger-effect reverse yielding
at ``-σ_y0 + 2·(σ_max - σ_y0)`` (the back-stress translation).
Isotropic hardening over-predicts reverse-yielding strength and
ignores the cyclic softening seen in real metals; it is mostly used
for monotonic / "ductile-up-to-fracture" analyses or as a reference
baseline for cyclic studies.

Sign and parameter conventions match :class:`UniaxialBilinear`:
``b`` is the post-yield-to-initial stiffness ratio, with
``H_iso = b · E / (1 - b)``.
"""
from __future__ import annotations

from femsolver.materials.uniaxial.base import UniaxialMaterial


class UniaxialIsotropicHardening(UniaxialMaterial):
    """Bilinear uniaxial plasticity with isotropic hardening.

    Parameters
    ----------
    E : float
        Initial elastic modulus.
    sigma_y : float
        Initial yield stress magnitude σ_y0 (positive).
    b : float, default 0.0
        Post-yield-to-initial stiffness ratio in ``[0, 1)``.
        ``b = 0`` is elastic-perfectly-plastic (no hardening at all).
    """

    def __init__(self, E: float, sigma_y: float, b: float = 0.0):
        if E <= 0.0:
            raise ValueError(f"E must be positive, got {E}")
        if sigma_y <= 0.0:
            raise ValueError(f"sigma_y must be positive, got {sigma_y}")
        if not (0.0 <= b < 1.0):
            raise ValueError(f"b must be in [0, 1), got {b}")
        self.E = float(E)
        self.sigma_y0 = float(sigma_y)
        self.b = float(b)
        self.H_iso = (self.b * self.E / (1.0 - self.b)) if self.b > 0.0 else 0.0
        # state
        self.eps_p_committed: float = 0.0
        self.eps_p_trial: float = 0.0
        self.p_committed: float = 0.0     # cumulative plastic strain
        self.p_trial: float = 0.0
        # most recent response — useful for output
        self.sigma_trial: float = 0.0
        self.Et: float = self.E

    def get_response(self, eps: float) -> tuple[float, float]:
        eps = float(eps)
        sigma_trial = self.E * (eps - self.eps_p_committed)
        sigma_y = self.sigma_y0 + self.H_iso * self.p_committed
        f_trial = abs(sigma_trial) - sigma_y
        if f_trial <= 0.0:
            self.eps_p_trial = self.eps_p_committed
            self.p_trial = self.p_committed
            self.sigma_trial = sigma_trial
            self.Et = self.E
            return sigma_trial, self.E
        # plastic step
        sign = 1.0 if sigma_trial >= 0.0 else -1.0
        d_lambda = f_trial / (self.E + self.H_iso)
        d_eps_p = d_lambda * sign
        self.eps_p_trial = self.eps_p_committed + d_eps_p
        self.p_trial = self.p_committed + d_lambda
        sigma = self.E * (eps - self.eps_p_trial)
        Et = (self.E * self.H_iso / (self.E + self.H_iso)
              if self.H_iso > 0.0 else 0.0)
        self.sigma_trial = sigma
        self.Et = Et
        return sigma, Et

    def commit_state(self) -> None:
        self.eps_p_committed = self.eps_p_trial
        self.p_committed = self.p_trial

    def revert_state(self) -> None:
        self.eps_p_trial = self.eps_p_committed
        self.p_trial = self.p_committed

    def __repr__(self) -> str:
        return (
            f"UniaxialIsotropicHardening(E={self.E:g}, "
            f"sigma_y={self.sigma_y0:g}, b={self.b:g})"
        )
