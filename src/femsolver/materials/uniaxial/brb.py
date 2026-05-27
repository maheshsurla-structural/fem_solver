"""Buckling-Restrained Brace (BRB) uniaxial material.

A simplified Black-Wada-Aiken style BRB model with the two
characteristic properties of buckling-restrained braces:

* **Asymmetric backbone** -- compression yield is larger than tension
  yield by a *compression overstrength factor* ``β >= 1`` (typically
  1.05-1.10 for laboratory BRBs, accounting for friction between the
  core and the buckling-restraint sleeve when the core is in
  compression).
* **Combined kinematic + isotropic hardening** -- both Bauschinger
  effect (back-stress translation) and cyclic strength growth
  (yield-surface expansion with cumulative plastic strain).

Yield function (asymmetric)
---------------------------
The yield surface is non-symmetric in the (σ - q) space:

    f(σ, q, p) = +(σ - q) - σ_y_t · (1 + a_iso · p)     when σ - q ≥ 0
    f(σ, q, p) = -(σ - q) - β · σ_y_t · (1 + a_iso · p) when σ - q < 0

where ``σ_y_t`` is the tension yield, ``β`` is the compression-over-
strength factor, ``q`` is the back-stress, and ``p`` is the
cumulative plastic strain. Setting ``β = 1`` and ``a_iso = 0`` gives
ordinary symmetric kinematic hardening (:class:`UniaxialBilinear`).

Algorithm (return mapping with combined hardening)
---------------------------------------------------
1. Predictor:    σ_trial = E · (ε - ε_p_committed)
2. ξ_trial    = σ_trial - q_committed
3. Direction:    sign = +1 if ξ_trial ≥ 0 else -1
4. σ_y_eff   = (σ_y_t if sign > 0 else β · σ_y_t)
                · (1 + a_iso · p_committed)
5. f_trial   = |ξ_trial| - σ_y_eff
6. If f_trial ≤ 0: elastic step
7. Else:
       H_total   = H_kin + σ_y_eff · a_iso  (kin + iso combined)
       Δλ        = f_trial / (E + H_total)
       Δε_p      = sign · Δλ
       ε_p_trial = ε_p_committed + Δε_p
       q_trial   = q_committed + H_kin · Δε_p
       p_trial   = p_committed + Δλ
       σ         = E · (ε - ε_p_trial)
       Et        = E · H_total / (E + H_total)

References
----------
Black, C. J., Makris, N., Aiken, I. D. (2004). "Component testing,
seismic evaluation, and characterization of buckling-restrained
braces." Journal of Structural Engineering, 130(6), 880-894.
"""
from __future__ import annotations

from femsolver.materials.uniaxial.base import UniaxialMaterial


class UniaxialBRB(UniaxialMaterial):
    """Buckling-restrained brace uniaxial material.

    Parameters
    ----------
    E : float
        Initial elastic modulus.
    sigma_y : float
        Tension yield stress σ_y_t (positive).
    b : float, default 0.02
        Strain-hardening ratio (kinematic component); typical 1-3%
        for BRB cores.
    beta : float, default 1.10
        Compression overstrength factor (≥ 1). Compression yield is
        ``β · σ_y_t``. Typical 1.05-1.15 for laboratory BRBs.
    a_iso : float, default 0.0
        Isotropic-growth rate (per unit cumulative plastic strain).
        ``0`` = pure kinematic (no cyclic strength growth);
        ``50`` ≈ 5% growth per 0.001 plastic strain (typical for
        cyclic strength buildup observed in BRB tests).
    """

    def __init__(self, E: float, sigma_y: float, *,
                 b: float = 0.02,
                 beta: float = 1.10,
                 a_iso: float = 0.0):
        if E <= 0.0:
            raise ValueError(f"E must be positive, got {E}")
        if sigma_y <= 0.0:
            raise ValueError(f"sigma_y must be positive, got {sigma_y}")
        if not (0.0 <= b < 1.0):
            raise ValueError(f"b must be in [0, 1), got {b}")
        if beta < 1.0:
            raise ValueError(f"beta must be >= 1, got {beta}")
        if a_iso < 0.0:
            raise ValueError(f"a_iso must be >= 0, got {a_iso}")
        self.E = float(E)
        self.sigma_y = float(sigma_y)
        self.b = float(b)
        self.beta = float(beta)
        self.a_iso = float(a_iso)
        self.H_kin = (self.b * self.E / (1.0 - self.b)) if self.b > 0.0 else 0.0
        # State
        self.eps_p_committed: float = 0.0
        self.q_committed: float = 0.0
        self.p_committed: float = 0.0
        self.eps_p_trial: float = 0.0
        self.q_trial: float = 0.0
        self.p_trial: float = 0.0
        self.sigma_trial: float = 0.0
        self.Et: float = self.E

    def get_response(self, eps: float) -> tuple[float, float]:
        eps = float(eps)
        sigma_trial = self.E * (eps - self.eps_p_committed)
        xi = sigma_trial - self.q_committed
        sign = 1.0 if xi >= 0.0 else -1.0
        # Effective yield strength in the loading direction
        sigma_y_dir = self.sigma_y if sign > 0.0 else self.beta * self.sigma_y
        sigma_y_eff = sigma_y_dir * (1.0 + self.a_iso * self.p_committed)
        f_trial = abs(xi) - sigma_y_eff
        if f_trial <= 0.0:
            self.eps_p_trial = self.eps_p_committed
            self.q_trial = self.q_committed
            self.p_trial = self.p_committed
            self.sigma_trial = sigma_trial
            self.Et = self.E
            return sigma_trial, self.E
        # Plastic step
        H_iso_rate = sigma_y_dir * self.a_iso
        H_total = self.H_kin + H_iso_rate
        d_lambda = f_trial / (self.E + H_total)
        d_eps_p = sign * d_lambda
        self.eps_p_trial = self.eps_p_committed + d_eps_p
        self.q_trial = self.q_committed + self.H_kin * d_eps_p
        self.p_trial = self.p_committed + d_lambda
        sigma = self.E * (eps - self.eps_p_trial)
        Et = (self.E * H_total / (self.E + H_total)
              if H_total > 0.0 else 0.0)
        self.sigma_trial = sigma
        self.Et = Et
        return sigma, Et

    def commit_state(self) -> None:
        self.eps_p_committed = self.eps_p_trial
        self.q_committed = self.q_trial
        self.p_committed = self.p_trial

    def revert_state(self) -> None:
        self.eps_p_trial = self.eps_p_committed
        self.q_trial = self.q_committed
        self.p_trial = self.p_committed

    def __repr__(self) -> str:
        return (
            f"UniaxialBRB(E={self.E:g}, sigma_y={self.sigma_y:g}, "
            f"b={self.b:g}, beta={self.beta:g}, a_iso={self.a_iso:g})"
        )
