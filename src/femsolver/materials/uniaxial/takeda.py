"""Modified Takeda hysteresis (Park-Reinhorn-Kunnath 1987).

A reinforced-concrete cyclic stress-strain rule with:

* **Bilinear backbone** -- elastic up to σ_y, then post-yield slope
  ``b · E`` (positive on both sides; the model is symmetric).
* **Stiffness-degrading unloading** -- the unloading slope from any
  extreme excursion ``ε_max`` degrades as ``ε_max`` grows past yield:

      K_u = E · (ε_y / |ε_max|) ** α

  where ``α >= 0`` is the unloading-stiffness-degradation parameter.
  ``α = 0`` recovers elastic unloading (no degradation);
  ``α ≈ 0.5`` is a typical value for RC.
* **Pinching-free reloading** -- after unloading reaches zero stress
  at residual strain ``ε_residual``, reloading is a straight line
  from ``(ε_residual, 0)`` to the opposite-side extreme
  ``(ε_max_opposite, σ_max_opposite)`` (or to the first-yield point
  if no opposite-side history exists). This is the "Modified Takeda
  without pinch" used in many RC software packages.

Limitations / scope
-------------------
* Symmetric backbone -- no asymmetric tension/compression strengths
  (use a fiber section of two opposed Takeda materials if asymmetry
  matters).
* No cracking branch (the simpler bilinear, not trilinear backbone).
* No strength deterioration with cumulative damage (extension would
  follow :class:`UniaxialHysteretic`).

These are deliberate simplifications -- the cyclic-loop shape is the
defining "Takeda" feature, and that is captured here.
"""
from __future__ import annotations

from femsolver.materials.uniaxial.base import UniaxialMaterial


class UniaxialTakeda(UniaxialMaterial):
    """Modified Takeda uniaxial cyclic material.

    Parameters
    ----------
    E : float
        Initial elastic modulus.
    sigma_y : float
        Symmetric yield stress magnitude (positive).
    b : float, default 0.01
        Post-yield-to-initial stiffness ratio in ``[0, 1)``.
    alpha : float, default 0.5
        Unloading-stiffness-degradation exponent (≥ 0). ``0`` =
        elastic unloading; ``0.5`` = typical RC value.
    """

    def __init__(self, E: float, sigma_y: float, b: float = 0.01,
                 alpha: float = 0.5):
        if E <= 0.0:
            raise ValueError(f"E must be positive, got {E}")
        if sigma_y <= 0.0:
            raise ValueError(f"sigma_y must be positive, got {sigma_y}")
        if not (0.0 <= b < 1.0):
            raise ValueError(f"b must be in [0, 1), got {b}")
        if alpha < 0.0:
            raise ValueError(f"alpha must be >= 0, got {alpha}")
        self.E = float(E)
        self.sigma_y = float(sigma_y)
        self.b = float(b)
        self.alpha = float(alpha)
        self.eps_y = self.sigma_y / self.E
        # Historical extremes (committed); initialised to first-yield so
        # the first elastic excursion produces the standard backbone
        # without any "phantom" stiffness degradation.
        self.eps_max_pos_committed: float = self.eps_y
        self.eps_max_neg_committed: float = -self.eps_y
        self.last_eps_committed: float = 0.0
        # Trial copies
        self.eps_max_pos_trial: float = self.eps_y
        self.eps_max_neg_trial: float = -self.eps_y
        self.last_eps_trial: float = 0.0
        # Output (most recent)
        self.sigma_trial: float = 0.0
        self.Et: float = self.E

    # ----------------------------------------------------- backbone
    def _backbone(self, eps: float) -> tuple[float, float]:
        """Bilinear envelope ``(σ, Et)`` at strain eps (symmetric)."""
        if eps >= self.eps_y:
            sigma = self.sigma_y + self.b * self.E * (eps - self.eps_y)
            Et = self.b * self.E
        elif eps <= -self.eps_y:
            sigma = -self.sigma_y + self.b * self.E * (eps + self.eps_y)
            Et = self.b * self.E
        else:
            sigma = self.E * eps
            Et = self.E
        return sigma, Et

    # ----------------------------------------------------- response
    def get_response(self, eps: float) -> tuple[float, float]:
        eps = float(eps)
        # Compute trial extremes (only push outward)
        em_pos = max(self.eps_max_pos_committed, eps)
        em_neg = min(self.eps_max_neg_committed, eps)
        self.eps_max_pos_trial = em_pos
        self.eps_max_neg_trial = em_neg
        self.last_eps_trial = eps

        # --- Case 1: at or past the +backbone (eps reaches new max +ve) ---
        if eps >= self.eps_max_pos_committed:
            sigma, Et = self._backbone(eps)
            self.sigma_trial = sigma
            self.Et = Et
            return sigma, Et

        # --- Case 2: at or past the -backbone (eps reaches new max -ve) ---
        if eps <= self.eps_max_neg_committed:
            sigma, Et = self._backbone(eps)
            self.sigma_trial = sigma
            self.Et = Et
            return sigma, Et

        # --- Case 3: interior point in the (ε_max_neg, ε_max_pos) box ---
        # Compute extremes and their backbone stresses
        ep_pos = self.eps_max_pos_committed
        ep_neg = self.eps_max_neg_committed
        sg_pos, _ = self._backbone(ep_pos)
        sg_neg, _ = self._backbone(ep_neg)
        # Degraded unloading slopes (from each side)
        K_u_pos = self.E * (self.eps_y / ep_pos) ** self.alpha
        K_u_neg = self.E * (self.eps_y / abs(ep_neg)) ** self.alpha
        # Zero-stress points on the unload lines
        ez_pos = ep_pos - sg_pos / K_u_pos     # >0 if ep_pos > eps_y
        ez_neg = ep_neg - sg_neg / K_u_neg     # <0
        # Direction from last_eps_committed to current eps
        deps = eps - self.last_eps_committed
        # Determine which interior branch we're on by looking at the
        # last committed state's location.
        # We need to know whether we're "going down from +" or "going
        # up from -". Decide using deps direction AND last position.
        last = self.last_eps_committed
        if deps >= 0.0:
            # Loading toward +max
            # If last was below ez_neg -> on the (-)-side unload branch
            # If last was between ez_neg and ep_pos -> on the reload-to-+max branch
            # If last was above ep_pos -> on the +backbone (already handled above)
            if last <= ez_neg:
                # On the -unload branch: continue with K_u_neg
                # sigma = sg_neg + K_u_neg * (eps - ep_neg)
                sigma = sg_neg + K_u_neg * (eps - ep_neg)
                # Check if this segment ends (when sigma >= 0): transitions to reload
                if eps > ez_neg:
                    # We've crossed zero -- switch to reload-to-+max line
                    # from (ez_neg, 0) to (ep_pos, sg_pos)
                    slope = sg_pos / (ep_pos - ez_neg)
                    sigma = slope * (eps - ez_neg)
                    Et = slope
                else:
                    Et = K_u_neg
            else:
                # On the reload-to-+max branch (or starting there)
                slope = sg_pos / (ep_pos - ez_neg)
                sigma = slope * (eps - ez_neg)
                Et = slope
        else:
            # Unloading from +side or loading toward -max
            if last >= ez_pos:
                # On the +unload branch
                sigma = sg_pos - K_u_pos * (ep_pos - eps)
                if eps < ez_pos:
                    # Crossed zero -- switch to reload-to--max line
                    slope = (-sg_neg) / (ez_pos - ep_neg)
                    # Line from (ez_pos, 0) to (ep_neg, sg_neg):
                    # sigma = 0 + (sg_neg - 0)/(ep_neg - ez_pos) * (eps - ez_pos)
                    # Since sg_neg < 0 and ep_neg < ez_pos, slope > 0
                    slope = sg_neg / (ep_neg - ez_pos)
                    sigma = slope * (eps - ez_pos)
                    Et = slope
                else:
                    Et = K_u_pos
            else:
                # On the reload-to--max branch
                slope = sg_neg / (ep_neg - ez_pos)
                sigma = slope * (eps - ez_pos)
                Et = slope

        self.sigma_trial = sigma
        self.Et = Et
        return sigma, Et

    def commit_state(self) -> None:
        self.eps_max_pos_committed = self.eps_max_pos_trial
        self.eps_max_neg_committed = self.eps_max_neg_trial
        self.last_eps_committed = self.last_eps_trial

    def revert_state(self) -> None:
        self.eps_max_pos_trial = self.eps_max_pos_committed
        self.eps_max_neg_trial = self.eps_max_neg_committed
        self.last_eps_trial = self.last_eps_committed

    def __repr__(self) -> str:
        return (
            f"UniaxialTakeda(E={self.E:g}, sigma_y={self.sigma_y:g}, "
            f"b={self.b:g}, alpha={self.alpha:g})"
        )
