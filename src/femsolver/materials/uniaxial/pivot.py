"""Single-pivot hysteresis (after Dowell-Seible-Hines 1998 "Pivot" model).

A reinforced-concrete cyclic stress-strain rule with **load-reversal
trajectories aimed at pivot points** outside the yield envelope. The
characteristic pinched-loop shape is produced because the trajectories
converge on a small number of fixed points, and the resulting cyclic
stiffness depends on how far the system has been pushed.

Pivot points
------------
This implementation uses **two primary pivot points**:

    P_pos = (+α · ε_y, +α · σ_y)        on the extended +elastic line
    P_neg = (-α · ε_y, -α · σ_y)        on the extended -elastic line

where ``α >= 1`` is the user-supplied pivot factor.  ``α = 1`` puts the
pivots exactly at yield (loops collapse to the backbone -- effectively
elastic-with-yield).  Typical values used in practice for RC are
``α ≈ 5-15``.

Trajectories
------------
* **Backbone** -- standard bilinear: elastic up to σ_y, then post-yield
  slope ``b · E`` (positive and negative sides symmetric).
* **Loading toward +max** (current ε > ε_max_pos): on +backbone.
* **Unloading from +max / loading toward -max**: a straight line from
  the most-recent positive extreme ``(ε_max_pos, σ_max_pos)`` aimed
  at ``P_neg``. Switches to the -backbone when ε passes ε_max_neg.
* **Loading toward +max from -max**: a straight line from
  ``(ε_max_neg, σ_max_neg)`` aimed at ``P_pos``. Switches to the
  +backbone when ε passes ε_max_pos.

This is a simplified Pivot variant -- the full Dowell-Seible-Hines
formulation includes a *second* pair of "pinching pivots" PP1 and PP2
at σ = 0 that introduce additional kinks and produce the
characteristic RC pinch.  For Phase 24 the single-pivot variant
captures the essential concept; pinching pivots are a future
extension.

Limitations
-----------
* Symmetric backbone.
* No strength deterioration beyond the static bilinear envelope.
* α applied symmetrically to both sides (asymmetric α1, α2 are a
  trivial extension of the present code).
"""
from __future__ import annotations

from femsolver.materials.uniaxial.base import UniaxialMaterial


class UniaxialPivot(UniaxialMaterial):
    """Single-pivot Dowell-Seible-Hines hysteresis.

    Parameters
    ----------
    E : float
        Initial elastic modulus.
    sigma_y : float
        Symmetric yield stress magnitude (positive).
    b : float, default 0.01
        Post-yield-to-initial stiffness ratio in ``[0, 1)``.
    alpha : float, default 5.0
        Pivot factor (≥ 1). ``α = 1`` collapses to backbone (no loop
        area). Larger α produces steeper interior reload slopes (more
        Bauschinger-like loops); typical RC values are 5-15.
    """

    def __init__(self, E: float, sigma_y: float, b: float = 0.01,
                 alpha: float = 5.0):
        if E <= 0.0:
            raise ValueError(f"E must be positive, got {E}")
        if sigma_y <= 0.0:
            raise ValueError(f"sigma_y must be positive, got {sigma_y}")
        if not (0.0 <= b < 1.0):
            raise ValueError(f"b must be in [0, 1), got {b}")
        if alpha < 1.0:
            raise ValueError(f"alpha must be >= 1, got {alpha}")
        self.E = float(E)
        self.sigma_y = float(sigma_y)
        self.b = float(b)
        self.alpha = float(alpha)
        self.eps_y = self.sigma_y / self.E
        # Pivots
        self._P_pos = (self.alpha * self.eps_y, self.alpha * self.sigma_y)
        self._P_neg = (-self.alpha * self.eps_y, -self.alpha * self.sigma_y)
        # History (committed)
        self.eps_max_pos_committed: float = self.eps_y
        self.eps_max_neg_committed: float = -self.eps_y
        self.last_eps_committed: float = 0.0
        # Trial copies
        self.eps_max_pos_trial: float = self.eps_y
        self.eps_max_neg_trial: float = -self.eps_y
        self.last_eps_trial: float = 0.0
        # Output
        self.sigma_trial: float = 0.0
        self.Et: float = self.E

    def _backbone(self, eps: float) -> tuple[float, float]:
        if eps >= self.eps_y:
            return (self.sigma_y + self.b * self.E * (eps - self.eps_y),
                    self.b * self.E)
        if eps <= -self.eps_y:
            return (-self.sigma_y + self.b * self.E * (eps + self.eps_y),
                    self.b * self.E)
        return self.E * eps, self.E

    def get_response(self, eps: float) -> tuple[float, float]:
        eps = float(eps)
        em_pos = max(self.eps_max_pos_committed, eps)
        em_neg = min(self.eps_max_neg_committed, eps)
        self.eps_max_pos_trial = em_pos
        self.eps_max_neg_trial = em_neg
        self.last_eps_trial = eps

        # On +backbone
        if eps >= self.eps_max_pos_committed:
            sigma, Et = self._backbone(eps)
            self.sigma_trial = sigma
            self.Et = Et
            return sigma, Et
        # On -backbone
        if eps <= self.eps_max_neg_committed:
            sigma, Et = self._backbone(eps)
            self.sigma_trial = sigma
            self.Et = Et
            return sigma, Et

        # Interior: line from one extreme to the opposite pivot
        ep_pos = self.eps_max_pos_committed
        ep_neg = self.eps_max_neg_committed
        sg_pos, _ = self._backbone(ep_pos)
        sg_neg, _ = self._backbone(ep_neg)
        deps = eps - self.last_eps_committed
        if deps >= 0.0:
            # Loading toward +max: line from (ep_neg, sg_neg) to P_pos
            P = self._P_pos
            x0, y0 = ep_neg, sg_neg
        else:
            # Unloading from +: line from (ep_pos, sg_pos) to P_neg
            P = self._P_neg
            x0, y0 = ep_pos, sg_pos
        # Linear interp
        denom = P[0] - x0
        if abs(denom) < 1.0e-30:
            # Degenerate: pivot coincides with extreme. Fall back to
            # elastic slope.
            sigma = y0 + self.E * (eps - x0)
            Et = self.E
        else:
            slope = (P[1] - y0) / denom
            sigma = y0 + slope * (eps - x0)
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
            f"UniaxialPivot(E={self.E:g}, sigma_y={self.sigma_y:g}, "
            f"b={self.b:g}, alpha={self.alpha:g})"
        )
