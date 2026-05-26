"""Bilinear hysteretic uniaxial material with pinching.

For a single load reversal, the trajectory follows a piecewise-linear
path from the *reversal point* ``(eps_r, sigma_r)`` through a
*pinch point* at ``(pinch_x * eps_target, pinch_y * sigma_target)``
and then to the opposite-direction envelope target. ``pinch_x = 1``
and ``pinch_y = 1`` recover the direct-reload (no pinching) case,
which is structurally identical to ``UniaxialBilinear`` with linear
reload. Smaller ``pinch_y`` produces the characteristic squashed
hourglass of a cracked-and-reopened RC section.

Sign convention
---------------
Strain and stress follow the standard solid-mechanics convention
(tension positive). The envelope is symmetric about the origin
with positive yield at ``(+eps_y, +sigma_y)`` and negative yield at
``(-eps_y, -sigma_y)``.

State
-----
The model tracks the most-extreme positive and negative strains
ever reached (``eps_max_pos``, ``eps_max_neg``), the current
reversal point ``(eps_rev, sigma_rev)``, and the current loading
direction. Initial extremes are set at the first-yield strains so
that a virgin loading from the origin reproduces pure elastic
behaviour ``sigma = E * eps`` regardless of the pinching ratios.
"""
from __future__ import annotations

from femsolver.materials.uniaxial.base import UniaxialMaterial


class UniaxialHysteretic(UniaxialMaterial):
    """Bilinear envelope + pinching cyclic uniaxial material with
    optional strength degradation.

    Parameters
    ----------
    E : float
        Initial elastic modulus.
    sigma_y : float
        Yield stress magnitude (positive).
    b : float, default 0.01
        Post-yield stiffness ratio. ``b = 0`` gives EPP envelope.
    pinch_x : float, default 1.0
        Pinching parameter for strain. ``1.0`` = no pinch (direct
        line). ``0`` = pinch through the origin (extreme pinching).
    pinch_y : float, default 1.0
        Pinching parameter for stress. ``1.0`` = no pinch. Smaller
        values squash the loop vertically.
    damage_factor : float, default 0.0
        Strength-degradation rate (dimensionless). The effective
        yield stress is

            sigma_y_eff = sigma_y / (1 + damage_factor * alpha/eps_y)

        where ``alpha`` is the cumulative plastic excursion magnitude.
        ``damage_factor = 0`` disables degradation (back-compat with
        the original undegraded model). Typical values for cyclic
        seismic RC: 0.05--0.3.
    min_strength_ratio : float, default 0.1
        Lower bound on ``sigma_y_eff / sigma_y``. Prevents the
        degraded yield from collapsing to zero under extreme cycling.
    """

    def __init__(self, E: float, sigma_y: float, b: float = 0.01,
                 pinch_x: float = 1.0, pinch_y: float = 1.0,
                 damage_factor: float = 0.0,
                 min_strength_ratio: float = 0.1):
        if E <= 0.0:
            raise ValueError(f"E must be positive, got {E}")
        if sigma_y <= 0.0:
            raise ValueError(f"sigma_y must be positive, got {sigma_y}")
        if not (0.0 <= b < 1.0):
            raise ValueError(f"b must be in [0, 1), got {b}")
        if not (0.0 < pinch_x <= 1.0):
            raise ValueError(f"pinch_x must be in (0, 1], got {pinch_x}")
        if not (0.0 < pinch_y <= 1.0):
            raise ValueError(f"pinch_y must be in (0, 1], got {pinch_y}")
        if damage_factor < 0.0:
            raise ValueError(f"damage_factor must be >= 0, got {damage_factor}")
        if not (0.0 < min_strength_ratio <= 1.0):
            raise ValueError(
                f"min_strength_ratio must be in (0, 1], "
                f"got {min_strength_ratio}"
            )
        self.E = float(E)
        self.sigma_y = float(sigma_y)
        self.eps_y = sigma_y / E
        self.b = float(b)
        self.pinch_x = float(pinch_x)
        self.pinch_y = float(pinch_y)
        self.damage_factor = float(damage_factor)
        self.min_strength_ratio = float(min_strength_ratio)
        # Committed state
        self.eps_committed: float = 0.0
        self.sigma_committed: float = 0.0
        self.eps_max_pos_committed: float = self.eps_y
        self.eps_max_neg_committed: float = -self.eps_y
        self.eps_rev_committed: float = 0.0
        self.sigma_rev_committed: float = 0.0
        self.direction_committed: int = +1
        self.alpha_committed: float = 0.0      # cumulative plastic excursion
        # Trial mirrors
        self.eps_trial: float = 0.0
        self.sigma_trial: float = 0.0
        self.eps_max_pos_trial: float = self.eps_y
        self.eps_max_neg_trial: float = -self.eps_y
        self.eps_rev_trial: float = 0.0
        self.sigma_rev_trial: float = 0.0
        self.direction_trial: int = +1
        self.alpha_trial: float = 0.0
        self.Et_trial: float = self.E

    # ------------------------------------------------------------ degradation
    def _strength_ratio(self, alpha: float) -> float:
        """Returns sigma_y_eff / sigma_y given cumulative plastic
        excursion ``alpha`` (>= 0)."""
        if self.damage_factor == 0.0 or alpha <= 0.0:
            return 1.0
        ratio = 1.0 / (1.0 + self.damage_factor * alpha / self.eps_y)
        return max(ratio, self.min_strength_ratio)

    # ------------------------------------------------------------ envelope
    def _envelope(self, eps: float, alpha: float = 0.0) -> tuple[float, float]:
        """Bilinear envelope with optional strength degradation.

        With ``alpha > 0`` and ``damage_factor > 0``, both sigma_y_eff
        and eps_y_eff shrink (yielding starts earlier at lower stress).
        The elastic modulus and hardening slope are unchanged.
        """
        ratio = self._strength_ratio(alpha)
        sigma_y_eff = self.sigma_y * ratio
        eps_y_eff = self.eps_y * ratio
        if eps >= eps_y_eff:
            sigma = sigma_y_eff + self.b * self.E * (eps - eps_y_eff)
            return sigma, self.b * self.E
        if eps <= -eps_y_eff:
            sigma = -sigma_y_eff + self.b * self.E * (eps + eps_y_eff)
            return sigma, self.b * self.E
        return self.E * eps, self.E

    # ------------------------------------------------------------ reload
    def _reload_segment(self, eps: float, direction: int,
                         eps_target: float, sigma_target: float,
                         eps_rev: float, sigma_rev: float
                         ) -> tuple[float, float]:
        """Piecewise-linear reload from ``(eps_rev, sigma_rev)`` through
        the pinch point to ``(eps_target, sigma_target)``."""
        eps_pinch = self.pinch_x * eps_target
        sigma_pinch = self.pinch_y * sigma_target
        # Identify which segment we're on. For direction +1 (going
        # positive, eps increasing from eps_rev toward eps_target), the
        # pinch lies between them on the strain axis when eps_pinch
        # is reached first. We branch on eps vs eps_pinch.
        if direction > 0:
            if eps <= eps_pinch:
                # Segment 1: reversal -> pinch
                denom = eps_pinch - eps_rev
            else:
                # Segment 2: pinch -> envelope target
                denom = eps_target - eps_pinch
                eps_rev = eps_pinch
                sigma_rev = sigma_pinch
                eps_target_local, sigma_target_local = eps_target, sigma_target
                if abs(denom) < 1.0e-30:
                    return sigma_pinch, 0.0
                slope = (sigma_target_local - sigma_rev) / denom
                return sigma_rev + slope * (eps - eps_rev), slope
        else:
            if eps >= eps_pinch:
                denom = eps_pinch - eps_rev
            else:
                denom = eps_target - eps_pinch
                eps_rev = eps_pinch
                sigma_rev = sigma_pinch
                if abs(denom) < 1.0e-30:
                    return sigma_pinch, 0.0
                slope = (sigma_target - sigma_rev) / denom
                return sigma_rev + slope * (eps - eps_rev), slope
        # Segment 1 fall-through
        if abs(denom) < 1.0e-30:
            return sigma_rev, 0.0
        slope = (sigma_pinch - sigma_rev) / denom
        return sigma_rev + slope * (eps - eps_rev), slope

    # ------------------------------------------------------------ get_response
    def get_response(self, eps: float) -> tuple[float, float]:
        eps = float(eps)
        self.eps_trial = eps
        delta = eps - self.eps_committed
        if delta > 0.0:
            inc_dir = +1
        elif delta < 0.0:
            inc_dir = -1
        else:
            inc_dir = self.direction_committed

        alpha = self.alpha_committed

        # ----- Pushing the envelope past previous peaks?
        if eps > self.eps_max_pos_committed:
            # Increment damage by the plastic excursion on envelope
            delta_alpha = eps - self.eps_max_pos_committed
            self.alpha_trial = alpha + delta_alpha
            sigma, Et = self._envelope(eps, alpha=self.alpha_trial)
            self.eps_max_pos_trial = eps
            self.eps_max_neg_trial = self.eps_max_neg_committed
            self.eps_rev_trial = self.eps_rev_committed
            self.sigma_rev_trial = self.sigma_rev_committed
            self.direction_trial = +1
        elif eps < self.eps_max_neg_committed:
            delta_alpha = self.eps_max_neg_committed - eps
            self.alpha_trial = alpha + delta_alpha
            sigma, Et = self._envelope(eps, alpha=self.alpha_trial)
            self.eps_max_neg_trial = eps
            self.eps_max_pos_trial = self.eps_max_pos_committed
            self.eps_rev_trial = self.eps_rev_committed
            self.sigma_rev_trial = self.sigma_rev_committed
            self.direction_trial = -1
        else:
            # Inside the loop.
            self.alpha_trial = alpha       # no new envelope excursion
            self.eps_max_pos_trial = self.eps_max_pos_committed
            self.eps_max_neg_trial = self.eps_max_neg_committed
            has_yielded_pos = self.eps_max_pos_trial > self.eps_y
            has_yielded_neg = self.eps_max_neg_trial < -self.eps_y
            if not has_yielded_pos and not has_yielded_neg:
                sigma = self.E * eps
                Et = self.E
                self.eps_rev_trial = self.eps_rev_committed
                self.sigma_rev_trial = self.sigma_rev_committed
                self.direction_trial = inc_dir if inc_dir != 0 else self.direction_committed
            else:
                if inc_dir != self.direction_committed:
                    self.eps_rev_trial = self.eps_committed
                    self.sigma_rev_trial = self.sigma_committed
                    self.direction_trial = inc_dir
                else:
                    self.eps_rev_trial = self.eps_rev_committed
                    self.sigma_rev_trial = self.sigma_rev_committed
                    self.direction_trial = self.direction_committed
                if self.direction_trial > 0:
                    eps_target = self.eps_max_pos_trial
                else:
                    eps_target = self.eps_max_neg_trial
                # Reload target uses the *current* (possibly degraded)
                # envelope value at the historical peak.
                sigma_target, _ = self._envelope(eps_target,
                                                   alpha=self.alpha_trial)
                sigma, Et = self._reload_segment(
                    eps, self.direction_trial,
                    eps_target, sigma_target,
                    self.eps_rev_trial, self.sigma_rev_trial,
                )
        self.sigma_trial = sigma
        self.Et_trial = Et
        return sigma, Et

    # ------------------------------------------------------------ state
    def commit_state(self) -> None:
        self.eps_committed = self.eps_trial
        self.sigma_committed = self.sigma_trial
        self.eps_max_pos_committed = self.eps_max_pos_trial
        self.eps_max_neg_committed = self.eps_max_neg_trial
        self.eps_rev_committed = self.eps_rev_trial
        self.sigma_rev_committed = self.sigma_rev_trial
        self.direction_committed = self.direction_trial
        self.alpha_committed = self.alpha_trial

    def revert_state(self) -> None:
        self.eps_trial = self.eps_committed
        self.sigma_trial = self.sigma_committed
        self.eps_max_pos_trial = self.eps_max_pos_committed
        self.eps_max_neg_trial = self.eps_max_neg_committed
        self.eps_rev_trial = self.eps_rev_committed
        self.sigma_rev_trial = self.sigma_rev_committed
        self.direction_trial = self.direction_committed
        self.alpha_trial = self.alpha_committed

    def __repr__(self) -> str:
        return (
            f"UniaxialHysteretic(E={self.E:g}, sigma_y={self.sigma_y:g}, "
            f"b={self.b:g}, pinch_x={self.pinch_x:g}, "
            f"pinch_y={self.pinch_y:g})"
        )
