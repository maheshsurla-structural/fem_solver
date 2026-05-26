"""Menegotto-Pinto cyclic steel model.

The Menegotto-Pinto (1973) curve smoothly transitions between an
elastic asymptote and a kinematic-hardening yield asymptote:

    sigma_star = b * eps_star + (1 - b) * eps_star / (1 + eps_star^R)^(1/R)

where

    eps_star   = (eps - eps_r) / (eps_0 - eps_r)
    sigma_star = (sigma - sigma_r) / (sigma_0 - sigma_r)

``(eps_r, sigma_r)`` is the most recent load-reversal point and
``(eps_0, sigma_0)`` is the *asymptotic yield point* -- the
intersection of the elastic reload line through the reversal point
and the next-direction hardening asymptote. The curvature parameter
``R`` controls the Bauschinger effect: large ``R`` (~20) gives a
sharp elastic-to-plastic transition, small ``R`` (~5) gives a soft,
round one typical of well-aged carbon steel.

Compared to ``UniaxialBilinear`` (kinematic hardening with a sharp
yield kink), Menegotto-Pinto produces realistic *smooth* hysteresis
loops -- the canonical input for seismic RC fiber-section analysis
together with the concrete models from Phase 16.0.

Notes
-----
This implementation uses a *fixed* curvature ``R``; the full
Giuffré-Menegotto-Pinto model evolves ``R`` with the previous
plastic excursion to model cyclic-hardening / softening, and is a
straightforward refinement on top of this base.
"""
from __future__ import annotations

from femsolver.materials.uniaxial.base import UniaxialMaterial


class UniaxialMenegottoPinto(UniaxialMaterial):
    """Cyclic-steel Menegotto-Pinto material with optional
    Giuffre-Menegotto-Pinto evolving R (Filippou et al. 1983).

    Parameters
    ----------
    E : float
        Initial elastic modulus.
    sigma_y : float
        Yield stress magnitude (positive).
    b : float, default 0.01
        Strain-hardening ratio in ``[0, 1)``. ``b = 0`` is EPP; values
        near ``0.005-0.02`` are typical for reinforcing steel.
    R0 : float, default 20.0
        Initial Menegotto-Pinto curvature parameter (used before any
        cyclic excursion). Backwards-compat: also accepted as ``R``.
    a1 : float, default 0.0
        First Giuffre curvature-evolution coefficient. ``a1 = 0``
        keeps R constant at ``R0`` (pure Menegotto-Pinto, no cyclic
        curvature evolution). Typical Filippou value: ``a1 = 18.5``.
    a2 : float, default 0.15
        Second Giuffre coefficient. Together with ``a1`` gives

            R(xi) = R0 - a1 * xi / (a2 + xi)

        where ``xi`` is a strain-excursion measure normalized by the
        yield strain.
    R : float, optional
        Backwards-compat alias for ``R0``. If given, ``R0`` is set to
        this value and the Giuffre evolution is disabled (``a1 = 0``).

    Notes
    -----
    With ``a1 = 0`` the model reduces to the classical Menegotto-Pinto
    of Phase 16.3. With ``a1 > 0``, each load reversal recomputes
    ``R`` based on the strain swing since the previous extreme in
    the opposite direction. Larger swings produce smaller R (rounder
    Bauschinger transition), matching real steel behaviour under
    accumulated cyclic loading.
    """

    def __init__(self, E: float, sigma_y: float, b: float = 0.01,
                 R0: float = 20.0, a1: float = 0.0, a2: float = 0.15,
                 R: float | None = None):
        if E <= 0.0:
            raise ValueError(f"E must be positive, got {E}")
        if sigma_y <= 0.0:
            raise ValueError(f"sigma_y must be positive, got {sigma_y}")
        if not (0.0 <= b < 1.0):
            raise ValueError(f"b must be in [0, 1), got {b}")
        if a1 < 0.0:
            raise ValueError(f"a1 must be >= 0, got {a1}")
        if a2 <= 0.0:
            raise ValueError(f"a2 must be positive, got {a2}")
        # Back-compat: if R is given, use it as R0 with no evolution.
        if R is not None:
            R0 = float(R)
            a1 = 0.0
        if R0 <= 1.0:
            raise ValueError(f"R0 must be > 1, got {R0}")
        self.E = float(E)
        self.sigma_y = float(sigma_y)
        self.eps_y = sigma_y / E
        self.b = float(b)
        self.R0 = float(R0)
        self.a1 = float(a1)
        self.a2 = float(a2)
        # R is updated on each reversal when a1 > 0; otherwise constant.
        self.R = self.R0
        # ----- state: committed -----
        self.eps_committed: float = 0.0
        self.sigma_committed: float = 0.0
        self.eps_r_committed: float = 0.0
        self.sigma_r_committed: float = 0.0
        self.eps0_committed: float = self.eps_y
        self.sigma0_committed: float = self.sigma_y
        self.direction_committed: int = +1
        self.eps_max_pos_committed: float = 0.0
        self.eps_min_neg_committed: float = 0.0
        self.R_committed: float = self.R0
        # ----- state: trial -----
        self.eps_trial: float = 0.0
        self.sigma_trial: float = 0.0
        self.eps_r_trial: float = 0.0
        self.sigma_r_trial: float = 0.0
        self.eps0_trial: float = self.eps_y
        self.sigma0_trial: float = self.sigma_y
        self.direction_trial: int = +1
        self.eps_max_pos_trial: float = 0.0
        self.eps_min_neg_trial: float = 0.0
        self.R_trial: float = self.R0
        self.Et_trial: float = self.E

    # ------------------------------------------------------------ helpers
    def _compute_asymptote(self, eps_r: float, sigma_r: float,
                            direction: int) -> tuple[float, float]:
        """Intersect the elastic reload line through ``(eps_r, sigma_r)``
        (slope E) with the hardening asymptote in the chosen direction:

            sigma = b E eps + direction * (1 - b) sigma_y

        Returns the asymptotic yield point ``(eps_0, sigma_0)``.
        """
        b = self.b
        E = self.E
        sigma_y = self.sigma_y
        eps0 = (
            direction * sigma_y / E
            + (E * eps_r - sigma_r) / (E * (1.0 - b))
        )
        sigma0 = sigma_r + E * (eps0 - eps_r)
        return eps0, sigma0

    def _giuffre_R(self, xi: float) -> float:
        """Giuffre-Menegotto-Pinto evolving curvature parameter.

            R = R0 - a1 * xi / (a2 + xi)

        With ``a1 = 0`` returns the constant ``R0``.
        """
        if self.a1 == 0.0 or xi <= 0.0:
            return self.R0
        R = self.R0 - self.a1 * xi / (self.a2 + xi)
        # Clamp to a reasonable minimum so the MP curve stays well-defined
        return max(R, 1.5)

    # ------------------------------------------------------------ response
    def get_response(self, eps: float) -> tuple[float, float]:
        eps = float(eps)
        self.eps_trial = eps
        delta = eps - self.eps_committed
        if delta == 0.0:
            inc_dir = self.direction_committed
        else:
            inc_dir = +1 if delta > 0.0 else -1

        # Update peak strain trackers based on the committed history.
        eps_max_pos = max(self.eps_max_pos_committed, self.eps_committed)
        eps_min_neg = min(self.eps_min_neg_committed, self.eps_committed)
        self.eps_max_pos_trial = eps_max_pos
        self.eps_min_neg_trial = eps_min_neg

        # Detect reversal vs the committed branch
        if inc_dir != self.direction_committed:
            eps_r = self.eps_committed
            sigma_r = self.sigma_committed
            eps0, sigma0 = self._compute_asymptote(eps_r, sigma_r, inc_dir)
            self.eps_r_trial = eps_r
            self.sigma_r_trial = sigma_r
            self.eps0_trial = eps0
            self.sigma0_trial = sigma0
            self.direction_trial = inc_dir
            # Giuffre R update: xi = magnitude of the just-completed
            # half-cycle (swing from previous reversal to current).
            xi = abs(eps_r - self.eps_r_committed) / self.eps_y
            self.R_trial = self._giuffre_R(xi)
        else:
            self.eps_r_trial = self.eps_r_committed
            self.sigma_r_trial = self.sigma_r_committed
            self.eps0_trial = self.eps0_committed
            self.sigma0_trial = self.sigma0_committed
            self.direction_trial = self.direction_committed
            self.R_trial = self.R_committed

        eps_r = self.eps_r_trial
        sigma_r = self.sigma_r_trial
        eps0 = self.eps0_trial
        sigma0 = self.sigma0_trial
        b = self.b
        R = self.R_trial

        denom_eps = eps0 - eps_r
        if abs(denom_eps) < 1.0e-30:
            # Degenerate branch -- treat as fully plastic
            self.sigma_trial = sigma_r
            self.Et_trial = self.E * b
            return sigma_r, self.E * b

        eps_star = (eps - eps_r) / denom_eps
        if eps_star < 0.0:
            # Tiny intra-iteration overshoot back across the reversal --
            # clamp to keep the formulas well-defined.
            eps_star = 0.0

        # Menegotto-Pinto curve
        u = eps_star ** R
        denom_factor = (1.0 + u) ** (1.0 / R)
        sigma_star = b * eps_star + (1.0 - b) * eps_star / denom_factor
        sigma = sigma_r + sigma_star * (sigma0 - sigma_r)

        # Tangent: d sigma / d eps = (sigma0 - sigma_r)/(eps0 - eps_r) * d sigma_star / d eps_star
        # d sigma_star / d eps_star = b + (1 - b) / (1 + eps_star^R)^(1 + 1/R)
        d_sigma_star = b + (1.0 - b) / (1.0 + u) ** (1.0 + 1.0 / R)
        Et = (sigma0 - sigma_r) / denom_eps * d_sigma_star

        self.sigma_trial = sigma
        self.Et_trial = Et
        return sigma, Et

    # ------------------------------------------------------------ state
    def commit_state(self) -> None:
        self.eps_committed = self.eps_trial
        self.sigma_committed = self.sigma_trial
        self.eps_r_committed = self.eps_r_trial
        self.sigma_r_committed = self.sigma_r_trial
        self.eps0_committed = self.eps0_trial
        self.sigma0_committed = self.sigma0_trial
        self.direction_committed = self.direction_trial
        self.eps_max_pos_committed = self.eps_max_pos_trial
        self.eps_min_neg_committed = self.eps_min_neg_trial
        self.R_committed = self.R_trial
        self.R = self.R_trial

    def revert_state(self) -> None:
        self.eps_trial = self.eps_committed
        self.sigma_trial = self.sigma_committed
        self.eps_r_trial = self.eps_r_committed
        self.sigma_r_trial = self.sigma_r_committed
        self.eps0_trial = self.eps0_committed
        self.sigma0_trial = self.sigma0_committed
        self.direction_trial = self.direction_committed
        self.eps_max_pos_trial = self.eps_max_pos_committed
        self.eps_min_neg_trial = self.eps_min_neg_committed
        self.R_trial = self.R_committed
        self.R = self.R_committed

    def __repr__(self) -> str:
        return (
            f"UniaxialMenegottoPinto(E={self.E:g}, sigma_y={self.sigma_y:g}, "
            f"b={self.b:g}, R0={self.R0:g}, a1={self.a1:g}, a2={self.a2:g})"
        )
