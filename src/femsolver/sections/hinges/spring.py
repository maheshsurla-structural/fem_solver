"""Bilinear moment-rotation spring with kinematic hardening.

A 1-D plasticity model with the canonical return-mapping algorithm.
Setting the post-yield slope ratio ``b = 0`` recovers
elastic-perfectly-plastic (EPP) behaviour, which is the simplest
practical hinge model.

State variables
---------------
* ``theta_p`` — accumulated plastic rotation
* ``q``        — back-stress (centre of yield surface in moment space);
                 only non-zero with kinematic hardening (``b > 0``)

The committed pair ``(theta_p_committed, q_committed)`` defines the
material's history at the end of the last converged Newton step. During
a Newton iteration, ``get_response`` consumes a *trial* total rotation
``theta`` and returns the corresponding moment and tangent, leaving
``(theta_p_trial, q_trial)`` updated. ``commit_state`` rolls the trial
values into the committed slot once the global solver converges;
``revert_state`` undoes any uncommitted updates if the step fails.

Algorithm (return mapping for 1-D J2 plasticity)
------------------------------------------------
1. Elastic predictor:  M_trial = K0 * (theta - theta_p_committed)
2. Yield-function trial:  f_trial = |M_trial - q_committed| - My
3. If f_trial <= 0  : elastic, no plastic flow
4. Else             : plastic flow,
                       d_lambda  = f_trial / (K0 + K_kin)
                       d_theta_p = sign(M_trial - q_committed) * d_lambda
                       theta_p_trial = theta_p_committed + d_theta_p
                       q_trial       = q_committed + K_kin * d_theta_p
                       M             = K0 * (theta - theta_p_trial)
                       K_tangent     = K0 * K_kin / (K0 + K_kin)  ==  b * K0

The kinematic-hardening modulus is
``K_kin = b * K0 / (1 - b)``, which is the relation that gives a
post-yield slope of exactly ``b * K0`` in the M-theta space.
"""
from __future__ import annotations

import math


class BilinearMomentRotationSpring:
    """Zero-length rotational spring with bilinear M-theta backbone.

    Parameters
    ----------
    K0 : float
        Initial (elastic) rotational stiffness, units of moment / radian.
    My : float
        Yield moment magnitude (positive). The yield surface in M-space
        is symmetric about the back-stress ``q``.
    b : float, default 0.0
        Post-yield stiffness ratio, ``b in [0, 1)``. ``b = 0`` gives
        elastic-perfectly-plastic; ``b > 0`` gives kinematic hardening
        with post-yield slope ``b * K0``.
    """

    def __init__(self, K0: float, My: float, b: float = 0.0):
        if K0 <= 0.0:
            raise ValueError(f"K0 must be positive, got {K0}")
        if My <= 0.0:
            raise ValueError(f"My must be positive, got {My}")
        if not (0.0 <= b < 1.0):
            raise ValueError(f"b must be in [0, 1), got {b}")
        self.K0 = float(K0)
        self.My = float(My)
        self.b = float(b)
        # Kinematic-hardening modulus that gives a post-yield slope of b*K0.
        # When b == 0 this is exactly 0 (perfectly plastic).
        self.K_kin = (self.b * self.K0 / (1.0 - self.b)) if self.b > 0.0 else 0.0
        # state
        self.theta_p_committed: float = 0.0
        self.theta_p_trial: float = 0.0
        self.q_committed: float = 0.0
        self.q_trial: float = 0.0
        # most recent response — useful for output and for the element
        self.M_trial: float = 0.0
        self.K_tangent: float = self.K0

    # ------------------------------------------------------------ response
    def get_response(self, theta: float) -> tuple[float, float]:
        """Return ``(M, K_tangent)`` for total rotation ``theta``.

        Side effect: updates the *trial* state. The committed state is
        unchanged until :meth:`commit_state` is called.
        """
        # Elastic predictor
        theta_e_trial = theta - self.theta_p_committed
        M_trial = self.K0 * theta_e_trial
        xi = M_trial - self.q_committed
        f_trial = abs(xi) - self.My

        if f_trial <= 0.0:
            # Elastic step
            self.theta_p_trial = self.theta_p_committed
            self.q_trial = self.q_committed
            self.M_trial = M_trial
            self.K_tangent = self.K0
            return M_trial, self.K0

        # Plastic step — return mapping
        sign = 1.0 if xi >= 0.0 else -1.0
        d_lambda = f_trial / (self.K0 + self.K_kin)
        d_theta_p = d_lambda * sign
        self.theta_p_trial = self.theta_p_committed + d_theta_p
        self.q_trial = self.q_committed + self.K_kin * d_theta_p
        M = self.K0 * (theta - self.theta_p_trial)
        K_t = self.K0 * self.K_kin / (self.K0 + self.K_kin) if self.K_kin > 0.0 else 0.0
        # Numerically: this equals b * K0 to within rounding
        self.M_trial = M
        self.K_tangent = K_t
        return M, K_t

    # -------------------------------------------------------------- state
    def commit_state(self) -> None:
        self.theta_p_committed = self.theta_p_trial
        self.q_committed = self.q_trial

    def revert_state(self) -> None:
        self.theta_p_trial = self.theta_p_committed
        self.q_trial = self.q_committed

    # ---------------------------------------------------------- diagnostics
    def yielded(self) -> bool:
        """True if the spring has accumulated any plastic rotation."""
        return not math.isclose(self.theta_p_committed, 0.0, abs_tol=0.0)

    def __repr__(self) -> str:
        return (
            f"BilinearMomentRotationSpring(K0={self.K0:g}, My={self.My:g}, "
            f"b={self.b:g})"
        )
