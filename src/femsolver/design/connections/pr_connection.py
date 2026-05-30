"""Partially-Restrained (PR) connection models.

A PR connection lies between the idealised pinned ("simple") and
fully-restrained ("rigid") limits. Its moment-rotation behaviour is
characterised by an initial rotational stiffness ``R_ki``, a
post-yield ("plastic") stiffness ``R_kp``, and a transition
parameter ``M_0`` (and shape exponent ``n``).

The **Richard-Abbott four-parameter** model (Richard & Abbott 1975)
gives a smooth single-curve relation::

    M(theta) = (R_ki - R_kp) theta
               / ( 1 + |(R_ki - R_kp) theta / M_0|^n )^(1/n)
              + R_kp theta

Parameters
----------
* ``R_ki`` (N·m/rad): initial rotational stiffness.
* ``R_kp`` (N·m/rad): post-yield (or strain-hardening) rotational
  stiffness, ``0 <= R_kp <= R_ki``.
* ``M_0`` (N·m): characteristic moment at which ``M`` transitions
  from the elastic to the plastic asymptote.
* ``n`` (-): sharpness exponent (typical 1..5; larger = sharper knee).

For commonly-encountered PR connection types AISC ASD/LRFD literature
suggests parameter ranges; we ship a small preset library
(:func:`Pr_preset`) covering:

* ``"top_seat_double_web"`` -- top + seat angle with double web angle.
* ``"end_plate_4_bolts"``  -- unstiffened 4-bolt end-plate connection.
* ``"end_plate_extended"`` -- extended (stiff) end-plate connection.
* ``"tee_stub"``            -- T-stub connection.

These presets give *order-of-magnitude* values; for design verification
use full member-level calibration.

References
----------
* Richard, R.M. & Abbott, B.J. (1975) "Versatile elastic-plastic
  stress-strain formula." *J. Eng. Mech.*, 101(4), 511-515.
* Chen, W.F., Goto, Y., Liew, J.Y.R. (1996). *Stability Design of
  Semi-Rigid Frames*. Wiley.
"""
from __future__ import annotations

from dataclasses import dataclass


@dataclass
class RichardAbbottParams:
    """Four-parameter Richard-Abbott M-theta curve.

    Attributes
    ----------
    R_ki : float
        Initial rotational stiffness (N·m/rad).
    R_kp : float
        Post-yield rotational stiffness (N·m/rad).
    M_0 : float
        Transition moment (N·m).
    n : float
        Sharpness exponent (dimensionless).
    """

    R_ki: float
    R_kp: float
    M_0: float
    n: float

    def __post_init__(self) -> None:
        if self.R_ki <= 0.0:
            raise ValueError("R_ki must be > 0")
        if self.R_kp < 0.0 or self.R_kp > self.R_ki:
            raise ValueError("R_kp must be in [0, R_ki]")
        if self.M_0 <= 0.0:
            raise ValueError("M_0 must be > 0")
        if self.n <= 0.0:
            raise ValueError("n must be > 0")

    def M(self, theta: float) -> float:
        """Moment at rotation ``theta``."""
        sign = 1.0 if theta >= 0.0 else -1.0
        t = abs(theta)
        dk = self.R_ki - self.R_kp
        if dk == 0.0:
            return float(sign * self.R_ki * t)
        x = dk * t
        denom = (1.0 + (x / self.M_0) ** self.n) ** (1.0 / self.n)
        return float(sign * (x / denom + self.R_kp * t))

    def tangent(self, theta: float) -> float:
        """Tangent stiffness ``dM/dtheta`` at ``theta``."""
        t = abs(theta)
        dk = self.R_ki - self.R_kp
        if dk == 0.0:
            return float(self.R_ki)
        x = dk * t / self.M_0
        x_n = x ** self.n
        denom_pow = (1.0 + x_n) ** (1.0 / self.n)
        # dM/dtheta = dk / (1 + x^n)^((n+1)/n) + R_kp
        dM_dtheta = dk / (denom_pow ** (self.n + 1.0)) ** (1.0 / self.n)
        # Equivalent simpler form using known relation:
        dM_dtheta = dk / ((1.0 + x_n)) ** ((self.n + 1.0) / self.n) + self.R_kp
        return float(dM_dtheta)


# ============================================================ presets

_PR_PRESETS = {
    "top_seat_double_web": dict(
        R_ki=2.5e7,                # ~ 25 MN·m/rad
        R_kp=1.0e6,
        M_0=80.0e3,                # ~ 80 kN·m
        n=1.2,
    ),
    "end_plate_4_bolts": dict(
        R_ki=4.0e7,
        R_kp=2.0e6,
        M_0=150.0e3,
        n=1.5,
    ),
    "end_plate_extended": dict(
        R_ki=1.2e8,
        R_kp=5.0e6,
        M_0=300.0e3,
        n=2.0,
    ),
    "tee_stub": dict(
        R_ki=6.0e7,
        R_kp=3.0e6,
        M_0=200.0e3,
        n=1.7,
    ),
}


def Pr_preset(name: str) -> RichardAbbottParams:
    """Return a Richard-Abbott parameter set for a named PR type.

    Order-of-magnitude values; calibrate against test data or
    full-section component models before design use.
    """
    if name not in _PR_PRESETS:
        raise ValueError(
            f"unknown PR preset {name!r}; available: "
            f"{list(_PR_PRESETS)}"
        )
    return RichardAbbottParams(**_PR_PRESETS[name])
