"""Compression-only / tension-only gap uniaxial material.

A "gap" material has zero stiffness over part of its strain range
(the *open* side) and linear-elastic stiffness over the other (the
*closed* side). It is the canonical building block for:

* Foundation uplift (column above ground, contact-only spring below)
* Pounding between adjacent buildings
* Stop-and-go isolator displacement limits
* Cracked-section / one-sided constitutive behaviour

Sign convention
---------------
``epsilon > 0`` means the element is **elongating** (the two end
nodes are moving apart). For a "compression-only" gap, the spring is
active when ``epsilon <= -gap`` (the gap has closed by at least
``gap``) and inactive otherwise. For "tension-only", the spring is
active when ``epsilon >= +gap``.

Initial state
-------------
With ``gap = 0`` the spring engages at zero relative displacement.
``gap > 0`` introduces an initial separation: the spring stays
inactive until the two nodes move toward each other by at least
``gap`` (compression-only) or apart by at least ``gap`` (tension-only).
"""
from __future__ import annotations

from femsolver.materials.uniaxial.base import UniaxialMaterial


class UniaxialGap(UniaxialMaterial):
    """One-sided linear-elastic gap material.

    Parameters
    ----------
    E : float
        Elastic stiffness when the gap is closed (positive).
    gap : float, default 0.0
        Initial gap distance (non-negative). Zero gives an "instant"
        contact at ``epsilon = 0``; positive values delay engagement.
    kind : ``"compression"`` (default) or ``"tension"``
        Which side of the relative-displacement axis the spring acts on.
    """

    def __init__(self, E: float, gap: float = 0.0,
                 kind: str = "compression"):
        if E <= 0.0:
            raise ValueError(f"E must be positive, got {E}")
        if gap < 0.0:
            raise ValueError(f"gap must be non-negative, got {gap}")
        if kind not in ("compression", "tension"):
            raise ValueError(
                f"kind must be 'compression' or 'tension', got {kind!r}"
            )
        self.E = float(E)
        self.gap = float(gap)
        self.kind = kind
        # Stateless gap: no history. Trial returns the same response
        # each time get_response is called.
        self.sigma_trial: float = 0.0
        self.Et_trial: float = 0.0

    def get_response(self, eps: float) -> tuple[float, float]:
        eps = float(eps)
        if self.kind == "compression":
            # Active for eps <= -gap: spring closes
            threshold = -self.gap
            if eps <= threshold:
                sigma = self.E * (eps - threshold)
                Et = self.E
            else:
                sigma = 0.0
                Et = 0.0
        else:  # tension
            threshold = +self.gap
            if eps >= threshold:
                sigma = self.E * (eps - threshold)
                Et = self.E
            else:
                sigma = 0.0
                Et = 0.0
        self.sigma_trial = sigma
        self.Et_trial = Et
        return sigma, Et

    def commit_state(self) -> None:
        # Stateless model
        return None

    def revert_state(self) -> None:
        return None

    def __repr__(self) -> str:
        return (
            f"UniaxialGap(E={self.E:g}, gap={self.gap:g}, kind={self.kind!r})"
        )
