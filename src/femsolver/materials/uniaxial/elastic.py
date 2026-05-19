"""Linear elastic uniaxial material — the simplest possible constitutive
law for a fiber. Stateless: ``sigma = E * eps`` regardless of history.
"""
from __future__ import annotations

from femsolver.materials.uniaxial.base import UniaxialMaterial


class UniaxialElastic(UniaxialMaterial):
    """``sigma = E * eps``.  Tangent ``Et = E`` is constant.

    Parameters
    ----------
    E : float
        Young's modulus, must be positive.
    """

    def __init__(self, E: float):
        if E <= 0.0:
            raise ValueError(f"E must be positive, got {E}")
        self.E = float(E)

    def get_response(self, eps: float) -> tuple[float, float]:
        return self.E * float(eps), self.E

    def __repr__(self) -> str:
        return f"UniaxialElastic(E={self.E:g})"
