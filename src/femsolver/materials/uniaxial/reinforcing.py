"""Reinforcing-steel monotonic stress-strain model (Park strain hardening).

:class:`UniaxialReinforcingSteel` implements the classic Park & Paulay /
Kent-Park monotonic backbone for reinforcing bars:

* **Elastic** ``0 <= eps <= eps_y``:            ``sigma = E * eps``
* **Yield plateau** ``eps_y < eps <= eps_sh``:  ``sigma = f_y``
* **Strain hardening** ``eps_sh < eps <= eps_su``: a curved rise from
  ``f_y`` at ``eps_sh`` to the ultimate stress ``f_su`` at ``eps_su``:

      sigma = f_y [ (m x + 2)/(60 x + 2) + x (60 - m)/(2 (30 r + 1)^2) ]

  with ``x = eps - eps_sh``, ``r = eps_su - eps_sh`` and
  ``m = [ (f_su/f_y)(30 r + 1)^2 - 60 r - 1 ] / (15 r^2)`` chosen so the
  curve passes through ``(eps_sh, f_y)`` and ``(eps_su, f_su)``.

Beyond ``eps_su`` the stress is held at ``f_su`` (fracture is governed by
the section driver's steel-rupture strain, not this curve). The response
is odd-symmetric in compression (``sigma(-eps) = -sigma(eps)``), and the
model is monotonic / history-independent -- appropriate for monotonic
moment-curvature and ULS section analysis.

The name in Midas GSD is "Park Strain Hardening".
"""
from __future__ import annotations

import math

from femsolver.materials.uniaxial.base import UniaxialMaterial


class UniaxialReinforcingSteel(UniaxialMaterial):
    """Park strain-hardening reinforcing-steel backbone (monotonic).

    Parameters
    ----------
    E : float
        Elastic modulus (positive).
    f_y : float
        Yield strength (positive magnitude).
    f_su : float
        Ultimate (tensile) strength at ``eps_su`` (positive, ``> f_y``).
    eps_sh : float
        Strain at the onset of strain hardening (``>= eps_y = f_y / E``),
        typically 0.005-0.015.
    eps_su : float
        Strain at the ultimate stress (``> eps_sh``), typically 0.08-0.12.
    """

    def __init__(self, E: float, f_y: float, f_su: float,
                 eps_sh: float, eps_su: float):
        if E <= 0.0:
            raise ValueError(f"E must be positive, got {E}")
        if f_y <= 0.0:
            raise ValueError(f"f_y must be positive, got {f_y}")
        if f_su <= f_y:
            raise ValueError(f"f_su must exceed f_y, got f_su={f_su}, "
                             f"f_y={f_y}")
        eps_y = f_y / E
        if eps_sh < eps_y:
            raise ValueError(f"eps_sh must be >= eps_y = f_y/E ({eps_y:g}), "
                             f"got {eps_sh}")
        if eps_su <= eps_sh:
            raise ValueError(f"eps_su must exceed eps_sh, got eps_su={eps_su}, "
                             f"eps_sh={eps_sh}")
        self.E = float(E)
        self.f_y = float(f_y)
        self.f_su = float(f_su)
        self.eps_y = float(eps_y)
        self.eps_sh = float(eps_sh)
        self.eps_su = float(eps_su)
        self.E0 = float(E)
        # Strain-hardening curve constants (independent of strain).
        r = self.eps_su - self.eps_sh
        self._r = r
        self._k = (30.0 * r + 1.0) ** 2                       # (30r + 1)^2
        self._m = (self.f_su / self.f_y * self._k - 60.0 * r - 1.0) \
            / (15.0 * r * r)

    # ------------------------------------------------------ backbone
    def _backbone(self, e: float) -> tuple[float, float]:
        """Tension-side ``(sigma, Et)`` magnitudes at strain ``e >= 0``."""
        if e <= self.eps_y:
            return self.E * e, self.E
        if e <= self.eps_sh:
            return self.f_y, 0.0
        if e <= self.eps_su:
            x = e - self.eps_sh
            m, k = self._m, self._k
            d = 60.0 * x + 2.0
            sigma = self.f_y * ((m * x + 2.0) / d + x * (60.0 - m) / (2.0 * k))
            Et = self.f_y * ((2.0 * m - 120.0) / (d * d)
                             + (60.0 - m) / (2.0 * k))
            return sigma, Et
        return self.f_su, 0.0            # past eps_su: hold (driver ends run)

    # ------------------------------------------------------ get_response
    def get_response(self, eps: float) -> tuple[float, float]:
        eps = float(eps)
        if eps == 0.0:
            return 0.0, self.E
        sigma, Et = self._backbone(abs(eps))
        return math.copysign(sigma, eps), Et

    def __repr__(self) -> str:
        return (
            f"UniaxialReinforcingSteel(E={self.E:g}, f_y={self.f_y:g}, "
            f"f_su={self.f_su:g}, eps_sh={self.eps_sh:g}, "
            f"eps_su={self.eps_su:g})"
        )
