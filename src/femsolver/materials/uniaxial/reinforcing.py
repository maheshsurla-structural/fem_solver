"""Reinforcing-steel stress-strain models (Park strain hardening).

:class:`UniaxialReinforcingSteel` implements the classic Park & Paulay /
Kent-Park **monotonic** backbone for reinforcing bars, and
:class:`ReinforcingSteelKinematic` wraps that same backbone in a kinematic-
hardening return map for **cyclic** analysis (the benchmark tools' rule).

The monotonic backbone:

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


class ReinforcingSteelKinematic(UniaxialMaterial):
    """Park strain-hardening steel with **kinematic hardening** (cyclic).

    The cyclic reinforcing-steel law used by both benchmark tools -- Midas
    "Park PM" and CSI "Simple" steel with a *Kinematic* hysteresis rule
    (plan §2.2 / §5.3 G3). It reuses the exact monotonic Park backbone of
    :class:`UniaxialReinforcingSteel`, so a monotonic tension push reproduces
    that curve fibre-for-fibre; on reversal it unloads elastically (slope
    ``E``) and translates the backbone through a back-stress ``q`` -- classic
    kinematic (Prager-type) hardening generalised to the Park backbone's
    varying hardening modulus.

    Formulation (return mapping, radius = ``f_y`` fixed, centre ``q`` moving)::

        sigma_trial = E (eps - eps_p)                    # elastic predictor
        f = |sigma_trial - q| - f_y                      # yield function
        f <= 0 : elastic, Et = E
        f  > 0 : plastic. Solve  R(dl) = |xi| - f_y - E dl - dQ(dl) = 0
                 dQ(dl) = alpha(p + dl) - alpha(p)        (= integral H dp)
                 eps_p += sign*dl ;  q += sign*dQ ;  p += dl
                 sigma = E (eps - eps_p) ;  Et = E H / (E + H)

    where ``alpha(p) = g(e(p)) - f_y`` is the virgin back-stress magnitude at
    accumulated plastic strain ``p`` (``g`` the Park backbone, ``e(p)`` its
    strain), and ``H(p) = d alpha / d p`` the kinematic modulus. Because
    ``integral H dp = alpha(p + dl) - alpha(p)`` exactly, no numerical
    integration of ``H`` is needed. With a constant ``H`` this reduces to the
    bilinear kinematic return map (:class:`UniaxialBilinear`).

    Parameters are identical to :class:`UniaxialReinforcingSteel`.
    """

    def __init__(self, E: float, f_y: float, f_su: float,
                 eps_sh: float, eps_su: float):
        self.backbone = UniaxialReinforcingSteel(E, f_y, f_su, eps_sh, eps_su)
        self.E = float(E)
        self.E0 = float(E)
        self.f_y = float(f_y)
        self.f_su = float(f_su)
        self.eps_y = self.backbone.eps_y
        self.eps_sh = float(eps_sh)
        self.eps_su = float(eps_su)
        # committed state
        self.eps_p_committed: float = 0.0
        self.q_committed: float = 0.0
        self.p_committed: float = 0.0            # accumulated plastic strain
        # trial mirrors
        self.eps_p_trial: float = 0.0
        self.q_trial: float = 0.0
        self.p_trial: float = 0.0
        # last response
        self.sigma_trial: float = 0.0
        self.Et: float = self.E

    # ---------------------------------------------- backbone helpers
    def _bb_strain_for_p(self, p: float) -> float:
        """Backbone strain ``e >= eps_y`` whose accumulated plastic strain is
        ``p``, i.e. the root of ``e - g(e)/E = p`` (monotone in ``e``)."""
        if p <= 0.0:
            return self.eps_y
        e = self.eps_sh + p            # good initial guess (plateau has p=0)
        for _ in range(60):
            sig, Et = self.backbone._backbone(e)
            h = e - sig / self.E - p
            hp = 1.0 - Et / self.E
            if abs(hp) < 1e-14:
                break
            step = h / hp
            e -= step
            if e < self.eps_y:
                e = self.eps_y
            if abs(step) < 1e-15:
                break
        return e

    def _alpha_H(self, p: float) -> tuple[float, float]:
        """``(alpha, H)`` at accumulated plastic strain ``p``: back-stress
        magnitude ``alpha = g(e) - f_y`` and kinematic modulus
        ``H = E*Et/(E - Et)``."""
        e = self._bb_strain_for_p(p)
        sig, Et = self.backbone._backbone(e)
        alpha = sig - self.f_y
        denom = self.E - Et
        H = (self.E * Et / denom) if denom > 1e-9 else 0.0
        return alpha, H

    # ---------------------------------------------- get_response
    def get_response(self, eps: float) -> tuple[float, float]:
        eps = float(eps)
        sigma_trial = self.E * (eps - self.eps_p_committed)
        xi = sigma_trial - self.q_committed
        f_trial = abs(xi) - self.f_y
        if f_trial <= 0.0:                        # elastic step
            self.eps_p_trial = self.eps_p_committed
            self.q_trial = self.q_committed
            self.p_trial = self.p_committed
            self.sigma_trial = sigma_trial
            self.Et = self.E
            return sigma_trial, self.E
        # plastic step: local Newton on the plastic multiplier dl >= 0
        sign = 1.0 if xi >= 0.0 else -1.0
        p0 = self.p_committed
        alpha0, _ = self._alpha_H(p0)
        absxi = abs(xi)
        dl = f_trial / (self.E + self._alpha_H(p0)[1])   # bilinear first step
        for _ in range(60):
            alpha1, H1 = self._alpha_H(p0 + dl)
            R = absxi - self.f_y - self.E * dl - (alpha1 - alpha0)
            dRdl = -self.E - H1
            step = R / dRdl
            dl -= step
            if dl < 0.0:
                dl = 0.0
            if abs(step) < 1e-14:
                break
        alpha1, H1 = self._alpha_H(p0 + dl)
        self.p_trial = p0 + dl
        self.eps_p_trial = self.eps_p_committed + sign * dl
        self.q_trial = self.q_committed + sign * (alpha1 - alpha0)
        sigma = self.E * (eps - self.eps_p_trial)
        Et = (self.E * H1 / (self.E + H1)) if H1 > 0.0 else 0.0
        self.sigma_trial = sigma
        self.Et = Et
        return sigma, Et

    # ---------------------------------------------- state
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
            f"ReinforcingSteelKinematic(E={self.E:g}, f_y={self.f_y:g}, "
            f"f_su={self.f_su:g}, eps_sh={self.eps_sh:g}, "
            f"eps_su={self.eps_su:g})"
        )
