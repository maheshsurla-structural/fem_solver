"""Cable elements with sag corrections + catenary closed-form refs.

A real cable stay (or any inclined cable under self-weight) does not
behave as a straight rod: gravity makes it sag, and that sag reduces
the effective axial stiffness because part of every axial force
change is absorbed by changes in the sag profile, not pure
elongation.

**Ernst (1965) equivalent modulus** is the standard structural-
engineering closed form for this::

    E_eq = E / (1 + (gamma_eff · L_h)^2 · A_c · E / (12 · T^3))

where ``gamma_eff`` is the cable's effective unit weight per unit
length, ``L_h`` is the horizontal chord length, ``A_c`` is the cross-
section area, and ``T`` is the chord tension at the operating point.
For high tensions the sag correction vanishes and ``E_eq -> E``; for
low tensions ``E_eq`` can drop to a fraction of ``E``.

This module provides:

* :class:`CableElement2D` -- a 2-node 2D truss element whose
  ``K_global`` uses ``E_eq`` instead of ``E``. Tension-only behavior
  is left to the caller (sized for the operating tension that
  determines E_eq).
* :func:`ernst_equivalent_modulus` -- the closed-form helper.
* :func:`catenary_sag` -- analytical sag profile for verification.
* :func:`catenary_max_tension` -- closed-form maximum tension at the
  cable ends for given chord, sag, and self-weight.
"""
from __future__ import annotations

import math
from dataclasses import dataclass

import numpy as np

from femsolver.elements.truss import Truss2D


# ============================================================ Ernst modulus

def ernst_equivalent_modulus(
    *,
    E: float, A: float,
    L_h: float, gamma_eff: float,
    T: float,
) -> float:
    """Ernst (1965) equivalent axial modulus for a sagging cable.

    ``E_eq = E / (1 + (gamma_eff · L_h)^2 · A · E / (12 · T^3))``

    Parameters
    ----------
    E : float
        Bare axial modulus (Pa).
    A : float
        Cable cross-section area (m^2).
    L_h : float
        Horizontal chord length (m).
    gamma_eff : float
        Effective weight per unit length (N/m). For a horizontal
        cable, this is the unit weight directly; for an inclined
        cable, it is the component perpendicular to the chord.
    T : float
        Chord tension at the operating point (N, > 0).
    """
    if E <= 0.0 or A <= 0.0 or L_h <= 0.0 or T <= 0.0:
        raise ValueError("E, A, L_h, T must all be > 0")
    if gamma_eff < 0.0:
        raise ValueError("gamma_eff must be >= 0")
    sag_term = (gamma_eff * L_h) ** 2 * A * E / (12.0 * T ** 3)
    return float(E / (1.0 + sag_term))


# ============================================================ cable element

class CableElement2D(Truss2D):
    """2D truss with the Ernst equivalent modulus.

    The element re-evaluates the equivalent modulus on every stiffness
    request: if the user supplies an operating-point tension
    ``T_operating`` at construction, that value is used for
    ``E_eq``; otherwise, the bare material modulus is used.

    Parameters
    ----------
    tag, nodes, material, area : as for :class:`Truss2D`.
    gamma_eff : float
        Cable weight per unit length (N/m). For an inclined cable,
        this is the component perpendicular to the chord direction.
    T_operating : float, optional
        Chord tension at the design / operating state (N). If supplied,
        the sag correction is applied and ``K_global`` returns the
        sag-corrected stiffness. If omitted, the element behaves as
        a regular Truss2D.
    """

    def __init__(
        self,
        tag: int,
        nodes,
        material,
        area: float,
        *,
        gamma_eff: float = 0.0,
        T_operating: float | None = None,
    ):
        super().__init__(tag, nodes, material, area)
        self.gamma_eff = float(gamma_eff)
        self.T_operating = (None if T_operating is None
                            else float(T_operating))

    def effective_modulus(self) -> float:
        """Return the (Ernst-corrected) axial modulus."""
        if self.T_operating is None or self.gamma_eff == 0.0:
            return self.material.E
        coords = self.node_coords()
        # Horizontal chord length
        L_h = abs(float(coords[1][0] - coords[0][0]))
        if L_h <= 0.0:
            return self.material.E
        return ernst_equivalent_modulus(
            E=self.material.E, A=self.area,
            L_h=L_h, gamma_eff=self.gamma_eff,
            T=self.T_operating,
        )

    def K_global(self) -> np.ndarray:
        """Stiffness with Ernst equivalent modulus."""
        # Mirror Truss2D.K_global but with E_eq
        coords = self.node_coords()
        d = coords[1] - coords[0]
        L = float(np.linalg.norm(d))
        n = d / L
        c, s = float(n[0]), float(n[1])
        E_eq = self.effective_modulus()
        EAoL = E_eq * self.area / L
        block = np.array([[c * c, c * s], [c * s, s * s]])
        K = np.zeros((4, 4))
        K[0:2, 0:2] = block
        K[2:4, 2:4] = block
        K[0:2, 2:4] = -block
        K[2:4, 0:2] = -block
        return EAoL * K


# ============================================================ catenary closed-form

@dataclass
class CatenaryResult:
    """Closed-form catenary cable properties.

    Attributes
    ----------
    H : float
        Horizontal tension component (N).
    T_max : float
        Maximum tension along the cable (N), at the upper end.
    sag_max : float
        Maximum vertical sag below the chord (m).
    L_arc : float
        Actual cable arc length (m, > chord length).
    """

    H: float
    T_max: float
    sag_max: float
    L_arc: float


def catenary_sag(
    *,
    L_h: float, w: float, H: float,
    x: float | None = None,
) -> float:
    """Catenary sag at horizontal distance ``x`` from one support.

    For a cable spanning two equal-elevation supports of horizontal
    spacing ``L_h`` under uniform weight ``w`` (N/m) with horizontal
    tension ``H``::

        y(x) = (H / w) (cosh(w x / H) - 1) - (H / w) (cosh(w L_h / (2H)) - 1)

    The mid-span sag is::

        f = (H / w)(cosh(w L_h / (2H)) - 1)

    Parameters
    ----------
    L_h : float
        Horizontal span (m).
    w : float
        Cable self-weight per unit length (N/m).
    H : float
        Horizontal tension at the supports (N).
    x : float, optional
        Horizontal position from one support (m). If omitted, the
        midspan sag is returned.
    """
    if L_h <= 0.0 or H <= 0.0:
        raise ValueError("L_h and H must be > 0")
    if w < 0.0:
        raise ValueError("w must be >= 0")
    arg_mid = w * L_h / (2.0 * H)
    f_mid = (H / w) * (math.cosh(arg_mid) - 1.0) if w > 0 else 0.0
    if x is None:
        return float(f_mid)
    # Measure from midspan
    arg_x = w * (x - L_h / 2.0) / H
    y = (H / w) * (math.cosh(arg_x) - 1.0) - f_mid if w > 0 else 0.0
    return float(-y)


def catenary_max_tension(
    *,
    L_h: float, w: float, H: float,
    h_chord: float = 0.0,
) -> CatenaryResult:
    """Closed-form max tension and sag for a catenary cable.

    ``T_max = sqrt(H^2 + (w L_arc/2 + V_offset)^2)``

    where ``V_offset`` is the vertical-component bias from a difference
    in support elevations ``h_chord``.

    Parameters
    ----------
    L_h : float
    w : float
    H : float
    h_chord : float, default 0.0
        Difference in elevation between the two supports (m). Positive
        if the right support is higher.
    """
    if L_h <= 0.0:
        raise ValueError("L_h must be > 0")
    if H <= 0.0:
        raise ValueError("H must be > 0")
    if w < 0.0:
        raise ValueError("w must be >= 0")
    # Arc length: L_arc = (2H/w) sinh(w L_h / (2H))   (for symmetric)
    arc = (2.0 * H / w) * math.sinh(w * L_h / (2.0 * H)) \
          if w > 0 else L_h
    # Vertical force at the higher support:
    V_high = 0.5 * w * arc + 0.5 * h_chord * w     # crude bias
    T_max = math.sqrt(H * H + V_high * V_high)
    sag = (H / w) * (math.cosh(w * L_h / (2.0 * H)) - 1.0) \
          if w > 0 else 0.0
    return CatenaryResult(
        H=float(H), T_max=float(T_max),
        sag_max=float(sag), L_arc=float(arc),
    )
